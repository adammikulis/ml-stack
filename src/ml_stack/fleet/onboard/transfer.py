"""Files from peers: served in ranges, fetched in chunks from several machines at once.

A peer is a source of bytes and nothing more. Size, digest and chunking come from the
manifest that verified under the pinned key; a chunk is kept only if it hashes to the digest
listed for it, a peer that sends a wrong one is dropped, and the finished file is hashed whole
before it is moved to staging (not installed: the caller's scan and quarantine step,
``on_staged``, decides). Resumable: chunks verified earlier are hashed again from disk before
they are trusted. The whole file plus a reserve must fit on disk before a byte is asked for.
Who may have which file is `sharing.py`: ``never`` files are not asked of a peer, ``owner`` files
go only to the owner's own devices with the licence acceptance on record.
"""

from __future__ import annotations

import base64
import contextlib
import hashlib
import http.client
import json
import queue
import shutil
import ssl
import threading
import urllib.parse
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ml_stack import macauth
from ml_stack.files import promote, sha256_file, write_json
from ml_stack.fleet import tls
from ml_stack.fleet.framing import Malformed, requested_range
from ml_stack.safenames import Unsafe, safe_filename, safe_join

from .events import BUS, Bus
from .manifest import Entry, Manifest
from .requests import Device
from .sharing import NEVER, Access, Licences, decide
from .web import Call, Listener, Reply, json_reply

__all__ = ["Downloader", "NotShareable", "PeerSource", "Settings", "Share", "ShareServer",
           "TransferError", "Withheld", "fetch_manifest", "mac_gate"]

API = "/onboard/v1"
RESERVE = 1 << 30
SEND_PIECE = 1 << 20


class TransferError(RuntimeError):
    pass


class NotShareable(TransferError):
    """The file may not be passed between peers; ``source`` is where its owner says to get it."""

    def __init__(self, name: str, source: str) -> None:
        super().__init__(f"{name} may not be shared between machines" +
                         (f"; get it from {source}" if source else ""))
        self.source = source


class Withheld(TransferError):
    """A peer refused the file under the owner's sharing rules (not a lie: it is not struck)."""


# -- serving -----------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class Share:
    """What a server offers: the files under ``root`` and the signed manifest that lists them."""

    root: Path
    manifest_raw: bytes
    manifest: Manifest
    licences: Licences | None = None
    """The owner's record of licences accepted, which ``owner`` files need."""


def mac_gate(cluster_secret: str, devices: Callable[[], Iterable[Device]] = lambda: (),
             ) -> Callable[[str, str, Any, str], Access | None]:
    """The fleet's own authentication as the gate a `ShareServer` asks, with each paired
    device able to sign with a key of its own: a request signed by the cluster secret alone is
    a member with no device identity; one signed by a device's key says which device it is."""
    def secrets_now() -> list[str]:
        return [cluster_secret, *(macauth.derive(base64.urlsafe_b64decode(d.secret))
                                  for d in devices() if d.secret and d.status == "active")]

    auth = macauth.Authenticator(secrets_now)

    def gate(method: str, target: str, headers: Any, who: str) -> Access | None:
        if not auth.check(method, target, headers, None, who).ok:
            return None
        given = dict(part.split("=", 1) for part in headers.get("Authorization", "")
                     .partition(" ")[2].split(",") if "=" in part).get("k", "")
        for d in devices():
            if d.secret and d.status == "active" and given == macauth.key_id(
                    macauth.derive(base64.urlsafe_b64decode(d.secret))):
                return Access(d.fingerprint, d.mine)
        return Access()
    return gate


def file_stream(path: Path, start: int, length: int) -> Iterator[bytes]:
    """``length`` bytes of ``path`` from ``start``, a piece at a time."""
    with path.open("rb") as fh:
        fh.seek(start)
        left = length
        while left > 0:
            piece = fh.read(min(SEND_PIECE, left))
            if not piece:
                return
            left -= len(piece)
            yield piece


def serve_file(share: Share, name: str, range_header: str, bus: Bus, access: Access) -> Reply:
    """The reply for one ranged GET of ``name``: 206 with the span, or why not."""
    try:
        entry = share.manifest.entry(safe_filename(name))
    except (KeyError, Unsafe):
        return json_reply(404, {"error": "no such file"})
    reason = decide(entry, access, share.licences)
    if reason:
        bus.emit("onboard.transfer.withheld", "notice", f"device:{access.device}",
                 file=entry.name, level=entry.sharing, reason=reason)
        return json_reply(403, {"error": reason})
    try:
        path = safe_join(share.root, entry.name)
        if not path.is_file() or path.stat().st_size != entry.size:
            raise OSError
        start, end = requested_range(range_header)
    except (Unsafe, OSError):
        return json_reply(404, {"error": "the file is not here"})
    except Malformed as bad:
        return json_reply(bad.status, {"error": bad.message})
    last = entry.size - 1 if end is None else min(end, entry.size - 1)
    if start >= entry.size or last - start + 1 > entry.chunk_size:
        return Reply(416, b'{"error":"range not satisfiable"}',
                     {"Content-Range": f"bytes */{entry.size}"}, "application/json")
    length = last - start + 1
    return Reply(206, headers={"Content-Range": f"bytes {start}-{last}/{entry.size}"},
                 stream=file_stream(path, start, length), length=length)


class ShareServer:
    """Serves the manifest and the files in it that each asking device may have. ``authenticate``
    ``(method, target, headers, client)`` says who is asking (an `Access`) or None; the
    fleet's is `mac_gate`."""

    def __init__(self, share: Share, *,
                 authenticate: Callable[[str, str, Any, str], Access | None],
                 ident: tls.Identity | None, address: tuple[str, int] = ("127.0.0.1", 0),
                 bus: Bus = BUS) -> None:
        self.share, self.authenticate, self.bus = share, authenticate, bus
        self.listener = Listener(self.dispatch, address,
                                 tls.server_context(ident) if ident else None)

    @property
    def port(self) -> int:
        return self.listener.port

    def start(self) -> ShareServer:
        self.listener.start()
        return self

    def stop(self) -> None:
        self.listener.stop()

    def __enter__(self) -> ShareServer:
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.stop()

    def dispatch(self, call: Call) -> Reply:
        access = self.authenticate(call.method, call.path, call.headers, call.client) \
            if call.method == "GET" else None
        if access is None:
            return json_reply(401, {"error": "not signed"})
        if call.path == f"{API}/manifest":
            return Reply(200, self.share.manifest_raw, content_type="application/json")
        prefix = f"{API}/files/"
        if not call.path.startswith(prefix):
            return json_reply(404, {"error": "no such path"})
        name = urllib.parse.unquote(call.path[len(prefix):].split("?")[0])
        return serve_file(self.share, name, call.headers.get("Range", ""), self.bus, access)


# -- fetching ----------------------------------------------------------------------------
@dataclass(slots=True)
class PeerSource:
    """A machine to fetch from: ``base_url`` (``https://host:port``), the MAC ``secret`` its
    requests are signed with (empty for a source that takes none), and the TLS ``context``
    that trusts that machine's one certificate."""

    base_url: str
    secret: str = ""
    context: ssl.SSLContext | None = None
    name: str = ""
    strikes: int = field(default=0, repr=False)
    lies: int = field(default=0, repr=False)
    """Answers that were wrong, as against merely missing."""
    banned: bool = field(default=False, repr=False)


def _disk_free(path: Path) -> int:
    return shutil.disk_usage(path).free


@dataclass(slots=True)
class Settings:
    """How a `Downloader` behaves."""

    reserve: int = RESERVE
    """Bytes that must stay free beyond the file."""
    workers: int = 4
    strikes: int = 2
    """Wrong answers from one peer before it is dropped."""
    timeout: float = 20.0
    free: Callable[[Path], int] = _disk_free
    bus: Bus = BUS


def fetch_manifest(peer: PeerSource, timeout: float = 20.0) -> bytes:
    """The signed manifest a peer serves, as bytes; the caller verifies it."""
    url = f"{peer.base_url}{API}/manifest"
    parts = urllib.parse.urlsplit(url)
    headers = macauth.sign(peer.secret, "GET", url, None) if peer.secret else {}
    conn = http.client.HTTPSConnection(parts.hostname or "", parts.port, timeout=timeout,
                                       context=peer.context) if parts.scheme == "https" \
        else http.client.HTTPConnection(parts.hostname or "", parts.port, timeout=timeout)
    try:
        conn.request("GET", parts.path, headers=headers)
        response = conn.getresponse()
        if response.status != 200:
            raise TransferError(f"{peer.name or peer.base_url} answered {response.status} "
                                "for the manifest")
        return response.read(8 * 1024 * 1024 + 1)
    finally:
        with contextlib.suppress(OSError):
            conn.close()


class Downloader:
    def __init__(self, manifest: Manifest, peers: list[PeerSource], staging: Path, *,
                 settings: Settings | None = None,
                 on_staged: Callable[[Path, Entry], None] | None = None) -> None:
        self.manifest, self.peers, self.staging = manifest, list(peers), Path(staging)
        self.cfg = settings or Settings()
        self.bus, self.on_staged = self.cfg.bus, on_staged
        self._lock = threading.Lock()
        self.fetched_from: dict[str, int] = {}

    def download(self, name: str) -> Path:
        """Fetch ``name`` into the staging directory; returns its path."""
        entry = self.manifest.entry(safe_filename(name))
        if entry.sharing == NEVER:
            raise NotShareable(entry.name, entry.source)
        if not self.peers:
            raise TransferError("no peer to fetch from")
        self.staging.mkdir(parents=True, exist_ok=True, mode=0o700)
        part, state = self.staging / f"{entry.name}.part", self.staging / f"{entry.name}.part.json"
        done = self._resume(entry, part, state)
        self._room(entry, len(done))
        if not part.exists():
            with part.open("wb") as fh:
                fh.truncate(entry.size)
        failure = self._fetch(entry, part, state, done)
        if failure is not None:
            raise failure
        if len(done) != len(entry.chunks):
            raise TransferError(f"{entry.name}: {len(entry.chunks) - len(done)} chunks missing")
        if sha256_file(part) != entry.sha256:
            self.bus.emit("onboard.transfer.bad_file", "critical", "", file=entry.name)
            part.unlink(missing_ok=True)
            state.unlink(missing_ok=True)
            raise TransferError(f"{entry.name} does not hash to the manifest's digest")
        target = promote(part, self.staging / entry.name)
        state.unlink(missing_ok=True)
        self.bus.emit("onboard.transfer.staged", "info", "", file=entry.name, size=entry.size)
        if self.on_staged is not None:
            self.on_staged(target, entry)
        return target

    def _room(self, entry: Entry, have: int) -> None:
        need = (len(entry.chunks) - have) * entry.chunk_size
        free = self.cfg.free(self.staging)
        if free < min(need, entry.size) + self.cfg.reserve:
            raise TransferError(f"{entry.name} needs about {entry.size >> 20} MiB and "
                                f"{self.cfg.reserve >> 20} MiB of reserve; {free >> 20} MiB "
                                "is free")

    def _fetch(self, entry: Entry, part: Path, state: Path,
               done: set[int]) -> TransferError | None:
        """Fetch every chunk not in ``done`` with several workers; returns why it stopped, or
        None."""
        todo: queue.Queue[int] = queue.Queue()
        for index in range(len(entry.chunks)):
            if index not in done:
                todo.put(index)
        failure: list[TransferError] = []

        def work() -> None:
            while not failure:
                try:
                    index = todo.get_nowait()
                except queue.Empty:
                    return
                try:
                    data, peer = self._chunk(entry, index)
                except TransferError as exc:
                    failure.append(exc)
                    return
                with self._lock:
                    with part.open("r+b") as fh:
                        fh.seek(index * entry.chunk_size)
                        fh.write(data)
                    done.add(index)
                    who = peer.name or peer.base_url
                    self.fetched_from[who] = self.fetched_from.get(who, 0) + len(data)
                    if len(done) % 16 == 0:
                        self._remember(entry, state, done)

        threads = [threading.Thread(target=work, daemon=True)
                   for _ in range(min(self.cfg.workers, max(1, todo.qsize())))]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self._remember(entry, state, done)
        return failure[0] if failure else None

    def _remember(self, entry: Entry, state: Path, done: set[int]) -> None:
        write_json(state, {"schema_version": 1, "sha256": entry.sha256, "done": sorted(done)},
                   indent=None)

    def _resume(self, entry: Entry, part: Path, state: Path) -> set[int]:
        """The chunks of an earlier run that are still good on disk."""
        try:
            held = json.loads(state.read_text())
            claimed = {int(i) for i in held["done"]}
            same = held["sha256"] == entry.sha256 and part.stat().st_size == entry.size
        except (OSError, ValueError, KeyError, TypeError):
            part.unlink(missing_ok=True)
            return set()
        if not same:
            part.unlink(missing_ok=True)
            return set()
        good: set[int] = set()
        with part.open("rb") as fh:
            for index in sorted(claimed):
                if not 0 <= index < len(entry.chunks):
                    continue
                fh.seek(index * entry.chunk_size)
                piece = fh.read(entry.chunk_size)
                if hashlib.sha256(piece).hexdigest() == entry.chunks[index]:
                    good.add(index)
        return good

    # -- one chunk --
    def _chunk(self, entry: Entry, index: int) -> tuple[bytes, PeerSource]:
        start = index * entry.chunk_size
        want = min(entry.chunk_size, entry.size - start)
        order = self.peers[index % len(self.peers):] + self.peers[:index % len(self.peers)]
        withheld = ""
        for peer in order:
            if peer.banned:
                continue
            try:
                data = self._get(peer, entry, start, want)
            except (OSError, http.client.HTTPException, ssl.SSLError) as exc:
                self._strike(peer, entry, f"unreachable: {type(exc).__name__}", hard=False)
                continue
            except Withheld as exc:
                withheld = str(exc)
                continue
            except TransferError as exc:
                self._strike(peer, entry, str(exc), hard=True)
                continue
            if hashlib.sha256(data).hexdigest() != entry.chunks[index]:
                self.bus.emit("onboard.transfer.bad_chunk", "critical", "",
                              file=entry.name, chunk=index, peer=peer.name or peer.base_url)
                self._strike(peer, entry, "sent a chunk that fails its digest", hard=True)
                continue
            return data, peer
        if withheld:
            raise Withheld(f"{entry.name}: {withheld}")
        raise TransferError(f"{entry.name}: no peer could supply chunk {index}")

    def _strike(self, peer: PeerSource, entry: Entry, why: str, *, hard: bool) -> None:
        with self._lock:
            peer.strikes += 1
            if hard:
                peer.lies += 1
            if (peer.lies >= self.cfg.strikes or peer.strikes >= 3 * self.cfg.strikes) \
                    and not peer.banned:
                peer.banned = True
                self.bus.emit("onboard.transfer.peer_dropped", "warning", "",
                              peer=peer.name or peer.base_url, file=entry.name, reason=why)

    def _get(self, peer: PeerSource, entry: Entry, start: int, want: int) -> bytes:
        url = f"{peer.base_url}{API}/files/{urllib.parse.quote(entry.name)}"
        parts = urllib.parse.urlsplit(url)
        headers = {"Range": f"bytes={start}-{start + want - 1}"}
        if peer.secret:
            headers.update(macauth.sign(peer.secret, "GET", url, None))
        if parts.scheme == "https":
            conn: http.client.HTTPConnection = http.client.HTTPSConnection(
                parts.hostname or "", parts.port, timeout=self.cfg.timeout, context=peer.context)
        else:
            conn = http.client.HTTPConnection(parts.hostname or "", parts.port,
                                              timeout=self.cfg.timeout)
        try:
            conn.request("GET", parts.path, headers=headers)
            response = conn.getresponse()
            if response.status == 403:
                try:
                    why = str(json.loads(response.read(2048)).get("error", "refused"))[:200]
                except ValueError:
                    why = "refused"
                raise Withheld(why)
            if response.status != 206:
                raise TransferError(f"answered {response.status}")
            if response.getheader("Content-Range") != \
                    f"bytes {start}-{start + want - 1}/{entry.size}":
                raise TransferError("answered a different range than asked")
            data = response.read(want + 1)
        finally:
            with contextlib.suppress(OSError):
                conn.close()
        if len(data) != want:
            raise TransferError("sent the wrong number of bytes")
        return data
