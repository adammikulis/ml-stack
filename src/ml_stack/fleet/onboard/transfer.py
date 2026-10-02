"""Files from peers: served in ranges, fetched in chunks from several machines at once, every
chunk checked against a signed manifest before it is kept.

A peer is a source of bytes and nothing more. What it says about the file (its size, its
digest, how it is cut up) is taken from the manifest that verified under the pinned key, and
whatever the peer sends is accepted only if it hashes to the digest listed for that chunk. A
peer that sends a wrong chunk is not retried for that chunk and is dropped after a second;
the chunk is fetched from another peer. The finished file is hashed whole once more before it
is moved to the staging directory -- it is *staged*, not installed: the caller's scan and
quarantine step (``on_staged``) decides whether it is used.

Resumable: the chunks already verified are listed beside the partial file, and on the next
run each of them is hashed again from disk before it is trusted, so a partial file damaged
between runs costs a re-fetch of the damaged chunks and nothing worse.

Disk: the whole file's size plus a reserve must fit before a byte is requested.

Licences: an entry marked ``shareable: false`` (a gated model, say) is never asked of a peer
and never served to one; `NotShareable` names where its own ``source`` says to get it.
"""

from __future__ import annotations

import contextlib
import hashlib
import http.client
import json
import queue
import shutil
import ssl
import threading
import urllib.parse
from collections.abc import Callable
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from typing import Any

from ml_stack import macauth
from ml_stack.files import promote, sha256_file, write_json
from ml_stack.fleet import tls
from ml_stack.fleet.framing import Limited, LimitedServer, Malformed, requested_range
from ml_stack.safenames import Unsafe, safe_filename, safe_join

from .events import BUS, Bus
from .manifest import Entry, Manifest

__all__ = ["Downloader", "NotShareable", "PeerSource", "ShareServer", "TransferError"]

API = "/onboard/v1"
RESERVE = 1 << 30


class TransferError(RuntimeError):
    pass


class NotShareable(TransferError):
    """The file may not be passed between peers; ``source`` is where its owner says to get it."""

    def __init__(self, name: str, source: str) -> None:
        super().__init__(f"{name} may not be shared between machines" +
                         (f"; get it from {source}" if source else ""))
        self.source = source


# -- serving -----------------------------------------------------------------------------
class ShareServer:
    """Serves the manifest and the shareable files in it from ``root`` to authenticated peers.

    ``authenticate(method, target, headers, who)`` answers whether a request may proceed;
    the fleet's own is `macauth.Authenticator` over the cluster secret (see `mac_gate`)."""

    def __init__(self, root: Path, manifest_raw: bytes, manifest: Manifest, *,
                 authenticate: Callable[[str, str, Any, str], bool], ident: tls.Identity | None,
                 host: str = "127.0.0.1", port: int = 0, bus: Bus = BUS) -> None:
        self.root, self.manifest_raw, self.manifest = Path(root), manifest_raw, manifest
        self.authenticate, self.bus = authenticate, bus
        self.served = 0
        self.httpd = LimitedServer((host, port), _share_handler(self),
                                   tls=tls.server_context(ident) if ident else None)
        self._thread: threading.Thread | None = None

    @property
    def port(self) -> int:
        return int(self.httpd.server_address[1])

    def start(self) -> ShareServer:
        self._thread = threading.Thread(target=self.httpd.serve_forever, daemon=True,
                                        name="onboard-share")
        self._thread.start()
        return self

    def stop(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()

    def __enter__(self) -> ShareServer:
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.stop()


def mac_gate(auth: macauth.Authenticator) -> Callable[[str, str, Any, str], bool]:
    def gate(method: str, target: str, headers: Any, who: str) -> bool:
        return auth.check(method, target, headers, None, who).ok
    return gate


def _share_handler(server: ShareServer) -> type[BaseHTTPRequestHandler]:
    class Handler(Limited, BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *_args: object) -> None:
            return

        def _say(self, status: int, body: bytes = b"", headers: dict[str, str] | None = None
                 ) -> None:
            self.send_response(status)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Content-Type", "application/octet-stream")
            for key, value in (headers or {}).items():
                self.send_header(key, value)
            self.end_headers()
            if body and self.command != "HEAD":
                self.wfile.write(body)

        def do_GET(self) -> None:
            who = self.client_address[0]
            if not server.authenticate("GET", self.path, self.headers, who):
                self._say(401, b'{"error":"not signed"}')
                return
            if self.path == f"{API}/manifest":
                self._say(200, server.manifest_raw)
                return
            prefix = f"{API}/files/"
            if not self.path.startswith(prefix):
                self._say(404, b'{"error":"no such path"}')
                return
            self._file(urllib.parse.unquote(self.path[len(prefix):].split("?")[0]), who)

        def _file(self, name: str, who: str) -> None:
            try:
                entry = server.manifest.entry(safe_filename(name))
            except (KeyError, Unsafe):
                self._say(404, b'{"error":"no such file"}')
                return
            if not entry.shareable:
                server.bus.emit("onboard.transfer.unshareable_asked", "notice", "",
                                file=entry.name, peer=who)
                self._say(403, b'{"error":"this file may not be shared"}')
                return
            try:
                path = safe_join(server.root, entry.name)
                if not path.is_file() or path.stat().st_size != entry.size:
                    raise OSError
                start, end = requested_range(self.headers.get("Range", ""))
            except (Unsafe, OSError):
                self._say(404, b'{"error":"the file is not here"}')
                return
            except Malformed as bad:
                self._say(bad.status, json.dumps({"error": bad.message}).encode())
                return
            last = entry.size - 1 if end is None else min(end, entry.size - 1)
            if start >= entry.size or last - start + 1 > entry.chunk_size * 4:
                self._say(416, b'{"error":"range not satisfiable"}',
                          {"Content-Range": f"bytes */{entry.size}"})
                return
            with path.open("rb") as fh:
                fh.seek(start)
                data = fh.read(last - start + 1)
            server.served += len(data)
            self._say(206, data, {"Content-Range": f"bytes {start}-{start + len(data) - 1}/"
                                                   f"{entry.size}"})

    return Handler


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


class Downloader:
    def __init__(self, manifest: Manifest, peers: list[PeerSource], staging: Path, *,
                 reserve: int = RESERVE, workers: int = 4, strikes: int = 2,
                 on_staged: Callable[[Path, Entry], None] | None = None, bus: Bus = BUS,
                 timeout: float = 20.0, free: Callable[[Path], int] | None = None) -> None:
        self.manifest, self.peers, self.staging = manifest, list(peers), Path(staging)
        self.reserve, self.workers, self.strikes = reserve, max(1, workers), strikes
        self.on_staged, self.bus, self.timeout = on_staged, bus, timeout
        self._free = free or (lambda p: shutil.disk_usage(p).free)
        self._lock = threading.Lock()
        self.fetched_from: dict[str, int] = {}

    # -- one file --
    def download(self, name: str) -> Path:
        """Fetch ``name`` into the staging directory; returns its path."""
        entry = self.manifest.entry(safe_filename(name))
        if not entry.shareable:
            raise NotShareable(entry.name, entry.source)
        if not self.peers:
            raise TransferError("no peer to fetch from")
        self.staging.mkdir(parents=True, exist_ok=True, mode=0o700)
        part, state = self.staging / f"{entry.name}.part", self.staging / f"{entry.name}.part.json"
        target = self.staging / entry.name
        done = self._resume(entry, part, state)
        need = (len(entry.chunks) - len(done)) * entry.chunk_size
        free = self._free(self.staging)
        if free < min(need, entry.size) + self.reserve:
            raise TransferError(f"{entry.name} needs about {entry.size >> 20} MiB and "
                                f"{self.reserve >> 20} MiB of reserve; {free >> 20} MiB is free")
        if not part.exists():
            with part.open("wb") as fh:
                fh.truncate(entry.size)
        todo: queue.Queue[int] = queue.Queue()
        for index in range(len(entry.chunks)):
            if index not in done:
                todo.put(index)
        failure: list[str] = []

        def work() -> None:
            while not failure:
                try:
                    index = todo.get_nowait()
                except queue.Empty:
                    return
                try:
                    data, peer = self._chunk(entry, index)
                except TransferError as exc:
                    failure.append(str(exc))
                    return
                with self._lock:
                    with part.open("r+b") as fh:
                        fh.seek(index * entry.chunk_size)
                        fh.write(data)
                    done.add(index)
                    self.fetched_from[peer.name or peer.base_url] = \
                        self.fetched_from.get(peer.name or peer.base_url, 0) + len(data)
                    if len(done) % 16 == 0:
                        self._remember(entry, state, done)

        threads = [threading.Thread(target=work, daemon=True)
                   for _ in range(min(self.workers, max(1, todo.qsize())))]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self._remember(entry, state, done)
        if failure:
            raise TransferError(failure[0])
        if len(done) != len(entry.chunks):
            raise TransferError(f"{entry.name}: {len(entry.chunks) - len(done)} chunks missing")
        if sha256_file(part) != entry.sha256:
            self.bus.emit("onboard.transfer.bad_file", "critical", "", file=entry.name)
            part.unlink(missing_ok=True)
            state.unlink(missing_ok=True)
            raise TransferError(f"{entry.name} does not hash to the manifest's digest")
        promote(part, target)
        state.unlink(missing_ok=True)
        self.bus.emit("onboard.transfer.staged", "info", "", file=entry.name, size=entry.size)
        if self.on_staged is not None:
            self.on_staged(target, entry)
        return target

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
        for peer in order:
            if peer.banned:
                continue
            try:
                data = self._get(peer, entry, start, want)
            except (OSError, http.client.HTTPException, ssl.SSLError) as exc:
                self._strike(peer, entry, f"unreachable: {type(exc).__name__}", hard=False)
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
        raise TransferError(f"{entry.name}: no peer could supply chunk {index}")

    def _strike(self, peer: PeerSource, entry: Entry, why: str, *, hard: bool) -> None:
        with self._lock:
            peer.strikes += 1
            if hard:
                peer.lies += 1
            if (peer.lies >= self.strikes or peer.strikes >= 3 * self.strikes) \
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
                parts.hostname or "", parts.port, timeout=self.timeout, context=peer.context)
        else:
            conn = http.client.HTTPConnection(parts.hostname or "", parts.port,
                                              timeout=self.timeout)
        try:
            conn.request("GET", parts.path, headers=headers)
            response = conn.getresponse()
            if response.status != 206:
                raise TransferError(f"answered {response.status}")
            if response.getheader("Content-Range") != f"bytes {start}-{start + want - 1}/{entry.size}":
                raise TransferError("answered a different range than asked")
            data = response.read(want + 1)
        finally:
            with contextlib.suppress(OSError):
                conn.close()
        if len(data) != want:
            raise TransferError("sent the wrong number of bytes")
        return data
