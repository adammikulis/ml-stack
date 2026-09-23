"""Model files this machine holds, and getting one it does not."""

from __future__ import annotations

import hashlib
import os
import secrets
import shutil
import threading
import time
import urllib.error
import urllib.parse
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ml_stack import hub
from ml_stack.files import UNVERSIONED, promote, read_json, version_of, versioned, write_json
from ml_stack.http import ServerError, ServerUnreachable, open_stream

from .weights import ModelError, resolve

__all__ = ["CHUNK", "Downloads", "Getting", "Model", "Models", "caches",
           "draft_beside", "holding", "sized"]

CHUNK = 1 << 20

#: 1 -- the url a partial download came from and its validator.
STAMP_VERSION = 1
MIN_SIZE = 1 << 20
# A download in progress writes continuously, so a part file untouched for this
# long belongs to one that stopped.
STALE_PART_S = 3600.0


def _read_stamp(stamp: Path) -> dict[str, Any]:
    """What a half-finished download recorded about where it came from.

    A stamp with no version key was written before the key existed and carries the same
    ``url`` and ``validator``; one from a version this code does not know is discarded, so
    the download starts again rather than resuming on a guess.
    """
    raw = read_json(stamp, None)
    if not isinstance(raw, dict):
        return {}
    return raw if version_of(raw) in (UNVERSIONED, STAMP_VERSION) else {}


def _write_stamp(stamp: Path, url: str, headers: Any) -> None:
    validator = headers.get("ETag") or headers.get("Last-Modified") or ""
    try:
        write_json(stamp, versioned({"url": url, "validator": validator}, STAMP_VERSION),
                   indent=None)
    except OSError:
        pass


@dataclass(frozen=True, slots=True)
class Model:
    name: str
    path: Path
    size: int
    modified: float

    def public(self) -> dict[str, Any]:
        return {"name": self.name, "size": self.size, "modified": self.modified}


WEIGHTS = (".gguf", ".safetensors")


def holding(directory: Path | str) -> tuple[int, int]:
    """How many weight files (GGUF, safetensors) are under ``directory``, and their bytes,
    read through symlinks."""
    files = total = 0
    todo = [str(Path(directory).expanduser())]
    while todo:
        try:
            with os.scandir(todo.pop()) as entries:
                for entry in entries:
                    try:
                        if entry.is_dir(follow_symlinks=False):
                            todo.append(entry.path)
                            continue
                        if entry.name.lower().endswith(WEIGHTS) and entry.is_file():
                            files += 1
                            total += os.stat(entry.path).st_size
                    except OSError:
                        continue
        except OSError:
            continue
    return files, total


def caches(root: Path | str) -> list[tuple[Path, int, int]]:
    """Each model root that exists on this machine, with its weight-file count and bytes."""
    out = []
    for where in hub.default_roots(root):
        if where.is_dir():
            files, total = holding(where)
            out.append((where, files, total))
    return out


def sized(count: int) -> str:
    """Bytes as a person reads them: ``86.2G``, ``412M``."""
    if count >= 2**30:
        return f"{count / 2**30:.1f}G"
    return f"{count / 2**20:.0f}M"


@dataclass
class Models:
    """The model files on this machine."""

    roots: list[Path]
    store: Path
    _digests: dict[tuple[str, int, int], str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.store = Path(self.store).expanduser()
        self.roots = [Path(r).expanduser() for r in self.roots]

    def all(self) -> list[Model]:
        seen: dict[str, Model] = {}
        for path in hub.weight_paths(self.roots):
            if hub.DRAFT_MARK in path.suffixes or path.name in seen:
                continue
            try:
                stat = path.stat()
            except OSError:
                continue
            if stat.st_size >= MIN_SIZE:
                seen[path.name] = Model(path.name, path, stat.st_size, stat.st_mtime)
        return sorted(seen.values(), key=lambda m: m.name.lower())

    def find(self, name: str) -> Model | None:
        """The model file this machine holds under that name, or ``None``."""
        found = hub.located(name.strip(), roots=self.roots, loose=True, min_size=MIN_SIZE)
        if found is None:
            return None
        stat = found.stat()
        return Model(found.name, found, stat.st_size, stat.st_mtime)

    def find_draft(self, name: str) -> Model | None:
        """A draft by its exact filename. Drafts are fetched, never listed."""
        for root in self.roots:
            path = root / Path(name).name
            if hub.DRAFT_MARK not in path.suffixes or not path.is_file():
                continue
            stat = path.stat()
            return Model(path.name, path, stat.st_size, stat.st_mtime)
        return None

    def digest(self, model: Model) -> str:
        """sha256, cached against size and mtime so a large file is read once."""
        stat = model.path.stat()
        key = (str(model.path), stat.st_size, stat.st_mtime_ns)
        found = self._digests.get(key)
        if found:
            return found
        h = hashlib.sha256()
        with model.path.open("rb") as fh:
            while True:
                block = fh.read(CHUNK)
                if not block:
                    break
                h.update(block)
        self._digests[key] = h.hexdigest()
        return self._digests[key]

    def public(self, limit: int = 24) -> list[dict[str, Any]]:
        """What goes in the beacon. Names and sizes only -- a beacon is a UDP packet,
        and a full path is needless disclosure."""
        return [m.public() for m in self.all()[:limit]]

    def beacon(self, limit: int = 24) -> dict[str, Any]:
        """The models for the beacon, and how many this machine holds. A peer reads the
        ones past ``limit`` from ``/models``."""
        held = self.all()
        return {"models": [m.public() for m in held[:limit]],
                "models_total": len(held)}

    # -- getting one ----------------------------------------------------
    def where(self, name: str, key: bytes, *, timeout_s: float = 2.0
              ) -> list[tuple[str, str, int]]:
        """Machines on this network holding a model. (peer, base_url, size)."""
        from .discovery import discover

        needle = name.strip().lower()
        out = []
        for beacon in discover(key, timeout_s=timeout_s):
            for row in (beacon.device.get("models") or []):
                if needle in str(row.get("name", "")).lower():
                    out.append((beacon.name, beacon.base_url, int(row.get("size") or 0)))
                    break
        return out

    def ensure(self, name: str, *, source: str = "", key: bytes | None = None,
               on_progress: "Callable[[int, int], None] | None" = None,
               on_note: "Callable[[str], None] | None" = None,
               autodownload: bool = True) -> Model:
        """Make sure this machine holds a model, preferring one on the network."""
        found = self.find(name)
        if found:
            return found
        if not autodownload:
            raise ModelError(
                f"{name} is not on this machine, and automatic downloading is off")

        if key is not None:
            for peer_name, base_url, _size in self.where(name, key):
                if on_note:
                    on_note(f"Copying {name} from {peer_name}")
                try:
                    return self._from_peer(name, base_url, key, on_progress)
                except (ModelError, OSError):
                    continue

        if not source:
            raise ModelError(
                f"no machine on this network has {name}, and no download was given")
        if on_note:
            on_note(f"Downloading {name}")
        return self._from_internet(name, source, on_progress)

    def _from_peer(self, name: str, base_url: str, key: bytes,
                   on_progress: Any) -> Model:
        from .discovery import derive_token
        from .remote import Peer

        peer = Peer(base_url, derive_token(key))
        holds = peer.models()
        wanted = name.strip().lower()
        match = next((m for m in holds
                      if str(m.get("name", "")).lower() == wanted), None)
        match = match or next((m for m in holds
                               if wanted in str(m.get("name", "")).lower()), None)
        if match is None:
            raise ModelError(f"{base_url} no longer has {name}")
        self.store.mkdir(parents=True, exist_ok=True)
        target = self.store / str(match["name"])
        peer.pull(str(match["name"]), target, on_progress=on_progress,
                  route="/models/")
        stat = target.stat()
        return Model(target.name, target, stat.st_size, stat.st_mtime)

    def _from_internet(self, name: str, source: str, on_progress: Any) -> Model:
        from .remote import range_total

        url = resolve(source)
        self.store.mkdir(parents=True, exist_ok=True)
        # The name that was asked for wins: saving it under whatever the URL happened
        # to call it means find() will not match it afterwards.
        wanted = Path(name).name
        if Path(wanted).suffix.lower() not in hub.WEIGHT_SUFFIXES:
            wanted = Path(urllib.parse.urlparse(url).path).name or wanted
        target = self.store / wanted
        partial = target.with_suffix(target.suffix + ".part")
        stamp = Path(str(partial) + ".from")

        start = partial.stat().st_size if partial.exists() else 0
        origin = _read_stamp(stamp)
        if start and origin.get("url") not in (None, url):
            partial.unlink(missing_ok=True)
            stamp.unlink(missing_ok=True)
            start, origin = 0, {}

        headers: dict[str, str] = {}
        if start:
            headers["Range"] = f"bytes={start}-"
            if origin.get("validator"):
                headers["If-Range"] = str(origin["validator"])
        try:
            response = open_stream(url, headers=headers, timeout=120)
        except ServerUnreachable as exc:
            raise ModelError(f"could not download {name}: {exc}") from None
        except ServerError as exc:
            if exc.status == 416 and start:
                size = range_total(exc.headers.get("Content-Range", ""))
                if size is not None and size == start:
                    promote(partial, target)
                    stamp.unlink(missing_ok=True)
                    stat = target.stat()
                    return Model(target.name, target, stat.st_size, stat.st_mtime)
                partial.unlink(missing_ok=True)
                stamp.unlink(missing_ok=True)
                raise ModelError(
                    f"{name}: the part here is {start} bytes but the file is "
                    f"{size}; discarded it, ask again") from None
            raise ModelError(f"could not download {name}: {exc.status}") from None

        # A server that does not honour Range answers 200 with the whole file.
        if start and response.status != 206:
            start = 0
        if start:
            total = range_total(response.headers.get("Content-Range", "")) or 0
        else:
            total = int(response.headers.get("Content-Length") or 0)

        _write_stamp(stamp, url, response.headers)
        done = start
        with response, partial.open("ab" if start else "wb") as fh:
            while True:
                block = response.read(CHUNK)
                if not block:
                    break
                fh.write(block)
                done += len(block)
                if on_progress:
                    on_progress(done, total)
        if total and partial.stat().st_size != total:
            raise ModelError(
                f"{name}: got {partial.stat().st_size} of {total} bytes; "
                f"left {partial.name} to resume from")
        promote(partial, target)
        stamp.unlink(missing_ok=True)
        stat = target.stat()
        return Model(target.name, target, stat.st_size, stat.st_mtime)

    def ensure_draft(self, model: Model, source: str, *, key: bytes | None = None,
                     on_progress: "Callable[[int, int], None] | None" = None) -> Path:
        """Fetch the small model that guesses ahead for ``model``, beside it.

        Taken from a machine on this network if one holds it, as the model itself is.
        """
        beside = model.path.with_suffix(hub.DRAFT_MARK + model.path.suffix)
        if beside.is_file():
            return beside
        if key is not None and self._draft_from_peers(model, beside, key, on_progress):
            return beside
        got = self._from_internet(beside.name, source, on_progress)
        if got.path != beside:
            promote(got.path, beside)
        return beside

    def _draft_from_peers(self, model: Model, beside: Path, key: bytes,
                          on_progress: Any) -> bool:
        """Ask the machines holding ``model`` for the draft that sits beside it."""
        from .discovery import derive_token
        from .remote import Peer, PeerError

        for _name, base_url, _size in self.where(model.name, key):
            try:
                Peer(base_url, derive_token(key)).pull(
                    beside.name, beside, on_progress=on_progress, route="/models/")
            except (PeerError, OSError):
                continue
            return True
        return False

    def remove(self, name: str) -> bool:
        found = self.find(name)
        if found is None or self.store not in found.path.parents:
            return False
        found.path.unlink(missing_ok=True)
        return True

    def unfinished(self, *, stale_s: float = STALE_PART_S) -> list[dict[str, Any]]:
        """Part files left by downloads that stopped, newest first."""
        import time

        if not self.store.exists():
            return []
        now = time.time()
        out = []
        for path in self.store.glob("*.part"):
            try:
                stat = path.stat()
            except OSError:
                continue
            if now - stat.st_mtime < stale_s:
                continue
            out.append({"name": path.name, "size": stat.st_size,
                        "modified": stat.st_mtime})
        out.sort(key=lambda r: r["modified"], reverse=True)
        return out

    def discard(self, name: str = "") -> list[str]:
        """Delete a part file and what it recorded, or every stale one. Returns names."""
        wanted = [r["name"] for r in self.unfinished()] if not name else [Path(name).name]
        gone = []
        for part in wanted:
            path = self.store / part
            if path.suffix != ".part" or self.store not in path.parents:
                continue
            if not path.exists():
                continue
            path.unlink(missing_ok=True)
            Path(str(path) + ".from").unlink(missing_ok=True)
            gone.append(part)
        return gone

    def free_gb(self) -> float:
        try:
            self.store.mkdir(parents=True, exist_ok=True)
            return round(shutil.disk_usage(self.store).free / 2**30, 1)
        except OSError:
            return 0.0


def draft_beside(model: Path) -> Path | None:
    """The small model kept next to ``model`` to guess ahead with, if one was got."""
    beside = model.with_suffix(hub.DRAFT_MARK + model.suffix)
    return beside if beside.is_file() else None




@dataclass
class Getting:
    """One model being fetched, and how far it has got."""

    id: str
    name: str
    source: str = ""
    state: str = "getting"          # getting | done | failed
    note: str = ""
    done: int = 0
    total: int = 0
    error: str = ""
    started_at: float = field(default_factory=time.time)
    finished_at: float = 0.0

    def public(self) -> dict[str, Any]:
        return {"id": self.id, "name": self.name, "source": self.source,
                "state": self.state, "note": self.note, "done": self.done,
                "total": self.total, "error": self.error,
                "started_at": self.started_at, "finished_at": self.finished_at}


class Downloads:
    """Model fetches running in the background, so a big one does not hold a request."""

    KEEP_S = 300.0

    def __init__(self, models: "Models", *, slots: int = 1) -> None:
        self.models = models
        self.getting: dict[str, Getting] = {}
        self._lock = threading.Lock()
        self._sem = threading.Semaphore(slots)

    def start(self, name: str, *, source: str = "", key: bytes | None = None,
              autodownload: bool = True, draft: str = "") -> Getting:
        with self._lock:
            for row in self.getting.values():
                if row.state == "getting" and row.name == name:
                    return row
        row = Getting(id=f"{int(time.time())}-{secrets.token_hex(3)}",
                      name=name, source=source)
        with self._lock:
            self.getting[row.id] = row
        threading.Thread(target=self._run, args=(row, key, autodownload, draft),
                         daemon=True, name=f"get-{row.id}").start()
        return row

    def _run(self, row: Getting, key: bytes | None, autodownload: bool,
             draft: str = "") -> None:
        with self._sem:
            try:
                def progress(done: int, total: int) -> None:
                    row.done, row.total = done, total

                def note(text: str) -> None:
                    row.note = text

                got = self.models.ensure(row.name, source=row.source, key=key,
                                         on_progress=progress, on_note=note,
                                         autodownload=autodownload)
                if draft:
                    row.note = f"Getting the draft for {got.name}"
                    try:
                        self.models.ensure_draft(got, draft, key=key,
                                                 on_progress=progress)
                    except (ModelError, OSError):
                        pass          # a model without its draft still runs
                row.state = "done"
                row.name = got.name
                row.done = row.total = got.size
            except Exception as exc:                  # noqa: BLE001
                row.state = "failed"
                row.error = str(exc)
            finally:
                row.finished_at = time.time()

    def active(self) -> list[Getting]:
        """Everything still running, plus anything that finished recently."""
        now = time.time()
        with self._lock:
            for key in [k for k, r in self.getting.items()
                        if r.finished_at and now - r.finished_at > self.KEEP_S]:
                del self.getting[key]
            rows = sorted(self.getting.values(), key=lambda r: r.started_at)
        return rows


