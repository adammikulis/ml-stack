"""Downloading a model from a Hugging Face endpoint, resumably, with progress.

``pull`` takes ``hf:owner/repo/file.gguf`` (every shard of that build comes down),
``hf:owner/repo:Q4_K_M`` or a folder-less file reference, and returns the path to serve.
A transfer interrupted by a cancel, a dropped connection or a crash continues from the
bytes already on disk.
"""

from __future__ import annotations

import hashlib
import shutil
import threading
import time
import urllib.parse
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from ml_stack import files, home, http, hub, lock
from ml_stack.hub import remote
from ml_stack.hub.remote import GatedRepo, NotFound, RemoteFile
from ml_stack.units import human_bytes

CHUNK = 1 << 20
MARGIN = 256 * 1024 * 1024
"""Free space left over after a download."""
TRIES = 3

__all__ = ["CancelToken", "Cancelled", "ChecksumMismatch", "GatedRepo", "NotEnoughSpace",
           "NotFound", "Progress", "plan", "pull"]


class Cancelled(Exception):
    """The caller's token was set. The partial file is kept for the next pull."""


class NotEnoughSpace(OSError):
    """The destination disk cannot hold the download."""


class ChecksumMismatch(OSError):
    """A finished file's sha256 is not the one the endpoint lists."""


class CancelToken:
    """A flag a UI thread sets to stop a pull running on another."""

    def __init__(self) -> None:
        self._flag = threading.Event()

    def cancel(self) -> None:
        self._flag.set()

    @property
    def cancelled(self) -> bool:
        return self._flag.is_set()


@dataclass(frozen=True, slots=True)
class Progress:
    """One progress report. ``phase`` is waiting, downloading, verifying or done; the byte
    counts cover every file of the pull, ``file_bytes`` the current one."""

    ref: str
    phase: str
    file: str
    file_index: int
    files_total: int
    done_bytes: int
    total_bytes: int
    file_bytes: int = 0
    file_total: int = 0
    bytes_per_second: float = 0.0

    @property
    def fraction(self) -> float:
        return self.done_bytes / self.total_bytes if self.total_bytes else 0.0


Report = Callable[[Progress], None]


def destination(dest: str | Path | None, ref: remote.Ref) -> Path:
    """The folder files land in: ``dest``, or the store's ``models/owner/repo``."""
    if dest is not None:
        return home.expand(dest)
    owner, name = ref.repo.split("/")
    return home.state("models", owner, name)


def plan(ref: str) -> tuple[remote.Ref, list[RemoteFile]]:
    """The reference parsed and the files it means, listed on the endpoint.

    A reference naming a repository only is resolved to the largest build under 85% of
    this machine's room when it is known, else to ``Q4_K_M``.
    """
    parsed = remote.parse(ref)
    listed = remote.listing(parsed.repo, parsed.revision)
    chosen = remote.members(listed, parsed)
    if not chosen and not parsed.file and not parsed.quant:
        chosen = _default_build(listed)
    if not chosen:
        held = sorted({f.build for f in listed if f.path.lower().endswith(".gguf")
                       and not f.companion})
        raise NotFound(f"{parsed.repo} has no {parsed.file or parsed.quant or 'model file'}"
                       + (f"; it holds {', '.join(held)}" if held else ""))
    return parsed, chosen


def _default_build(listed: list[RemoteFile]) -> list[RemoteFile]:
    models = [f for f in listed if f.path.lower().endswith(".gguf") and not f.companion]
    builds: dict[str, list[RemoteFile]] = {}
    for one in models:
        builds.setdefault(one.build, []).append(one)
    room = hub.room()
    fits = [b for b in builds.values() if not room or sum(f.size for f in b) < room * 0.85]
    pool = fits or list(builds.values())
    if not pool:
        return []
    named = [b for b in pool if "Q4_K_M" in b[0].path.upper()]
    pick = named[0] if named else max(pool, key=lambda b: sum(f.size for f in b))
    return sorted(pick, key=lambda f: f.path)


def _headers(offset: int) -> dict[str, str]:
    return {"Range": f"bytes={offset}-"} if offset else {}


def _source(parsed: remote.Ref, one: RemoteFile, auth: str) -> tuple[str, str]:
    """``(url, token to send there)``: the redirect target, without the token, when the
    endpoint hands out another host, else the endpoint's own address."""
    quoted = urllib.parse.quote(one.path)
    rev = urllib.parse.quote(parsed.revision, safe="")
    url = f"{remote.endpoint()}/{parsed.repo}/resolve/{rev}/{quoted}"
    try:
        first = http.head_once(url, token=auth)
    except http.ServerError as exc:
        if exc.status in (401, 403):
            raise GatedRepo(remote.hint(parsed.repo, exc.status)) from exc
        if exc.status == 404:
            raise NotFound(f"{parsed.repo} has no {one.path}") from exc
        raise
    where = first.headers.get("Location") if 300 <= first.status < 400 else ""
    if not where:
        return url, auth
    target = urllib.parse.urljoin(url, where)
    if urllib.parse.urlsplit(target).netloc != urllib.parse.urlsplit(url).netloc:
        http.check(target)
        return target, ""
    return target, auth


class _Meter:
    """Bytes done across the whole pull, reported at most ten times a second."""

    def __init__(self, ref: str, total: int, count: int, report: Report | None) -> None:
        self.ref, self.total, self.count, self.report = ref, total, count, report
        self.base, self.index, self.last, self.began = 0, 0, 0.0, time.monotonic()
        self.resumed = 0

    def send(self, phase: str, one: RemoteFile, got: int, *, force: bool = False) -> None:
        now = time.monotonic()
        if not self.report or (not force and now - self.last < 0.1):
            return
        self.last = now
        speed = (self.base + got - self.resumed) / max(now - self.began, 1e-6)
        self.report(Progress(self.ref, phase, one.path, self.index, self.count,
                             self.base + got, self.total, got, one.size, speed))


def _stream(source: tuple[str, str], part: Path, one: RemoteFile, meter: _Meter,
            cancel: CancelToken | None) -> None:
    """Append to ``part`` from where it ends until ``one.size`` bytes are there."""
    url, auth = source
    offset = part.stat().st_size if part.exists() else 0
    meter.resumed += offset
    try:
        response = http.open_stream(url, headers=_headers(offset), token=auth, timeout=60.0)
    except http.ServerError as exc:
        if exc.status != 416:
            raise
        part.unlink(missing_ok=True)
        offset = 0
        response = http.open_stream(url, token=auth, timeout=60.0)
    with response:
        if offset and response.status != 206:
            offset = 0
        with part.open("ab" if offset else "wb") as out:
            got = offset
            while True:
                if cancel and cancel.cancelled:
                    raise Cancelled(one.path)
                block = response.read(CHUNK)
                if not block:
                    break
                out.write(block)
                got += len(block)
                meter.send("downloading", one, got)
    meter.send("downloading", one, got, force=True)


def _fetch(parsed: remote.Ref, one: RemoteFile, final: Path, meter: _Meter,
           cancel: CancelToken | None) -> None:
    part = final.with_name(final.name + ".part")
    auth = remote.token()
    for attempt in range(TRIES):
        try:
            _stream(_source(parsed, one, auth), part, one, meter, cancel)
            break
        except http.ServerUnreachable:
            if attempt == TRIES - 1:
                raise
            time.sleep(0.5 * (attempt + 1))
    size = part.stat().st_size
    if one.size and size != one.size:
        raise OSError(f"{one.path}: got {size} bytes of {one.size}; pull again to continue")
    if one.sha256:
        meter.send("verifying", one, size, force=True)
        if files.sha256_file(part) != one.sha256:
            part.unlink(missing_ok=True)
            raise ChecksumMismatch(f"{one.path}: sha256 differs from the one {remote.endpoint()} lists")
    files.promote(part, final)


def _lock_for(final: Path) -> Path:
    return home.cache("pulls", hashlib.sha256(str(final).encode()).hexdigest()[:24] + ".lock")


def _present(final: Path, one: RemoteFile) -> bool:
    try:
        return final.stat().st_size == one.size
    except OSError:
        return False


def _remaining(final: Path, one: RemoteFile) -> int:
    """Bytes still to come for ``one``: all of them, less what a partial file holds."""
    if _present(final, one):
        return 0
    try:
        return max(0, one.size - final.with_name(final.name + ".part").stat().st_size)
    except OSError:
        return one.size


def _space(folder: Path, need: int) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(folder).free
    if free < need + MARGIN:
        raise NotEnoughSpace(
            f"{folder} has {human_bytes(free)} free; this needs {human_bytes(need)} "
            f"and {human_bytes(MARGIN)} to spare")


def pull(ref: str, dest: str | Path | None = None, on_progress: Report | None = None,
         cancel: CancelToken | None = None) -> Path:
    """Download ``ref`` into ``dest`` and return the path of its first file.

    ``dest`` defaults to ``<store>/models/owner/repo``. Every shard of a sharded build comes
    down; files already complete are kept. Raises `NotFound`, `GatedRepo` (with how to
    get a token), `NotEnoughSpace`, `ChecksumMismatch` or `Cancelled`; a cancelled or
    failed pull leaves ``<file>.part`` and the next pull of the same reference continues
    it. Two processes pulling the same file take turns.
    """
    parsed, chosen = plan(ref)
    folder = destination(dest, parsed)
    _space(folder, sum(_remaining(folder / f.path, f) for f in chosen))
    meter = _Meter(parsed.text, sum(f.size for f in chosen), len(chosen), on_progress)
    for index, one in enumerate(chosen):
        meter.index = index
        final = folder / one.path
        final.parent.mkdir(parents=True, exist_ok=True)
        with lock.only_one(_lock_for(final), announce=lambda _t: None):
            if not _present(final, one):
                _fetch(parsed, one, final, meter, cancel)
        meter.base += one.size
        meter.send("done" if index == len(chosen) - 1 else "downloading", one, 0, force=True)
    return folder / chosen[0].path
