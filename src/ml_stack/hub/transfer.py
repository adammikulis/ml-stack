"""Downloading a model from a Hugging Face endpoint, resumably, with progress.

``pull`` takes ``hf:owner/repo/file.gguf`` (every shard of that build comes down),
``hf:owner/repo:Q4_K_M`` or a folder-less file reference. Safetensors references
retrieve the complete snapshot folder; GGUF references return the first file.
A transfer interrupted by a cancel, a dropped connection or a crash continues from the
bytes already on disk.
"""

from __future__ import annotations

import hashlib
import logging
import shutil
import threading
import time
import urllib.parse
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from ml_stack import home, http, hub, lock, net
from ml_stack.hub import peers as peering, remote
from ml_stack.hub.remote import GatedRepo, NotFound, RemoteFile
from ml_stack.net.download import staged_part
from ml_stack.units import human_bytes

logger = logging.getLogger(__name__)

CHUNK = 1 << 20
MARGIN = 256 * 1024 * 1024
"""Free space left over after a download."""

__all__ = ["CancelToken", "Cancelled", "ChecksumMismatch", "GatedRepo", "NotEnoughSpace",
           "NotFound", "Progress", "held_snapshot", "plan", "pull", "snapshot"]


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


def _url(parsed: remote.Ref, one: RemoteFile) -> str:
    quoted = urllib.parse.quote(one.path)
    rev = urllib.parse.quote(parsed.revision, safe="")
    return f"{remote.endpoint()}/{parsed.repo}/resolve/{rev}/{quoted}"


class _Meter:
    """Bytes done across the whole pull, reported at most ten times a second."""

    def __init__(self, ref: str, total: int, count: int, report: Report | None) -> None:
        self.ref, self.total, self.count, self.report = ref, total, count, report
        self.base, self.index, self.last, self.began = 0, 0, 0.0, time.monotonic()
        self.resumed = 0
        self.step = 0
        self.cancel: CancelToken | None = None
        self.session: peering.Session | None = None

    def send(self, phase: str, one: RemoteFile, got: int, *, force: bool = False) -> None:
        now = time.monotonic()
        step = (self.base + got) // CHUNK
        if not self.report or (not force and now - self.last < 0.1 and step == self.step):
            return
        self.last, self.step = now, step
        speed = (self.base + got - self.resumed) / max(now - self.began, 1e-6)
        self.report(Progress(self.ref, phase, one.path, self.index, self.count,
                             self.base + got, self.total, got, one.size, speed))


def _kind(path: str) -> str:
    low = path.lower()
    return "gguf" if low.endswith(".gguf") else "safetensors" if low.endswith(".safetensors") else ""


def _want(one: RemoteFile, auth: str) -> net.Want:
    return net.Want(kind=_kind(one.path), sha256=one.sha256, size=one.size, token=auth,
                    max_bytes=int(one.size * 1.001) + MARGIN if one.size else 1 << 41,
                    purpose="model download")


def _from_peers(parsed: remote.Ref, one: RemoteFile, final: Path, meter: _Meter) -> bool:
    """Try the paired devices for ``one``; True when ``final`` is in place. Whatever goes
    wrong with them is a reason to ask the Hub, a cancel is not."""
    cancel = meter.cancel
    try:
        return meter.session.fetch(  # type: ignore[union-attr]
            peering.Wanted(parsed.repo, one.path, one.size, one.sha256), final,
            cancelled=lambda: bool(cancel and cancel.cancelled),
            progress=lambda done: meter.send("downloading", one, done),
            phase=lambda name: meter.send(name, one, one.size, force=True))
    except peering.Stopped as exc:
        raise Cancelled(one.path) from exc
    except (OSError, ValueError, RuntimeError) as exc:
        logger.warning("peers failed for %s, using the Hub: %s", one.path, exc)
        return False


def _fetch(parsed: remote.Ref, one: RemoteFile, final: Path, meter: _Meter) -> None:
    if meter.session is not None and _from_peers(parsed, one, final, meter):
        return
    cancel = meter.cancel
    url, auth = _url(parsed, one), remote.token()
    meter.resumed += _staged(url, final)
    hooks = net.Hooks(cancel=cancel, progress=lambda done, _t: meter.send("downloading", one, done),
                      phase=lambda name: meter.send(name, one, one.size, force=True))
    try:
        net.download(url, final, _want(one, auth), hooks=hooks)
    except net.Truncated as exc:
        if cancel and cancel.cancelled:
            raise Cancelled(one.path) from exc
        raise OSError(f"{one.path}: {exc}; pull again to continue") from exc
    except http.ServerError as exc:
        if exc.status in (401, 403):
            raise GatedRepo(remote.hint(parsed.repo, exc.status)) from exc
        if exc.status == 404:
            raise NotFound(f"{parsed.repo} has no {one.path}") from exc
        raise
    except net.ChecksumMismatch as exc:
        raise ChecksumMismatch(
            f"{one.path}: sha256 differs from the one {remote.endpoint()} lists") from exc


def _staged(url: str, final: Path) -> int:
    part = staged_part(url, final)
    return part.stat().st_size if part.exists() else 0


def _lock_for(final: Path) -> Path:
    return home.cache("pulls", hashlib.sha256(str(final).encode()).hexdigest()[:24] + ".lock")


def _present(final: Path, one: RemoteFile) -> bool:
    try:
        return final.stat().st_size == one.size
    except OSError:
        return False


def _remaining(parsed: remote.Ref, final: Path, one: RemoteFile) -> int:
    """Bytes still to come for ``one``: all of them, less what a partial file holds."""
    if _present(final, one):
        return 0
    return max(0, one.size - _staged(_url(parsed, one), final))


def _space(folder: Path, need: int) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(folder).free
    if free < need + MARGIN:
        raise NotEnoughSpace(
            f"{folder} has {human_bytes(free)} free; this needs {human_bytes(need)} "
            f"and {human_bytes(MARGIN)} to spare")


def pull(ref: str, dest: str | Path | None = None, on_progress: Report | None = None,
         cancel: CancelToken | None = None, *, peers: bool | None = None) -> Path:
    """Download ``ref`` into ``dest`` and return its GGUF file or safetensors folder.

    ``dest`` defaults to ``<store>/models/owner/repo``. Every shard of a sharded build comes
    down; files already complete are kept. Raises `NotFound`, `GatedRepo` (with how to
    get a token), `NotEnoughSpace`, `ChecksumMismatch` or `Cancelled`; a cancelled or
    failed pull leaves ``<file>.part`` and the next pull of the same reference continues
    it. Two processes pulling the same file take turns.

    Paired devices are asked first (``peers=False`` skips them; `ml_stack.hub.peers`): their
    bytes count only if they hash to the Hub's own digest, else the Hub is used.
    """
    parsed = remote.parse(ref)
    if parsed.file.lower().endswith(".safetensors"):
        files = remote.listing(parsed.repo, parsed.revision)
        if parsed.file not in {one.path for one in files}:
            raise NotFound(f"{parsed.repo} has no {parsed.file}")
        return _snapshot_files(parsed, files, dest, _Run(on_progress, cancel, peers))
    try:
        parsed, chosen = plan(ref)
    except NotFound:
        if parsed.quant or parsed.file:
            raise
        files = remote.listing(parsed.repo, parsed.revision)
        names = {one.path for one in files}
        if "config.json" not in names or not any(name.lower().endswith(".safetensors")
                                                   for name in names):
            raise
        return _snapshot_files(parsed, files, dest, _Run(on_progress, cancel, peers))
    folder = destination(dest, parsed)
    _bring(parsed, chosen, folder, _Run(on_progress, cancel, peers))
    return folder / chosen[0].path


@dataclass(frozen=True, slots=True)
class _Run:
    """What the caller of a pull asked for besides the files."""

    report: Report | None = None
    cancel: CancelToken | None = None
    peers: bool | None = None


def _bring(parsed: remote.Ref, chosen: list[RemoteFile], folder: Path, run: _Run) -> None:
    root = folder.resolve()
    for one in chosen:
        relative = Path(one.path)
        if (relative.is_absolute() or ".." in relative.parts
                or not (folder / relative).resolve().is_relative_to(root)):
            raise ValueError(f"Unsafe model file path: {one.path}")
    _space(folder, sum(_remaining(parsed, folder / f.path, f) for f in chosen))
    meter = _Meter(parsed.text, sum(f.size for f in chosen), len(chosen), run.report)
    meter.cancel = run.cancel
    if any(not _present(folder / f.path, f) for f in chosen):
        meter.session = peering.session(run.peers)
    for index, one in enumerate(chosen):
        meter.index = index
        final = folder / one.path
        final.parent.mkdir(parents=True, exist_ok=True)
        with lock.only_one(_lock_for(final), announce=lambda _t: None):
            if not _present(final, one):
                _fetch(parsed, one, final, meter)
        meter.base += one.size
        meter.send("done" if index == len(chosen) - 1 else "downloading", one, 0, force=True)


PICKLES = (".bin", ".pt", ".pth", ".ckpt", ".pkl", ".pickle", ".h5", ".msgpack", ".npy", ".npz")


def snapshot(repo: str, revision: str = "main", on_progress: Report | None = None,
             cancel: CancelToken | None = None, *, dest: str | Path | None = None) -> Path:
    """Every file of ``repo`` but its pickle-based weights, in ``<store>/models/owner/repo``;
    returns that folder. For a repository that holds safetensors."""
    parsed = remote.Ref(repo, "", "", revision)
    return _snapshot_files(parsed, remote.listing(repo, revision), dest,
                           _Run(on_progress, cancel))


def _snapshot_files(parsed: remote.Ref, files: list[RemoteFile], dest: str | Path | None,
                    run: _Run) -> Path:
    chosen = [f for f in files
              if not f.path.lower().endswith(PICKLES) and not f.name.startswith(".git")]
    if not chosen:
        raise NotFound(f"{parsed.repo} has no files")
    folder = destination(dest, parsed)
    _bring(parsed, chosen, folder, run)
    return folder


def held_snapshot(repo: str) -> Path | None:
    """The folder `snapshot` made for ``repo``, or the newest Hugging Face cache snapshot of
    it, when either is on this machine; nothing is fetched."""
    ours = home.state("models", *repo.split("/")[:2])
    if ours.is_dir() and any(ours.iterdir()):
        return ours
    snaps = hub.hub_cache() / ("models--" + repo.replace("/", "--")) / "snapshots"
    found = sorted((d for d in snaps.glob("*") if d.is_dir()), key=lambda d: d.stat().st_mtime)
    return found[-1] if found else None
