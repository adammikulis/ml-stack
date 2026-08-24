"""A training daemon: one GPU box, one job at a time, reachable over the LAN.

Training happens on whatever machine has the card, which is usually not the
machine you are working on. Without something like this, "run it on the other
box" means ssh, tmux, a hand-rolled rsync, and no way to ask what is happening
except logging in again.

Three decisions shape this:

**One job at a time.** A GPU is not shareable in any useful sense: two jobs on
one card contend for memory and both get slower, and the failure is silent --
a run that should take 3 hours quietly takes 11 and looks like it is working.
Submitting while busy queues rather than competing. This is ``RunLock``'s
lesson applied across machines instead of within one.

**Stop means SIGTERM, not SIGKILL.** A training loop that checkpoints on
SIGTERM can be stopped safely at any moment; killing it outright throws away
everything since the last checkpoint. The daemon sends TERM, waits, and only
escalates if the process ignores it.

**It executes commands you send it.** That is the point, and it is also
remote code execution: a token is mandatory, there is no anonymous mode, and
this belongs on a trusted LAN and nowhere else. Path traversal is blocked on
the file endpoints because "upload a dataset" and "overwrite /etc/passwd"
otherwise differ only by how you spell the path.

**It announces itself only to machines that already hold the cluster key.**
With a key present at ``~/.ml-stack/cluster.key`` the daemon advertises over
UDP (see ``discovery``) and derives its bearer token from that key, so a client
on the LAN finds it and authenticates to it with nothing configured. With no
key it stays silent and falls back to a random token you copy by hand -- which
is the old behaviour, and the right default for a machine that has not opted
into a cluster.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import platform
import re
import secrets
import socket
import shlex
import signal
import subprocess
import sys
import threading
import time
import urllib.parse
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from .availability import Availability, parse_window
from .discovery import (
    Advertiser,
    Beacon,
    DiscoveryError,
    derive_token,
    discover,
    key_path,
    load_cluster_key,
)

DEFAULT_PORT = 8770
_SAFE_SEGMENT = re.compile(r"^[A-Za-z0-9._-]+$")
#: How much of a file is held in memory at once while serving it. Fixed, and
#: unrelated to the file's size -- that is the entire point.
FILE_CHUNK = 1 << 20
DIGEST_HEADER = "X-ML-Stack-SHA256"

#: Digests are cached because a resumed download asks for the same one over and
#: over, and hashing a gigabyte checkpoint on every attempt would cost more
#: than the transfer. Keyed on size and mtime as well as path, so rewriting a
#: file invalidates its entry rather than serving a digest for bytes that are
#: no longer there. Bounded: this is a cache, not a record.
_DIGESTS: dict[tuple[str, int, int], str] = {}
_DIGEST_LOCK = threading.Lock()
_DIGEST_CACHE_MAX = 256


def file_digest(path: Path) -> str:
    """sha256 of a file, read in fixed pieces and cached by (path, size, mtime)."""
    st = path.stat()
    key = (str(path), st.st_size, st.st_mtime_ns)
    with _DIGEST_LOCK:
        hit = _DIGESTS.get(key)
    if hit is not None:
        return hit
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            block = fh.read(FILE_CHUNK)
            if not block:
                break
            h.update(block)
    digest = h.hexdigest()
    with _DIGEST_LOCK:
        if len(_DIGESTS) >= _DIGEST_CACHE_MAX:
            _DIGESTS.pop(next(iter(_DIGESTS)))
        _DIGESTS[key] = digest
    return digest


def remember_digest(path: Path, digest: str) -> None:
    """Record a digest already known, so the next download need not recompute it."""
    try:
        st = path.stat()
    except OSError:
        return
    with _DIGEST_LOCK:
        if len(_DIGESTS) >= _DIGEST_CACHE_MAX:
            _DIGESTS.pop(next(iter(_DIGESTS)))
        _DIGESTS[(str(path), st.st_size, st.st_mtime_ns)] = digest


class DaemonError(RuntimeError):
    pass


@dataclass
class Job:
    id: str
    name: str
    argv: list[str]
    cwd: str
    state: str = "queued"          # queued | running | done | failed | stopped
    pid: int | None = None
    returncode: int | None = None
    submitted_at: float = 0.0
    started_at: float | None = None
    finished_at: float | None = None
    env: dict[str, str] = field(default_factory=dict)

    def public(self) -> dict[str, Any]:
        d = asdict(self)
        if self.started_at:
            end = self.finished_at or time.time()
            d["elapsed_s"] = round(end - self.started_at, 1)
        return d


def safe_relpath(root: Path, relpath: str) -> Path:
    """Resolve ``relpath`` under ``root``, refusing anything that escapes it.

    Checked after resolution, not by string inspection: '..' filtering misses
    symlinks, and a symlink inside the file root pointing at / is exactly how
    an upload endpoint becomes a way to write anywhere on the machine.
    """
    relpath = urllib.parse.unquote(relpath)
    if not relpath:
        raise DaemonError("empty path")
    # Refuse absolute paths rather than silently rewriting them as relative.
    # Quietly turning "/etc/passwd" into "<root>/etc/passwd" is contained and
    # therefore safe, but it hides a confused client instead of telling it.
    if relpath.startswith("/") or (len(relpath) > 1 and relpath[1] == ":"):
        raise DaemonError(f"absolute path not allowed: {relpath!r}")
    for seg in Path(relpath).parts:
        if seg in ("..", "/") or not _SAFE_SEGMENT.match(seg):
            raise DaemonError(f"unsafe path segment: {seg!r}")
    target = (root / relpath).resolve()
    root_resolved = root.resolve()
    if root_resolved != target and root_resolved not in target.parents:
        raise DaemonError("path escapes the file root")
    return target


class JobRunner:
    """Runs jobs, ``slots`` at a time, and owns their processes.

    One slot is the default because a GPU is not shareable in any useful sense -- see the
    module docstring. A box whose work is tokenizing text has no such contention and
    should be told so with ``--slots``; capacity here is the number of worker threads
    rather than a counter, so the limit is structural and there is no arithmetic to get
    wrong.
    """

    def __init__(self, root: Path, files_root: Path | None = None, *,
                 slots: int = 1, gate: "Callable[[], tuple[bool, str]] | None" = None) -> None:
        if slots < 1:
            raise DaemonError(f"slots must be at least 1, got {slots}")
        self.root = root
        #: Job directories live UNDER the file root, so a checkpoint a job writes to
        #: $ML_STACK_JOB_DIR can actually be pulled. With them beside it instead, the
        #: obvious call -- pull("jobs/<id>/ckpt/model.safetensors") -- resolves to a
        #: path nothing creates and 404s, which is a confusing way to discover that
        #: the artifact you waited three hours for is unreachable.
        self.files_root = Path(files_root) if files_root is not None else root / "files"
        #: Asked before a queued job is STARTED, never before it is queued. "Come back
        #: at six" is something a scheduler can act on; a rejection is not, and work
        #: submitted at 3pm to a box that frees up at 5pm should simply run at 5pm.
        self.gate = gate
        self.slots = slots
        self.jobs: dict[str, Job] = {}
        self._queue: list[str] = []
        #: job id -> process. Keyed, not a single handle: with more than one slot
        #: "the running process" is not a thing, and stopping job B by reaching for
        #: it would kill job A.
        self._running: dict[str, subprocess.Popen] = {}
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._threads = [
            threading.Thread(target=self._run_loop, daemon=True, name=f"jobrunner-{i}")
            for i in range(slots)
        ]
        for t in self._threads:
            t.start()

    # -- state, under the lock -------------------------------------------
    def status(self) -> dict[str, Any]:
        """Capacity and what is on it. Taken under the lock, because a placement loop
        polling this while a worker thread mutates it is now the normal case."""
        with self._lock:
            running = sorted(self._running)
            return {"slots": self.slots, "free": self.slots - len(running),
                    "busy": bool(running), "running": running,
                    "queued": len(self._queue)}

    def snapshot(self) -> list[dict[str, Any]]:
        """Every job's public view. Copied under the lock -- iterating ``self.jobs``
        directly races a submit and raises 'dictionary changed size during iteration',
        which reaches the caller as a 500 on a route that should never fail."""
        with self._lock:
            return [j.public() for j in list(self.jobs.values())]

    # -- paths -----------------------------------------------------------
    def job_dir(self, job_id: str) -> Path:
        return self.files_root / "jobs" / job_id

    def log_path(self, job_id: str) -> Path:
        return self.job_dir(job_id) / "job.log"

    # -- submission ------------------------------------------------------
    def submit(self, name: str, argv: Iterable[str], cwd: str,
               env: dict[str, str] | None = None) -> Job:
        argv = [str(a) for a in argv]
        if not argv:
            raise DaemonError("empty argv")
        job_id = f"{int(time.time())}-{secrets.token_hex(3)}"
        job = Job(id=job_id, name=name or argv[0], argv=argv, cwd=cwd,
                  submitted_at=time.time(), env=dict(env or {}))
        d = self.job_dir(job_id)
        d.mkdir(parents=True, exist_ok=True)
        (d / "job.json").write_text(json.dumps(job.public(), indent=2))
        with self._lock:
            self.jobs[job_id] = job
            self._queue.append(job_id)
        self._wake.set()
        return job

    def stop(self, job_id: str, *, grace_s: float = 30.0) -> Job:
        with self._lock:
            job = self.jobs.get(job_id)
            if job is None:
                raise DaemonError(f"unknown job {job_id}")
            if job.state == "queued":
                self._queue.remove(job_id)
                job.state = "stopped"
                return job
            proc = self._running.get(job_id)
        if proc is None:
            return job
        # TERM first: a loop that checkpoints on TERM loses nothing. KILL is
        # for a process that ignored TERM, and costs everything since the last
        # checkpoint.
        proc.send_signal(signal.SIGTERM)
        deadline = time.time() + grace_s
        while time.time() < deadline and proc.poll() is None:
            time.sleep(0.25)
        if proc.poll() is None:
            proc.kill()
        job.state = "stopped"
        return job

    # -- the loop --------------------------------------------------------
    def _run_loop(self) -> None:
        while not self._stop.is_set():
            self._wake.wait(timeout=1.0)
            self._wake.clear()
            if self.gate is not None:
                allowed, _why = self.gate()
                if not allowed:
                    continue
            with self._lock:
                if not self._queue:
                    continue
                job_id = self._queue.pop(0)
                job = self.jobs[job_id]
            self._run_one(job)

    def _run_one(self, job: Job) -> None:
        log = self.log_path(job.id)
        log.parent.mkdir(parents=True, exist_ok=True)
        env = {**os.environ, **job.env, "PYTHONUNBUFFERED": "1",
               "ML_STACK_JOB_ID": job.id,
               "ML_STACK_JOB_DIR": str(self.job_dir(job.id)),
               # A job that must produce something the coordinator can fetch has to be
               # told where fetchable is. Without this it defaults to cwd, and a caller
               # who passes cwd lands the artifact somewhere nothing can see.
               "ML_STACK_FILES_ROOT": str(self.files_root),
               "ML_STACK_OUT": str(self.job_dir(job.id) / "out")}
        try:
            with log.open("ab") as fh:
                proc = subprocess.Popen(job.argv, cwd=job.cwd or None, env=env,
                                        stdout=fh, stderr=subprocess.STDOUT,
                                        start_new_session=True)
                with self._lock:
                    self._running[job.id] = proc
                    job.state = "running"
                    job.pid = proc.pid
                    job.started_at = time.time()
                rc = proc.wait()
        except Exception as exc:                      # noqa: BLE001
            job.state = "failed"
            job.returncode = -1
            log.write_text(f"failed to start: {exc}\n")
        else:
            job.returncode = rc
            if job.state != "stopped":
                job.state = "done" if rc == 0 else "failed"
        finally:
            job.finished_at = time.time()
            with self._lock:
                self._running.pop(job.id, None)
            (self.job_dir(job.id) / "job.json").write_text(
                json.dumps(job.public(), indent=2))
            self._wake.set()

    def stop_running(self, *, grace_s: float = 30.0) -> list[str]:
        """TERM everything in flight and put it back on the queue.

        Used when someone takes their machine back. TERM rather than KILL because a loop
        that checkpoints on TERM loses nothing, which the daemon already relies on --
        and requeueing rather than failing means the work resumes when the box does.
        """
        with self._lock:
            running = list(self._running)
        for job_id in running:
            job = self.jobs.get(job_id)
            try:
                self.stop(job_id, grace_s=grace_s)
            except DaemonError:
                continue
            if job is not None:
                job.state = "queued"
                job.pid = job.returncode = None
                with self._lock:
                    self._queue.insert(0, job_id)
        self._wake.set()
        return running

    def shutdown(self) -> None:
        self._stop.set()
        self._wake.set()


@dataclass
class Fetch:
    """One peer-to-peer transfer in flight."""

    id: str
    source: str
    relpath: str
    to: str
    state: str = "fetching"        # fetching | done | failed
    done: int = 0
    total: int = 0
    error: str = ""
    started_at: float = field(default_factory=time.time)
    finished_at: float = 0.0

    def public(self) -> dict[str, Any]:
        return {"id": self.id, "source": self.source, "relpath": self.relpath,
                "to": self.to, "state": self.state, "done": self.done,
                "total": self.total, "error": self.error,
                "started_at": self.started_at, "finished_at": self.finished_at}


class Fetcher:
    """Pulls files from other daemons, without going through the job queue.

    Three decisions shape this.

    **It is not a job.** Making it one would burn a training slot on the GPU box to run
    a download, which defeats the point of moving the bytes peer-to-peer in the first
    place. It would also put the source's bearer token in an argv, and therefore in
    ``ps`` output on a machine other people can see.

    **The destination pulls; the source is never told to push.** The destination is the
    only side that knows its own disk pressure, ``Peer.pull`` already resumes from a
    ``.part`` and verifies the whole reassembled file, and a pull inherits all of that
    for free.

    **The source is named, never addressed.** See ``resolve``.
    """

    def __init__(self, files_root: Path, key: bytes | None, *, slots: int = 2,
                 discover_timeout_s: float = 2.0) -> None:
        self.files_root = files_root
        self.key = key
        self.slots = slots
        self.discover_timeout_s = discover_timeout_s
        self.fetches: dict[str, Fetch] = {}
        self._lock = threading.Lock()
        self._sem = threading.Semaphore(slots)
        self._peers: dict[str, tuple[float, str]] = {}

    def resolve(self, name: str) -> str:
        """A peer name to a base URL, via signed discovery. Never from the caller.

        A route that accepted ``{"from_url": ...}`` would be a general "fetch any URL
        you are told to, with the cluster token attached" primitive -- server-side
        request forgery, and a confused deputy handing out a credential the requester
        did not have. Resolving the name here means the address comes from a signed
        beacon, which is ``discovery``'s own rule -- a peer's address comes from the
        packet, never from a self-reported field -- applied one level up.
        """
        if self.key is None:
            raise DaemonError(
                "this daemon holds no cluster key, so it cannot authenticate to a peer "
                "-- run 'ml-stack-peers init' and copy the key here")
        cached = self._peers.get(name)
        if cached and cached[0] > time.time():
            return cached[1]
        for beacon in discover(self.key, timeout_s=self.discover_timeout_s):
            self._peers[beacon.name] = (time.time() + 30.0, beacon.base_url)
        found = self._peers.get(name)
        if not found:
            raise DaemonError(f"no peer named {name!r} answered on this LAN")
        return found[1]

    def start(self, *, source: str, relpath: str, to: str,
              sha256: str = "") -> Fetch:
        target = safe_relpath(self.files_root, to)
        fetch = Fetch(id=f"{int(time.time())}-{secrets.token_hex(3)}",
                      source=source, relpath=relpath, to=to)
        with self._lock:
            self.fetches[fetch.id] = fetch
        threading.Thread(target=self._run, args=(fetch, target, sha256),
                         daemon=True, name=f"fetch-{fetch.id}").start()
        return fetch

    def _run(self, fetch: Fetch, target: Path, sha256: str) -> None:
        from .remote import Peer, sha256_file
        with self._sem:
            try:
                base_url = self.resolve(fetch.source)
                peer = Peer(base_url, derive_token(self.key))  # type: ignore[arg-type]

                def progress(done: int, total: int) -> None:
                    fetch.done, fetch.total = done, total

                target.parent.mkdir(parents=True, exist_ok=True)
                peer.pull(fetch.relpath, target, on_progress=progress)
                if sha256:
                    # The caller said which file it wanted. "A file arrived at that
                    # path" and "the file I asked for arrived" are different facts, and
                    # a pipeline stage that consumes the wrong shard fails much later,
                    # somewhere that looks nothing like a transfer.
                    got = sha256_file(target)
                    if not hmac.compare_digest(got, sha256.strip().lower()):
                        target.unlink(missing_ok=True)
                        raise DaemonError(
                            f"fetched {fetch.relpath} but its digest is {got[:16]}..., "
                            f"not the {sha256[:16]}... that was asked for")
                fetch.state = "done"
                fetch.done = fetch.total = target.stat().st_size
            except Exception as exc:                  # noqa: BLE001
                fetch.state = "failed"
                fetch.error = str(exc)
            finally:
                fetch.finished_at = time.time()

    def status(self) -> dict[str, Any]:
        with self._lock:
            active = [f for f in self.fetches.values() if f.state == "fetching"]
            return {"slots": self.slots, "fetching": len(active)}


REPORT_GROUP = "ml_stack.device_report"
"""Entry-point group a higher tier registers a richer device probe under.

The dependency points the right way: this package cannot import ``ml_stack.backend`` to
ask about CUDA -- that is a lab module and this is a device one -- so instead a lab
package *registers itself here* and the daemon finds it at runtime. ``ml-stack-train``
declares one in its ``pyproject.toml``; a box without it simply reports less.
"""


def _total_ram_gb() -> float | None:
    """Physical RAM, from whatever this platform exposes to the standard library."""
    try:                                    # Linux, and macOS via SC_PHYS_PAGES
        return round(os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / 2**30, 2)
    except (AttributeError, ValueError, OSError):
        return None


def stdlib_device_report() -> dict[str, Any]:
    """What this box is, using only what the standard library can see.

    Deliberately says nothing about accelerators. A device-tier package cannot import a
    framework to ask, and *guessing* is worse than staying quiet: a beacon claiming no
    GPU is read by placement as "not a training box", so an unverified negative would
    park every run on the wrong machine.
    """
    out: dict[str, Any] = {"backends": [], "cpus": os.cpu_count() or 1,
                           "arch": platform.machine(), "platform": sys.platform}
    ram = _total_ram_gb()
    if ram is not None:
        out["ram_gb"] = ram
    return out


def registered_reports() -> list[Callable[[], dict[str, Any]]]:
    """Every device probe a higher tier has registered under ``REPORT_GROUP``."""
    try:
        from importlib.metadata import entry_points
        found = entry_points(group=REPORT_GROUP)
    except Exception:                                 # noqa: BLE001
        return []
    out = []
    for ep in found:
        try:
            out.append(ep.load())
        except Exception:                             # noqa: BLE001
            # A broken plugin must not stop the daemon booting. The box then
            # advertises less than it could, which placement handles; failing
            # to start would take the machine out of the fleet entirely.
            continue
    return out


def resolve_report(spec: str) -> Callable[[], dict[str, Any]]:
    """Turn ``"ml_stack.train.accelerator:report"`` into the callable it names.

    The explicit form exists because entry-point discovery cannot be the thing
    correctness rests on here: this repo's tests run from source directories on
    ``sys.path`` with nothing pip-installed, so an entry point would be invisible to
    every test that matters -- an untested default is not a default, it is a hope.
    """
    module, _, attr = spec.partition(":")
    if not module or not attr:
        raise DaemonError(
            f"bad report spec {spec!r} -- expected 'module.path:callable'")
    try:
        import importlib
        return getattr(importlib.import_module(module), attr)
    except (ImportError, AttributeError) as exc:
        raise DaemonError(f"cannot load report {spec!r}: {exc}") from None


def device_report(extra: Callable[[], dict[str, Any]] | None = None) -> dict[str, Any]:
    """What is on this box. Best effort, and additive.

    ``extra`` overrides the registered probes rather than adding to them, so a caller
    that passes one gets exactly what it asked for.
    """
    out = stdlib_device_report()
    for fn in ([extra] if extra is not None else registered_reports()):
        try:
            out.update(fn() or {})
        except Exception:                             # noqa: BLE001
            pass
    return out


_DEFAULT_REPORT = device_report
"""Bound here so ``serve_forever``'s ``device_report=`` parameter can shadow the name
without losing the default."""


def make_handler(runner: JobRunner, files_root: Path,
                 token: "str | Callable[[], str]",
                 name: str = "",
                 report: Callable[[], dict[str, Any]] = device_report,
                 fetcher: "Fetcher | None" = None,
                 ui: "Any | None" = None,
                 schedule: "Availability | None" = None,
                 on_paused: str = "stop",
                 schedule_path: "Path | None" = None):
    class Handler(BaseHTTPRequestHandler):
        server_version = "ml-stack-traind/0.1"

        def log_message(self, fmt: str, *args: Any) -> None:  # quieter
            pass

        # -- helpers --
        def _token(self) -> str:
            """Read at request time, not captured at startup.

            A daemon started before its box had joined a cluster minted a random token.
            When someone then joins through the setup wizard, the derived token is what
            every peer will compute -- so a token frozen at startup means the machine
            appears in the fleet, answers /health, and rejects every job with a 401
            until somebody restarts it. Which is not a setup wizard.
            """
            return token() if callable(token) else token

        def _authed(self) -> bool:
            got = self.headers.get("Authorization", "")
            if got.startswith("Bearer "):
                # compare_digest: token comparison should not leak length or
                # prefix through timing, cheap to get right.
                return secrets.compare_digest(got[7:], self._token())
            return False

        def _send(self, code: int, payload: Any, *, raw: bytes | None = None,
                  headers: dict[str, str] | None = None,
                  content_type: str = "") -> None:
            body = raw if raw is not None else json.dumps(payload).encode()
            self.send_response(code)
            # One Content-Type, chosen here. Passing another through `headers` sends the
            # header twice, and a browser takes the FIRST -- so an HTML page announced
            # second arrives as application/octet-stream and is offered as a download
            # rather than rendered.
            self.send_header("Content-Type", content_type or (
                "application/octet-stream" if raw is not None else "application/json"))
            self.send_header("Content-Length", str(len(body)))
            for k, v in (headers or {}).items():
                self.send_header(k, v)
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)

        def _send_file(self, target: Path, start: int = 0,
                       end: int | None = None) -> None:
            """Send a file without ever holding more than FILE_CHUNK of it.

            The obvious version -- read_bytes() then slice for the Range --
            costs one full copy of the file, and two if a Range is present.
            Checkpoints here run to a gigabyte, so serving a resumed download
            of one cost about 2GB on the machine that is also running the
            training job it came from. Seek and stream instead: memory is now
            a function of the buffer, not of the file.
            """
            size = target.stat().st_size
            if start and start >= size:
                # The caller's .part is at or past the end. Saying so with 416
                # matters: the alternative is a 206 carrying zero bytes and a
                # nonsensical Content-Range, which a client reasonably reads as
                # "you already have it all" and promotes a stale file.
                self._send(416, {"error": "range beyond end of file",
                                 "size": size},
                           headers={"Content-Range": f"bytes */{size}"})
                return
            last = size - 1 if end is None else min(end, size - 1)
            length = max(0, last - start + 1)
            ranged = start > 0 or (end is not None and last < size - 1)
            self.send_response(206 if ranged else 200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(length))
            self.send_header("Accept-Ranges", "bytes")
            # Always the digest of the WHOLE file, never of the range being
            # sent: the client is reassembling a file across resumes and can
            # only check the thing it ends up with.
            self.send_header(DIGEST_HEADER, file_digest(target))
            if ranged:
                self.send_header("Content-Range", f"bytes {start}-{last}/{size}")
            self.end_headers()
            if self.command == "HEAD":
                return
            remaining = length
            with target.open("rb") as fh:
                fh.seek(start)
                while remaining > 0:
                    chunk = fh.read(min(FILE_CHUNK, remaining))
                    if not chunk:
                        break
                    try:
                        self.wfile.write(chunk)
                    except (BrokenPipeError, ConnectionResetError):
                        # The client hung up mid-download. Its .part is intact
                        # and it will resume; there is nothing to clean up and
                        # nothing worth logging.
                        return
                    remaining -= len(chunk)

        def _guard(self) -> bool:
            if not self._authed():
                self._send(401, {"error": "bad or missing bearer token"})
                return False
            return True

        # -- routes --
        def _ui(self) -> bool:
            if ui is None or not self.path.startswith("/ui"):
                return False
            from .ui import routes
            return routes(ui, self)

        def do_GET(self) -> None:                      # noqa: N802
            if self._ui():
                return
            parsed = urllib.parse.urlparse(self.path)
            path, q = parsed.path, urllib.parse.parse_qs(parsed.query)
            if path == "/health":
                # unauthenticated on purpose: liveness must be checkable
                # without handing a probe the token.
                status = runner.status()
                sched = schedule.public() if schedule is not None else None
                if sched is not None and not sched["available"]:
                    status = {**status, "free": 0}
                self._send(200, {"ok": True, "name": name, **status, **report(),
                                 **({"availability": sched} if sched else {})})
                return
            if not self._guard():
                return
            if path == "/availability":
                if schedule is None:
                    self._send(501, {"error": "no schedule on this daemon"}); return
                self._send(200, schedule.public()); return
            if path == "/jobs":
                self._send(200, {"jobs": runner.snapshot()})
                return
            m = re.match(r"^/jobs/([^/]+)(/log|/metrics)?$", path)
            if m:
                job = runner.jobs.get(m.group(1))
                if job is None:
                    self._send(404, {"error": "unknown job"}); return
                kind = m.group(2)
                if kind is None:
                    self._send(200, job.public()); return
                if kind == "/log":
                    n = int(q.get("tail", ["200"])[0])
                    p = runner.log_path(job.id)
                    text = ""
                    if p.exists():
                        text = "".join(p.read_text(errors="replace").splitlines(True)[-n:])
                    self._send(200, {"log": text}); return
                since = int(q.get("since", ["0"])[0])
                mp = runner.job_dir(job.id) / "metrics.jsonl"
                rows: list[Any] = []
                if mp.exists():
                    for line in mp.read_text(errors="replace").splitlines()[since:]:
                        try:
                            rows.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass
                self._send(200, {"metrics": rows, "next": since + len(rows)}); return
            m = re.match(r"^/fetch/([^/]+)$", path)
            if m:
                if fetcher is None:
                    self._send(501, {"error": "peer-to-peer fetch is not enabled"})
                    return
                fetch = fetcher.fetches.get(m.group(1))
                if fetch is None:
                    self._send(404, {"error": "unknown fetch"}); return
                self._send(200, fetch.public()); return
            if path == "/fetch":
                if fetcher is None:
                    self._send(501, {"error": "peer-to-peer fetch is not enabled"})
                    return
                self._send(200, {"fetches": [f.public() for f in
                                             list(fetcher.fetches.values())],
                                 **fetcher.status()})
                return
            if path.startswith("/files/"):
                try:
                    target = safe_relpath(files_root, path[len("/files/"):])
                except DaemonError as e:
                    self._send(400, {"error": str(e)}); return
                if not target.is_file():
                    self._send(404, {"error": "not found"}); return
                start, end = 0, None
                rng = self.headers.get("Range", "")
                if rng.startswith("bytes="):
                    spec = rng.split("=", 1)[1].split(",")[0].strip()
                    head, _, tail = spec.partition("-")
                    if not head:
                        # "bytes=-500" means the LAST 500 bytes. Nothing here
                        # sends it, and guessing wrong would serve the wrong
                        # part of a checkpoint, so refuse rather than mis-serve.
                        self._send(400, {"error": "suffix ranges not supported"})
                        return
                    try:
                        start = int(head)
                        end = int(tail) if tail else None
                    except ValueError:
                        self._send(400, {"error": f"bad Range: {rng!r}"}); return
                self._send_file(target, start, end)
                return
            self._send(404, {"error": "no such route"})

        # _send and _send_file already branch on HEAD; without this the method they
        # were written for answers 501. Data-locality scoring asks "do you already hold
        # this exact file" and should not have to download it to find out.
        do_HEAD = do_GET

        def do_POST(self) -> None:                     # noqa: N802
            if self._ui():
                return
            if not self._guard():
                return
            parsed = urllib.parse.urlparse(self.path)
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length) if length else b"{}"
            if parsed.path == "/jobs":
                try:
                    req = json.loads(body or b"{}")
                    argv = req.get("argv")
                    if isinstance(argv, str):
                        argv = shlex.split(argv)
                    job = runner.submit(req.get("name", ""), argv or [],
                                        req.get("cwd", str(files_root)),
                                        req.get("env"))
                except (DaemonError, ValueError) as e:
                    self._send(400, {"error": str(e)}); return
                self._send(201, job.public()); return
            if parsed.path == "/availability":
                if schedule is None:
                    self._send(501, {"error": "no schedule on this daemon"}); return
                req = json.loads(body or b"{}")
                action = str(req.get("action") or "")
                try:
                    if action == "pause":
                        minutes = req.get("minutes")
                        schedule.pause(minutes=float(minutes) if minutes else None,
                                       reason=str(req.get("reason") or ""))
                        if on_paused == "stop":
                            # The point of pausing is to get the machine back. Leaving
                            # the current job holding the GPU would make the toggle mean
                            # "no NEW work", which is not what someone reaching for it
                            # in the middle of a game is asking for.
                            runner.stop_running()
                    elif action == "resume":
                        schedule.resume()
                    elif action == "reserve":
                        schedule.reserve(str(req.get("holder") or "someone"),
                                         float(req.get("seconds") or 3600),
                                         str(req.get("reason") or ""))
                    elif action == "release":
                        schedule.release(str(req.get("holder") or ""))
                    elif action == "window":
                        schedule.windows.append(
                            parse_window(str(req.get("spec") or ""),
                                         busy=bool(req.get("busy", True))))
                    elif action == "clear_windows":
                        schedule.windows.clear()
                    else:
                        raise DaemonError(f"unknown action {action!r}")
                except (DaemonError, ValueError, PermissionError) as e:
                    self._send(400, {"error": str(e)}); return
                if schedule_path is not None:
                    schedule.save(schedule_path)
                self._send(200, schedule.public()); return
            if parsed.path == "/fetch":
                if fetcher is None:
                    self._send(501, {"error": "peer-to-peer fetch is not enabled"})
                    return
                try:
                    req = json.loads(body or b"{}")
                    if "from_url" in req or "url" in req or "token" in req:
                        # Refused rather than ignored, so a caller reaching for the
                        # obvious-looking field is told why instead of quietly getting
                        # something else. See Fetcher.resolve.
                        self._send(400, {"error": "name the source peer, not a URL: a "
                                                  "route that fetches any URL it is "
                                                  "given is a forgery primitive"})
                        return
                    source = str(req.get("peer") or "")
                    relpath = str(req.get("relpath") or "")
                    if not source or not relpath:
                        raise DaemonError("both 'peer' and 'relpath' are required")
                    fetch = fetcher.start(source=source, relpath=relpath,
                                          to=str(req.get("to") or relpath),
                                          sha256=str(req.get("sha256") or ""))
                except (DaemonError, ValueError) as e:
                    self._send(400, {"error": str(e)}); return
                self._send(202, fetch.public()); return
            m = re.match(r"^/jobs/([^/]+)/stop$", parsed.path)
            if m:
                try:
                    job = runner.stop(m.group(1))
                except DaemonError as e:
                    self._send(404, {"error": str(e)}); return
                self._send(200, job.public()); return
            self._send(404, {"error": "no such route"})

        def do_DELETE(self) -> None:                   # noqa: N802
            if self._ui():
                return
            self._send(404, {"error": "no such route"})

        def do_PUT(self) -> None:                      # noqa: N802
            if self._ui():
                return
            if not self._guard():
                return
            parsed = urllib.parse.urlparse(self.path)
            if not parsed.path.startswith("/files/"):
                self._send(404, {"error": "no such route"}); return
            try:
                target = safe_relpath(files_root, parsed.path[len("/files/"):])
            except DaemonError as e:
                self._send(400, {"error": str(e)}); return
            target.parent.mkdir(parents=True, exist_ok=True)
            length = int(self.headers.get("Content-Length", "0"))
            # Content-Range lets a killed upload resume instead of restarting a
            # multi-gigabyte dataset from zero.
            cr = self.headers.get("Content-Range", "")
            offset = 0
            if cr.startswith("bytes "):
                offset = int(cr.split(" ", 1)[1].split("-")[0])
            partial = target.with_suffix(target.suffix + ".part")
            held = partial.stat().st_size if partial.exists() else 0
            if offset > held:
                # Seeking past the end of what we hold writes a sparse run of zeros and
                # then fails the digest check with "checksum mismatch" -- which sends the
                # uploader hunting a corruption that never happened. The truth is that
                # this resume has nothing to resume from, so say the size we actually
                # hold, the same answer the download side already gives on a bad Range.
                self._send(416, {"error": "cannot resume past what this daemon holds",
                                 "held": held, "requested_offset": offset},
                           headers={"Content-Range": f"bytes */{held}"})
                return
            mode = "r+b" if offset else "wb"
            with partial.open(mode) as fh:
                if offset:
                    fh.seek(offset)
                remaining = length
                while remaining > 0:
                    chunk = self.rfile.read(min(1 << 20, remaining))
                    if not chunk:
                        break
                    fh.write(chunk)
                    remaining -= len(chunk)
            if self.headers.get("X-ML-Stack-Complete", "1") == "1":
                want = self.headers.get(DIGEST_HEADER, "").strip().lower()
                got = file_digest(partial) if want else ""
                if want and not secrets.compare_digest(got, want):
                    # Delete rather than keep for resume. A length mismatch is
                    # a short transfer and resuming fixes it; a digest mismatch
                    # means some byte already written is wrong, and resuming
                    # from it preserves the corruption forever.
                    partial.unlink(missing_ok=True)
                    self._send(422, {"error": "checksum mismatch, upload discarded",
                                     "expected": want, "got": got})
                    return
                os.replace(partial, target)
                size = target.stat().st_size
                if got:
                    remember_digest(target, got)
                self._send(200, {"ok": True, "path": str(target), "bytes": size,
                                 "sha256": got or file_digest(target)})
            else:
                self._send(200, {"ok": True, "partial": str(partial),
                                 "bytes": partial.stat().st_size})

    return Handler


def load_or_create_token(root: Path, cluster_key: bytes | None = None) -> str:
    """The bearer token: derived from the cluster key, or random and local.

    A cluster key deliberately overrides an existing random token rather than
    deferring to it. The alternative -- keeping whatever the file says -- means
    a box that joins a cluster keeps answering on a token no peer can compute,
    and presents as "found but unauthorised", which is a far more confusing
    failure than a token that changed when you asked it to change.
    """
    p = root / "token"
    if cluster_key is not None:
        tok = derive_token(cluster_key)
        root.mkdir(parents=True, exist_ok=True)
        if not p.exists() or p.read_text().strip() != tok:
            p.write_text(tok)
        p.chmod(0o600)
        return tok
    if p.exists():
        return p.read_text().strip()
    root.mkdir(parents=True, exist_ok=True)
    tok = secrets.token_urlsafe(24)
    p.write_text(tok)
    p.chmod(0o600)
    return tok


def serve_forever(root: Path | str = "~/.ml-stack/traind",
                  host: str = "0.0.0.0", port: int = DEFAULT_PORT, *,
                  name: str = "", announce: bool = True,
                  cluster_key_path: Path | str | None = None,
                  device_report: Callable[[], dict[str, Any]] | None = None,
                  slots: int = 1, labels: Iterable[str] = (),
                  fetch_slots: int = 2, web: bool = True,
                  setup_from_lan: bool = False,
                  busy_hours: Iterable[str] = (), free_hours: Iterable[str] = (),
                  on_paused: str = "stop") -> None:
    root = Path(root).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    live_token: list[str] = [""]
    files_root = root / "files"
    files_root.mkdir(exist_ok=True)
    name = name or os.environ.get("ML_STACK_PEER_NAME") or socket.gethostname()
    key = load_cluster_key(cluster_key_path)
    token = load_or_create_token(root, key)
    live_token[0] = token
    schedule_path = root / "availability.json"
    schedule = Availability.load(schedule_path)
    for spec in busy_hours:
        schedule.windows.append(parse_window(spec))
    for spec in free_hours:
        schedule.windows.append(parse_window(spec, busy=False))
    if busy_hours or free_hours:
        schedule.save(schedule_path)

    runner = JobRunner(root, files_root, slots=slots,
                       gate=lambda: schedule.may_start())
    fetcher = Fetcher(files_root, key, slots=fetch_slots)
    interface = None
    setup_token = ""
    if web:
        from .ui import UI
        if key is None and setup_from_lan:
            # Printed to the console below. Someone who can read this machine's console
            # is someone who is already on the machine, which is the property that makes
            # relaxing the loopback rule safe to offer at all.
            setup_token = secrets.token_urlsafe(9)
        interface = UI(name=name, cluster_key_path=cluster_key_path, peer_port=port,
                       setup_token=setup_token)
    base_report = device_report or _DEFAULT_REPORT
    # Filtered: an unset $ML_STACK_LABELS splits to [""], and an empty label would be
    # advertised, shown in the UI, and matchable by Requires(labels=("",)).
    labels = sorted({s.strip() for s in labels if s and s.strip()})

    def report() -> dict[str, Any]:
        # Declared, not detected. A box cannot prove it has no GPU, so "keep prep off the
        # training boxes" has to be something an operator says rather than something the
        # daemon infers -- see stdlib_device_report.
        return {**base_report(), "labels": labels}
    httpd = ThreadingHTTPServer((host, port),
                                make_handler(runner, files_root,
                                             lambda: live_token[0], name, report,
                                             fetcher, interface, schedule, on_paused,
                                             schedule_path))
    advertiser: Advertiser | None = None

    def refresh(b: Beacon) -> None:
        """Bring the beacon's mutable half up to date before it goes on the wire."""
        status = runner.status()
        available = schedule.public()
        b.busy, b.queued = status["busy"], status["queued"]
        b.slots, b.free = status["slots"], status["free"]
        if not available["available"]:
            # Advertised as having no capacity rather than as absent: the box is still
            # there, still reachable, and still worth showing -- it just will not take
            # work right now, and placement should say so rather than silently skip it.
            b.free = 0
        b.device = {**report(), "availability": available}

    def start_announcing() -> None:
        """Begin advertising, now that there is a key to sign beacons with."""
        nonlocal advertiser, key
        if advertiser is not None or not announce:
            return
        key = load_cluster_key(cluster_key_path)
        if key is None:
            return
        fetcher.key = key
        # The token every peer will now compute. Without this the box is discoverable
        # and undrivable, which is the worst of both.
        live_token[0] = load_or_create_token(root, key)
        beacon = Beacon(name=name, port=port, device=report(),
                        slots=runner.slots, free=runner.slots)
        advertiser = Advertiser(beacon, key, refresh=refresh).start()

    if interface is not None:
        interface.on_join = start_announcing

    if announce and key is not None:
        beacon = Beacon(name=name, port=port, device=report(),
                        slots=runner.slots, free=runner.slots)
        try:
            advertiser = Advertiser(beacon, key, refresh=refresh).start()
        except DiscoveryError as exc:
            # Not fatal. A daemon nobody can find is still a daemon you can
            # reach by address, and refusing to start would turn a flaky
            # network into an outage.
            print(f"  discovery OFF: {exc}")
    print(f"ml-stack traind on http://{host}:{port}")
    print(f"  name  {name}")
    print(f"  root  {root}")
    print(f"  slots {slots}")
    state = schedule.public()
    for window in schedule.windows:
        print(f"  busy  {window.describe()}" if window.busy
              else f"  free  {window.describe()}")
    if not state["available"]:
        print(f"  NOT TAKING WORK -- {state['unavailable_because']}")
    if labels:
        print(f"  labels {' '.join(labels)}")
    if advertiser is not None:
        print(f"  peers announcing on {advertiser.group}:{advertiser.port} "
              f"(key {key_path(cluster_key_path)})")
        print("  token derived from the cluster key -- peers compute it themselves")
    elif key is None:
        print(f"  token {token}")
        print(f"  discovery OFF: no cluster key at {key_path(cluster_key_path)}")
        print("  run 'ml-stack-peers setup' to join one")
    print(f"  device {json.dumps(report())}")
    if slots > 1:
        # Said out loud next to the other warning, because --slots on a box with a card
        # is how you quietly halve your own training throughput: two jobs on one GPU
        # contend for memory and both get slower, and nothing in the logs says so.
        print(f"  {slots} jobs will run at once. Correct for CPU work; on a GPU box "
              "this makes every job slower.")
    if interface is not None:
        shown = "127.0.0.1" if host in ("0.0.0.0", "") else host
        print(f"  open   http://{shown}:{port}/ui/")
        if key is None:
            print("  this machine has not joined a cluster yet -- open the address "
                  "above ON THIS MACHINE to set it up")
            if setup_token:
                print(f"  setup from the LAN with this one-time code: {setup_token}")
    print("  THIS EXECUTES COMMANDS YOU SEND IT. Trusted LAN only.", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        if advertiser is not None:
            advertiser.stop()
        runner.shutdown()
        httpd.server_close()


def main(argv: list[str] | None = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="ml-stack-traind")
    ap.add_argument("--root", default="~/.ml-stack/traind")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--name", default="",
                    help="how this box identifies itself to peers "
                         "(default: $ML_STACK_PEER_NAME, else the hostname)")
    ap.add_argument("--cluster-key", default=None,
                    help="path to the cluster key (default: ~/.ml-stack/cluster.key)")
    ap.add_argument("--no-announce", action="store_true",
                    help="serve, but stay invisible to peer discovery")
    ap.add_argument("--busy", action="append", default=[], metavar="WHEN",
                    help="when this machine is NOT available for work, e.g. "
                         "'mon-fri 09:00-17:00' or '22:00-06:00'. Repeatable. Queued "
                         "work waits for the window to close rather than failing.")
    ap.add_argument("--free", action="append", default=[], metavar="WHEN",
                    help="carve an exception out of a busy window, e.g. "
                         "'mon-fri 12:00-13:00'")
    ap.add_argument("--on-paused", choices=("stop", "finish"), default="stop",
                    help="what happens to work already running when the machine is "
                         "paused: stop it (default -- SIGTERM, so a loop that "
                         "checkpoints keeps its progress, and it is requeued) or let "
                         "it finish")
    ap.add_argument("--no-web", action="store_true",
                    help="serve the API but not the web interface")
    ap.add_argument("--setup-from-lan", action="store_true",
                    help="on a machine that has not joined a cluster, allow first-run "
                         "setup from another machine using the one-time code printed "
                         "at startup. For a headless box you cannot open a browser on.")
    ap.add_argument("--fetch-slots", type=int, default=2,
                    help="concurrent peer-to-peer file transfers (default 2). Not job "
                         "slots: a transfer is I/O against another box, not compute, "
                         "and must not occupy a training slot.")
    ap.add_argument("--label", action="append", default=[], metavar="LABEL",
                    help="a role this box declares, e.g. 'prep'. Repeatable. Work can "
                         "require or exclude labels; nothing is inferred from them.")
    ap.add_argument("--report", action="append", default=[], metavar="MODULE:CALLABLE",
                    help="a richer device probe, e.g. "
                         "'ml_stack.train.accelerator:report' on a box with a card. "
                         "Repeatable; later ones win. Without any, the daemon reports "
                         "only what the standard library can see, plus whatever is "
                         "registered under the 'ml_stack.device_report' entry point.")
    ap.add_argument("--slots", type=int,
                    default=int(os.environ.get("ML_STACK_SLOTS") or 1),
                    help="how many jobs to run at once (default 1; raise it only on a "
                         "box whose work does not contend for one accelerator)")
    a = ap.parse_args(argv)
    probes = [resolve_report(spec) for spec in a.report]

    def report() -> dict[str, Any]:
        out = stdlib_device_report()
        for probe in probes:
            try:
                out.update(probe() or {})
            except Exception:                         # noqa: BLE001
                pass
        return out

    serve_forever(a.root, a.host, a.port, name=a.name,
                  announce=not a.no_announce, cluster_key_path=a.cluster_key,
                  slots=a.slots, device_report=report if probes else None,
                  labels=a.label or os.environ.get("ML_STACK_LABELS", "").split(","),
                  fetch_slots=a.fetch_slots, web=not a.no_web,
                  setup_from_lan=a.setup_from_lan,
                  busy_hours=a.busy, free_hours=a.free, on_paused=a.on_paused)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
