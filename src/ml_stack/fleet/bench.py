"""Benchmarks over the fleet.

Two things, one module. The first half is a short calibration a peer runs once when it
joins the swarm (`measure`, `calibrate`), so placement has a number for a machine nobody
has measured. The second half is the fleet side of ``ml-stack-bench sweep --fleet``: a
model sweep spread over every machine with room for it.

- `Job` is one ``ml-stack-bench`` invocation a peer must serve; the daemon takes it beside
  training jobs (``POST /bench``) through a `BenchHost`, which refuses a peer whose code
  differs from the dispatcher's, whose measuring lock is held, or whose memory the model
  would not fit -- and otherwise starts ``ml-stack-bench ... --detach`` on itself, adopts
  the detached pid into its job list, and settles it ``done`` or ``failed`` from the log.
- `plan` puts each model on the idle peer with the most room that fits it; `dispatch`
  sends the jobs; `wait` polls the daemons; `gather` brings each peer's runs home into
  the dispatcher's store with ``server["host"]`` set, and `import_runs` does that by hand
  for a peer with no daemon.

Everything the bench side needs to call is `plan`, `jobs_from`, `dispatch`, `wait` and
`gather`, in that order.
"""

from __future__ import annotations

import functools
import hashlib
import json
import os
import platform
import re
import socket
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .daemon import Job as DaemonJob, JobRunner
    from .rates import Rates
    from .remote import Peer

__all__ = ["BENCH_KIND", "BUDGET_S", "BenchHost", "Handle", "Job", "Local", "Plan",
           "Refused", "argv", "bench_home", "calibrate", "dispatch", "estimate", "gather",
           "here", "import_runs", "installed_commit", "jobs_from", "main", "measure",
           "plan", "same_commit", "wait"]

BENCH_KIND = "_bench"
"""The ``Rates`` kind this is filed under. Leading underscore because it is a proxy for"""

BUDGET_S = 1.5
"""Seconds of work. Long enough that scheduler noise does not dominate, short enough"""

BLOCK = b"ml-stack-fleet-bench" * 64
CHUNK = 200
"""Rounds between clock checks. Reading the clock every iteration would measure the"""


def measure(budget_s: float = BUDGET_S) -> dict[str, Any]:
    """Hash and float throughput over a fixed time budget."""
    half = max(0.05, budget_s / 2)

    digest = hashlib.sha256()
    hashed = 0
    started = time.perf_counter()
    while time.perf_counter() - started < half:
        for _ in range(CHUNK):
            digest.update(BLOCK)
        hashed += CHUNK
    hash_s = time.perf_counter() - started

    acc = 0.0
    floats = 0
    fstart = time.perf_counter()
    while time.perf_counter() - fstart < half:
        for i in range(CHUNK):
            acc += (i * 1.000001) ** 0.5
        floats += CHUNK
    float_s = time.perf_counter() - fstart

    hash_rate = hashed / hash_s if hash_s > 0 else 0.0
    float_rate = floats / float_s if float_s > 0 else 0.0
    return {
        "score": round((hash_rate * float_rate) ** 0.5, 2),
        "hash_per_s": round(hash_rate, 2),
        "float_per_s": round(float_rate, 2),
        "seconds": round(hash_s + float_s, 3),
        "mb_hashed": round(hashed * len(BLOCK) / 2**20, 1),
        "cpus": os.cpu_count() or 1,
        "arch": platform.machine(),
        "python": platform.python_version(),
        "checksum": digest.hexdigest()[:16],
    }


def argv(budget_s: float = BUDGET_S) -> list[str]:
    """What to submit to a peer to have it benchmark itself."""
    return ["python3", "-m", "ml_stack.fleet.bench", "--budget", str(budget_s)]


def main(argv_: list[str] | None = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="ml-stack-bench")
    ap.add_argument("--budget", type=float, default=BUDGET_S,
                    help="seconds to spend measuring (default 1.5)")
    a = ap.parse_args(argv_)
    result = measure(a.budget)
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


def calibrate(peers: "Sequence[Peer]", rates: "Rates", *,
              budget_s: float = BUDGET_S, timeout_s: float = 120.0,
              force: bool = False,
              on_event: "Callable[[str, dict[str, Any]], None] | None" = None,
              ) -> dict[str, float]:
    """Benchmark every peer that has never been benchmarked. Returns name -> score."""
    def emit(event: str, **fields: Any) -> None:
        if on_event is not None:
            try:
                on_event(event, fields)
            except Exception:                         # noqa: BLE001
                pass

    scores: dict[str, float] = {}
    for peer in peers:
        try:
            name = peer.health().get("name") or peer.name
        except Exception as exc:                      # noqa: BLE001
            emit("skip", peer=peer.base_url, why=str(exc))
            continue
        if not force and rates.get(name, BENCH_KIND) is not None:
            scores[name] = rates.get(name, BENCH_KIND)      # type: ignore[assignment]
            continue
        emit("bench", peer=name)
        try:
            job = peer.submit(argv(budget_s), name="ml-stack-bench")
            final = peer.wait(job["id"], poll_s=1.0, timeout_s=timeout_s)
            if final.get("state") != "done":
                emit("skip", peer=name, why=f"bench {final.get('state')}")
                continue
            result = json.loads(peer.log(job["id"], tail=5).strip().splitlines()[-1])
            score = float(result["score"])
        except Exception as exc:                      # noqa: BLE001
            emit("skip", peer=name, why=str(exc))
            continue
        rates.record(name, BENCH_KIND, units=score, seconds=1.0)
        scores[name] = score
        emit("benched", peer=name, score=score)
    rates.save()
    return scores


# ======================================================================================
# The fleet side of ``ml-stack-bench sweep --fleet``
# ======================================================================================

LOCK = "measuring.lock"
"""What ``ml-stack-bench`` holds while it measures, under `bench_home`. A peer whose lock
is held is measuring already, whoever started it, and gets no second job."""

STORE = "runs.ladybug"
"""Where a peer's ``ml-stack-bench`` keeps its runs, under `bench_home`: the store its
``--kept`` defaults to, which is why a `Job` may not name one."""

FAILED_MARKS = ("error:", "Traceback (most recent call last)", "[killed]", "selfcheck: FAILED")
"""What the end of a detached bench's log says when it did not finish. A detached child is
nobody's to ``wait()`` on -- it was reparented the moment it started -- so its exit code is
gone, and the log is the record: `ml-stack-bench` prints ``error:`` on every failing path,
``[killed]`` when `stop` reached it, and a traceback when something else did."""

TAIL = 5
"""Lines of a peer's log `wait` prints when a job ends."""


def bench_home() -> Path:
    """Where ``ml-stack-bench`` keeps everything on this machine: the lock, the store, the
    logs and ``measuring.json``. The same path `ml_stack.graph.bench.HOME` names, written
    here rather than imported so a daemon that never measures never loads the bench."""
    return Path("~/.ml-stack/bench").expanduser()


# -- the pin -------------------------------------------------------------------------
def installed_commit() -> str:
    """Which ml-stack this process runs: the short sha of the checkout the package is
    imported from, ``(dirty)`` appended when that tree has changes; ``v<version>`` from
    ``importlib.metadata`` for a package installed from a wheel; "" when neither answers.

    Found from the package's own path -- this file's, up to the nearest ``.git``, a
    directory or a worktree's file (``ml_stack`` is a namespace package and has no
    ``__file__``) -- because that is the code that will measure, whatever ``pip`` thinks
    is installed. A dispatcher and a peer on different commits measure
    different things and file them under one name, so `BenchHost.submit` refuses a `Job`
    whose ``commit`` differs from this. Compared with `same_commit`, which ignores dirtiness
    -- a tree edited on one side is a warning in the record, not a reason to refuse.
    """
    from ml_stack.paths import repo_root

    where = repo_root(Path(__file__).resolve().parent)
    if where is not None:
        try:
            def git(*words: str) -> str:
                return subprocess.run(["git", "-C", str(where), *words], capture_output=True,
                                      text=True, timeout=15, check=True).stdout.strip()

            sha = git("rev-parse", "--short", "HEAD")
            if sha:
                return f"{sha} (dirty)" if git("status", "--porcelain") else sha
        except Exception:  # noqa: BLE001 - no git on the box: the version below still answers
            pass
    try:
        from importlib.metadata import version

        return f"v{version('ml-stack')}"
    except Exception:  # noqa: BLE001
        return ""


def same_commit(mine: str, theirs: str) -> bool:
    """Whether two `installed_commit` answers name the same code: the sha or version,
    ``(dirty)`` ignored. Two empty answers are not the same code -- they are no answer."""
    a, b = str(mine or "").split()[:1], str(theirs or "").split()[:1]
    return bool(a) and a == b


@functools.lru_cache(maxsize=1)
def machine_room() -> int:
    """`hub.room()`: what a model may use here, in bytes, or 0 when unknown. Cached, since
    it asks the kernel and the answer does not change while the daemon runs."""
    from ml_stack.hub import room

    return int(room() or 0)


def _human(size: int | float) -> str:
    return f"{size / 2**30:.1f}G"


# -- a job ---------------------------------------------------------------------------
class Refused(RuntimeError):
    """A well-formed `Job` this peer will not run now. ``kind`` says which of the three
    reasons: ``commit`` (its code differs), ``lock`` (it is measuring) or ``room`` (a
    model would not fit its memory) -- or ``launch`` when the bench would not start."""

    def __init__(self, kind: str, why: str) -> None:
        super().__init__(why)
        self.kind = kind


@dataclass(frozen=True, slots=True)
class Job:
    """One ``ml-stack-bench`` invocation for one peer.

    ``argv`` is the command line after ``ml-stack-bench``, with only the ``--serve`` flags
    for the models this peer owns; ``models`` names them again so the peer can size them
    without parsing the line, and ``needs`` is the dispatcher's estimate of each in bytes
    (0 when unknown, which is not the same as enormous). ``commit`` is the dispatcher's
    `installed_commit`, which the peer must match. ``kept_label`` names the sweep, so the
    peer's job list says what is measuring.

    The line may not carry ``--kept``: the peer keeps its runs in its own store and
    `gather` brings them home. Nor ``--detach`` or ``--no-queue``: the peer adds both.
    """

    argv: tuple[str, ...]
    models: tuple[str, ...]
    commit: str
    kept_label: str = ""
    needs: Mapping[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.argv:
            raise ValueError("a bench job needs an argv")
        for flag in ("--kept", "--detach", "--no-queue"):
            if flag in self.argv:
                raise ValueError(f"a bench job's argv may not carry {flag}: the peer keeps "
                                 f"its runs in its own store and detaches the run itself")
        if not self.commit:
            raise ValueError("a bench job needs the dispatcher's commit, so the peer can "
                             "refuse to measure with different code")

    def public(self) -> dict[str, Any]:
        """The request body ``POST /bench`` takes."""
        return {"argv": list(self.argv), "models": list(self.models), "commit": self.commit,
                "kept_label": self.kept_label,
                "needs": {str(k): int(v) for k, v in self.needs.items()}}

    @classmethod
    def from_request(cls, req: Mapping[str, Any]) -> "Job":
        """A `Job` out of a request body, or a `ValueError` saying what was wrong."""
        if not isinstance(req, Mapping):
            raise ValueError("a bench job is a JSON object")
        argv_ = req.get("argv")
        if isinstance(argv_, str):
            import shlex

            argv_ = shlex.split(argv_)
        if not isinstance(argv_, (list, tuple)) or not all(isinstance(a, str) for a in argv_):
            raise ValueError("'argv' must be a list of strings")
        models = req.get("models") or ()
        if not isinstance(models, (list, tuple)) or not all(isinstance(m, str) for m in models):
            raise ValueError("'models' must be a list of strings")
        needs = req.get("needs") or {}
        if not isinstance(needs, Mapping):
            raise ValueError("'needs' must map each model to its estimated bytes")
        return cls(argv=tuple(argv_), models=tuple(models),
                   commit=str(req.get("commit") or ""),
                   kept_label=str(req.get("kept_label") or ""),
                   needs={str(k): int(v) for k, v in needs.items()})

    @property
    def name(self) -> str:
        """What the peer's job list calls it."""
        return f"bench:{self.kept_label or ' '.join(self.models) or self.argv[0]}"


def jobs_from(planned: Mapping[Any, Sequence[str]], base_argv: Sequence[str], *,
              commit: str = "", needs: Mapping[str, int] | None = None,
              drafts: Mapping[str, str] | None = None, kept_label: str = "",
              ) -> dict[Any, Job]:
    """One `Job` per peer in a `plan`: ``base_argv`` -- the sweep's line without any
    ``--serve`` -- with ``--serve MODEL`` for each model the peer got, and ``--serve-draft``
    beside it when ``drafts`` names one for that model. ``commit`` defaults to this
    process's `installed_commit`."""
    commit = commit or installed_commit()
    needs = dict(needs or {})
    out: dict[Any, Job] = {}
    for peer, models in planned.items():
        if not models:
            continue
        line = list(base_argv)
        for model in models:
            line += ["--serve", model]
            if drafts and model in drafts:
                line += ["--serve-draft", drafts[model]]
        out[peer] = Job(argv=tuple(line), models=tuple(models), commit=commit,
                        kept_label=kept_label,
                        needs={m: int(needs.get(m, 0)) for m in models})
    return out


# -- what a model needs --------------------------------------------------------------
def estimate(model: str, *, context: int = 32768, draft: str = "auto",
             preflight: Callable[..., Any] | None = None, binary: str = "") -> int:
    """Bytes ``model`` will take to serve: weights, its draft head, the KV cache at
    ``context`` and the runtime allowance -- `serve.preflight`'s own estimate when the
    file is at hand, else the weights' size from the Hub listing, else 0 for unknown.

    Unknown is not enormous: `plan` sends a model it could not size to the roomiest idle
    peer and lets that peer's preflight be the judge, as `weight_of` leaves it to the load.
    ``preflight`` is `serve.preflight.Preflight` unless a test hands in a fake.
    """
    from ml_stack.serve.manager import weight_of
    from ml_stack.serve.preflight import RUNTIME_ALLOWANCE_BYTES

    path = _at_hand(model)
    if path is None:
        return _hub_bytes(model, draft=draft)
    head: str | Path | None = None
    if draft == "auto":
        from .models import draft_beside

        head = draft_beside(path)
    elif draft:
        head = _at_hand(draft) or draft
    try:
        from ml_stack.serve.backend import ServerSpec

        if preflight is None:
            from ml_stack.serve.preflight import Preflight as preflight
        if not binary:
            from ml_stack.serve.binary import find_binary

            binary = str(find_binary() or "llama-server")
        report = preflight(ServerSpec(model=path, context=int(context), draft=head),
                           binary=binary, limit_bytes=0)
        weights = int(report.weights_bytes) or weight_of(path)
        kv = int(report.kv_estimate_bytes)
    except Exception:  # noqa: BLE001 - a preflight that cannot read it still leaves the weights
        weights, kv = weight_of(path), 0
    return weights + kv + (weight_of(head) if head else 0) + RUNTIME_ALLOWANCE_BYTES


def _at_hand(model: str) -> Path | None:
    """The file ``model`` names on this machine, or None: a path, a Hub-cached file by its
    exact name, or an ``hf:`` reference whose file has been fetched."""
    if not model:
        return None
    where = Path(model).expanduser()
    if where.is_file():
        return where
    from ml_stack.hub import located

    if model.startswith("hf:"):
        parts = [p for p in model[3:].split("/") if p]
        return located("/".join(parts[2:])) if len(parts) > 2 else None
    if "/" in model:
        return None
    return located(model)


def _hub_bytes(model: str, *, draft: str = "auto") -> int:
    """The weights' size from the Hub listing for an ``hf:`` reference not yet fetched --
    its draft head's beside it when ``draft`` is ``auto`` -- or 0 when nothing answers."""
    if not model.startswith("hf:"):
        return 0
    parts = [p for p in model[3:].split("/") if p]
    if len(parts) < 3:
        return 0
    repo, name = "/".join(parts[:2]), "/".join(parts[2:])
    try:
        from ml_stack.hub import draft_for, files

        listed = dict(files(repo))
        total = int(listed.get(name, 0))
        if total and draft == "auto":
            head = draft_for(repo)
            total += int(listed.get(head.rsplit("/", 2)[-1], 0)) if head else 0
        return total
    except Exception:  # noqa: BLE001 - the Hub is somebody else's machine
        return 0


# -- the daemon side -----------------------------------------------------------------
def detach_bench(line: Sequence[str], home: Path) -> tuple[int, Path]:
    """Start ``ml-stack-bench <line> --no-queue --detach`` here and return the child's pid
    and log. ``--detach`` re-runs the command in its own session with its output in a log
    under ``home/logs`` and writes the pid into ``home/measuring.json``, which is where
    this reads it back; ``--no-queue`` makes a race for the lock a failed job rather than a
    run queued behind another, since the lock was checked before this was called."""
    done = subprocess.run([sys.executable, "-m", "ml_stack.graph.bench", *line,
                           "--no-queue", "--detach"],
                          capture_output=True, text=True, timeout=120)
    said = (done.stdout or "") + (done.stderr or "")
    if done.returncode != 0:
        raise RuntimeError(f"ml-stack-bench --detach exited {done.returncode}: {said.strip()}")
    found = re.search(r"log: (.+)", said)
    try:
        held = json.loads((home / "measuring.json").read_text(encoding="utf-8"))
        pid = int(held.get("pid") or 0)
        log = Path(str(held.get("log") or (found.group(1).strip() if found else "")))
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"ml-stack-bench detached but {home / 'measuring.json'} did not "
                           f"say what: {exc}") from None
    if not pid:
        raise RuntimeError(f"ml-stack-bench detached but recorded no pid; it said: {said.strip()}")
    return pid, log


def _alive(pid: int) -> bool:
    """Whether ``pid`` is still doing something -- a zombie is not."""
    if not pid or pid <= 0:
        return False
    try:
        import psutil

        return psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
    except ImportError:
        pass
    except Exception:  # noqa: BLE001 - gone, or not ours to ask about
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def ended_badly(log: Path, *, lines: int = 60) -> str:
    """The line near the end of a detached bench's log that says it failed, or "" when
    nothing does. See `FAILED_MARKS` for why the log, and not an exit code, is the record."""
    try:
        tail = log.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:]
    except OSError:
        return "no log was written"
    for line in tail:
        if any(mark in line for mark in FAILED_MARKS):
            return line.strip()
    return ""


class BenchHost:
    """The bench side of one daemon: takes a `Job`, starts it detached, watches it, and
    hands back what it measured.

    ``runner`` is the daemon's `JobRunner`: an accepted job is adopted into it, so it is
    listed under ``/jobs``, polled at ``/jobs/<id>``, read at ``/jobs/<id>/log`` and
    stopped at ``/jobs/<id>/stop`` exactly as a training job is. ``home`` is where this
    machine's ``ml-stack-bench`` keeps its lock, store and logs; ``commit`` is what this
    machine runs (`installed_commit` unless given); ``room`` and ``launch`` are
    `machine_room` and `detach_bench` unless a test hands in fakes.
    """

    def __init__(self, runner: "JobRunner", *, home: Path | str | None = None,
                 commit: str | None = None, room: Callable[[], int] = machine_room,
                 launch: Callable[[Sequence[str], Path], tuple[int, Path]] = detach_bench,
                 name: str = "", poll_s: float = 1.0) -> None:
        self.runner = runner
        self.home = Path(home).expanduser() if home is not None else bench_home()
        self.commit = installed_commit() if commit is None else commit
        self.room = room
        self.launch = launch
        self.name = name or socket.gethostname()
        self.poll_s = poll_s
        self._mine: dict[str, "DaemonJob"] = {}
        self._lock = threading.Lock()

    # -- what this machine says about itself --
    def lock_held(self) -> str:
        """Who holds the measuring lock -- ``pid N`` -- or "" when nobody does.

        A plain non-blocking ``flock`` on the same file `only_one` takes, and nothing
        else: asking through `only_one(wait=False)` would run its ``finally`` on the
        refused attempt, which truncates the holder's pid record, so the next person to
        ask would be told "somebody".
        """
        import fcntl

        path = self.home / LOCK
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            handle = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
        except OSError:
            return ""
        try:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                return os.pread(handle, 32, 0).decode("utf-8", "replace").strip() or "somebody"
            fcntl.flock(handle, fcntl.LOCK_UN)
            return ""
        finally:
            os.close(handle)

    def measuring(self) -> bool:
        """Whether something is measuring here: a job this host started that has not
        ended, or the lock held by anyone -- a run started at the keyboard counts."""
        with self._lock:
            if any(j.state == "running" for j in self._mine.values()):
                return True
        return bool(self.lock_held())

    def report(self) -> dict[str, Any]:
        """What the beacon and ``/health`` carry for `plan`: the memory a model may use,
        the code this machine runs, and whether it is measuring."""
        return {"room_bytes": int(self.room() or 0), "bench_commit": self.commit,
                "measuring": self.measuring()}

    def snapshot(self) -> list[dict[str, Any]]:
        with self._lock:
            return [j.public() for j in self._mine.values()]

    # -- taking a job --
    def submit(self, job: Job) -> "DaemonJob":
        """Start ``job`` here, or raise `Refused` saying why not.

        Three refusals, checked in this order: the dispatcher's ``commit`` is not this
        machine's (`same_commit`); the measuring lock is held or a job of ours is still
        running; a model's ``needs`` exceeds `room` (a need of 0 is unknown, and passes --
        the peer's own preflight sizes it before the load). Then `launch` starts the
        bench detached, the pid is adopted into the runner under ``bench:<label>`` with
        the bench's own log, and a thread watches the pid and settles the job.
        """
        from .daemon import Job as DaemonJob

        if not same_commit(self.commit, job.commit):
            raise Refused("commit", f"{self.name} runs ml-stack {self.commit or '(unknown)'}, "
                                    f"the dispatcher {job.commit}; measuring with different "
                                    f"code files two measurements under one name -- update "
                                    f"one of them")
        with self._lock:
            running = [j for j in self._mine.values() if j.state == "running"]
        if running:
            raise Refused("lock", f"{self.name} is measuring already: {running[0].name} "
                                  f"(job {running[0].id}, pid {running[0].pid})")
        holder = self.lock_held()
        if holder:
            raise Refused("lock", f"{self.name} is measuring already: {self.home / LOCK} is "
                                  f"held by {holder}")
        room = int(self.room() or 0)
        if room:
            for model in job.models:
                need = int(job.needs.get(model, 0))
                if need > room:
                    raise Refused("room", f"{model} needs {_human(need)} and {self.name} may "
                                          f"use {_human(room)}")
        try:
            pid, log = self.launch(job.argv, self.home)
        except Exception as exc:  # noqa: BLE001 - said in the answer, not swallowed
            raise Refused("launch", f"{self.name} could not start ml-stack-bench: {exc}") from exc
        import secrets

        mine = DaemonJob(id=f"{int(time.time())}-{secrets.token_hex(3)}", name=job.name,
                         argv=["ml-stack-bench", *job.argv], cwd=str(self.home), pid=pid,
                         submitted_at=time.time(), log=str(log))
        self.runner.adopt(mine)
        with self._lock:
            self._mine[mine.id] = mine
        threading.Thread(target=self._watch, args=(mine,), daemon=True,
                         name=f"bench-watch-{mine.id}").start()
        return mine

    def _watch(self, job: "DaemonJob") -> None:
        """Wait for the detached pid to go, then settle the job from its log. A job
        `stop` already marked ``stopped`` stays so."""
        while _alive(int(job.pid or 0)):
            time.sleep(self.poll_s)
        if job.state == "running":
            why = ended_badly(Path(job.log)) if job.log else "no log was written"
            job.state = "failed" if why else "done"
            job.returncode = 1 if why else 0
        job.finished_at = job.finished_at or time.time()
        self.runner.record(job)

    # -- handing back what was measured --
    def export(self, *, since: str = "", job: str = "", full: bool = False,
               anyway: bool = False) -> dict[str, Any]:
        """The runs kept here since ``since`` (``%FT%T``, this machine's clock) or since
        the job ``job`` started -- what ``ml-stack-bench show --export`` writes, flattened
        and gated to the invented community, or the whole records with ``full``. The
        gate is `show --export`'s own (`_over_invented`): ``anyway`` lifts it, as
        ``--anyway-export`` does, and not into a repository."""
        from .daemon import DaemonError

        if job:
            found = self.runner.jobs.get(job)
            if found is None:
                raise DaemonError(f"unknown job {job}")
            since = time.strftime("%FT%T", time.localtime(found.started_at or found.submitted_at))
        store = self.home / STORE
        out: dict[str, Any] = {"runs": [], "skipped": 0, "since": since,
                               "commit": self.commit, "store": str(store), "full": full}
        if not store.exists():
            return out
        from ml_stack.graph import bench as measuring

        kept = [r for r in measuring.runs(store) if str(r.get("at", "")) >= since]
        if full:
            over, skipped = measuring._over_invented(kept, anyway=anyway)
            out["runs"] = [dict(one) for one in over]
        else:
            out["runs"], skipped = measuring._exportable(kept, anyway=anyway)
        out["skipped"] = skipped
        return out


# -- this machine as a peer ----------------------------------------------------------
class Local:
    """This machine, answering as a peer would, in-process: for a dispatcher that runs no
    daemon and still counts itself. The same calls `plan`, `dispatch`, `wait` and `gather`
    make of a `Peer` -- ``health``, ``job``, ``log`` and the bench ones -- answered by a
    `BenchHost` of its own over a private `JobRunner`."""

    def __init__(self, host: BenchHost | None = None, *, name: str = "",
                 home: Path | str | None = None) -> None:
        if host is None:
            from .daemon import JobRunner

            home = Path(home).expanduser() if home is not None else bench_home()
            runner = JobRunner(home / "local", slots=1)
            host = BenchHost(runner, home=home, name=name)
        self.host = host
        self.name = name or host.name
        self.base_url = f"local://{self.name}"

    def health(self) -> dict[str, Any]:
        status = self.host.runner.status()
        measuring = self.host.measuring()
        return {"ok": True, "name": self.name, **status,
                "busy": status["busy"] or measuring,
                "free": 0 if measuring else status["free"], **self.host.report()}

    def job(self, job_id: str) -> dict[str, Any]:
        found = self.host.runner.jobs.get(job_id)
        if found is None:
            raise KeyError(job_id)
        return found.public()

    def log(self, job_id: str, tail: int = 200) -> str:
        path = self.host.runner.log_path(job_id)
        if not path.exists():
            return ""
        return "".join(path.read_text(errors="replace").splitlines(True)[-tail:])

    def stop(self, job_id: str) -> dict[str, Any]:
        return self.host.runner.stop(job_id).public()


def here(*, name: str = "", home: Path | str | None = None) -> Local:
    """The dispatcher itself as a peer, so `plan` can count it."""
    return Local(name=name, home=home)


# -- the calls a peer answers, over the daemon or in-process --------------------------
def submit_bench(peer: Any, job: Job) -> dict[str, Any]:
    """``POST /bench`` on ``peer``: the daemon's job record, or `Refused`."""
    if isinstance(peer, Local):
        return peer.host.submit(job).public()
    from .remote import PeerError

    try:
        return peer._json("POST", "/bench", job.public())
    except PeerError as exc:
        answered = _answered(str(exc))
        if answered and answered[0] == 409:
            raise Refused(str(answered[1].get("refused") or "refused"),
                          str(answered[1].get("error") or exc)) from None
        raise


def bench_export(peer: Any, *, since: str = "", job: str = "", full: bool = True,
                 anyway: bool = False) -> dict[str, Any]:
    """``GET /bench/export`` on ``peer``: its runs since ``since`` or since ``job`` began."""
    if isinstance(peer, Local):
        return {**peer.host.export(since=since, job=job, full=full, anyway=anyway),
                "host": peer.name}
    import urllib.parse

    query = urllib.parse.urlencode({k: v for k, v in (("since", since), ("job", job),
                                                        ("full", "1" if full else ""),
                                                        ("anyway", "1" if anyway else ""))
                                    if v})
    return peer._json("GET", f"/bench/export?{query}")


def _answered(message: str) -> tuple[int, dict[str, Any]] | None:
    """The status and JSON body out of a `PeerError`'s message, or None."""
    found = re.search(r"-> (\d{3}): (\{.*\})", message, re.S)
    if not found:
        return None
    try:
        return int(found.group(1)), json.loads(found.group(2))
    except ValueError:
        return None


# -- dispatch ------------------------------------------------------------------------
class Plan(dict):
    """``{peer: [model, ...]}`` -- and ``unplaced``, the models that fit nowhere, each with
    why not on every peer, so nothing is dropped silently."""

    def __init__(self) -> None:
        super().__init__()
        self.unplaced: list[tuple[str, str]] = []


def plan(models: Sequence[str], peers: Sequence[Any], *, needs: Mapping[str, int] | None = None,
         context: int = 32768, log: Callable[[str], None] = print) -> Plan:
    """Which peer serves which model.

    Each peer is asked ``health()`` for its name, whether it is idle (not busy, a free slot,
    not measuring) and its ``room_bytes``. Each model -- sized by ``needs`` when given, else
    by `estimate` at ``context`` -- goes, largest first, to an idle peer with room for it:
    the one holding the fewest models so far, the roomiest on a tie, so two models land on
    two machines rather than both on the biggest. A model nobody fits is listed under
    ``unplaced`` with every peer's reason, and printed. ``peers`` may include `here()`.
    """
    sized = dict(needs or {})
    for model in models:
        if model not in sized:
            sized[model] = int(estimate(model, context=context))
    seen: list[tuple[Any, str, dict[str, Any]]] = []
    for peer in peers:
        try:
            health = peer.health()
        except Exception as exc:  # noqa: BLE001 - a peer that does not answer is not idle
            log(f"  {getattr(peer, 'name', peer)}: did not answer ({exc})")
            continue
        seen.append((peer, _name_of(peer, health), health))
    out = Plan()
    for peer, _name, _health in seen:
        out[peer] = []
    log(f"planning {len(models)} model(s) over {len(seen)} peer(s):")
    for model in sorted(models, key=lambda m: -sized.get(m, 0)):
        need = sized.get(model, 0)
        fitting: list[tuple[Any, str, int]] = []
        reasons: list[str] = []
        for peer, name, health in seen:
            room = int(health.get("room_bytes") or 0)
            why = _not_idle(health)
            if why:
                reasons.append(f"{name}: {why}")
            elif room and need and need > room:
                reasons.append(f"{name}: room {_human(room)} < {_human(need)}")
            else:
                fitting.append((peer, name, room))
        if not fitting:
            said = "; ".join(reasons) or "no peers answered"
            out.unplaced.append((model, said))
            log(f"  {model:<40} {_human(need):>7} fits nowhere: {said}")
            continue
        peer, name, room = min(fitting, key=lambda f: (len(out[f[0]]), -f[2]))
        out[peer].append(model)
        log(f"  {model:<40} {_human(need):>7} -> {name} (room {_human(room) if room else '?'})"
            + ("" if need else "  [size unknown; the peer's preflight decides]"))
    return out


def _name_of(peer: Any, health: Mapping[str, Any] | None = None) -> str:
    """What a peer calls itself: its beacon's name, else what ``/health`` says, else
    whatever it is addressed by -- a `Peer` made from a URL alone is named by the URL."""
    beacon = getattr(peer, "beacon", None)
    if beacon is not None and getattr(beacon, "name", ""):
        return str(beacon.name)
    if health is None:
        try:
            health = peer.health()
        except Exception:  # noqa: BLE001
            health = {}
    return str(health.get("name") or getattr(peer, "name", peer))


def _not_idle(health: Mapping[str, Any]) -> str:
    if health.get("measuring"):
        return "measuring"
    if health.get("busy"):
        return "busy"
    if int(health.get("free", 1) or 0) < 1:
        return "no free slot"
    return ""


@dataclass
class Handle:
    """One dispatched job: where it went, the daemon's id for it, and how it stands."""

    peer: Any
    job: Job
    id: str = ""
    name: str = ""
    state: str = "pending"
    """pending | running | done | failed | stopped | refused"""
    log: str = ""
    why: str = ""
    host: str = ""
    """What the peer calls itself, which is what `gather` files its runs under."""

    @property
    def peer_name(self) -> str:
        return self.host or str(getattr(self.peer, "name", self.peer))

    @property
    def ended(self) -> bool:
        return self.state in ("done", "failed", "stopped", "refused")


def dispatch(jobs: Mapping[Any, Job], *, log: Callable[[str], None] = print) -> list[Handle]:
    """Send each job to its peer. A refusal is a `Handle` in state ``refused`` with the
    reason, printed, not an exception: the rest of the sweep still goes out."""
    out: list[Handle] = []
    for peer, job in jobs.items():
        handle = Handle(peer=peer, job=job, host=_name_of(peer))
        try:
            answered = submit_bench(peer, job)
        except Refused as why:
            handle.state, handle.why = "refused", f"{why.kind}: {why}"
            log(f"  {handle.peer_name}: refused ({why.kind}) -- {why}")
        except Exception as exc:  # noqa: BLE001 - one unreachable peer must not stop the rest
            handle.state, handle.why = "refused", f"unreachable: {exc}"
            log(f"  {handle.peer_name}: unreachable -- {exc}")
        else:
            handle.id = str(answered.get("id") or "")
            handle.name = str(answered.get("name") or job.name)
            handle.state = str(answered.get("state") or "running")
            handle.log = str(answered.get("log") or "")
            log(f"  {handle.peer_name}: {handle.name} {handle.state} (job {handle.id}"
                + (f", log {handle.log}" if handle.log else "") + ")")
        out.append(handle)
    return out


def wait(handles: Sequence[Handle], *, poll_s: float = 20.0, timeout_s: float | None = None,
         log: Callable[[str], None] = print) -> list[Handle]:
    """Poll each peer's job until every one has ended, printing one line per change and
    the last `TAIL` lines of the log when a job ends. ``timeout_s`` bounds the whole wait;
    a job still running then is left in state ``running`` and said so."""
    deadline = time.monotonic() + timeout_s if timeout_s else None
    while True:
        for handle in handles:
            if handle.ended:
                continue
            try:
                current = handle.peer.job(handle.id)
            except Exception as exc:  # noqa: BLE001 - said, and asked again next round
                log(f"  {handle.peer_name}: could not read job {handle.id}: {exc}")
                continue
            state = str(current.get("state") or "")
            if state == handle.state:
                continue
            handle.state = state
            log(f"  {handle.peer_name}: {handle.name} {state}")
            if handle.ended:
                try:
                    tail = handle.peer.log(handle.id, tail=TAIL).rstrip()
                except Exception:  # noqa: BLE001
                    tail = ""
                for line in tail.splitlines():
                    log(f"      {line}")
        if all(h.ended for h in handles):
            return list(handles)
        if deadline is not None and time.monotonic() > deadline:
            for handle in handles:
                if not handle.ended:
                    log(f"  {handle.peer_name}: {handle.name} still {handle.state} after "
                        f"{timeout_s:.0f}s; left running")
            return list(handles)
        time.sleep(poll_s)


# -- gather --------------------------------------------------------------------------
SERVER_KEYS = ("model", "draft_model", "binary", "context", "slots", "cache_type",
               "reasoning_budget", "load_s", "resident_bytes", "kv_and_run_bytes", "mmapped",
               "sampling", "finder", "concurrency")
"""The fields ``show --export`` flattens out of a run's ``server``, put back there on import."""


def _doc_from(record: Mapping[str, Any], *, host: str, commit: str) -> dict[str, Any]:
    """A ``bench:`` doc out of one record -- a whole run as `runs` reads it, or a flat one
    as `show --export` writes it. Either way ``server["host"]`` and ``server["commit"]``
    are set. A flat record has no rows, so its totals go under ``totals`` and its
    ``derived`` is precomputed, which is what `rates`, `pareto` and `composed` read."""
    if "rows" in record:
        one = {k: v for k, v in record.items() if k != "key"}
        one["server"] = {**(one.get("server") or {}), "host": host, "commit": commit}
        return one
    server = {k: record[k] for k in SERVER_KEYS if _said(record.get(k))}
    right = float(record.get("f1") or 0)
    seconds = float(record.get("seconds") or 0)
    paid = float(record.get("read_tokens") or 0) + float(record.get("written_tokens") or 0)
    memory = float(record.get("kv_and_run_bytes") or 0)
    derived: dict[str, float] = {
        "right": right, "recall": float(record.get("recall") or 0),
        "precision": float(record.get("precision") or 0),
        "shown_per_question": float(record.get("lit_per_question") or 0),
        "wanted_per_question": 0.0, "seconds": seconds, "paid_tokens": paid,
        "calls": float(record.get("calls") or 0), "kv_bytes": memory,
        "questions": float(record.get("questions") or 0)}
    if seconds > 0:
        derived["right_per_minute"] = right * 60.0 / seconds
        if right:
            derived["seconds_per_right"] = seconds / right
    if paid > 0:
        derived["right_per_1k"] = right * 1000.0 / paid
        if right:
            derived["tokens_per_right"] = paid / right
    if memory > 0:
        derived["right_per_gb"] = right * 2**30 / memory
    return {"at": str(record.get("at") or ""), "label": str(record.get("label") or ""),
            "server": {**server, "host": host, "commit": commit}, "rows": [],
            "totals": dict(record), "derived": derived}


def _said(value: Any) -> bool:
    """Whether a flattened field carries anything: not None, "", {} or False -- and 0 is
    a number, which ``0 == False`` would have dropped."""
    if value is None or isinstance(value, bool):
        return bool(value)
    return value not in ("", {})


def import_runs(path_or_json: str | Path | Sequence[Mapping[str, Any]] | Mapping[str, Any],
                into: str | Path, *, host: str, commit: str = "",
                log: Callable[[str], None] = print) -> list[str]:
    """Put a peer's runs into ``into`` as new ``bench:`` docs with ``server["host"]`` and
    ``server["commit"]`` set. Returns the keys written.

    ``path_or_json`` is a file ``ml-stack-bench show --export`` wrote on the peer, the
    JSON text of one, the list it holds, or what `bench_export` answered (``{"runs":
    [...]}``); a whole run record, rows and all, is taken as it is. A run already in
    ``into`` -- same label, ``at`` and host -- is skipped, and nothing there is ever
    overwritten: a key that exists gets ``-n`` on the newcomer. Usable by hand for a peer
    with no daemon: export there, copy the file, import here.
    """
    from ml_stack.graph.store import GraphStore

    records, said = _records(path_or_json)
    commit = commit or said
    with GraphStore(into) as writer:
        kept = writer.docs()
        present = {(str(v.get("label", "")), str(v.get("at", "")),
                    str((v.get("server") or {}).get("host") or ""))
                   for k, v in kept.items() if k.startswith("bench:") and isinstance(v, dict)}
        written: list[str] = []
        skipped = 0
        for record in records:
            doc = _doc_from(record, host=host, commit=commit)
            mark = (doc["label"], doc["at"], host)
            if mark in present:
                skipped += 1
                continue
            present.add(mark)
            stem = f"bench:{doc['label']}:{doc['at'].replace('-', '').replace(':', '')}@{host}"
            key, n = stem, 1
            while writer.get_doc(key) is not None:
                key, n = f"{stem}-{n}", n + 1
            writer.put_doc(key, json.loads(json.dumps(doc)))
            written.append(key)
    log(f"  {host}: imported {len(written)} run(s) into {into}"
        + (f", {skipped} already there" if skipped else ""))
    return written


def _records(source: Any) -> tuple[list[Mapping[str, Any]], str]:
    """The runs in an export -- a file, JSON text, a list, or a `bench_export` answer --
    and the commit the answer named, "" when it named none."""
    if isinstance(source, (str, Path)):
        text = str(source)
        if not text.lstrip().startswith(("[", "{")):
            text = Path(source).expanduser().read_text(encoding="utf-8")
        source = json.loads(text)
    commit = ""
    if isinstance(source, Mapping):
        commit = str(source.get("commit") or "")
        source = source.get("runs") or []
    if not isinstance(source, (list, tuple)):
        raise ValueError("an export is a list of runs, or {'runs': [...]}")
    return [r for r in source if isinstance(r, Mapping)], commit


def gather(handles: Sequence[Handle], *, into: str | Path,
           log: Callable[[str], None] = print) -> dict[str, list[str]]:
    """Bring home what each dispatched job measured: every peer's runs kept since its job
    started, imported into ``into`` by `import_runs` with the peer's name as host and the
    peer's commit. A refused or never-started job has nothing to gather. Returns the keys
    written per peer; a peer whose export holds nothing is said, since a job marked done
    that kept no run is the thing worth noticing."""
    out: dict[str, list[str]] = {}
    for handle in handles:
        if not handle.id:
            continue
        try:
            answered = bench_export(handle.peer, job=handle.id, full=True)
        except Exception as exc:  # noqa: BLE001 - said, and the others still come home
            log(f"  {handle.peer_name}: could not export: {exc}")
            continue
        host = str(answered.get("host") or handle.peer_name)
        if not answered.get("runs"):
            log(f"  {host}: kept no run since {answered.get('since', '?')}"
                + (f" ({answered['skipped']} not over the invented community)"
                   if answered.get("skipped") else ""))
            out[host] = []
            continue
        out[host] = import_runs(answered, into, host=host,
                                commit=str(answered.get("commit") or ""), log=log)
    return out
