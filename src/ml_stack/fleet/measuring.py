"""One machine measuring: the code it runs, the job a dispatcher asks it for, and the
detached ``ml-stack-bench`` it starts and watches.

`installed_commit` is the ml-stack this process runs and `same_commit` compares two of
them. `Job` is one ``ml-stack-bench`` invocation on the wire, `jobs_from` builds one per
peer out of a plan, and `Refused` is what a peer answers when it will not take one.
`BenchHost` is the bench side of a daemon; `Local` is this machine answering as a peer
would, for a dispatcher with no daemon of its own.
"""

from __future__ import annotations

import json
import os
import re
import socket
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ml_stack.hub import room as machine_room
from ml_stack.lock import held_by
from ml_stack.paths import repo_root
from ml_stack.units import human_bytes

from .jobs import DaemonError, JobRunner
from .jobs import Job as DaemonJob

__all__ = [
    "FAILED_MARKS",
    "LOCK",
    "STORE",
    "TAIL",
    "BenchHost",
    "Job",
    "Local",
    "Refused",
    "bench_home",
    "detach_bench",
    "ended_badly",
    "here",
    "installed_commit",
    "jobs_from",
    "same_commit",
]

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


def bench_home(traind_root: Path | str | None = None) -> Path:
    """Where ``ml-stack-bench`` keeps everything on this machine: the lock, the store, the
    logs and ``measuring.json``.

    Given a daemon's root, the ``bench`` directory beside it -- ``~/.ml-stack/traind``
    measures in ``~/.ml-stack/bench``, and a daemon rooted in a test's directory looks
    beside *that*, never in the real home. A daemon that read ``~/.ml-stack/bench``
    whatever its root was told the truth about the wrong machine: a training job sent to a
    daemon under test stayed queued for as long as a real benchmark was measuring on the
    developer's box. Without a root, ``~/.ml-stack/bench``: the same path
    `ml_stack.bench.home_dir` names, written here rather than imported so a daemon that
    never measures never loads the bench.
    """
    if traind_root is not None:
        return Path(traind_root).expanduser().parent / "bench"
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



# -- the daemon side -----------------------------------------------------------------
def detach_bench(line: Sequence[str], home: Path) -> tuple[int, Path]:
    """Start ``ml-stack-bench <line> --no-queue --detach`` here and return the child's pid
    and log. ``--detach`` re-runs the command in its own session with its output in a log
    under ``home/logs`` and writes the pid into ``home/measuring.json``, which is where
    this reads it back; ``--no-queue`` makes a race for the lock a failed job rather than a
    run queued behind another, since the lock was checked before this was called."""
    done = subprocess.run([sys.executable, "-m", "ml_stack.bench", *line,
                           "--no-queue", "--detach"],
                          capture_output=True, text=True, timeout=120,
                          env={**os.environ, "MLSTACK_BENCH_HOME": str(home)})
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
    machine's ``ml-stack-bench`` keeps its lock, store and logs -- always given, never
    defaulted, because the caller knows its root and this does not (`bench_home`);
    ``commit`` is what this machine runs (`installed_commit` unless given); ``room`` and
    ``launch`` are `hub.room` and `detach_bench` unless a test hands in fakes.
    """

    def __init__(self, runner: "JobRunner", *, home: Path | str,
                 commit: str | None = None, room: Callable[[], int] | None = None,
                 launch: Callable[[Sequence[str], Path], tuple[int, Path]] = detach_bench,
                 name: str = "", poll_s: float = 1.0) -> None:
        self.runner = runner
        self.home = Path(home).expanduser()
        self.commit = installed_commit() if commit is None else commit
        self.room = machine_room if room is None else room
        self.launch = launch
        self.name = name or socket.gethostname()
        self.poll_s = poll_s
        self._mine: dict[str, "DaemonJob"] = {}
        self._lock = threading.Lock()

    # -- what this machine says about itself --
    def lock_held(self) -> str:
        """Who holds the measuring lock -- ``pid N`` -- or "" when nobody does."""
        return held_by(self.home / LOCK)

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
                    raise Refused("room", f"{model} needs {human_bytes(need)} and {self.name} may "
                                          f"use {human_bytes(room)}")
        try:
            pid, log = self.launch(job.argv, self.home)
        except Exception as exc:
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
        from ml_stack import bench as measured

        kept = [r for r in measured.runs(store) if str(r.get("at", "")) >= since]
        if full:
            over, skipped = measured._over_invented(kept, anyway=anyway)
            out["runs"] = [dict(one) for one in over]
        else:
            out["runs"], skipped = measured._exportable(kept, anyway=anyway)
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

