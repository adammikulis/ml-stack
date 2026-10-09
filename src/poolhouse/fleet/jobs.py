"""The daemon's work queue: a `Job`, and the `JobRunner` that owns their processes."""

from __future__ import annotations

import contextlib
import os
import secrets
import subprocess
import threading
import time
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from poolhouse.credentials import child_environment
from poolhouse.log import say
from poolhouse.platform import process_group_kwargs, stop_gently, stop_pid

from . import job_exit
from .environment import Environment
from .job_records import Records, ownership, owns, process_started


class DaemonError(RuntimeError):
    pass


@dataclass
class Job:
    id: str
    name: str
    argv: list[str]
    cwd: str
    state: str = "queued"          # queued | preparing | running | done | failed | stopped
    pid: int | None = None
    process_started: float | None = None
    recovery_note: str = ""
    capacity_held: bool = False
    returncode: int | None = None
    submitted_at: float = 0.0
    started_at: float | None = None
    finished_at: float | None = None
    env: dict[str, str] = field(default_factory=dict)
    log: str = ""
    """Where its output goes, when that is not the runner's own ``job.log``: a bench job
    started with ``--detach`` writes under the bench's home, and the runner reads that."""

    def public(self) -> dict[str, Any]:
        """The view clients see. A launch in progress is still waiting from where they stand;
        the durable record keeps the exact state."""
        d = asdict(self)
        if self.state == "launching":
            d["state"] = "queued"
        if self.started_at:
            end = self.finished_at or time.time()
            d["elapsed_s"] = round(end - self.started_at, 1)
        return d


class JobRunner:
    """Runs jobs, ``slots`` at a time, and owns their processes."""

    def __init__(self, root: Path, files_root: Path | None = None, *,
                 slots: int = 1, gate: Callable[[], tuple[bool, str]] | None = None,
                 environment: Environment | None = None) -> None:
        if slots < 1:
            raise DaemonError(f"slots must be at least 1, got {slots}")
        self.root = root
        self.files_root = Path(files_root) if files_root is not None else root / "files"
        self.gate = gate
        self.environment = environment
        self.slots = slots
        self.jobs: dict[str, Job] = {}
        self._queue: list[str] = []
        self._running: dict[str, subprocess.Popen] = {}
        self._adopted: set[str] = set()
        self._starting: set[str] = set()
        self._recovered: set[str] = set()
        self._uncertain: set[str] = set()
        self._settled: dict[str, threading.Event] = {}
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._held_because = ""
        self._records = Records(self.files_root / "jobs")
        with contextlib.ExitStack() as cleanup:
            cleanup.callback(self._records.close)
            self._restore()
            self._spawn(slots)
            cleanup.pop_all()

    def _restore(self):
        for row in self._records.load():
            values = {key: value for key, value in row.items() if key in Job.__dataclass_fields__}
            job = Job(**values)
            self.jobs[job.id] = job
            if job.state == 'queued' and row.get('version') == 2:
                self._queue.append(job.id)
            elif row.get('version') == 2 and job.state == 'running' and owns(job):
                self._adopted.add(job.id)
                self._recovered.add(job.id)
            elif self._finished(job):
                pass
            elif job.state in ('queued', 'launching', 'preparing', 'running'):
                job.capacity_held = ownership(job) is None
                job.state = 'interrupted'
                job.finished_at = time.time()
                job.recovery_note = 'The prior execution outcome is unknown; the job was not replayed.'
            if job.capacity_held:
                self._uncertain.add(job.id)
            self.record(job)
        self._queue.sort(key=lambda ident: (self.jobs[ident].submitted_at, ident))

    def _finished(self, job):
        """Settle a job whose process is gone from the exit status its own launcher recorded."""
        if job.state not in ('launching', 'running') or ownership(job) is not False:
            return False
        code = job_exit.read(self.job_dir(job.id), job.pid)
        if code is None:
            return False
        job.returncode, job.finished_at = code, time.time()
        job.state = 'done' if code == 0 else 'failed'
        job.capacity_held = False
        job.recovery_note = 'The job finished while the daemon was restarting; its exit code was recovered.'
        return True

    def _reconcile(self):
        for ident in list(self._recovered):
            job = self.jobs[ident]
            saved = self._records.read(self.job_dir(ident) / 'job.json')
            if saved['state'] in ('done', 'failed', 'stopped'):
                for key in Job.__dataclass_fields__:
                    if key in saved:
                        setattr(job, key, saved[key])
                self._recovered.discard(ident)
            elif job.state == 'running' and not owns(job) and self._finished(job):
                self._recovered.discard(ident)
                self.record(job)
            elif job.state == 'running' and not owns(job):
                job.capacity_held = ownership(job) is None
                if job.capacity_held:
                    self._uncertain.add(ident)
                job.state = 'interrupted'
                job.finished_at = time.time()
                job.recovery_note = 'The recovered process exited without a durable terminal outcome.'
                self.record(job)

    def _spawn(self, upto: int) -> None:
        while len(self._threads) < upto:
            index = len(self._threads)
            worker = threading.Thread(target=self._run_loop, args=(index,), daemon=True,
                                      name=f"jobrunner-{index}")
            self._threads.append(worker)
            worker.start()

    def set_slots(self, slots: int) -> int:
        """Change how many jobs run at once, without a restart."""
        if slots < 1:
            raise DaemonError(f"slots must be at least 1, got {slots}")
        self.slots = slots
        self._spawn(slots)
        self._wake.set()
        return slots

    # -- state, under the lock -------------------------------------------
    def status(self) -> dict[str, Any]:
        """Capacity and what is on it. Taken under the lock, because a placement loop"""
        with self._lock:
            self._reconcile()
            running = sorted({*self._running, *self._starting, *self._uncertain, *(j for j in self._adopted
                                                 if self.jobs[j].state == "running")})
            return {"slots": self.slots, "free": 0 if self._uncertain else max(0, self.slots - len(running)),
                    "busy": bool(running), "running": running,
                    "queued": len(self._queue)}

    def snapshot(self) -> list[dict[str, Any]]:
        """Every job's public view. Copied under the lock -- iterating ``self.jobs``"""
        with self._lock:
            self._reconcile()
            return [j.public() for j in list(self.jobs.values())]

    # -- paths -----------------------------------------------------------
    def job_dir(self, job_id: str) -> Path:
        return self.files_root / "jobs" / job_id

    def log_path(self, job_id: str) -> Path:
        job = self.jobs.get(job_id)
        if job is not None and job.log:
            return Path(job.log)
        return self.job_dir(job_id) / "job.log"

    def hold(self, job: Job) -> Job:
        """List a job some other process will own once it starts, as ``preparing``, so it
        is polled and stopped like any other until `adopt` takes it in."""
        job.submitted_at = job.submitted_at or time.time()
        job.state = "preparing"
        self.job_dir(job.id).mkdir(parents=True, exist_ok=True)
        with self._lock:
            self.jobs[job.id] = job
            self._adopted.add(job.id)
        self.record(job)
        return job

    def adopt(self, job: Job) -> Job:
        """Take in a job some other process owns -- a bench started detached, whose pid
        and log are known -- so it is listed, polled and stopped like one of ours; a job
        `stop` ended while it was preparing stays stopped and its pid is stopped.

        Whoever adopts it settles it: the runner has no ``Popen`` to wait on, so the
        adopter watches the pid and calls `record` with the state it ended in.
        """
        if not job.pid:
            raise DaemonError("an adopted job needs the pid of the process that owns it")
        job.submitted_at = job.submitted_at or time.time()
        job.started_at = job.started_at or time.time()
        job.process_started = process_started(job.pid)
        self.job_dir(job.id).mkdir(parents=True, exist_ok=True)
        with self._lock:
            stopped = job.state == "stopped"
            if not stopped:
                job.state = "running"
            self.jobs[job.id] = job
            self._adopted.add(job.id)
        if owns(job):
            job.capacity_held = False
            self._uncertain.discard(job.id)
        if stopped and owns(job):
            with contextlib.suppress(OSError):
                stop_pid(job.pid)
        self.record(job)
        return job

    def record(self, job: Job) -> None:
        """Write a job's public view beside its log, as `_run_one` does on every change."""
        self._records.write(job)

    # -- submission ------------------------------------------------------
    def submit(self, name: str, argv: Iterable[str], cwd: str,
               env: dict[str, str] | None = None) -> Job:
        argv = [str(a) for a in argv]
        if not argv:
            raise DaemonError("empty argv")
        job_id = f"{int(time.time())}-{secrets.token_hex(3)}"
        job = Job(id=job_id, name=name or argv[0], argv=argv, cwd=cwd,
                  submitted_at=time.time(),
                  env={str(k): str(v) for k, v in (env or {}).items()})
        with self._lock:
            if self._stop.is_set() or self._records.fd is None:
                raise DaemonError('the Fleet runner is frozen; submit to its active replacement')
            self.record(job)
            self.jobs[job_id] = job
            self._queue.append(job_id)
        self._wake.set()
        return job

    def stop(self, job_id: str, *, grace_s: float = 30.0) -> Job:
        with self._lock:
            job = self.jobs.get(job_id)
            if job is None:
                raise DaemonError(f"unknown job {job_id}")
            if job_id in self._uncertain:
                raise DaemonError('this interrupted launch requires verified process recovery before releasing capacity')
            if job.state == "queued":
                self._queue.remove(job_id)
                job.state = "stopped"
                job.finished_at = time.time()
                self.record(job)
                return job
            proc = self._running.get(job_id)
            adopted = job_id in self._adopted
        if proc is None:
            with self._lock:
                preparing = adopted and job.state == "preparing"
                if preparing:
                    job.state = "stopped"
            if preparing:
                job.finished_at = time.time()
                self.record(job)
            if adopted and job.state == "running" and job.pid and owns(job):
                # By pid, never by name: the bench turns the signal into an exit that
                # takes its served model down, and nobody else's server with it. There
                # is no Popen to hand `stop_gently`, so `stop_pid` does the same by pid.
                with contextlib.suppress(OSError):
                    stop_pid(job.pid)
                job.state = "stopped"
                job.finished_at = time.time()
                self.record(job)
            return job
        # SIGTERM, or on Windows a CTRL_BREAK_EVENT aimed at the job's own process group
        # -- the one thing there a checkpointing loop can catch (as SIGBREAK).
        stop_gently(proc)
        deadline = time.time() + grace_s
        while time.time() < deadline and proc.poll() is None:
            time.sleep(0.25)
        if proc.poll() is None:
            proc.kill()
        job.state = "stopped"
        self.record(job)
        return job

    # -- the loop --------------------------------------------------------
    def _run_loop(self, index: int = 0) -> None:
        while not self._stop.is_set():
            if index >= self.slots:
                time.sleep(0.5)
                continue
            self._wake.wait(timeout=1.0)
            self._wake.clear()
            if self.gate is not None:
                allowed, why = self.gate()
                self._holding(why if not allowed else "")
                if not allowed:
                    continue
            with self._lock:
                self._reconcile()
                busy = len(self._running) + len(self._starting) + len(self._uncertain) + sum(self.jobs[ident].state == "running" for ident in self._adopted)
                if self._stop.is_set() or self._uncertain or busy >= self.slots or not self._queue:
                    continue
                job_id = self._queue.pop(0)
                job = self.jobs[job_id]
                self._starting.add(job_id)
                job.state = "launching"
                self.record(job)
            self._run_one(job)

    def _holding(self, why: str) -> None:
        """Say, once each time it changes, why queued work is waiting. A job that sits
        queued with no word from the daemon looks like a hang, and its log -- not the
        gate's return value, which nobody sees -- is where a person goes to find out."""
        with self._lock:
            waiting = len(self._queue)
            if why and not waiting:
                return
            if why == self._held_because:
                return
            self._held_because = why
        say(f"  holding {waiting} queued job(s): {why}" if why
            else "  taking queued work again", flush=True)

    def _environment(self, job):
        env = {**child_environment(), **job.env, "PYTHONUNBUFFERED": "1",
               "POOLHOUSE_JOB_ID": job.id,
               "POOLHOUSE_JOB_DIR": str(self.job_dir(job.id)),
               "POOLHOUSE_FILES_ROOT": str(self.files_root),
               "POOLHOUSE_OUT": str(self.job_dir(job.id) / "out")}
        if self.environment is not None and self.environment.exists:
            env["POOLHOUSE_PYTHON"] = str(self.environment.python)
            env["PATH"] = os.pathsep.join(
                [str(self.environment.python.parent), env.get("PATH", "")])
        return env

    def _launch(self, job, fh, env):
        """Start the job's process and checkpoint its ownership, or None when admission is frozen."""
        with self._lock:
            if self._stop.is_set():
                return None
            job.state = 'launching'
            self.record(job)
            self._settled[job.id] = threading.Event()
            job_exit.clear(self.job_dir(job.id))
            proc = subprocess.Popen(job_exit.command(self.job_dir(job.id), job.argv),
                                    cwd=job.cwd or None, env=env, stdout=fh, stderr=subprocess.STDOUT,
                                    **process_group_kwargs())
            self._starting.discard(job.id)
            self._running[job.id] = proc
            job.state = "running"
            job.pid = proc.pid
            job.process_started = process_started(proc.pid)
            job.started_at = time.time()
            self.record(job)
            return proc

    def _run_one(self, job: Job) -> None:
        log = self.log_path(job.id)
        log.parent.mkdir(parents=True, exist_ok=True)
        env = self._environment(job)
        try:
            program = Path(job.argv[0])
            managed = (program.name in {"python", "python3", "python.exe"}
                       or program.name.startswith("poolhouse-")
                       or (self.environment is not None and program == self.environment.python))
            if self.environment is not None and managed:
                self.environment.require_current_runtime()
            with log.open("ab") as fh:
                # Its own process group (a session on POSIX, CREATE_NEW_PROCESS_GROUP on
                # Windows), so a stop reaches this job and nothing beside it.
                proc = self._launch(job, fh, env)
                if proc is None:
                    return
                rc = proc.wait()
                recorded = job_exit.read(self.job_dir(job.id), proc.pid)
                rc = rc if recorded is None else recorded
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            if proc := self._running.get(job.id):
                self._stop.set()
                job.recovery_note = 'Process ownership checkpoint failed; admission is frozen.'
                with log.open('a') as failure_log:
                    failure_log.write(f'ownership checkpoint failed: {exc}\n')
                job.returncode = proc.wait()
                job.state = 'done' if job.returncode == 0 else 'failed'
            else:
                job.state = "failed"
                job.returncode = -1
                log.write_text(f"failed to start: {exc}\n")
        else:
            job.returncode = rc
            if job.state != "stopped":
                job.state = "done" if rc == 0 else "failed"
        finally:
            if job.id in self._running or not self._stop.is_set():
                job.finished_at = time.time()
                with self._lock:
                    self._running.pop(job.id, None)
                    self._starting.discard(job.id)
                    self.record(job)
                if settled := self._settled.get(job.id):
                    settled.set()
            self._wake.set()

    def stop_running(self, *, grace_s: float = 30.0) -> list[str]:
        """TERM everything in flight and put it back on the queue."""
        with self._lock:
            running = list(self._running)
        for job_id in running:
            job = self.jobs.get(job_id)
            try:
                self.stop(job_id, grace_s=grace_s)
            except DaemonError:
                continue
            settled = self._settled.get(job_id)
            if job is not None and settled and settled.wait(timeout=5):
                job.state = "queued"
                job.pid = job.returncode = job.process_started = None
                job.started_at = job.finished_at = None
                self.record(job)
                with self._lock:
                    self._queue.insert(0, job_id)
        self._wake.set()
        return running

    def checkpoint_restart(self) -> dict[str, int]:
        """Freeze execution and persist process ownership without stopping child processes."""
        with self._lock:
            original_queue = list(self._queue)
            original_states = {ident: self.jobs[ident].state for ident in self._starting}
            with contextlib.ExitStack() as rollback:
                rollback.callback(self._restore_checkpoint, original_queue, original_states)
                for ident in self._starting:
                    if ident not in self._running:
                        self.jobs[ident].state = 'queued'
                        if ident not in self._queue:
                            self._queue.append(ident)
                for job in self.jobs.values():
                    self.record(job)
                self._stop.set()
                rollback.pop_all()
            running = len(self._running) + len(self._uncertain) + sum(self.jobs[ident].state == 'running'
                                               for ident in self._adopted)
            return {'queued': len(self._queue), 'running': running}

    def _restore_checkpoint(self, queue, states):
        self._queue = queue
        for ident, state in states.items():
            self.jobs[ident].state = state

    def shutdown(self) -> None:
        with self._lock:
            self._stop.set()
            self._records.close()
        self._wake.set()
