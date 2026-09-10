"""The daemon's work queue: a `Job`, and the `JobRunner` that owns their processes."""

from __future__ import annotations

import contextlib
import json
import os
import secrets
import subprocess
import threading
import time
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ml_stack.log import say
from ml_stack.platform import process_group_kwargs, stop_gently, stop_pid

from .environment import Environment


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
    log: str = ""
    """Where its output goes, when that is not the runner's own ``job.log``: a bench job
    started with ``--detach`` writes under the bench's home, and the runner reads that."""

    def public(self) -> dict[str, Any]:
        d = asdict(self)
        if self.started_at:
            end = self.finished_at or time.time()
            d["elapsed_s"] = round(end - self.started_at, 1)
        return d


class JobRunner:
    """Runs jobs, ``slots`` at a time, and owns their processes."""

    def __init__(self, root: Path, files_root: Path | None = None, *,
                 slots: int = 1, gate: "Callable[[], tuple[bool, str]] | None" = None,
                 environment: "Environment | None" = None) -> None:
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
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._held_because = ""
        self._spawn(slots)

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
            running = sorted({*self._running, *(j for j in self._adopted
                                                 if self.jobs[j].state == "running")})
            return {"slots": self.slots, "free": max(0, self.slots - len(running)),
                    "busy": bool(running), "running": running,
                    "queued": len(self._queue)}

    def snapshot(self) -> list[dict[str, Any]]:
        """Every job's public view. Copied under the lock -- iterating ``self.jobs``"""
        with self._lock:
            return [j.public() for j in list(self.jobs.values())]

    # -- paths -----------------------------------------------------------
    def job_dir(self, job_id: str) -> Path:
        return self.files_root / "jobs" / job_id

    def log_path(self, job_id: str) -> Path:
        job = self.jobs.get(job_id)
        if job is not None and job.log:
            return Path(job.log)
        return self.job_dir(job_id) / "job.log"

    def adopt(self, job: Job) -> Job:
        """Take in a job some other process owns -- a bench started detached, whose pid
        and log are known -- so it is listed, polled and stopped like one of ours.

        Whoever adopts it settles it: the runner has no ``Popen`` to wait on, so the
        adopter watches the pid and calls `record` with the state it ended in.
        """
        if not job.pid:
            raise DaemonError("an adopted job needs the pid of the process that owns it")
        job.submitted_at = job.submitted_at or time.time()
        job.started_at = job.started_at or time.time()
        job.state = "running"
        self.job_dir(job.id).mkdir(parents=True, exist_ok=True)
        with self._lock:
            self.jobs[job.id] = job
            self._adopted.add(job.id)
        self.record(job)
        return job

    def record(self, job: Job) -> None:
        """Write a job's public view beside its log, as `_run_one` does on every change."""
        (self.job_dir(job.id) / "job.json").write_text(json.dumps(job.public(), indent=2))

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
            adopted = job_id in self._adopted
        if proc is None:
            if adopted and job.state == "running" and job.pid:
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
                if not self._queue:
                    continue
                job_id = self._queue.pop(0)
                job = self.jobs[job_id]
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

    def _run_one(self, job: Job) -> None:
        log = self.log_path(job.id)
        log.parent.mkdir(parents=True, exist_ok=True)
        env = {**os.environ, **job.env, "PYTHONUNBUFFERED": "1",
               "ML_STACK_JOB_ID": job.id,
               "ML_STACK_JOB_DIR": str(self.job_dir(job.id)),
               "ML_STACK_FILES_ROOT": str(self.files_root),
               "ML_STACK_OUT": str(self.job_dir(job.id) / "out")}
        if self.environment is not None and self.environment.exists:
            # A job that asks for "python" gets the environment the machine was set up
            # with, not whatever interpreter happens to be first on PATH -- and in a
            # bundled install there is no other one.
            env["ML_STACK_PYTHON"] = str(self.environment.python)
            env["PATH"] = os.pathsep.join(
                [str(self.environment.python.parent), env.get("PATH", "")])
        try:
            with log.open("ab") as fh:
                # Its own process group (a session on POSIX, CREATE_NEW_PROCESS_GROUP on
                # Windows), so a stop reaches this job and nothing beside it.
                proc = subprocess.Popen(job.argv, cwd=job.cwd or None, env=env,
                                        stdout=fh, stderr=subprocess.STDOUT,
                                        **process_group_kwargs())
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
        """TERM everything in flight and put it back on the queue."""
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
