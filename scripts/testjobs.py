"""Background test jobs: records, a detached runner, status, wait, result and cancel."""

from __future__ import annotations

import json
import os
import secrets
import signal
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import testreuse_store as storage

KEEP_S = 3 * 86400.0
POLL_S = 0.4
TERMINAL = ("done", "failed", "cancelled")
FULL_TIERS = ("fast", "full", "slow", "all", "quick")


class Refused(Exception):
    """A request the job table will not take."""


def is_full(argv: list[str], root: Path) -> bool:
    """Whether a job runs a whole tier: a tier name with no test path in its arguments."""
    return bool(argv) and argv[0] in FULL_TIERS and not any(
        not a.startswith("-") and (root / a.split("::")[0]).exists() for a in argv[1:])


class Jobs:
    """The jobs of one project, under ``folder``."""

    def __init__(self, folder: Path) -> None:
        self.folder = folder / "jobs"

    def path(self, job: str) -> Path:
        """The directory of ``job``."""
        if not job.replace("-", "").isalnum():
            raise Refused(f"no job {job!r}")
        return self.folder / job

    def read(self, job: str, name: str = "status.json") -> dict:
        """A job's status or spec; ``{}`` when absent."""
        try:
            return json.loads((self.path(job) / name).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    def write(self, job: str, name: str, data: dict) -> None:
        """Replace a job's status or spec."""
        storage.write_atomic(self.path(job) / name, json.dumps(data))

    def ids(self) -> list[str]:
        """Every job id, oldest first."""
        return sorted(p.name for p in self.folder.iterdir() if p.is_dir()) if self.folder.is_dir() else []

    def state(self, job: str) -> dict:
        """The job's status with a dead runner reported as failed."""
        status = self.read(job)
        if status.get("state") in ("queued", "running") and not storage.alive(
                int(status.get("pid", 0)), float(status.get("started", 0))) and time.time() - status.get("at", 0) > 5:
            status = {**status, "state": "failed", "detail": "the runner process is gone"}
            self.write(job, "status.json", status)
        return status

    def live_full(self, root: Path) -> list[str]:
        """Ids of unfinished jobs that run a whole tier."""
        return [j for j in self.ids() if self.state(j).get("state") in ("queued", "running")
                and is_full(self.read(j, "spec.json").get("argv", []), root)]

    def prune(self) -> None:
        """Remove finished jobs older than three days."""
        for job in self.ids():
            status = self.state(job)
            if status.get("state") in TERMINAL and time.time() - status.get("finished", time.time()) > KEEP_S:
                for child in self.path(job).iterdir():
                    child.unlink(missing_ok=True)
                self.path(job).rmdir()

    def submit(self, argv: list[str], owner: dict, root: Path, script: Path, extra: dict) -> str:
        """Record a job and start its detached runner; returns the id."""
        self.prune()
        held = self.live_full(root) if is_full(argv, root) else []
        if held:
            who = self.read(held[0], "spec.json").get("owner", {}).get("id") or "unknown"
            raise Refused(f"job {held[0]} ({who}) already runs a whole tier; wait for it or cancel it")
        job = f"{time.strftime('%Y%m%d-%H%M%S')}-{secrets.token_hex(2)}"
        self.write(job, "spec.json", {"id": job, "argv": argv, "owner": owner, "created": time.time(), **extra})
        self.write(job, "status.json", {"state": "queued", "at": time.time()})
        log = (self.path(job) / "log").open("ab")
        child = subprocess.Popen([sys.executable, str(script), "_job", job, *self.passthrough(extra)], cwd=root,
                                 stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                 start_new_session=True, env={**os.environ, "DEV_TEST_JOB": job})
        self.write(job, "status.json", {"state": "queued", "pid": child.pid,
                                        "started": storage.process_start(child.pid), "at": time.time()})
        return job

    @staticmethod
    def passthrough(extra: dict) -> list[str]:
        """Arguments that re-establish the submitter's agent in the runner."""
        return [x for k in ("agent", "label", "task") if extra.get(k) for x in (f"--{k}", extra[k])]

    def wait(self, job: str, timeout: float, tail: bool = False) -> dict:
        """Block until the job ends or ``timeout`` seconds pass; optionally copy its log to stdout."""
        deadline, sent = time.monotonic() + timeout, 0
        while True:
            status = self.state(job)
            if tail:
                sent = self.copy_log(job, sent)
            if status.get("state") in TERMINAL or time.monotonic() >= deadline:
                return status
            time.sleep(POLL_S)

    def copy_log(self, job: str, offset: int) -> int:
        """Write the job's log from ``offset`` to stdout and return the new offset."""
        try:
            data = (self.path(job) / "log").read_bytes()
        except OSError:
            return offset
        sys.stdout.write(data[offset:].decode("utf-8", errors="replace"))
        sys.stdout.flush()
        return len(data)

    def cancel(self, job: str, who: str) -> str:
        """Stop a job owned by ``who``; returns what happened."""
        spec, status = self.read(job, "spec.json"), self.state(job)
        if not spec:
            raise Refused(f"no job {job}")
        if spec["owner"].get("id", "") != who:
            raise Refused(f"job {job} belongs to {spec['owner'].get('id') or 'another run'}")
        if status.get("state") in TERMINAL:
            return f"job {job} already {status['state']}"
        with_pid = int(status.get("pid", 0))
        if storage.alive(with_pid, float(status.get("started", 0))):
            os.kill(with_pid, signal.SIGTERM)
        return f"job {job} cancelled"

    def result(self, job: str, root: Path) -> dict:
        """The job's status, summary, failing node ids and junit files."""
        status = self.state(job)
        junits = sorted(self.path(job).glob("junit-*.xml"))
        failing = []
        for junit in junits:
            for case in ET.parse(junit).getroot().iter("testcase"):  # noqa: S314 - written by pytest
                if any(child.tag in ("failure", "error") for child in case):
                    failing.append(f"{case.get('classname', '').replace('.', '/')}.py::{case.get('name')}")
        return {**status, "failing": failing, "junit": [str(j) for j in junits]}
