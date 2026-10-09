"""Durable agent runtime repair through the owned replacement boundary."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

from ml_stack import agent_dependency
from ml_stack.http import ServerError
from ml_stack.lock import Busy, only_one
from ml_stack.log import warn
from ml_stack.platform import start_process
from ml_stack.redact.secrets import env_secrets, redact
from ml_stack.serve.process import pid_exists, started_at

from . import updates
from .daemon_control import ControlError, request_replacement
from .launch import _wait_for_exit, already_running, wait_for_health
from .runtime_candidates import Candidates, runtime_environment
from .setup_jobs import Jobs, jobs

KIND = "agent-runtime"


def job_id(value: str) -> str:
    """Return a bounded canonical setup-job identifier."""
    if not isinstance(value, str) or len(value) != 32 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("Agent runtime jobs require a 32-character lowercase hexadecimal identifier.")
    return value


def safe_error(error):
    """Return a bounded credential-redacted installation error."""
    return redact(str(error), env_secrets(os.environ))[0][:1000]


def resume_stored(ui, root):
    """Resume a repair persisted before the daemon restarted, when a queue exists."""
    if ui is not None and (Path(root) / "setup-jobs.db").exists():
        resume(ui, jobs(ui))


def run(root, job):
    """Install as the replacement process and return its exit status."""
    try:
        return install(root, job)
    except ValueError as exc:
        warn(safe_error(exc))
        return 1


def _waiting(queue):
    return next((row for row in queue.all() if row["kind"] == KIND and row["state"] == "waiting"), None)


def _alive(row):
    pid = row.get("worker_pid")
    return bool(pid and pid_exists(pid) and started_at(pid) == row.get("worker_born"))


def resume(ui, queue):
    """Resume a persisted repair without occupying the setup download worker."""
    row = _waiting(queue)
    if row is None or _alive(row):
        return
    active = getattr(ui, "_runtime_repair_thread", None)
    if active and active.is_alive():
        return
    ui._runtime_repair_thread = threading.Thread(target=_prepare, args=(ui, queue),
                                                daemon=True, name="runtime-repair-preparation")
    ui._runtime_repair_thread.start()


def _prepare(ui, queue):
    row = _waiting(queue)
    if row is None:
        return
    try:
        job_id(row["id"])
        target = updates.running_path()
        if target is None or target.suffix != ".app":
            raise ValueError("Agent runtime installation requires an installed macOS app.")
        if not agent_dependency.problem() and not row["request"].get("repair"):
            queue._update(row, state="done", note="Agent runtime ready", result={"runtime_ready": True})
            return
        registry = Candidates(ui.root)
        candidate = registry.select(target.name)
        while candidate is None:
            queue._update(row, note="Waiting for a verified agent runtime build. Preparation is queued.")
            time.sleep(10)
            if not _waiting(queue):
                return
            candidate = registry.select(target.name)
        running = wait_for_health(ui.peer_port, seconds=60)
        if not running or not running.get("launcher_control"):
            raise ValueError("The running app has no owned update boundary; update the app before repairing its runtime.")
        result = {"candidate": candidate, "port": ui.peer_port,
                  "instance": running["launcher_control"]}
        queue._update(row, note="Waiting for active work to finish before installing", result=result)
        process = start_process(
            [sys.executable, "-m", "ml_stack.cli.daemon", "--root", str(ui.root),
             "--agent-runtime-job", row["id"]],
            env=runtime_environment(), stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        queue._update(row, worker_pid=process.pid, worker_born=started_at(process.pid))
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        queue._update(row, state="failed", error=safe_error(error), note="Agent runtime installation failed")


def request(ui, provenance, *, repair=False):
    """Queue the fixed local agent runtime installation action."""
    queue = jobs(ui)
    row = _waiting(queue) or queue.start(KIND, {"repair": repair}, None, provenance=provenance)
    resume(ui, queue)
    return row


def _admit(root, row, candidate):
    port, instance = row["result"]["port"], row["result"]["instance"]
    while True:
        running = already_running(port)
        if running is None or running.get("launcher_control") != instance:
            raise ControlError("The running app changed before repair admission; retry the installation.")
        try:
            request_replacement(root, port, running, candidate["commit"])
            break
        except Busy:
            time.sleep(2)
        except ServerError as error:
            if error.status != 409:
                raise
            time.sleep(2)
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        if already_running(port) is None:
            return port
        time.sleep(0.2)
    raise ControlError("The admitted app did not exit; its installation was left unchanged.")


def install(root, job):
    """Install a registered artifact and confirm readiness from the replacement app."""
    job = job_id(job)
    queue = Jobs(root)
    row = next((entry for entry in queue.all() if entry["id"] == job and entry["kind"] == KIND), None)
    if row is None or row["state"] != "waiting":
        raise ValueError("This agent runtime repair job is not waiting.")
    drained = False
    process = None
    target = None
    replacement_instance = None
    try:
        with only_one(Path(root) / "runtime-repair-worker.lock", wait=False):
            target = updates.running_path()
            if target is None or target.suffix != ".app":
                raise ValueError("Runtime repair must execute from its installed app.")
            candidate = Candidates(root).select(target.name)
            if candidate is None or candidate != row["result"].get("candidate"):
                raise ValueError("The selected immutable artifact changed; retry the installation.")
            if target.with_name(target.name + ".old").exists():
                raise ValueError("An earlier runtime backup needs recovery before installation.")
            port = _admit(root, row, candidate)
            drained = True
            with only_one(Path(root) / "runtime-install.lock", wait=False):
                queue._update(row, note="Installing the verified agent runtime")
                if Candidates(root).select(target.name) != candidate:
                    raise ValueError("The registered artifact changed at installation admission.")
                archive = Path(root) / "runtime-candidates" / (candidate["sha256"] + ".zip")
                updates.install(archive, app_path=target, keep_backup=True)
                executable = target / "Contents/MacOS/ml-stack-headless"
                process = _start(target, root, port)
            queue._update(row, note="Waiting for the installed agent runtime to become ready")
            health = wait_for_health(port, seconds=60)
            if not health or health.get("commit") != candidate["commit"]:
                raise ValueError("The installed app did not report the selected source identity.")
            replacement_instance = health.get("launcher_control")
            checked = subprocess.run([str(executable), "--check-agent-runtime"], env=runtime_environment(),
                                     capture_output=True, text=True, check=True, timeout=30)
            report = json.loads(checked.stdout)
            if report.get("runtime_ready") is not True or report.get("commit") != candidate["commit"]:
                raise ValueError("The installed agent runtime is not ready.")
            queue._update(row, state="done", note="Agent runtime ready",
                          result={**row["result"], "runtime_ready": True})
            updates.complete_install(target)
    except (OSError, ValueError, ControlError, Busy, ServerError, subprocess.SubprocessError) as error:
        recovery = ""
        if drained:
            recovery = _recover_runtime(root, row, target, process, replacement_instance)
        queue._update(row, state="failed", error=safe_error(error) + recovery, note="Agent runtime installation failed")
        return 1
    return 0


def _start(target, root, port):
    return start_process([str(target / "Contents/MacOS/ml-stack-headless"), "--port", str(port),
                             "--root", str(root), "--no-browser"], env=runtime_environment(),
                            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL)


def _recover_runtime(root, row, target, process, instance):
    port = row["result"]["port"]
    try:
        running = already_running(port)
        if running is not None:
            if not instance or running.get("launcher_control") != instance:
                raise ControlError("The running replacement generation changed; its backup was retained.")
            request_replacement(root, port, running, "rollback")
            if not _wait_for_exit(port):
                raise ControlError("The replacement is still exiting; its backup was retained.")
        elif process is not None and process.poll() is None:
            raise ControlError("The replacement is still starting; its backup was retained for recovery.")
        with only_one(Path(root) / "runtime-install.lock", wait=False):
            updates.restore_install(target)
            _start(target, root, port)
        if wait_for_health(port, seconds=60) is None:
            raise ControlError("The previous app was restored but did not become ready.")
        return " The previous app was restored and restarted."
    except (OSError, ValueError, ControlError, Busy, ServerError, subprocess.SubprocessError) as error:
        return " Recovery requires retry at a safe boundary: " + safe_error(error)
