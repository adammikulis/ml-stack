"""``land up|down|status``: keep the landing runner (``land serve``) running, safe to leave alone.

``up`` starts a detached supervisor and returns; the supervisor runs ``land serve`` and starts it
again with a growing delay whenever it exits, so a crash costs seconds, not a queue. Its state is one
JSON file and its output one log, both in the workspace directory, so ``digest --status`` can show
whether it is alive. Only one supervisor runs per workspace (a lock), and stopping it stops the runner
cleanly (the runner releases its claim and lock on SIGTERM).
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from ml_stack.lock import Busy, only_one, pid_alive
from ml_stack.workspace import landing

HERE = Path(__file__).resolve().parent
MIN_DELAY_S = 1.0
MAX_DELAY_S = 60.0
STABLE_S = 120.0
LOG_LIMIT = 5 * 1024 * 1024
STOP_GRACE_S = 30.0


def files(base: Path) -> tuple[Path, Path, Path]:
    """The status file, the log and the lock of the supervisor for the workspace at ``base``."""
    return base / landing.SUPERVISOR_STATUS, base / "land-runner.log", base / "land-runner.lock"


def status(base: Path) -> dict:
    """The supervisor's recorded state with ``alive`` set from its pid; empty when never started."""
    return landing.supervisor_state(base)


def flag_argv(root: Path, flags: dict) -> list[str]:
    """The command-line flags a supervisor and the runner it keeps share."""
    return ["--repo", str(root), "--interval", str(flags["interval"]), "--stall-minutes", str(flags["stall_minutes"]),
            "--remote", flags["remote"], *(["--target", flags["target"]] if flags.get("target") else [])]


def serve_argv(root: Path, flags: dict) -> list[str]:
    """The ``land serve`` command the supervisor keeps running."""
    return [sys.executable, str(HERE / "land"), "serve", *flag_argv(root, flags)]


def up(base: Path, root: Path, flags: dict) -> dict:
    """Start the supervisor unless one is alive; its state once it answers."""
    now = status(base)
    if now.get("alive"):
        return {**now, "started": False}
    _, log, _ = files(base)
    log.parent.mkdir(parents=True, exist_ok=True)
    argv = [sys.executable, str(HERE / "land"), "supervise", *flag_argv(root, flags)]
    with log.open("ab") as out:
        child = subprocess.Popen(argv, cwd=root, stdin=subprocess.DEVNULL, stdout=out, stderr=subprocess.STDOUT,
                                 start_new_session=True)
    end = time.monotonic() + 10
    while time.monotonic() < end:
        now = status(base)
        if now.get("alive") and now.get("supervisor") == child.pid:
            return {**now, "started": True}
        time.sleep(0.1)
    return {"alive": False, "started": False, "error": f"no answer from the supervisor; see {log}"}


def down(base: Path) -> dict:
    """Ask the supervisor to stop and wait for it; whether it is gone."""
    now = status(base)
    pid = int(now.get("supervisor") or 0)
    if not now.get("alive"):
        return {"alive": False, "stopped": False, "note": "no supervisor was running"}
    os.kill(pid, signal.SIGTERM)
    end = time.monotonic() + STOP_GRACE_S + 5
    while time.monotonic() < end and pid_alive(pid):
        time.sleep(0.1)
    return {"alive": pid_alive(pid), "stopped": not pid_alive(pid)}


def record(base: Path, **fields: object) -> None:
    """Write the supervisor's state file."""
    state = files(base)[0]
    tmp = state.with_suffix(".tmp")
    tmp.write_text(json.dumps({"supervisor": os.getpid(), "updated": time.time(), **fields}), encoding="utf-8")
    tmp.replace(state)


def rotate(log: Path) -> None:
    """Keep the log bounded: the old one becomes ``.1`` once it passes the limit."""
    if log.exists() and log.stat().st_size > LOG_LIMIT:
        log.replace(log.with_name(log.name + ".1"))


def stop_child(child: subprocess.Popen) -> None:
    """Ask the runner to stop (it releases its claim and lock); kill it only after the grace."""
    child.send_signal(signal.SIGTERM)
    try:
        child.wait(STOP_GRACE_S)
    except subprocess.TimeoutExpired:
        child.kill()
        child.wait()


def supervise(base: Path, argv: list[str], delays: tuple[float, float, float] = (MIN_DELAY_S, MAX_DELAY_S, STABLE_S)) -> int:
    """Run ``argv`` until told to stop, starting it again after each exit; 1 if another supervisor runs."""
    low, high, stable = delays
    stopping = {"now": False}
    signal.signal(signal.SIGTERM, lambda *_: stopping.update(now=True))
    signal.signal(signal.SIGINT, lambda *_: stopping.update(now=True))
    _, log, lock = files(base)
    try:
        with only_one(lock, wait=False, note="landing runner supervisor"):
            return loop(base, argv, log, stopping, (low, high, stable))
    except Busy:
        print("land: a supervisor already runs for this workspace", file=sys.stderr)
        return 1


def loop(base: Path, argv: list[str], log: Path, stopping: dict, delays: tuple[float, float, float]) -> int:
    """The restart loop: start, wait, record, back off, repeat."""
    low, high, stable = delays
    delay, restarts, began = low, 0, time.time()
    while not stopping["now"]:
        rotate(log)
        with log.open("ab") as out:
            child = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=out, stderr=subprocess.STDOUT)
        record(base, child=child.pid, restarts=restarts, since=began, log=str(log), state="running")
        ran = time.monotonic()
        while child.poll() is None and not stopping["now"]:
            time.sleep(0.2)
        if stopping["now"]:
            if child.poll() is None:
                stop_child(child)
            break
        delay = low if time.monotonic() - ran >= stable else min(delay * 2, high)
        restarts += 1
        record(base, child=0, restarts=restarts, since=began, log=str(log), state=f"restarting in {delay:g}s",
               last_exit=child.returncode)
        end = time.monotonic() + delay
        while time.monotonic() < end and not stopping["now"]:
            time.sleep(0.2)
    record(base, child=0, restarts=restarts, since=began, log=str(log), state="stopped")
    return 0
