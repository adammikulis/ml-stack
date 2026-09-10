"""Keeping the build fresh on its own: a weekly LaunchAgent on macOS, a Scheduled Task on
Windows, both rerunning ``ml-stack-serve build`` -- a run that fails verification changes
nothing, which is what makes leaving it unattended safe."""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

from ml_stack import home
from ml_stack.log import say, warn
from ml_stack.serve.build_paths import BuildFailed, root

__all__ = ["PERSIST_LABEL", "PERSIST_PLIST", "PERSIST_TASK", "WEEK_SECONDS", "cmd_persist"]

PERSIST_LABEL = "com.ml-stack.llama-build"
PERSIST_PLIST = home.user_home() / "Library" / "LaunchAgents" / f"{PERSIST_LABEL}.plist"
PERSIST_TASK = "MLStackLlamaBuild"
WEEK_SECONDS = 7 * 24 * 60 * 60


def _persist_argv() -> list[str]:
    found = shutil.which("ml-stack-serve")
    if found:
        return [found, "build"]
    return [sys.executable, "-m", "ml_stack.serve.cli", "build"]


def _install_persist_macos(*, every: int = WEEK_SECONDS) -> Path:
    import plistlib

    log_dir = root() / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    plist = {
        "Label": PERSIST_LABEL,
        "ProgramArguments": _persist_argv(),
        "StartInterval": every,
        "StandardOutPath": str(log_dir / "build.log"),
        "StandardErrorPath": str(log_dir / "build.log"),
        "EnvironmentVariables": {"PATH": os.environ.get("PATH", "/usr/bin:/bin")},
    }
    PERSIST_PLIST.parent.mkdir(parents=True, exist_ok=True)
    PERSIST_PLIST.write_bytes(plistlib.dumps(plist))
    subprocess.run(["launchctl", "unload", str(PERSIST_PLIST)],
                   capture_output=True, check=False)
    done = subprocess.run(["launchctl", "load", "-w", str(PERSIST_PLIST)],
                          capture_output=True, text=True, check=False)
    if done.returncode != 0:
        raise BuildFailed(f"launchctl refused the job: {(done.stderr or '').strip()}")
    return PERSIST_PLIST


def _install_persist_windows() -> str:
    """A weekly Scheduled Task, since Windows has no LaunchAgent. Written against a faked
    ``subprocess.run`` only -- report to Adam that this is untested on a real Windows
    machine before relying on it."""
    argv = _persist_argv()
    quoted = " ".join(f'"{a}"' if " " in a else a for a in argv)
    done = subprocess.run(
        ["schtasks", "/Create", "/F", "/TN", PERSIST_TASK, "/TR", quoted,
         "/SC", "WEEKLY", "/RL", "LIMITED"],
        capture_output=True, text=True)
    if done.returncode != 0:
        raise BuildFailed(f"schtasks refused the job: {(done.stderr or '').strip()}")
    return PERSIST_TASK


def cmd_persist() -> int:
    try:
        system = platform.system()
        if system == "Windows":
            name = _install_persist_windows()
            say(f"installed the scheduled task {name!r}")
        elif system == "Darwin":
            path = _install_persist_macos()
            say(f"installed {path}")
        else:
            warn("no scheduled-refresh install for this platform yet; run "
                 "'ml-stack-serve build' from cron or a timer of your own")
            return 1
    except BuildFailed as exc:
        warn(f"error: {exc}")
        return 2
    say(f"  ml-stack-serve build will run every {WEEK_SECONDS // 86400} days -- a "
        "refresh that fails verification changes nothing, which is what makes this safe "
        "to leave unattended")
    return 0
