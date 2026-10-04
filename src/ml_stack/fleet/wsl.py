"""Run the Windows application in a Linux WSL distribution."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

from ml_stack.log import say

__all__ = ["WSLError", "command", "prepare", "start"]


class WSLError(RuntimeError):
    """The WSL application runtime could not start."""


def command(*args: str) -> list[str]:
    """A command in the selected WSL distribution."""
    executable = shutil.which("wsl.exe")
    if not executable:
        raise WSLError("Install Ubuntu with WSL, then run ml-stack again.")
    distro = os.environ.get("ML_STACK_WSL_DISTRO", "Ubuntu")
    return [executable, "--distribution", distro, "--exec", *args]


def _read(*args: str) -> str:
    done = subprocess.run(command(*args), capture_output=True, text=True, timeout=60)
    if done.returncode:
        detail = (done.stderr or done.stdout).strip().replace("\x00", "")
        raise WSLError(f"WSL could not run {args[0]}: {detail}")
    return done.stdout.strip()


def prepare() -> str:
    """Check WSL and install the application in its Linux user environment."""
    probe = """import json, os, platform, shutil, sys
print(json.dumps(dict(kernel=platform.release(), python=list(sys.version_info[:2]),
 bwrap=shutil.which('bwrap'), home=os.path.expanduser('~'), gpu=os.path.exists('/dev/dxg'))))
"""
    found = json.loads(_read("python3", "-c", probe))
    if "microsoft" not in found["kernel"].lower() or "wsl2" not in found["kernel"].lower():
        raise WSLError("ml-stack needs a WSL 2 distribution. Convert Ubuntu to WSL 2.")
    if tuple(found["python"]) < (3, 12):
        raise WSLError("Install the Python required by ml-stack in Ubuntu.")
    if not found["bwrap"]:
        raise WSLError("Install bubblewrap in Ubuntu: sudo apt install bubblewrap python3-venv")
    _read(found["bwrap"], "--unshare-all", "--ro-bind", "/", "/", "--", "/usr/bin/true")
    source = Path(__file__).resolve().parents[3]
    if not (source / "pyproject.toml").is_file():
        raise WSLError("The Windows launcher needs the local ml-stack source checkout.")
    linux_source = _read("wslpath", "-a", "-u", str(source))
    runtime = found["home"] + "/.local/share/ml-stack/runtime"
    python = runtime + "/bin/python"
    ready = _read("python3", "-c", "import os,sys; print(int(os.path.isfile(sys.argv[1])))", python)
    if ready != "1":
        say("Preparing ml-stack's Linux runtime in Ubuntu.")
        _read("python3", "-m", "venv", runtime)
    extras = "graph,store,web,hub,memory,plot,fleet-tls,fleet-update,fleet-onboard"
    fingerprint = hashlib.sha256((linux_source + "\n" + extras + "\n").encode()
                                 + (source / "pyproject.toml").read_bytes()).hexdigest()
    marker = runtime + "/ml-stack-install"
    installed = _read(python, "-c", "import pathlib,sys; p=pathlib.Path(sys.argv[1]); "
                      "print(p.read_text() if p.is_file() else '')", marker)
    if installed != fingerprint:
        say("Installing ml-stack's Linux dependencies.")
        done = subprocess.run(command(python, "-m", "pip", "install", "--disable-pip-version-check",
                                      "-e", linux_source + "[" + extras + "]"))
        if done.returncode:
            raise WSLError("The ml-stack Linux dependency installation did not complete.")
        _read(python, "-c", "import pathlib,sys; pathlib.Path(sys.argv[1]).write_text(sys.argv[2])",
              marker, fingerprint)
    return python


def start(argv: list[str], *, executable: str | None = None) -> int:
    """Run the Linux daemon with model confinement enabled."""
    executable = executable or prepare()
    arguments = list(argv)
    environment = ["ML_STACK_SANDBOX_SERVE=1"]
    for name in ("ML_STACK_HOME", "ML_STACK_CACHE"):
        if value := os.environ.get(name):
            translated = _read("wslpath", "-a", "-u", str(Path(value).resolve()))
            environment.append(name + "=" + translated)
    for index, arg in enumerate(arguments):
        name, separator, value = arg.partition("=")
        if name not in {"--root", "--bench-home", "--cluster-key"}:
            continue
        if not separator:
            if index + 1 >= len(arguments):
                continue
            value = arguments[index + 1]
        if not value.startswith("/"):
            value = _read("wslpath", "-a", "-u", str(Path(value).resolve()))
            if separator:
                arguments[index] = name + "=" + value
            else:
                arguments[index + 1] = value
    process = subprocess.Popen(command("env", *environment, executable,
                                       "-m", "ml_stack.fleet.wsl_daemon", *arguments),
                               stdin=subprocess.PIPE)
    try:
        return process.wait()
    except KeyboardInterrupt:
        return 130
    finally:
        if process.stdin:
            process.stdin.close()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.terminate()
