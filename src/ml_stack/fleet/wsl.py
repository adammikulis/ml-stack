"""Run the Windows application in a Linux WSL distribution."""

from __future__ import annotations

import hashlib
import ipaddress
import json
import os
import shutil
import socket
import subprocess
import sys
from pathlib import Path

from ml_stack.log import say, warn
from ml_stack.platform import launch

from . import discovery, runtime_wheel, wsl_network, wsl_ui

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
    done = subprocess.run(command(*args), capture_output=True, text=True,
                          encoding="utf-8", timeout=60)
    if done.returncode:
        detail = (done.stderr or done.stdout).strip().replace("\x00", "")
        raise WSLError(f"WSL could not run {args[0]}: {detail}")
    return done.stdout.strip()


_CACHE_RUNTIME = """import pathlib, shutil, sys, tempfile
from ml_stack.fleet import runtime_wheel
wheel = pathlib.Path(sys.argv[1])
commit = runtime_wheel.wheel_commit(wheel)
with tempfile.TemporaryDirectory(prefix='ml-stack-wsl-wheel-') as temporary:
    copied = pathlib.Path(temporary) / wheel.name
    shutil.copyfile(wheel, copied)
    source = pathlib.Path(sys.argv[2]) if sys.argv[2] else None
    if source is not None and source.is_absolute() and (source / '.git').exists():
        runtime_wheel.stamp(copied, commit, source)
    print(runtime_wheel.cache_wheel(copied, commit, prefix=pathlib.Path(sys.argv[3])))
"""


def prepare() -> str:
    """Check WSL and install the application in its Linux user environment."""
    probe = """import json, os, platform, shutil, sys
print(json.dumps(dict(kernel=platform.release(), python=list(sys.version_info[:2]),
 bwrap=shutil.which('bwrap'), home=os.path.expanduser('~'), gpu=os.path.exists('/dev/dxg'))))
"""
    found = json.loads(_read("python3", "-c", probe))
    if "microsoft" not in found["kernel"].lower() or "wsl2" not in found["kernel"].lower():
        raise WSLError("ml-stack needs a WSL 2 distribution. Use WSL 2 for Ubuntu.")
    if tuple(found["python"]) < (3, 12):
        raise WSLError("Install the Python required by ml-stack in Ubuntu.")
    if not found["bwrap"]:
        raise WSLError("Install bubblewrap in Ubuntu: sudo apt install bubblewrap python3-venv")
    _read(found["bwrap"], "--unshare-all", "--ro-bind", "/", "/", "--", "/usr/bin/true")
    try:
        wheel = runtime_wheel.current_wheel()
        if wheel is None:
            raise OSError("no committed runtime wheel is installed")
        wheel_digest = hashlib.sha256(wheel.read_bytes()).hexdigest()
    except (OSError, ValueError) as exc:
        raise WSLError("Update the Windows ml-stack installation to provide its committed runtime wheel.") from exc
    linux_wheel = _read("wslpath", "-a", "-u", str(wheel))
    source = runtime_wheel.source_checkout()
    linux_source = _read("wslpath", "-a", "-u", str(source)) if source else ""
    runtime = found["home"] + "/.local/share/ml-stack/runtime"
    python = runtime + "/bin/python"
    ready = _read("python3", "-c", "import os,sys; print(int(os.path.isfile(sys.argv[1])))", python)
    if ready != "1":
        say("Preparing ml-stack's Linux runtime in Ubuntu.")
        _read("python3", "-m", "venv", runtime)
    extras = "agents,pi,graph,store,web,hub,memory,plot,fleet-tls,fleet-update,fleet-onboard"
    fingerprint = hashlib.sha256((wheel_digest + "\n" + extras + "\n" + linux_source + "\ncache-v1\n").encode()).hexdigest()
    marker = runtime + "/ml-stack-install"
    installed = _read(python, "-c", "import pathlib,sys; p=pathlib.Path(sys.argv[1]); "
                      "print(p.read_text() if p.is_file() else '')", marker)
    if installed != fingerprint:
        say("Installing ml-stack's Linux dependencies.")
        done = subprocess.run(command(python, "-m", "pip", "install", "--disable-pip-version-check",
                                      "--force-reinstall", "--no-deps", "--no-index", linux_wheel))
        if done.returncode:
            raise WSLError("The ml-stack Linux runtime installation did not complete.")
        done = subprocess.run(command(python, "-m", "pip", "install", "--disable-pip-version-check",
                                      "--upgrade", linux_wheel + "[" + extras + "]"))
        if done.returncode:
            raise WSLError("The ml-stack Linux dependency installation did not complete.")
        cached = _read(python, "-c", _CACHE_RUNTIME, linux_wheel, linux_source, runtime)
        done = subprocess.run(command(python, "-m", "pip", "install", "--disable-pip-version-check",
                                      "--force-reinstall", "--no-deps", "--no-index", cached))
        if done.returncode:
            raise WSLError("The ml-stack Linux runtime installation did not complete.")
        linux_path = _read("printenv", "PATH")
        done = subprocess.run(command("env", "PATH=" + runtime + "/bin:" + linux_path,
                                      runtime + "/bin/npm", "install", "--global", "--prefix", runtime,
                                      "@earendil-works/pi-coding-agent"),
                              capture_output=True, text=True, encoding="utf-8")
        if done.returncode:
            detail = ((done.stderr or done.stdout) or "").strip()[-2000:]
            raise WSLError("The Pi coding-agent installation did not complete: " + detail)
        _read(python, "-c", "import pathlib,sys; pathlib.Path(sys.argv[1]).write_text(sys.argv[2])",
              marker, fingerprint)
    return python


def _port(arguments: list[str]) -> int:
    port = 8770
    for index, arg in enumerate(arguments):
        if arg == "--port" and index + 1 < len(arguments):
            port = int(arguments[index + 1])
        elif arg.startswith("--port="):
            port = int(arg.partition("=")[2])
    return port


def _gateway(host: str) -> str:
    route = _read("ip", "-4", "route", "show", "default").split()
    if "via" not in route:
        return ""
    address = ipaddress.IPv4Address(route[route.index("via") + 1])
    if not address.is_private or address.is_loopback or address.is_unspecified or str(address) == host:
        return ""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.bind((str(address), 0))
    except OSError:
        return ""
    return str(address)


def _bridge(executable: str, arguments: list[str]) -> wsl_network.NetworkBridge | None:
    host = discovery.primary_ip()
    if not host:
        warn("LAN discovery is unavailable while Windows has no LAN address.")
        return None
    linux_host = _read(executable, "-c", "from ml_stack.fleet.discovery import primary_ip; print(primary_ip())")
    port = _port(arguments)
    bridge = wsl_network.NetworkBridge(host, (linux_host, port), discovery.default_group(),
                                      discovery.default_port(), discovery._native_socket)
    bridge.gateway = _gateway(host)
    try:
        return bridge.start()
    except OSError as exc:
        raise WSLError(f"The Windows LAN bridge could not start: {exc}") from exc


def _bridges(executable: str, arguments: list[str]) -> tuple[wsl_ui.LocalUIBridge, wsl_network.NetworkBridge | None]:
    port = _port(arguments)
    local_ui = wsl_ui.LocalUIBridge(command(executable, "-m", "ml_stack.fleet.wsl_ui", str(port)), port)
    try:
        local_ui.start()
        bridge = _bridge(executable, arguments)
    except OSError as exc:
        local_ui.close()
        raise WSLError(f"The Windows local UI bridge could not start: {exc}") from exc
    except WSLError:
        local_ui.close()
        raise
    return local_ui, bridge


def start(argv: list[str], *, executable: str | None = None) -> int:
    """Run the Linux daemon with model confinement enabled."""
    executable = executable or prepare()
    arguments = list(argv)
    host_python = _read("wslpath", "-a", "-u", sys.executable)
    linux_path = _read("printenv", "PATH")
    environment = ["PATH=" + executable.rsplit("/", 1)[0] + ":" + linux_path,
                   "ML_STACK_SANDBOX_SERVE=1", "ML_STACK_WINDOWS_PYTHON=" + host_python]
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
    local_ui, bridge = _bridges(executable, arguments)
    if bridge:
        environment.append(wsl_network.ENV + "=" + bridge.config)
    try:
        process = launch(command("env", *environment, executable,
                                 "-m", "ml_stack.cli.wsl_daemon", *arguments), stdin=subprocess.PIPE)
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
    finally:
        if bridge:
            bridge.close()
        local_ui.close()
