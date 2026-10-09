"""``ml-stack`` -- the thing a person double-clicks."""

from __future__ import annotations

import argparse
import socket
import sys
import threading
import time
import webbrowser
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ml_stack import runtime
from ml_stack.http import ServerError, request_json
from ml_stack.log import say, warn

from . import launch_open
from .cluster_modes import notice
from .daemon_control import ControlError, request_replacement
from .discovery import (
    DEFAULT_PORT as DISCOVERY_PORT,  # noqa: F401  (keeps ports in view)
    memberships,
)
from .measuring import same_commit
from .runtime_paths import default_root
from .updates import state
from .wsl import WSLError, prepare, replace_running, start

__all__ = ["already_running", "last_screen", "main", "wait_for_health"]

HTTP_PORT = 8770


#: /health carries the device report, and a first telemetry sample on a loaded machine
#: takes over a second. A port with nothing on it is refused at once either way.
HEALTH_TIMEOUT_S = 5.0


def _health(port: int, timeout: float = HEALTH_TIMEOUT_S) -> dict[str, Any] | None:
    """What the ml-stack daemon on ``port`` says about itself, or None."""
    try:
        said = request_json(f"http://127.0.0.1:{port}/health", timeout=timeout)
    except ServerError:
        return None
    return said if isinstance(said, dict) else {}


SLOW_HEALTH_TIMEOUT_S = 30.0
SLOW_HEALTH_TRIES = 4
"""How a held port is asked again after one missed answer: patiently, since a loaded daemon is slow, not absent."""


def port_held(port: int, timeout: float = 2.0) -> bool:
    """Whether something owns the port. Only a refused connection says it is free; a slow or odd answer does not."""
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=timeout):
            return True
    except ConnectionRefusedError:
        return False
    except OSError:
        return True


def _held_health(port: int) -> dict[str, Any] | None:
    for _ in range(SLOW_HEALTH_TRIES):
        said = _health(port, SLOW_HEALTH_TIMEOUT_S)
        if said is not None:
            return said
    return None


def already_running(port: int = HTTP_PORT) -> dict[str, Any] | None:
    """A healthy daemon already on this port, or None."""
    return _health(port)


def wait_for_health(port: int = HTTP_PORT, *, seconds: float = 20.0
                    ) -> dict[str, Any] | None:
    """Block until the daemon answers, or give up. Returns its health."""
    deadline = time.time() + seconds
    while time.time() < deadline:
        found = _health(port)
        if found is not None:
            return found
        time.sleep(0.15)
    return None


def last_screen(name: str, *, track: str = "", port: int = HTTP_PORT) -> list[str]:
    """The lines an installer ends on: what this machine runs, and how to open its page."""
    said = state()
    version = said["version"] or "?"
    commit = said["commit"]
    lines = [f"  machine     {name}",
             f"  running     {version}" + ("" if commit in ("", f"v{version}") else f"  {commit}")]
    groups = memberships()
    if groups:
        lines.append(f"  cluster     {groups[0].group}")
    url = f"http://127.0.0.1:{port}/ui/"
    if already_running(port) is None:
        return [*lines, "",
                f"  next        ml-stack                        -- starts it and opens {url}",
                "              ml-stack-cluster join --persist   -- joins the cluster and starts "
                "it at every login"]
    return [*lines,
            f"  open        {url}",
            f"  updates     {'follows ' + track if track else 'releases'}, "
            "whenever nothing is running here",
            "",
            "  next        ml-stack-cluster status    -- other devices in the cluster"]


def _root(arguments: list[str]) -> Path:
    for index, arg in enumerate(arguments):
        if arg.startswith("--root="):
            return Path(arg.partition("=")[2]).expanduser()
        if arg == "--root" and index + 1 < len(arguments):
            return Path(arguments[index + 1]).expanduser()
    return default_root()


def _wait_for_exit(port: int, seconds: float = 20.0) -> bool:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if not port_held(port, 0.5):
            return True
        time.sleep(0.15)
    return False


def _open_when_ready(port: int, browser: bool, stopped: threading.Event,
                     root: Path | None = None) -> None:
    waiting = time.monotonic() + 20.0
    notified = False
    while not stopped.is_set():
        if _health(port) is not None:
            if browser:
                webbrowser.open(launch_open.page_url(port, root))
            return
        if not notified and time.monotonic() >= waiting:
            say("ml-stack is still starting; waiting for its web interface.")
            notified = True
        stopped.wait(0.15)


def _arguments(argv):
    ap = argparse.ArgumentParser(
        prog="ml-stack",
        description="Start ml-stack on this machine and open it in your browser.")
    ap.add_argument("--port", type=int, default=HTTP_PORT)
    ap.add_argument("--no-browser", action="store_true",
                    help="start the daemon but do not open a browser")
    ap.add_argument('--restart', action='store_true',
                    help='restart the daemon while preserving jobs and independent model workers')
    return ap.parse_known_args(argv)


def _reuse(running: dict[str, Any], known: argparse.Namespace, rest: list[str]) -> int:
    groups = memberships()
    say(notice(running.get("cluster_mode") or (groups[0].mode if groups else "dev")))
    say(f"ml-stack is already running as '{running.get('name', '?')}'.")
    if not known.no_browser:
        webbrowser.open(launch_open.page_url(known.port, _root(rest)))
    say(f"  http://127.0.0.1:{known.port}/ui/")
    return 0


def main(argv: list[str] | None = None, *,
         daemon_main: Callable[[list[str]], int] | None = None) -> int:
    daemon_main is None and runtime.forward("ml_stack.fleet.launch", list(sys.argv[1:] if argv is None else argv))
    known, rest = _arguments(argv)

    linux_executable = None
    running = already_running(known.port)
    if running is None and port_held(known.port):
        running = _held_health(known.port)
        if running is None:
            warn(f"Something holds port {known.port} and did not answer /health; not starting a second daemon. Retry shortly.")
            return 1
    expected = str(state().get("commit") or "")
    if running is not None and (known.restart or not same_commit(str(running.get("commit") or ""), expected)):
        try:
            if sys.platform == "win32":
                replace_running(rest, known.port, running, expected, restart="preserve")
            else:
                answer = request_replacement(_root(rest), known.port, running, expected, restart="preserve")
                if isinstance(answer, dict) and answer.get("preserved"):
                    say(f"Restart preserves jobs: {answer['preserved']}")
            if not _wait_for_exit(known.port):
                raise ControlError("The previous daemon is still exiting; retry shortly.")
            running = None
        except (ControlError, ServerError, WSLError, OSError, ValueError) as exc:
            warn(str(exc))
            return 1
    if running is not None:
        return _reuse(running, known, rest)

    if sys.platform == "win32" and linux_executable is None:
        try:
            linux_executable = prepare()
        except (WSLError, OSError) as exc:
            warn(str(exc))
            return 1

    stopped = threading.Event()
    threading.Thread(target=_open_when_ready,
                     args=(known.port, not known.no_browser, stopped, _root(rest)), daemon=True).start()
    arguments = ["--port", str(known.port), *rest]
    if not known.no_browser:
        arguments.append("--initial-setup")
    if sys.platform == "win32":
        try:
            return start(arguments, executable=linux_executable)
        except (WSLError, OSError) as exc:
            warn(str(exc))
            return 1
        finally:
            stopped.set()
    if daemon_main is None:
        from .daemon import run as daemon_main

    try:
        return daemon_main(arguments)
    finally:
        stopped.set()


if __name__ == "__main__":
    raise SystemExit(main())
