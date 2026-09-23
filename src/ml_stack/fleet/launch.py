"""``ml-stack`` -- the thing a person double-clicks."""

from __future__ import annotations

import threading
import time
import webbrowser
from typing import Any

from ml_stack.http import ServerError, request_json
from ml_stack.log import say, warn

from .discovery import (
    DEFAULT_PORT as DISCOVERY_PORT,  # noqa: F401  (keeps ports in view)
    memberships,
)
from .updates import state

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
                "              ml-stack-fleet join --persist   -- joins the fleet and starts "
                "it at every login"]
    return [*lines,
            f"  open        {url}",
            f"  updates     {'follows ' + track if track else 'releases'}, "
            "whenever nothing is running here",
            "",
            "  next        ml-stack-fleet status    -- who else is in the fleet"]


def main(argv: list[str] | None = None) -> int:
    import argparse

    from .daemon import main as daemon_main

    ap = argparse.ArgumentParser(
        prog="ml-stack",
        description="Start ml-stack on this machine and open it in your browser.")
    ap.add_argument("--port", type=int, default=HTTP_PORT)
    ap.add_argument("--no-browser", action="store_true",
                    help="start the daemon but do not open a browser")
    known, rest = ap.parse_known_args(argv)

    url = f"http://127.0.0.1:{known.port}/ui/"

    running = already_running(known.port)
    if running is not None:
        say(f"ml-stack is already running as '{running.get('name', '?')}'.")
        if not known.no_browser:
            webbrowser.open(url)
        say(f"  {url}")
        return 0

    def open_when_ready() -> None:
        if wait_for_health(known.port) is None:
            warn("The daemon did not start. Its output is above.")
            return
        if not known.no_browser:
            webbrowser.open(url)

    # Started before the daemon takes over this thread, which never returns.
    threading.Thread(target=open_when_ready, daemon=True).start()
    return daemon_main(["--port", str(known.port), *rest])


if __name__ == "__main__":
    raise SystemExit(main())
