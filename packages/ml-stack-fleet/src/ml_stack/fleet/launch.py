"""``ml-stack`` -- the thing a person double-clicks."""

from __future__ import annotations

import json
import sys
import threading
import time
import urllib.error
import urllib.request
import webbrowser
from typing import Any

from .discovery import DEFAULT_PORT as DISCOVERY_PORT  # noqa: F401  (keeps ports in view)

__all__ = ["main", "wait_for_health", "already_running"]

HTTP_PORT = 8770


def _health(port: int, timeout: float = 1.0) -> dict[str, Any] | None:
    try:
        with urllib.request.urlopen(
                f"http://127.0.0.1:{port}/health", timeout=timeout) as r:
            return json.loads(r.read() or b"{}")
    except (urllib.error.URLError, OSError, ValueError):
        return None


def already_running(port: int = HTTP_PORT) -> dict[str, Any] | None:
    """A healthy daemon already on this port, or None."""
    return _health(port)


def wait_for_health(port: int = HTTP_PORT, *, seconds: float = 20.0
                    ) -> dict[str, Any] | None:
    """Block until the daemon answers, or give up. Returns its health."""
    deadline = time.time() + seconds
    while time.time() < deadline:
        found = _health(port, timeout=0.5)
        if found is not None:
            return found
        time.sleep(0.15)
    return None


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
        print(f"ml-stack is already running as '{running.get('name', '?')}'.")
        if not known.no_browser:
            webbrowser.open(url)
        print(f"  {url}")
        return 0

    def open_when_ready() -> None:
        if wait_for_health(known.port) is None:
            print("The daemon did not start. Its output is above.", file=sys.stderr)
            return
        if not known.no_browser:
            webbrowser.open(url)

    # Started before the daemon takes over this thread, which never returns.
    threading.Thread(target=open_when_ready, daemon=True).start()
    return daemon_main(["--port", str(known.port), *rest])


if __name__ == "__main__":
    raise SystemExit(main())
