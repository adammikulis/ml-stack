"""Open this machine's page with a one-shot sign-in ticket the running daemon issued."""

from __future__ import annotations

from pathlib import Path

from poolhouse.http import ServerError, request_json

from . import launch_secret
from .launch_secret import HEADER, LaunchError
from .runtime_paths import default_root

__all__ = ["page_url", "ticket_url"]

TIMEOUT_S = 5.0


def ticket_url(root: Path | None = None) -> str:
    """The page URL carrying a fresh ticket from the daemon recorded under ``root``;
    `LaunchError` when no daemon answers."""
    record = launch_secret.read(root or default_root())
    port = int(record["port"])
    if not 0 < port < 65536:
        raise LaunchError("the recorded daemon has no port")
    try:
        got = request_json(f"http://127.0.0.1:{port}/ui/launch/ticket", payload={}, method="POST",
                           headers={"X-Poolhouse-UI": "1", HEADER: str(record["secret"])},
                           timeout=TIMEOUT_S)
    except ServerError as error:
        raise LaunchError(f"the daemon on port {port} refused a launch ticket: {error}") from error
    if not isinstance(got, dict) or not isinstance(got.get("ticket"), str):
        raise LaunchError(f"the daemon on port {port} did not issue a ticket")
    return f"http://127.0.0.1:{port}/ui/?launch_ticket={got['ticket']}"


def page_url(port: int, root: Path | None = None) -> str:
    """The URL a window for the daemon on ``port`` opens: carrying a ticket when one is issued,
    else the bare page."""
    bare = f"http://127.0.0.1:{port}/ui/"
    try:
        url = ticket_url(root)
    except LaunchError:
        return bare
    return url if url.startswith(bare + "?") else bare
