"""How the public API talks to this device's node: one call, with the node's failures as `poolhouse.Error`s."""

from __future__ import annotations

from typing import Any

from poolhouse import errors, node_health, node_launch
from poolhouse.board import client, session as board_session

__all__ = ["call", "session"]


def call(method: str, params: dict[str, Any] | None = None, *, token: str = "") -> dict[str, Any]:
    """One request to this device's node (started when it is not running); raises `NotRunning` or `Error`."""
    try:
        node_launch.ensure_node(node_launch.default_state())
        return node_health.call(node_launch.default_state(), method, params, token=token)
    except OSError as exc:
        raise errors.NotRunning(f"no node answered on this device ({exc}); run `poolhouse` to start it") from None
    except ValueError as exc:
        text = str(exc)
        raise (errors.Denied if "grant" in text or "denied" in text else errors.Error)(text) from None


def session() -> Any:
    """A session of the pool board of this device (the one that may change the pool), as the node knows the caller."""
    node = client.Client(node_launch.default_state())
    got = node.call("register", "pool", "", model="poolhouse-api", harness="poolhouse-api", session="python-api")
    return board_session.Session(node, "pool", got["token"], got["name"])
