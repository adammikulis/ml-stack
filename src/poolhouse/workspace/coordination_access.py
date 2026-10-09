"""Read access to shared project coordination messages."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from poolhouse.workspace.identity import AGENT, Identity

if TYPE_CHECKING:
    from poolhouse.workspace.service import Workspace


def project_key(ws: Workspace, name: str) -> str:
    """Return the trusted project key of an identity's registered root."""
    root = ws.registry.root_of(name)
    info = ws.registry.info(root)
    return str(info["project"].get("key", "")) if info["role"] else ""


def shared_pair(ws: Workspace, who: Identity, a: str, b: str) -> bool:
    """Whether both conversation participants share the reader's project."""
    key = project_key(ws, who.id)
    return bool(key and project_key(ws, a) == key and project_key(ws, b) == key)


def can_read_row(ws: Workspace, who: Identity, row: dict[str, Any],
                 boards: dict[str, Any] | None = None) -> bool:
    """Whether the reader may view a board, addressed or shared project message."""
    if row["to"].startswith("#"):
        return ws.board.can_read(who, row["to"], boards)
    return (who.role != AGENT or who.id in (row["from"], row["to"])
            or row["to"] == "*" or shared_pair(ws, who, row["from"], row["to"]))
