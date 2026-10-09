"""A subagent is its own identity: the board names it, records who spawned it and mints its token.

Nothing here reads a name, a parent or a sender from the caller. The name comes from the
subagent's native session identity (`session_name`), the parent is the authenticated caller, and
the model is a claim like any other agent's."""

from __future__ import annotations

from typing import Any

from ml_stack.workspace import session_name, tokens
from ml_stack.workspace.chain import held
from ml_stack.workspace.identity import Denied
from ml_stack.workspace.modelid import CLAIMED, clean_harness, clean_model

__all__ = ["retire", "spawn", "store"]


def spawn(ws: Any, token: str, harness: str, session: str, model: str = "") -> dict[str, str]:
    """Register the native session ``session`` of ``harness`` as a child of the token's owner.

    Returns ``{"id", "parent", "token"}``; the token is empty when the child is already registered
    (its file then exists) and is never written anywhere but the child's own token file by the
    caller. `Denied` when the name is already another agent's child."""
    parent = ws.auth(token)
    ws._may(parent, "claim")
    harness = clean_harness(harness)
    model = clean_model(model) if model else ""
    name = session_name.assign(ws.base, model or parent_model(ws, parent.id), harness, session)
    known = ws.registry.info(name)
    if not known["revoked"]:
        if known["parent"] != parent.id:
            raise Denied(f"{name} is already registered under another parent")
        return {"id": name, "parent": parent.id, "token": ""}
    lim = ws.limits
    made = ws.registry.adopt(parent.id, name, lim.child_ttl_s, parent.can,
                             (lim.max_children, lim.agent_tree_live, lim.agents_live))
    ws.registry.record_model(name, model, harness, CLAIMED)
    ws.audit("subagent.spawn", parent.id, child=name, harness=harness, model=model)
    return {"id": name, "parent": parent.id, "token": made}


def retire(ws: Any, token: str) -> dict[str, str]:
    """End a subagent's own identity: its token stops working and its claims are released."""
    if not ws.auth(token).parent:
        raise Denied("only a spawned subagent retires; a main session ends with its harness")

    def release(actor):
        with held(ws.claims.lock):
            claims = ws.claims._load()
            ws.claims._save({key: row for key, row in claims.items() if row["owner"] != actor.id})

    who = ws.registry.revoke_self(token, release)
    ws.audit("subagent.retire", who.id, parent=who.parent)
    return {"id": who.id, "parent": who.parent}


def parent_model(ws: Any, parent: str) -> str:
    """The model family source for a subagent that names none: its parent's model."""
    return ws.registry.model_of(parent)[0]


def store(ws: Any, made: dict[str, str]) -> None:
    """Write a freshly minted child token to its private file."""
    if made["token"]:
        tokens.store(ws.base, made["id"], made["token"])
