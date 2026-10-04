"""Serving a placement on each peer, and showing what each peer then serves."""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from ml_stack.log import say

from .pausing import peer_clients
from .remote import PeerError

__all__ = ["apply_plan", "serving_table"]


def apply_plan(placement: Any, rows: Sequence[dict[str, Any]], *,
               cluster_key_path: Path | str | None = None,
               say: Callable[[str], None] = say) -> list[dict[str, Any]]:
    """``POST /serve`` on each placed peer, and what each answered.

    Each answer is ``{"peer", "model", "slots", "status", "served" | "error", "serving"}``;
    ``serving`` is the peer's ``/health`` serving rows after the call.
    """
    clients = peer_clients(rows, cluster_key_path=cluster_key_path, timeout=600.0)
    out: list[dict[str, Any]] = []
    for row in placement.rows:
        answer: dict[str, Any] = {"peer": row.peer, "model": row.model, "slots": row.slots}
        peer = clients.get(row.peer)
        if peer is None:
            answer.update(status=0, error="no daemon answered for this peer")
            out.append(answer)
            say(f"{row.peer}: {row.model}: {answer['error']}")
            continue
        try:
            served = peer._json("POST", "/serve", {"model": row.model, "context": row.context,
                                                   "parallel": row.slots})
            answer.update(status=201, served=served)
            say(f"{row.peer}: {row.model}: {served.get('slots', row.slots)} slot(s) on "
                f"port {served.get('port', '?')}")
        except PeerError as exc:
            status, body = _refusal(str(exc))
            answer.update(status=status, error=body.get("error") or str(exc))
            say(f"{row.peer}: {row.model}: {answer['error']}")
        try:
            answer["serving"] = list(peer.health().get("serving") or [])
        except (PeerError, OSError, ValueError):
            answer["serving"] = []
        out.append(answer)
    return out


def serving_table(applied: Sequence[dict[str, Any]]) -> str:
    """What each peer serves after an apply, as text."""
    lines = [f"{'PEER':<16} SERVING"]
    for answer in applied:
        cells = [f"{m}:{one.get('port', '?')} ({int(one.get('slots') or 1)} slot(s))"
                 for one in answer.get("serving") or [] for m in one.get("models") or []]
        lines.append(f"{answer['peer']:<16} {', '.join(cells) or '-'}")
    return "\n".join(lines)


def _refusal(message: str) -> tuple[int, dict[str, Any]]:
    """The status and JSON body out of a `PeerError`'s message; ``(0, {})`` for none."""
    found = re.search(r"-> (\d{3}): (\{.*\})", message, re.S)
    if not found:
        return 0, {}
    try:
        return int(found.group(1)), json.loads(found.group(2))
    except ValueError:
        return int(found.group(1)), {}
