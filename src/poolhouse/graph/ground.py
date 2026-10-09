"""Whether the entries an answer is about point at a source, and whether the model read them."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

__all__ = ["grounded", "ungrounded"]


def _held(store: Any) -> dict[str, Mapping[str, Any]]:
    """``{id: node}`` from a store or from a graph mapping."""
    found = store.nodes() if callable(getattr(store, "nodes", None)) \
        else (store.get("nodes") or ())
    return {str(n["id"]): n for n in found}


def grounded(answer: Any, store: Any) -> list[dict[str, Any]]:
    """One row per id the answer is about: whether the store holds it, what it points at,
    whether it carries a span, and whether the model read it."""
    held = _held(store)
    read = {str(one) for one in (getattr(answer, "read", None) or ())}
    rows: list[dict[str, Any]] = []
    for node_id in (getattr(answer, "show", None) or ()):
        node = held.get(str(node_id)) or {}
        units = [str(u) for u in (node.get("provenance") or ())]
        spans = {str(u): list(span) for u, span in (node.get("spans") or {}).items()}
        rows.append({"id": str(node_id), "label": str(node.get("label") or ""),
                     "known": str(node_id) in held, "grounded": bool(units),
                     "read": str(node_id) in read, "units": units, "spans": spans})
    return rows


def ungrounded(rows: list[dict[str, Any]]) -> str:
    """The entries of a `grounded` report that have nothing behind them, named on one line."""
    said = [f"{r['label'] or r['id']} ({'not in the graph' if not r['known'] else 'no source'})"
            for r in rows if not r["grounded"]]
    return "; ".join(said)
