"""The graph of model leases that have ended: one node per lease, joined to its model and branch."""

from __future__ import annotations

import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from poolhouse.graph.store import GraphStore

__all__ = ["append", "ended_row", "rows"]

KIND = "lease"


def ended_row(model: str, lease: str, record: Mapping[str, Any], *, ended: float | None = None) -> dict[str, Any]:
    """The history row for a lease that ended now: the record, the model and the duration."""
    stop = ended if ended is not None else time.time()
    taken = float(record.get("taken") or stop)
    return {**record, "lease": lease, "model": Path(model).name, "ended": stop,
            "duration_s": round(max(0.0, stop - taken), 1)}


def append(file: Path, row: Mapping[str, Any]) -> None:
    """Add one ended lease to the history at ``file``."""
    model, branch = f"model:{row['model']}", str(row.get("branch") or "")
    with GraphStore(file) as store:
        store.upsert_node({"id": f"lease:{row['lease']}", "kind": KIND,
                           "label": f"{row['model']} for {row.get('reason') or ''}"[:120],
                           "attrs": dict(row)})
        store.upsert_node({"id": model, "kind": "model", "label": str(row["model"])})
        store.upsert_edge({"source": f"lease:{row['lease']}", "rel": "served", "target": model})
        if branch:
            store.upsert_node({"id": f"branch:{branch}", "kind": "branch", "label": branch})
            store.upsert_edge({"source": f"lease:{row['lease']}", "rel": "taken-on",
                               "target": f"branch:{branch}"})


def rows(file: Path, *, model: str = "", since: float = 0.0) -> list[dict[str, Any]]:
    """Every ended lease at ``file``, oldest first, optionally of a model whose name contains
    ``model`` and ended at or after ``since`` (epoch seconds)."""
    if not file.exists():
        return []
    with GraphStore(file, read_only=True) as store:
        found = [n["attrs"] for n in store.nodes(KIND)]
    wanted = [r for r in found if model.lower() in str(r.get("model", "")).lower()
              and float(r.get("ended") or 0.0) >= since]
    return sorted(wanted, key=lambda r: float(r.get("ended") or 0.0))
