"""Making a store hold a different graph without losing the one it holds: the whole-graph
replace and the counts it is judged against."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from poolhouse.graph.store import MOST, GraphStore, StoreMismatch, WouldLoseTooMuch

__all__ = ["count_store", "replace"]


def replace(path: str | Path, graph: Mapping[str, Any], *, force: bool = False) -> dict[str, int]:
    """Make the store hold this graph and nothing else, safely.

    Anything no longer in the graph goes, but a write that would take most of the store
    (`MOST`) is refused rather than performed. The store is a derived index: what a wrong
    write loses is rebuilt from the board's entries.
    """
    live = {str(n["id"]) for n in (graph.get("nodes") or ())}
    if not Path(path).expanduser().exists():
        # nothing to lose yet
        with GraphStore(path) as store:
            return store.write(graph)
    with GraphStore(path, read_only=True) as reader:
        held = [n["id"] for n in reader.nodes()]
    gone = [i for i in held if i not in live]
    if not force and held and len(gone) > len(held) * MOST:
        raise WouldLoseTooMuch(
            f"{len(gone)} of {len(held)} nodes would go in one write. If that is really meant, "
            "pass force=True; if it is not, something upstream read nothing.")
    with GraphStore(path) as store:
        with store.transaction():
            store.drop(gone, force=True)  # already judged, above, against the whole store
            written = store.write(graph)
            # counted before the commit, so a store that did not take the write keeps none of it
            held = store.counts()["nodes"]
            if held != len(live):
                raise StoreMismatch(
                    f"{path}: wrote {len(live)} nodes and counts {held} afterwards")
        store.index()                     # outside the transaction: see GraphStore.index
        return written


def count_store(path: str | Path) -> dict[str, int]:
    """Open a store read-only on a fresh handle and count its nodes, edges and documents."""
    with GraphStore(path, read_only=True) as store:
        return store.counts()
