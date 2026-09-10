"""Making a store hold a different graph without losing the one it holds: the whole-graph
replace, the counts it is judged against, and the snapshots either side of it."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ml_stack.graph.snapshots import restore, take
from ml_stack.graph.store import MOST, GraphStore, StoreMismatch, WouldLoseTooMuch

__all__ = ["COPY_OVER", "count_store", "fold_log", "replace", "roll_back", "snapshot"]

# how much of a store one write may remove before a verified copy is taken on the way past
COPY_OVER = 0.1


def replace(path: str | Path, graph: Mapping[str, Any], *, force: bool = False,
            keep_copy: bool = True) -> dict[str, int]:
    """Make the store hold this graph and nothing else, safely.

    Anything no longer in the graph goes, but a write that would take most of the store
    (`MOST`) is refused rather than performed, and one that would take a tenth
    (`COPY_OVER`) leaves a verified copy behind first.
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
    if keep_copy and held and len(gone) > len(held) * COPY_OVER:
        take(path, reason=f"before dropping {len(gone)} of {len(held)} nodes",
             count=count_store, fold=fold_log)
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
    """Open a store read-only on a fresh handle and count it.

    A bulk write can report every row written while reading back on the same connection,
    and be short when reopened; only a fresh open sees what reached the disk.
    """
    with GraphStore(path, read_only=True) as store:
        return store.counts()


def fold_log(path: str | Path) -> None:
    """Open a store writable once and close it, which checkpoints its log away."""
    GraphStore(path).close()


def snapshot(path: str | Path, *, reason: str, keep: int = 10):
    """A verified copy of a store, taken before something that cannot be undone."""
    return take(path, reason=reason, count=count_store, fold=fold_log, keep=keep)


def roll_back(snapshot_path: str | Path):
    """Put a snapshot back, saving what is there now first."""
    return restore(snapshot_path, count=count_store, fold=fold_log)
