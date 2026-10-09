"""Bringing a version 1 store (one sealed JSON file of plain facts) into the graph."""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ml_stack.graph.store import GraphStore
from ml_stack.memory import vault
from ml_stack.memory.entities import parse
from ml_stack.memory.facts import BUILD_BOUND, MAX_FACTS, Fact, Refused, check
from ml_stack.sentinel.sealed import SealedFile

if TYPE_CHECKING:
    from ml_stack.memory.store import Store

__all__ = ["import_v1"]

_ID = re.compile(r"m\d{4,6}")


def _row(store: Store, g: GraphStore, row: dict[str, Any]) -> bool:
    try:
        fact = Fact.from_json(row)
        if not _ID.fullmatch(fact.id):
            raise Refused("bad id")
        fact.text = check(fact.text, person=fact.source == "user-said")
        scope = {k: v for k, v in fact.scope.items() if k in ("machine", "build")}
        names = [f"model:{fact.scope['model']}"] if fact.scope.get("model") else []
        if fact.kind in BUILD_BOUND and scope.get("build"):
            names.append(f"build:{scope['build']}")
        ents = parse(names)
    except Refused:
        return False
    fact.scope = scope
    store._put_fact(g, fact, ents)
    return True


def import_v1(store: Store, legacy: Path) -> dict[str, int]:
    """Import the facts of the sealed JSON file ``legacy`` into ``store`` once, re-running the
    write checks, then keep the file encrypted as ``<name>.v1`` and delete the plain copies.
    Returns ``{"imported", "refused"}``; a file that fails its seal is left alone."""
    sealed = SealedFile(legacy)
    loaded = sealed.load()
    if loaded.status in ("fresh", "tampered"):
        return {"imported": 0, "refused": 0}
    rows = [r for r in loaded.payload.get("facts", []) if isinstance(r, dict)][:MAX_FACTS]
    kept = [0]

    def change(g: GraphStore) -> None:
        for row in rows:
            kept[0] += _row(store, g, row)
        top = max((int(str(r.get("id", "m0"))[1:] or 0) for r in rows if _ID.fullmatch(str(r.get("id", "")))), default=0)
        g.put_doc("memory", {"next": top + 1})

    store._edit(change)
    original = legacy.read_bytes()
    key = store.keys.keys(store._salt, create=True)[0]
    blob = vault.seal_blob(original, key, mode=store.keys.mode, salt=store._salt, owner=store.owner)
    target = legacy.with_name(legacy.name + ".v1")
    target.write_bytes(blob)
    target.chmod(0o600)
    for each in (legacy, sealed.prev, sealed.keyfile):
        each.unlink(missing_ok=True)
    return {"imported": kept[0], "refused": len(rows) - kept[0]}
