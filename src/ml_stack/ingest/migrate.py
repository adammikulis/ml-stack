"""A store written when a source was called a book, brought up to date."""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from ml_stack.ingest.progress import Progress
from ml_stack.ingest.reads import _read_json, _write_json

__all__ = ["NEW_NAME", "NEW_PREFIX", "OLD_NAME", "OLD_PREFIX", "migrate", "pending",
           "reads_beside"]

OLD_PREFIX = "book:"
NEW_PREFIX = "source:"
OLD_NAME = "shelf"
NEW_NAME = "sources"


class NotSound(RuntimeError):
    """The migrated store did not read back as the store that went in."""


def reads_beside(out: str | Path) -> list[Path]:
    """Every ``<store>.<slug>.reads.json`` beside a store."""
    where = Path(out).expanduser()
    if not where.parent.is_dir():
        return []
    return sorted(where.parent.glob(f"{where.name}.*.reads.json"))


def pending(out: str | Path) -> dict[str, int]:
    """What still speaks of books: nodes, edges, documents, reads rows, progress entries."""
    where = Path(out).expanduser()
    found = {"nodes": 0, "edges": 0, "docs": 0, "rows": 0, "progress": 0}
    if where.exists():
        from ml_stack.graph.store import GraphStore

        with GraphStore(where, read_only=True) as store:
            old = {str(n["id"]) for n in store.nodes()
                   if str(n["id"]).startswith(OLD_PREFIX) or n.get("kind") == "book"}
            found["nodes"] = len(old)
            found["edges"] = sum(1 for e in store.edges()
                                 if str(e["source"]) in old or str(e["target"]) in old)
            found["docs"] = sum(1 for key in store.doc_keys()
                                if _doc_changed(key, store.get_doc(key), old)[1])
    held = _read_json(Progress.beside(where))
    if isinstance(held, dict) and isinstance(held.get("books"), dict):
        found["progress"] = len(held["books"])
    for path in reads_beside(where):
        rows = _read_json(path)
        if isinstance(rows, dict):
            found["rows"] += sum(1 for row in rows.values()
                                 if isinstance(row, Mapping) and "book" in row)
    return found


def migrate(out: str | Path, *, say: Callable[[str], None] = print) -> int:
    """``ml-stack-ingest migrate --out STORE``: ``book:`` node ids into ``source:``.

    The nodes, the ``read_from`` edges that name them, every unit document's provenance,
    the progress file and the reads files beside the store; a store file named ``shelf``
    and its files move to ``sources``. A verified copy of the store is taken first and put
    back if the store does not read back whole afterwards. A store that already speaks of
    sources is left alone.
    """
    from ml_stack.graph.store import GraphStore, count_store, roll_back, snapshot

    where = Path(out).expanduser()
    if not where.exists():
        say(f"no store at {out}")
        return 1
    found = pending(where)
    if not any(found.values()) and _moves(where) is None:
        say(f"{out} already names its sources `source:`; nothing to migrate")
        return 0
    say(f"{out}: {found['nodes']} node(s), {found['edges']} edge(s), {found['docs']} "
        f"document(s), {found['rows']} reads row(s) and {found['progress']} progress "
        f"entry(s) still name books")
    before = count_store(where)
    kept = {path: path.read_bytes() for path in
            [Progress.beside(where), *reads_beside(where)] if path.is_file()}
    snap = snapshot(where, reason="before book: node ids became source:")
    say(f"copied the store to {Path(snap.path).name}")
    try:
        with GraphStore(where) as store:
            changed = _rewrite(store)
        changed["progress"] = _rewrite_progress(Progress.beside(where))
        changed["rows"] = sum(_rewrite_reads(path) for path in reads_beside(where))
        after = count_store(where)
        with GraphStore(where, read_only=True) as store:
            findings = store.check()
            left = sorted(str(n["id"]) for n in store.nodes()
                          if str(n["id"]).startswith(OLD_PREFIX) or n.get("kind") == "book")
        if after != before:
            raise NotSound(f"{before} went in and {after} came back")
        if findings:
            raise NotSound(f"{len(findings)} finding(s), e.g. {findings[0]!r}")
        if left:
            raise NotSound(f"{len(left)} node(s) still named books, e.g. {left[0]}")
    except BaseException as why:  # noqa: BLE001 - the store goes back whatever stopped it
        for path, raw in kept.items():
            path.write_bytes(raw)
        roll_back(snap.path)
        if isinstance(why, (KeyboardInterrupt, SystemExit)):
            raise
        say(f"the migration did not verify: {why}")
        say(f"{out} is as it was, put back from {Path(snap.path).name}")
        return 1
    say(f"{out}: {changed['nodes']} node(s) renamed, {changed['edges']} edge(s) moved, "
        f"{changed['docs']} document(s), {changed['rows']} reads row(s) and "
        f"{changed['progress']} progress entry(s) rewritten")
    say(f"{after['nodes']} node(s), {after['edges']} edge(s), {after['docs']} document(s); "
        f"reads back whole")
    moves = _moves(where)
    if moves is None:
        return 0
    for src, dst in moves:
        os.replace(src, dst)
    say(f"moved {where.name} and its {len(moves) - 1} file(s) beside it to "
        f"{moves[0][1].name}")
    return 0


def _rewrite(store: Any) -> dict[str, int]:
    """The store's book nodes, their edges and the documents naming them, as sources."""
    nodes = store.nodes()
    old = {str(n["id"]): n for n in nodes
           if str(n["id"]).startswith(OLD_PREFIX) or n.get("kind") == "book"}
    moved = {node_id: _renamed_id(node_id) for node_id in old}
    edges = [e for e in store.edges()
             if str(e["source"]) in moved or str(e["target"]) in moved]
    changed = {"nodes": len(old), "edges": len(edges), "docs": 0}
    with store.transaction():
        for node_id, node in old.items():
            attrs = dict(node.get("attrs") or {})
            if "book" in attrs:
                attrs["source"] = attrs.pop("book")
            store.upsert_node({**node, "id": moved[node_id], "kind": "source",
                               "attrs": attrs})
        for edge in edges:
            store.upsert_edge({**edge, "source": moved.get(str(edge["source"]),
                                                           str(edge["source"])),
                               "target": moved.get(str(edge["target"]), str(edge["target"]))})
        store.drop(sorted(old), force=True)
        for key in store.doc_keys():
            value, differs = _doc_changed(key, store.get_doc(key), set(moved))
            if differs:
                store.put_doc(key, value)
                changed["docs"] += 1
    return changed


def _renamed_id(node_id: str) -> str:
    """``book:<slug>`` as ``source:<slug>``; any other id unchanged."""
    return (NEW_PREFIX + node_id[len(OLD_PREFIX):]
            if node_id.startswith(OLD_PREFIX) else node_id)


def _doc_changed(key: str, value: Any, old: set[str]) -> tuple[Any, bool]:
    """One document with its book ids and, for a unit, its ``book`` field, as sources."""
    out = _renamed_in(value, old)
    if key.startswith("ingest:unit:") and isinstance(out, dict):
        out = dict(out)
        if "book" in out:
            out["source"] = out.pop("book")
        where = out.get("where")
        if isinstance(where, Mapping) and "book" in where:
            out["where"] = {**{k: v for k, v in where.items() if k != "book"},
                            "source": where["book"]}
    return out, out != value


def _renamed_in(value: Any, old: set[str]) -> Any:
    """Every ``book:<slug>`` id inside a document, at any depth, as ``source:<slug>``."""
    if isinstance(value, str):
        return _renamed_id(value) if value in old else value
    if isinstance(value, Mapping):
        return {(_renamed_id(k) if isinstance(k, str) and k in old else k):
                _renamed_in(v, old) for k, v in value.items()}
    if isinstance(value, list):
        return [_renamed_in(v, old) for v in value]
    return value


def _rewrite_progress(path: Path) -> int:
    """The progress file's ``books`` as ``sources``. Returns the entries rewritten."""
    held = _read_json(path)
    if not isinstance(held, dict) or not isinstance(held.get("books"), dict):
        return 0
    books = held.pop("books")
    held["sources"] = {**books, **(held.get("sources") or {})}
    _write_json(path, held)
    return len(books)


def _rewrite_reads(path: Path) -> int:
    """One reads file's ``book`` field as ``source``. Returns the rows rewritten."""
    held = _read_json(path)
    if not isinstance(held, dict):
        return 0
    out, changed = {}, 0
    for unit, row in held.items():
        if isinstance(row, Mapping) and "book" in row:
            out[unit] = {**{k: v for k, v in row.items() if k != "book"},
                         "source": row["book"]}
            changed += 1
        else:
            out[unit] = row
    if changed:
        _write_json(path, out)
    return changed


def _moves(out: str | Path) -> list[tuple[Path, Path]] | None:
    """``[(from, to)]`` for a store named ``shelf`` and every file beside it, else None.

    None where the name is already ``sources``, where the store is not there, or where
    anything is in the way at the new name.
    """
    where = Path(out).expanduser()
    if where.stem != OLD_NAME or not where.exists():
        return None
    held = [where, Path(str(where) + ".wal"), Progress.beside(where), *reads_beside(where)]
    out_pairs = [(one, one.with_name(NEW_NAME + one.name[len(OLD_NAME):]))
                 for one in held if one.exists()]
    if any(dst.exists() for _, dst in out_pairs):
        return None
    return out_pairs
