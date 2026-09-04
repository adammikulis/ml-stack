"""The record of a run, the pointers back to it, and the judge a fold hands its
close spellings to."""

from __future__ import annotations

import json
import sys
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from ml_stack.ingest.extract import WITH_IMAGES, instructions, schema
from ml_stack.ingest.sources import Sources

__all__ = ["located", "origin", "run_record", "sources_for", "write_run"]


def write_run(out: str | Path, record: Mapping[str, Any]) -> str:
    """The hidden ``run`` node one ingest run hangs its units on: the model that read them,
    its build, head, sampling, the schema and instructions it read with, the version, the
    host, when. Written once; every unit document points at it (``run``), and every node
    and edge points at its units (``provenance``), so "which model extracted this" is a
    walk along pointers rather than a string copied a thousand times. Adam: "a hidden
    node for each metadata with hidden edges ... probably use less mem to use pointers"."""
    from ml_stack.graph.store import GraphStore

    run_id = str(record.get("id") or f"run:{time.strftime('%Y%m%dT%H%M%S')}")
    node = {"id": run_id, "kind": "run", "label": str(record.get("label") or run_id),
            "mentions": 1, "attrs": {**{k: v for k, v in record.items() if k != "id"},
                                     "hidden": True}}
    with GraphStore(out) as store:
        store.write({"nodes": [node], "edges": []})
    return run_id


def run_record(args: Any, *, model: str = "", serving: str = "") -> dict[str, Any]:
    """What one run read with, for `write_run`: everything a person would ask later."""
    import hashlib
    import platform as _platform
    from importlib import metadata

    def sha(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]

    core_only = bool(getattr(args, "core_only", False))

    try:
        version = metadata.version("ml-stack")
    except metadata.PackageNotFoundError:  # pragma: no cover - a checkout without install
        version = "unknown"
    stamp = time.strftime("%Y%m%dT%H%M%S")
    return {"id": f"run:{stamp}", "label": f"ingest {stamp}",
            "model": model or str(getattr(args, "model", "") or ""),
            "serving": serving, "images": bool(getattr(args, "images", False)),
            "n_max": getattr(args, "n_max", None),
            "sampling": {k: v for k, v in (("temperature", getattr(args, "temperature", None)),
                                           ("top_p", getattr(args, "top_p", None)),
                                           ("top_k", getattr(args, "top_k", None)),
                                           ("min_p", getattr(args, "min_p", None)))
                         if v is not None},
            "core_only": core_only,
            "schema_sha": sha(json.dumps(schema(core_only=core_only), sort_keys=True)),
            "instructions_sha": sha(instructions(core_only=core_only) + WITH_IMAGES),
            "ml_stack": version, "host": _platform.node(),
            "started": time.strftime("%FT%T"), "argv": list(sys.argv[1:])}


def located(store: Any, thing: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Where a node or edge was read: ``[{source, title, chapter, section, pages, unit}]``,
    one per unit in its provenance, resolved through the unit documents -- the pointers
    turned back into pages. A unit the store no longer holds comes back as its id alone."""
    out = []
    for unit_id in thing.get("provenance") or ():
        doc = store.get_doc(f"ingest:unit:{unit_id}") or {}
        where = doc.get("where") or {}
        out.append({"unit": unit_id,
                    "source": str(doc.get("source") or where.get("source") or ""),
                    "title": str(doc.get("title") or ""),
                    "chapter": str(where.get("chapter") or ""),
                    "section": str(where.get("section") or ""),
                    "pages": list(where.get("pages") or ()) or ([where["page"]]
                                                                if where.get("page") else [])})
    return out


def origin(store: Any, thing: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Which run -- and so which model, build, head and instructions -- read a node or an
    edge: one record per distinct run behind its provenance, oldest first, each naming the
    units it read. A unit read before runs were recorded says so (``run: ""``)."""
    runs: dict[str, dict[str, Any]] = {}
    for unit_id in thing.get("provenance") or ():
        doc = store.get_doc(f"ingest:unit:{unit_id}") or {}
        run_id = str(doc.get("run") or "")
        held = runs.get(run_id)
        if held is None:
            attrs = {}
            if run_id:
                found = store.nodes(kind="run")
                attrs = next((n.get("attrs") or {} for n in found if n["id"] == run_id), {})
            held = runs[run_id] = {"run": run_id, "model": str(attrs.get("model") or ""),
                                   "serving": str(attrs.get("serving") or ""),
                                   "started": str(attrs.get("started") or ""),
                                   "units": []}
        held["units"].append(unit_id)
    return sorted(runs.values(), key=lambda r: r["started"])


def sources_for(out: str | Path, *, texts: Mapping[str, str] | None = None
                ) -> Callable[[str], str]:
    """``unit id -> the text it was read from``, for the judge's second look.

    ``texts`` are units already in memory (the run has them); anything else is found by
    reading the document again from the path the progress file recorded -- once per
    source, and kept for the rest of the pass. Adam: "allow it to go back over the source
    material if needed".
    """
    from ml_stack.sources import pdf

    view = Sources(out)
    known: dict[str, str] = dict(texts or {})
    read: set[str] = set()

    def text_of(unit_id: str) -> str:
        if unit_id in known:
            return known[unit_id]
        slug = unit_id.split(":", 1)[0]
        if slug in read:
            return ""
        read.add(slug)
        held = view.source(slug)
        if held is None or not held.path or not Path(held.path).expanduser().is_file():
            return ""
        document = pdf.read(held.path)
        for unit in pdf.units(document, keep_questions=True):
            known.setdefault(unit.id, unit.text)
        return known.get(unit_id, "")

    return text_of


def _judge(client: Any, out: str | Path, *, model: str = "",
           texts: Mapping[str, str] | None = None) -> Any:
    """The pass's judge over this store: the run's model, and the sources to re-read."""
    from ml_stack.graph.tidy import ModelJudge

    return ModelJudge(client, sources=sources_for(out, texts=texts), model=model)
