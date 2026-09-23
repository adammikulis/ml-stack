"""A peer's exported runs into this machine's store. `import_runs` writes them as
``bench:`` docs with ``server["host"]``, ``server["machine"]`` and ``server["commit"]`` set,
so a sweep spread over the fleet reads back as one set of runs that says which machine
measured each."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from ml_stack.bench.score import machine_of
from ml_stack.log import say

__all__ = ["SERVER_KEYS", "import_runs"]


SERVER_KEYS = ("model", "draft_model", "binary", "context", "slots", "cache_type",
               "reasoning_budget", "load_s", "resident_bytes", "kv_and_run_bytes", "mmapped",
               "sampling", "finder", "concurrency")
"""The fields ``show --export`` flattens out of a run's ``server``, put back there on import."""


def _doc_from(record: Mapping[str, Any], *, host: str, machine: str,
              commit: str) -> dict[str, Any]:
    """A ``bench:`` doc out of one record -- a whole run as `runs` reads it, or a flat one
    as `show --export` writes it. Either way ``server["host"]`` and ``server["commit"]``
    are set, and ``server["machine"]`` when ``machine`` names one. A flat record has no rows, so its totals go under ``totals`` and its
    ``derived`` is precomputed, which is what `rates`, `pareto` and `composed` read."""
    named = {"host": host, "commit": commit, **({"machine": machine} if machine else {})}
    if "rows" in record:
        one = {k: v for k, v in record.items() if k != "key"}
        one["server"] = {**(one.get("server") or {}), **named}
        return one
    server = {k: record[k] for k in SERVER_KEYS if _said(record.get(k))}
    right = float(record.get("f1") or 0)
    seconds = float(record.get("seconds") or 0)
    paid = float(record.get("read_tokens") or 0) + float(record.get("written_tokens") or 0)
    memory = float(record.get("kv_and_run_bytes") or 0)
    derived: dict[str, float] = {
        "right": right, "recall": float(record.get("recall") or 0),
        "precision": float(record.get("precision") or 0),
        "shown_per_question": float(record.get("lit_per_question") or 0),
        "wanted_per_question": 0.0, "seconds": seconds, "paid_tokens": paid,
        "calls": float(record.get("calls") or 0), "kv_bytes": memory,
        "questions": float(record.get("questions") or 0)}
    if seconds > 0:
        derived["right_per_minute"] = right * 60.0 / seconds
        if right:
            derived["seconds_per_right"] = seconds / right
    if paid > 0:
        derived["right_per_1k"] = right * 1000.0 / paid
        if right:
            derived["tokens_per_right"] = paid / right
    if memory > 0:
        derived["right_per_gb"] = right * 2**30 / memory
    return {"at": str(record.get("at") or ""), "label": str(record.get("label") or ""),
            "server": {**server, **named}, "rows": [],
            "totals": dict(record), "derived": derived}


def _said(value: Any) -> bool:
    """Whether a flattened field carries anything: not None, "", {} or False -- and 0 is
    a number, which ``0 == False`` would have dropped."""
    if value is None or isinstance(value, bool):
        return bool(value)
    return value not in ("", {})


def import_runs(path_or_json: str | Path | Sequence[Mapping[str, Any]] | Mapping[str, Any],
                into: str | Path, *, host: str, commit: str = "",
                log: Callable[[str], None] = say) -> list[str]:
    """Put a peer's runs into ``into`` as new ``bench:`` docs with ``server["host"]``,
    ``server["machine"]`` and ``server["commit"]`` set. Returns the keys written.

    ``path_or_json`` is a file ``ml-stack-bench show --export`` wrote on the peer, the
    JSON text of one, the list it holds, or what `bench_export` answered (``{"runs":
    [...]}``, which names the peer's commit and `home.machine_id`); a whole run record,
    rows and all, is taken as it is. A run already in ``into`` -- same label, ``at`` and
    machine, or host where no machine is named -- is skipped, and nothing there is ever
    overwritten: a key that exists gets ``-n`` on the newcomer. Usable by hand for a peer
    with no daemon: export there, copy the file, import here.
    """
    from ml_stack.graph.store import GraphStore

    records, said = _records(path_or_json)
    commit = commit or str(said.get("commit") or "")
    machine = str(said.get("machine") or "")
    with GraphStore(into) as writer:
        kept = writer.docs()
        present = {(str(v.get("label", "")), str(v.get("at", "")), machine_of(v))
                   for k, v in kept.items() if k.startswith("bench:") and isinstance(v, dict)}
        written: list[str] = []
        skipped = 0
        for record in records:
            doc = _doc_from(record, host=host, machine=machine, commit=commit)
            mark = (doc["label"], doc["at"], machine_of(doc))
            if mark in present:
                skipped += 1
                continue
            present.add(mark)
            stem = f"bench:{doc['label']}:{doc['at'].replace('-', '').replace(':', '')}@{host}"
            key, n = stem, 1
            while writer.get_doc(key) is not None:
                key, n = f"{stem}-{n}", n + 1
            writer.put_doc(key, json.loads(json.dumps(doc)))
            written.append(key)
    log(f"  {host}: imported {len(written)} run(s) into {into}"
        + (f", {skipped} already there" if skipped else ""))
    return written


def _records(source: Any) -> tuple[list[Mapping[str, Any]], Mapping[str, Any]]:
    """The runs in an export -- a file, JSON text, a list, or a `bench_export` answer --
    and the answer itself, {} for a bare list."""
    if isinstance(source, (str, Path)):
        text = str(source)
        if not text.lstrip().startswith(("[", "{")):
            text = Path(source).expanduser().read_text(encoding="utf-8")
        source = json.loads(text)
    said: Mapping[str, Any] = {}
    if isinstance(source, Mapping):
        said, source = source, source.get("runs") or []
    if not isinstance(source, (list, tuple)):
        raise ValueError("an export is a list of runs, or {'runs': [...]}")
    return [r for r in source if isinstance(r, Mapping)], said
