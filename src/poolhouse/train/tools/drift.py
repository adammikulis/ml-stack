"""A fingerprint of the tool schemas a dataset was made for, and the check that a model or
dataset is only used against the schemas it was made for."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

from poolhouse.train.tools.schemas import schemas_of

__all__ = ["SchemaDrift", "check", "fingerprint", "schema_hash", "signatures"]


class SchemaDrift(RuntimeError):
    """The tools in front of a dataset or model are not the ones it was made for."""


def schema_hash(tools: Any) -> str:
    """``sha256:...`` over the schemas, whatever their key order or the order of the tools."""
    ordered = sorted(schemas_of(tools), key=lambda s: s["function"]["name"])
    canon = json.dumps(ordered, sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(canon.encode("utf-8")).hexdigest()


def signatures(tools: Any) -> dict[str, dict[str, list[str]]]:
    """``{tool: {"required": [...], "properties": [...]}}``, the diffable part of a schema."""
    out = {}
    for schema in schemas_of(tools):
        fn = schema["function"]
        params = fn.get("parameters") or {}
        out[fn["name"]] = {"required": sorted(params.get("required") or []),
                           "properties": sorted((params.get("properties") or {}).keys())}
    return out


def fingerprint(tools: Any) -> dict[str, Any]:
    """The manifest fields that pin ``tools``: ``schema_hash`` and ``signatures``."""
    return {"schema_hash": schema_hash(tools), "signatures": signatures(tools)}


def check(pinned: Mapping[str, Any], live: Any) -> None:
    """Raise `SchemaDrift` naming the tools added, removed and changed, unless ``live``
    hashes to what ``pinned`` (a manifest) recorded. A manifest with no hash is not checked."""
    want = pinned.get("schema_hash")
    if not want or want == schema_hash(live):
        return
    old, new = pinned.get("signatures") or {}, signatures(live)
    parts = [f"{label}: {', '.join(names)}" for label, names in (
        ("added", sorted(set(new) - set(old))),
        ("removed", sorted(set(old) - set(new))),
        ("changed", sorted(n for n in set(old) & set(new) if old[n] != new[n]))) if names]
    raise SchemaDrift("the tool schemas differ from the ones this was made for ("
                      + ("; ".join(parts) or "descriptions or nested schemas changed") + ")")
