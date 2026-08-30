"""A free-text request about a graph turned into a checked list of edits."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Collection, Iterable, Mapping

OPERATIONS = ("rename", "remove", "merge", "set_attr", "add_relation", "remove_relation")

# op -> what target must be, what name must be, what value must be
_TARGET_NODE = frozenset({"merge", "add_relation"})
_TARGET_EDGE = frozenset({"remove_relation"})
_VALUE_NODE = frozenset({"merge", "add_relation"})
_VALUE_TEXT = frozenset({"rename", "set_attr"})
_NEEDS_NAME = frozenset({"set_attr", "add_relation"})

EDIT_INSTRUCTIONS = """You turn a request about a graph into edits to that graph.
Return only JSON matching the schema. Rules:
- target, and value where an id is called for, are ids copied exactly from the listing.
  Never invent one.
- rename: target is the item to rename, value its new label.
- remove: target is the item to delete. name and value are "".
- merge: target is the node that disappears, value the node id it merges into.
- set_attr: target is the item, name the attribute, value what to set it to.
- add_relation: target is the source node, name the relation, value the destination node id.
- remove_relation: target is the edge id to delete. name and value are "".
- reason: the words from the request that ask for this edit.
- Only edits the request asks for. An empty list is fine."""

EDITS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "edits": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "op": {"type": "string", "enum": list(OPERATIONS)},
                    "target": {"type": "string"},
                    "name": {"type": "string"},
                    "value": {"type": "string"},
                    "reason": {"type": "string"},
                },
                "required": ["op", "target", "name", "value", "reason"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["edits"],
    "additionalProperties": False,
}


@dataclass(frozen=True)
class Edit:
    """One operation on one node or edge, with the words that asked for it."""

    op: str
    target: str
    name: str = ""
    value: str = ""
    reason: str = ""


def validate_edits(raw: Any, *, node_ids: Collection[str],
                   edge_ids: Collection[str] = ()) -> list[Edit]:
    """The edits in ``raw`` naming a known operation and existing ids. Anything else is dropped."""
    nodes = frozenset(node_ids)
    ids = nodes | frozenset(edge_ids)
    items = raw.get("edits") if isinstance(raw, Mapping) else raw
    if isinstance(items, (str, bytes)) or not isinstance(items, Iterable):
        return []
    out: list[Edit] = []
    seen: set[tuple[str, str, str, str]] = set()
    for item in items:
        edit = _edit(item, nodes, ids, frozenset(edge_ids))
        if edit is None:
            continue
        key = (edit.op, edit.target, edit.name, edit.value)
        if key not in seen:
            seen.add(key)
            out.append(edit)
    return out


def plan_edits(request: str, *, nodes: Any, edges: Any = (), client: Any,
               instructions: str | None = None, n_predict: int | None = None,
               tries: int = 2) -> list[Edit]:
    """The edits a model reads out of ``request``, keeping only those that check out."""
    node_ids = ids_of(nodes)
    edge_ids = ids_of(edges)
    if not request.strip() or not (node_ids or edge_ids):
        return []
    text = "\n\n".join([
        "Nodes:\n" + _listing(nodes),
        "Edges:\n" + _listing(edges),
        "Request:\n" + request.strip(),
    ])
    raw = client.extract(
        text, EDITS_SCHEMA,
        instructions=instructions or EDIT_INSTRUCTIONS,
        think=False, schema_name="graph_edits", n_predict=n_predict, tries=tries,
        check=lambda obj: objections(obj, node_ids=node_ids, edge_ids=edge_ids),
    )
    return validate_edits(raw, node_ids=node_ids, edge_ids=edge_ids)


def objections(raw: Any, *, node_ids: Collection[str],
               edge_ids: Collection[str] = ()) -> list[str]:
    """One line per edit in ``raw`` that validation drops."""
    items = raw.get("edits") if isinstance(raw, Mapping) else raw
    if isinstance(items, (str, bytes)) or not isinstance(items, Iterable):
        return ["the reply has no edits array"]
    nodes = frozenset(node_ids)
    ids = nodes | frozenset(edge_ids)
    out: list[str] = []
    for index, item in enumerate(items, 1):
        if _edit(item, nodes, ids, frozenset(edge_ids)) is not None:
            continue
        op = item.get("op") if isinstance(item, Mapping) else None
        target = item.get("target") if isinstance(item, Mapping) else None
        if op not in OPERATIONS:
            out.append(f"edit {index}: {op!r} is not one of {', '.join(OPERATIONS)}")
        elif target not in ids:
            out.append(f"edit {index}: {target!r} is not an id in the listing")
        else:
            out.append(f"edit {index}: {op} is missing the name or value it needs")
    return out


def ids_of(items: Any) -> list[str]:
    """The ids in a mapping keyed by id, in records carrying an ``id``, or in a sequence of ids."""
    if isinstance(items, Mapping):
        return [k for k in items if isinstance(k, str)]
    if isinstance(items, (str, bytes)) or not isinstance(items, Iterable):
        return []
    out = []
    for item in items:
        if isinstance(item, Mapping):
            found = item.get("id")
            if isinstance(found, str):
                out.append(found)
        elif isinstance(item, str):
            out.append(item)
    return out


def _edit(item: Any, nodes: frozenset[str], ids: frozenset[str],
          edges: frozenset[str]) -> Edit | None:
    """``item`` as an ``Edit``, or None when any field fails its check."""
    if not isinstance(item, Mapping):
        return None
    op, target, name, value, reason = (_text(item.get(f)) for f in
                                       ("op", "target", "name", "value", "reason"))
    if op is None or target is None or name is None or value is None or reason is None:
        return None
    if op not in OPERATIONS:
        return None
    allowed = nodes if op in _TARGET_NODE else edges if op in _TARGET_EDGE else ids
    if target not in allowed:
        return None
    if op in _VALUE_NODE:
        if value not in nodes or value == target:
            return None
    elif op in _VALUE_TEXT:
        if not value:
            return None
    else:
        value = ""
    if op in _NEEDS_NAME:
        if not name:
            return None
    else:
        name = ""
    return Edit(op=op, target=target, name=name, value=value, reason=reason)


def _text(field: Any) -> str | None:
    """A stripped string for a string or a missing field, None for anything else."""
    if field is None:
        return ""
    return field.strip() if isinstance(field, str) else None


def _listing(items: Any) -> str:
    """One line per item: its id, and its label when it has one."""
    lines = []
    if isinstance(items, Mapping):
        pairs: Iterable[tuple[Any, Any]] = items.items()
    elif isinstance(items, (str, bytes)) or not isinstance(items, Iterable):
        pairs = ()
    else:
        pairs = ((item.get("id") if isinstance(item, Mapping) else item, item) for item in items)
    for key, body in pairs:
        if not isinstance(key, str):
            continue
        label = body.get("label") if isinstance(body, Mapping) else None
        lines.append(f"{key}: {label}" if isinstance(label, str) and label else key)
    return "\n".join(lines)
