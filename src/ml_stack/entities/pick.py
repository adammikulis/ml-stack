"""A question about a set of records answered with the ids it is about."""

from __future__ import annotations

from typing import Any, Collection

from .edits import _listing, ids_of

PICK_INSTRUCTIONS = """You answer a question about a graph by naming the items it is about.
Return only JSON matching the schema. Rules:
- ids are copied exactly from the listing. Never invent one, and never return an id that is
  not in the listing.
- Only what the question asks for. An empty list is the right answer when nothing fits.
- why: one short sentence saying what these have in common."""

PICK_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "ids": {"type": "array", "items": {"type": "string"}},
        "why": {"type": "string"},
    },
    "required": ["ids", "why"],
    "additionalProperties": False,
}


def validate_pick(raw: Any, *, ids: Collection[str], limit: int | None = None) -> tuple[list[str], str]:
    """The ids that are really in ``ids``, in the order given, with duplicates dropped."""
    allowed = set(ids)
    body = raw if isinstance(raw, dict) else {}
    out: list[str] = []
    seen = set()
    for item in body.get("ids") or []:
        if not isinstance(item, str):
            continue
        item = item.strip()
        if item in allowed and item not in seen:
            seen.add(item)
            out.append(item)
    why = body.get("why")
    return (out[:limit] if limit is not None else out), (why.strip() if isinstance(why, str) else "")


def objections(raw: Any, *, ids: Collection[str]) -> list[str]:
    """One line per id that is not in the listing."""
    allowed = set(ids)
    body = raw if isinstance(raw, dict) else {}
    return [f"{item!r} is not an id in the listing" for item in body.get("ids") or []
            if isinstance(item, str) and item.strip() not in allowed]


def pick(question: str, *, records: Any, client: Any, instructions: str | None = None,
         limit: int | None = None, n_predict: int | None = None,
         tries: int = 2) -> tuple[list[str], str]:
    """The ids a model reads out of ``question``, keeping only those that are really there."""
    known = ids_of(records)
    if not question.strip() or not known:
        return [], ""
    text = "\n\n".join(["Items:\n" + _listing(records), "Question:\n" + question.strip()])
    raw = client.extract(
        text, PICK_SCHEMA,
        instructions=instructions or PICK_INSTRUCTIONS,
        think=False, schema_name="graph_pick", n_predict=n_predict, tries=tries,
        check=lambda obj: objections(obj, ids=known),
    )
    return validate_pick(raw, ids=known, limit=limit)
