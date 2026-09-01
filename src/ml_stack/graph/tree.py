"""A hierarchy read out of prose or a picture, as graph entries and the links between them.

An org chart, a family tree, a subject taxonomy and a parts breakdown are the same object:
named things, and a link from each to the one above it. What differs is only what an entry
is called, what the link means, and which details are worth keeping — so that is what a
:class:`Shape` carries, and everything else here is shared.

The reading is done by whatever model the caller hands in. A picture needs one served with
its vision projector (`ml-stack-serve up --mmproj auto`), which is the failure worth knowing
about: without a projector a multimodal model serves as a text model and ignores the image
in silence, answering confidently about nothing.

Nothing here writes to a store. It returns entries and links in the shape `GraphStore.write`
takes, so a caller can look at what was read before letting it near a real graph -- which
matters, because a model reading a blurry chart will invent a plausible reporting line and
say nothing about having done so.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = ["FAMILY", "ORG", "PARTS", "PictureUnreadable", "Shape", "TAXONOMY", "cycles",
           "read", "roots", "schema_for", "to_graph", "transcribe"]


class PictureUnreadable(RuntimeError):
    """An image was handed in and none of it survived being prepared.

    Raised rather than carried on with, because the alternative is asking a model to read a
    chart with no chart attached: it answers, confidently, about nothing, and the reason is
    three layers down in a warnings list nobody read.
    """


@dataclass(frozen=True)
class Shape:
    """What kind of hierarchy this is.

    ``relation`` reads from the entry *upwards*: a person ``reports_to`` their manager, a
    child is ``child_of`` a parent, a subject is ``part_of`` a broader one. Keeping the
    direction fixed means one traversal answers "who is above me" for every shape.
    """

    name: str = "hierarchy"
    kind: str = "person"                    # the node kind each entry becomes
    relation: str = "reports_to"            # the edge from an entry to the one above it
    above: str = "the person they report to"   # how to name the parent when asking
    entry: str = "person"                   # how to name an entry when asking
    attributes: tuple[str, ...] = ("title",)
    # Some hierarchies allow two parents and some do not. A family tree's second parent is
    # ordinary; an org chart's is a mistake worth seeing rather than silently keeping one.
    parents: int = 1
    hint: str = ""


ORG = Shape(name="org chart", kind="person", relation="reports_to",
            above="the person they report to", entry="person",
            attributes=("title", "team"),
            hint="Titles as written. If someone appears with no manager they are at the top.")

FAMILY = Shape(name="family tree", kind="person", relation="child_of",
               above="a parent", entry="person",
               attributes=("born", "died", "note"), parents=2,
               hint="A person may have two parents. Keep dates exactly as written, and "
                    "leave them out entirely rather than guessing at one.")

TAXONOMY = Shape(name="subject taxonomy", kind="topic", relation="part_of",
                 above="the broader subject it belongs to", entry="subject",
                 attributes=(), hint="")

PARTS = Shape(name="parts breakdown", kind="thing", relation="part_of",
              above="the assembly it belongs to", entry="part",
              attributes=("quantity", "code"), hint="")


def schema_for(shape: Shape) -> dict[str, Any]:
    """The JSON schema a model is asked to fill in for this shape."""
    props: dict[str, Any] = {
        "name": {"type": "string", "description": f"the {shape.entry}'s name, as written"},
    }
    for attr in shape.attributes:
        props[attr] = {"type": "string", "description": f"{attr}, or an empty string"}
    props["above"] = {
        "type": "array", "items": {"type": "string"},
        "description": f"the name of {shape.above}. Empty when there is none — "
                       f"do not guess.",
    }
    return {"type": "object", "properties": {"entries": {
        "type": "array", "items": {
            "type": "object", "properties": props, "required": ["name", "above"]}}},
        "required": ["entries"]}


def transcribe(reader: Any, images: Sequence[Path | str | bytes], *,
               ask: str = "", n_predict: int | None = None) -> str:
    """A picture as text, using a model served to read pictures.

    Worth its own step because the best model at *reading* a crowded chart is rarely the
    best at deciding what it means. A document model -- DeepSeek-OCR and its successors,
    GLM-OCR, surya -- transcribes boxes and lines that a general multimodal model skims;
    a chat model then turns that text into structure. Either can be the same model.

    The model must be served with its vision projector or it will answer about nothing at
    all, confidently and without erroring: `ml-stack-serve up <model> --mmproj auto`.
    """
    wanted = ask or (
        "Transcribe everything in this image as plain text. Keep the layout: one name per "
        "line, and where a line sits under another, indent it under that one. Do not "
        "summarise, do not explain, and do not add anything that is not written there.")
    message = _picture(wanted, images)
    reply = reader.chat([message], think=False, **({"n_predict": n_predict} if n_predict else {}))
    return (getattr(reply, "content", "") or "").strip()


def _picture(text: str, images: Sequence[Path | str | bytes]) -> dict[str, Any]:
    """A multimodal message, or a refusal saying which picture could not be read."""
    from ml_stack.vision.payloads import build_message

    message, report = build_message(text, list(images))
    kept = [p for p in message["content"] if p.get("type") == "image_url"]
    if not kept:
        why = "; ".join(report.warnings) or "no reason given"
        raise PictureUnreadable(
            f"none of the {len(list(images))} image(s) could be prepared: {why}")
    return message


def read(client: Any, shape: Shape = ORG, *, text: str = "",
         images: Sequence[Path | str | bytes] = (), n_predict: int | None = None,
         instructions: str = "", reader: Any = None) -> list[dict[str, Any]]:
    """Read a hierarchy out of ``text``, a picture, or both.

    Returns one row per entry, each with a ``name``, the shape's attributes, and ``above``:
    the names it is under. Names, not ids — resolving those is :func:`to_graph`'s job, and
    keeping them apart means a caller can correct a misread name before anything is written.

    ``reader`` is a second model that reads the picture first, and is how a document model
    gets used for what it is good at: it transcribes, ``client`` structures. Without one the
    picture goes to ``client`` directly, which needs it served with a projector.
    """
    if not text and not images:
        raise ValueError("nothing to read: pass text, images, or both")

    if images and reader is not None:
        seen = transcribe(reader, images, n_predict=n_predict)
        text = f"{text}\n\n{seen}".strip() if text else seen
        images = ()

    asking = instructions or (
        f"Read the {shape.name} and list every {shape.entry} in it exactly once. "
        f"For each, give {shape.above} under 'above', using the same spelling you used for "
        f"their own name so the two can be matched. Someone at the top has an empty "
        f"'above'. Do not invent anyone who is not there, and do not guess at a link you "
        f"cannot see. " + (shape.hint or "")
    ).strip()

    messages = [_picture(text or asking, images)] if images else None

    out = client.extract(text, schema_for(shape), instructions=asking,
                         messages=messages, n_predict=n_predict,
                         schema_name=shape.name.replace(" ", "_"))
    rows = out.get("entries")
    return [r for r in rows if isinstance(r, Mapping) and str(r.get("name") or "").strip()] \
        if isinstance(rows, list) else []


def _slug(name: str, kind: str) -> str:
    plain = re.sub(r"[^a-z0-9]+", "", str(name).casefold())
    return f"{kind}:{plain or 'unnamed'}"


def to_graph(rows: Iterable[Mapping[str, Any]], shape: Shape = ORG) -> dict[str, Any]:
    """Entries and links, in the shape ``GraphStore.write`` takes.

    A name given as somebody's parent but never listed as an entry of its own still becomes
    an entry: a chart that says "reports to the board" and never draws the board is a chart
    about the board. Matching is on a squashed-down form of the name, so "Jo Ash" and
    "jo  ash" are one entry and not two.
    """
    nodes: dict[str, dict[str, Any]] = {}
    edges: list[dict[str, Any]] = []

    def put(name: str, attrs: Mapping[str, Any] | None = None) -> str:
        node_id = _slug(name, shape.kind)
        held = nodes.setdefault(node_id, {"id": node_id, "label": str(name).strip(),
                                          "kind": shape.kind, "mentions": 0,
                                          "attrs": {}, "messages": []})
        held["mentions"] += 1
        for key, value in (attrs or {}).items():
            if str(value or "").strip() and not held["attrs"].get(key):
                held["attrs"][key] = str(value).strip()
        return node_id

    for row in rows:
        name = str(row.get("name") or "").strip()
        if not name:
            continue
        mine = put(name, {a: row.get(a) for a in shape.attributes})
        above = row.get("above")
        above = [above] if isinstance(above, str) else list(above or ())
        for parent in [str(p).strip() for p in above if str(p).strip()][: max(1, shape.parents)]:
            theirs = put(parent)
            if theirs == mine:
                continue                  # a chart that says somebody manages themselves
            edges.append({"source": mine, "rel": shape.relation, "target": theirs,
                          "weight": 2, "messages": []})

    return {"nodes": list(nodes.values()), "edges": edges, "messages": {},
            "stats": {"entries": len(nodes), "links": len(edges)},
            "meta": {"shape": shape.name, "relation": shape.relation}}


def roots(graph: Mapping[str, Any], shape: Shape = ORG) -> list[str]:
    """The entries with nobody above them. More than one is common and not an error."""
    under = {str(e["source"]) for e in (graph.get("edges") or ())
             if e.get("rel") == shape.relation}
    return sorted(str(n["id"]) for n in (graph.get("nodes") or ()) if str(n["id"]) not in under)


def cycles(graph: Mapping[str, Any], shape: Shape = ORG) -> list[list[str]]:
    """Any loop in what should be a tree.

    A model reading a crowded chart will happily draw one, and a loop makes "who is above
    me" run until it runs out of graph. Finding them is cheap; being surprised by one later
    is not.
    """
    up: dict[str, list[str]] = {}
    for edge in (graph.get("edges") or ()):
        if edge.get("rel") == shape.relation:
            up.setdefault(str(edge["source"]), []).append(str(edge["target"]))

    found: list[list[str]] = []
    seen: set[str] = set()

    def walk(node: str, trail: list[str]) -> None:
        if node in trail:
            loop = trail[trail.index(node):] + [node]
            if sorted(loop) not in [sorted(f) for f in found]:
                found.append(loop)
            return
        if node in seen:
            return
        seen.add(node)
        for parent in up.get(node, ()):
            walk(parent, [*trail, node])

    for node in list(up):
        walk(node, [])
    return found
