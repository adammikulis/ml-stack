"""The tools a model reads a graph with: find entries by name, read what is held on them,
read a neighbourhood, trace a path, list a kind, read the graph at a glance, quote a source."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from typing import Any

from ml_stack.asking import ASKING, Asking
from ml_stack.client.tokens import estimate_tokens
from ml_stack.entities.paths import between, shortest_path
from ml_stack.graph.prompts import QUOTE_SCHEMA, SUMMARY_SCHEMA, TERSE, TOOLS, as_asked
from ml_stack.graph.search import MATCHED_BY_RANK

FOUND = 12
JOINED = 12
# How many people a rich look_up hit brings with it. A topic found is only halfway to who
# has it, and measured against a real graph every staffing question spent rounds guessing
# spellings because nothing joined a skill to its people in one call. Eight is enough to
# staff from and few enough that a hit with its people is still a line, not a page.
JOINED_HITS = 8
SAID = 2
SAID_CHARS = 220
# How much of an entry's own definition `look_at` reads out. A definition a document was
# read into is cut to 400 characters where it is built, so this cuts nothing that a
# document put there.
DEFINED_CHARS = 400


# How many entries list_kind reads out of one kind. Enough for every organisation a
# community of a few hundred people works for; a kind bigger than that is read most
# mentioned first, and the total is stated so the model knows what it did not see.
LISTED = 40
# How many joined entries `look_around` reads out beside each entry it was given, and how
# many entries it will read out in all. Twelve is `JOINED` -- what look_at already shows as
# names -- said properly, with each one's kind, id and a line of its own words, so a model
# can select a neighbour it never looked up.
AROUND = 12
AROUND_ENTRIES = 24
# The flat cut on one tool result: the characters a tool message has always been trimmed to.
# A conversation given a `reach` cuts by tokens instead -- see `cut`.
CUT = 6000
# How many units a cited entry names, and how many passages `quote` returns per entry.
CITED = 2
QUOTED = 2


# ------------------------------------------------------------------ the graph at a glance
#
# The broad question -- "what is this community about?" -- has no name in it to look up, so
# a search answers it by finding whatever the words happen to hit, and the answer is about
# five arbitrary entries. What it wants is the shape of the whole graph, and that is
# arithmetic over the nodes and edges: no model, no round trip, the same text every time.
SUMMARY_TOP = 10          # entries read out per kind, most mentioned first
SUMMARY_RELATIONS = 8     # how many relations are named as the busiest
SUMMARY_SAID = 140        # characters of one entry's own words


def summarise(graph: Mapping[str, Any], *, top: int = SUMMARY_TOP,
              relations: int = SUMMARY_RELATIONS) -> str:
    """What the whole graph holds, in one text: counts per kind, the most-mentioned
    entries of each kind with a line of their own words, and the busiest relations.

    Computed from the graph and nothing else, so it costs no model call and reads the same
    way twice. Ids are in brackets, as `look_around` writes them, so a model may select an
    entry the summary named without looking it up.
    """
    nodes = list(graph.get("nodes") or ())
    messages = graph.get("messages") or {}
    by_kind: dict[str, list[Mapping[str, Any]]] = {}
    for node in nodes:
        by_kind.setdefault(kind_of(node) or "entry", []).append(node)
    order = sorted(by_kind, key=lambda k: (-len(by_kind[k]), k))
    edges = list(graph.get("edges") or ())
    counted = ", ".join(f"{len(by_kind[k])} {k}" for k in order)
    lines = [f"This graph holds {len(nodes)} entries and {len(edges)} joins"
             + (f": {counted}." if counted else ".")]
    for kind in order:
        rows = sorted(by_kind[kind], key=lambda n: (-int(n.get("mentions") or 0),
                                                    str(n.get("label") or "")))
        lines.append(f"{kind} ({len(rows)}), most mentioned first:")
        for node in rows[:top]:
            attrs = node.get("attrs") or {}
            line = f"  - {node.get('label')} [{node['id']}]"
            for key in ("role", "type", "location"):
                if str(attrs.get(key) or "").strip():
                    line += f", {attrs[key]}"
            said = _said_lines(node, messages, most=1, chars=SUMMARY_SAID, indent="")
            if said:
                line += " " + said[0]
            lines.append(line)
        if len(rows) > top:
            lines.append(f"  ... and {len(rows) - top} more {kind}")
    busiest: dict[str, int] = {}
    for edge in edges:
        rel = str(edge.get("rel") or "joined to")
        busiest[rel] = busiest.get(rel, 0) + 1
    if busiest:
        best = sorted(busiest.items(), key=lambda kv: (-kv[1], kv[0]))[:relations]
        lines.append("Busiest joins: " + ", ".join(f"{rel} ({n})" for rel, n in best) + ".")
    return "\n".join(lines)


def look_up(graph: Mapping[str, Any], text: str, *, limit: int = FOUND,
            rich: bool = False) -> list[dict[str, Any]]:
    """Entries whose name, attributes or own words carry that text, best match first.

    Characters only. For a search that also stems and also knows what a word means, pass
    ``finder=`` to converse — see ``ml_stack.graph.search.hybrid``. With ``rich``, each hit
    also carries ``"score"`` (the rank it was found at) and ``"matched"`` (what found it:
    ``label``, ``attribute`` or ``said``), so an exact name is told apart from one word in
    one quote. Off, a hit is ``{"id", "label", "kind"}`` and nothing else.
    """
    want = " ".join((text or "").split()).casefold()
    if not want:
        return []
    messages = graph.get("messages") or {}
    scored: list[tuple[int, int, Mapping[str, Any]]] = []
    for node in graph.get("nodes") or ():
        label = str(node.get("label") or "").casefold()
        attrs = node.get("attrs") or {}
        if label == want:
            score = 4
        elif want in label:
            score = 3
        elif any(want in str(v).casefold() for v in attrs.values()):
            score = 2
        elif any(want in str((messages.get(mid) or {}).get("text") or "").casefold()
                 for mid in (node.get("messages") or ())[:20]):
            score = 1
        else:
            continue
        scored.append((score, int(node.get("mentions") or 0), node))
    scored.sort(key=lambda row: (-row[0], -row[1], str(row[2].get("label") or "")))
    rows: list[dict[str, Any]] = []
    for score, _, n in scored[:limit]:
        row: dict[str, Any] = {"id": str(n["id"]), "label": str(n.get("label") or ""),
                               "kind": str(n.get("kind") or "")}
        if rich:
            row["score"] = score
            row["matched"] = [MATCHED_BY_RANK[score]]
        rows.append(row)
    return rows


def joined_people(graph: Mapping[str, Any], node_id: str, *,
                  limit: int = JOINED_HITS) -> list[dict[str, str]]:
    """The people joined to an entry by any edge, in either direction, most mentioned first.

    For the staffing question: a topic found is halfway to who has it, and reading the
    people off the topic in the same call is what saves the rounds spent guessing their
    spellings. A person is whatever the graph calls ``person``, by kind or by id prefix.
    """
    by_id = {str(n["id"]): n for n in (graph.get("nodes") or ())}
    people: dict[str, Mapping[str, Any]] = {}
    for edge in graph.get("edges") or ():
        source, target = str(edge.get("source") or ""), str(edge.get("target") or "")
        other = target if source == node_id else source if target == node_id else ""
        if not other or other == node_id:
            continue
        node = by_id.get(other)
        if node is not None and kind_of(node) == "person":
            people[other] = node
    rows = sorted(people.values(), key=lambda n: (-int(n.get("mentions") or 0),
                                                  str(n.get("label") or ""), str(n["id"])))
    return [{"id": str(n["id"]), "label": str(n.get("label") or "")} for n in rows[:limit]]


def enriched(graph: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """look_up's hits with, on each that is not a person, the people joined to it.

    A finder of the caller's own may already say why a hit matched (``score``,
    ``matched``); what it did not say is left out rather than guessed, and ``joined`` is
    read from the graph either way. The caller's rows are copied, never written to.
    """
    by_id = {str(n["id"]): n for n in (graph.get("nodes") or ())}
    out: list[dict[str, Any]] = []
    for r in rows:
        row = dict(r)
        node = by_id.get(str(row.get("id") or ""))
        kind = str(row.get("kind") or "") or (kind_of(node) if node is not None else "")
        if kind != "person":
            row["joined"] = joined_people(graph, str(row.get("id") or ""))
        out.append(row)
    return out


def _packed(blocks: Sequence[tuple[int, str, str]], budget: int | None) -> str:
    """Those blocks as one text, and how much of them a ``budget`` of tokens will carry.

    Without a budget every block goes, in the order it was built, which is what this always
    did. With one, and only when they do not all fit, the most-mentioned go first and each
    goes *whole* -- its quotes with it -- rather than every entry going with its words cut
    out. The quote is the evidence; four entries entire answer a question that twelve
    clipped ones do not.

    Packing rather than stopping at the first block that will not fit: a small entry after a
    large one still fits, and the budget is there to be spent. Ties are broken by id, so the
    same graph and the same budget read out the same way twice.
    """
    if budget is None:
        return "\n".join(text for _mentions, _node_id, text in blocks)
    whole = "\n".join(text for _mentions, _node_id, text in blocks)
    if not whole or estimate_tokens(whole) <= budget:
        return whole
    kept: list[str] = []
    left = int(budget)
    for mentions, node_id, text in sorted(blocks, key=lambda b: (-b[0], b[1])):
        cost = estimate_tokens(text) + 1
        if kept and cost > left:
            continue
        left -= cost
        kept.append(text)
    return "\n".join(kept)


def _said_lines(node: Mapping[str, Any], messages: Mapping[str, Any], *, most: int,
                chars: int = SAID_CHARS, indent: str = "    ") -> list[str]:
    """What an entry said, at most ``most`` of them, each on one line."""
    out: list[str] = []
    for mid in (node.get("messages") or ())[:most]:
        text = str((messages.get(mid) or {}).get("text") or "")
        if text:
            out.append(f'{indent}said: "{" ".join(text.split())[:chars]}"')
    return out


def _defined(node: Mapping[str, Any], *, chars: int = DEFINED_CHARS,
             indent: str = "    ") -> list[str]:
    """An entry's own definition, when the graph holds one.

    A graph read out of documents holds what a thing *is* in ``attrs.definition``, which is
    the only thing there is to answer from; one read out of a conversation holds what was
    said instead. `look_around` already reads out every attribute an entry has; this is
    that one, for `look_at`.
    """
    text = " ".join(str((node.get("attrs") or {}).get("definition") or "").split())
    return [f'{indent}defined: "{text[:chars]}"'] if text else []


def _read_at(where: Mapping[str, Any], unit_id: str) -> str:
    """One unit as a reader would name it: the source, the section, the pages."""
    said = str(where.get("source") or "") or str(unit_id)
    for key in ("section", "title"):
        if where.get(key):
            said += f" {where[key]}"
    pages = [p for p in (where.get("pages") or ()) if p]
    if pages:
        said += (f", p. {pages[0]}" if len(set(pages)) == 1
                 else f", pp. {pages[0]}-{pages[-1]}")
    return said


def _cited(graph: Mapping[str, Any], thing: Mapping[str, Any]) -> str:
    """Where an entry was read, on one line; empty when it points at nothing."""
    units = graph.get("units") or {}
    return "; ".join(_read_at(units.get(u) or {}, str(u))
                     for u in (thing.get("provenance") or ())[:CITED])


def _text_of(texts: Any, unit_id: str) -> str:
    """One unit's own text, from a mapping or a lookup the graph carries."""
    if texts is None:
        return ""
    found = texts(unit_id) if callable(texts) else texts.get(unit_id)
    return str(found or "")


def quotes(graph: Mapping[str, Any], ids: Sequence[str], *,
           budget: int | None = None) -> list[dict[str, Any]]:
    """The words behind those entries: the passage, whether it is the source's own words,
    the entry it belongs to, and where it was read."""
    by_id = {str(n["id"]): n for n in (graph.get("nodes") or ())}
    units, texts = graph.get("units") or {}, graph.get("texts")
    rows: list[dict[str, Any]] = []
    for node_id in ids:
        node = by_id.get(str(node_id))
        if node is None:
            continue
        attrs = node.get("attrs") or {}
        spans = node.get("spans") or {}
        # what the entry records is only quotable when the fold found it in the source
        said = "" if attrs.get("unsourced") else \
            " ".join(str(attrs.get("definition") or "").split())
        cited = list(node.get("provenance") or ())[:QUOTED] or [""]
        for unit_id in cited:
            span = spans.get(unit_id)
            text = _text_of(texts, unit_id) if span else ""
            words = text[int(span[0]):int(span[1])] if text else ""
            rows.append({"id": str(node_id), "label": str(node.get("label") or ""),
                         "quote": " ".join((words or said).split())[:DEFINED_CHARS],
                         "verbatim": bool(words),
                         "read_at": _read_at(units.get(unit_id) or {}, str(unit_id))
                                    if unit_id else ""})
    return _within(rows, budget) if budget is not None else rows


def look_at(graph: Mapping[str, Any], ids: Sequence[str], *,
            budget: int | None = None, cite: bool = False) -> str:
    """What the graph holds on those entries, as text a model can answer from.

    ``budget`` is a conversation's `reach`: a ceiling in tokens on what one tool result may
    carry. Without one nothing is packed and nothing is dropped -- what a caller asked for
    is what it gets. See `_packed`. ``cite`` puts where each entry was read on its first
    line, from the unit documents the graph carries under ``units``.
    """
    by_id = {str(n["id"]): n for n in (graph.get("nodes") or ())}
    messages = graph.get("messages") or {}
    blocks: list[tuple[int, str, str]] = []
    for node_id in ids:
        node = by_id.get(str(node_id))
        if node is None:
            continue
        attrs = node.get("attrs") or {}
        joined = [f"{e.get('rel', 'joined to')} {by_id[e['target']].get('label')}"
                  for e in (graph.get("edges") or ())
                  if e.get("source") == node_id and e.get("target") in by_id]
        joined += [f"{by_id[e['source']].get('label')} {e.get('rel', 'joined to')} this"
                   for e in (graph.get("edges") or ())
                   if e.get("target") == node_id and e.get("source") in by_id]
        line = f"- {node.get('label')} ({attrs.get('type') or node.get('kind') or 'entry'})"
        if cite and (read_at := _cited(graph, node)):
            line += f" [read at {read_at}]"
        for key in ("role", "location"):
            if attrs.get(key):
                line += f", {attrs[key]}"
        if joined:
            line += ": " + "; ".join(joined[:JOINED])
        lines = [line, *_defined(node), *_said_lines(node, messages, most=SAID)]
        blocks.append((int(node.get("mentions") or 0), str(node_id), "\n".join(lines)))
    return _packed(blocks, budget)


def _edges_around(graph: Mapping[str, Any], node_id: str) -> list[tuple[str, str, float]]:
    """Everything joined to ``node_id``: ``(other id, how it reads, weight)``, best first.

    Both directions, and the direction is kept in the reading -- ``-> works_at`` against
    ``<- employs`` -- because "Wren employs the foundry" and "the foundry employs Wren" are
    not the same fact and a model given the wrong one writes it down.

    Heaviest edge first, then by id: a neighbourhood that reads out in a different order
    each time cannot have two answers about it compared with each other.
    """
    best: dict[str, tuple[float, str]] = {}
    for edge in graph.get("edges") or ():
        source, target = str(edge.get("source") or ""), str(edge.get("target") or "")
        if source == node_id:
            other, reads = target, f"-> {edge.get('rel') or 'joined to'}"
        elif target == node_id:
            other, reads = source, f"<- {edge.get('rel') or 'joined to'}"
        else:
            continue
        if not other or other == node_id:
            continue
        weight = float(edge.get("weight") or 0)
        if other not in best or weight > best[other][0]:
            best[other] = (weight, reads)
    return sorted(((one, reads, weight) for one, (weight, reads) in best.items()),
                  key=lambda row: (-row[2], row[0]))


def look_around(graph: Mapping[str, Any], ids: Sequence[str], *, hops: int = 1,
                joined: int = AROUND, entries: int = AROUND_ENTRIES,
                budget: int | None = None, cite: bool = False) -> str:
    """The neighbourhood of those entries, read out in one call.

    Each entry as `look_at` gives it -- label, kind, what is held on it, what it said -- and
    then, indented under it, everything joined to it: the relation and its direction, the
    other entry's label, its id in brackets, its kind, and one line of its own words. The id
    is the point of the brackets: a model may write about a neighbour and select it without
    ever having looked it up, which is the four calls this replaces.

    ``hops`` of 2 makes each neighbour an entry in its own right and reads its neighbourhood
    too; ``entries`` caps how many entries are read out in all, whatever the hops.

    For a model whose context is cheap and whose reading is fast, this is the shape that
    suits it: one fat result instead of five thin ones, since a tool call costs a round trip
    through the slow half of the model and reading the answer back costs the fast half.
    ``budget`` -- a conversation's `reach` -- is what keeps that from being unbounded.
    ``cite`` puts where each entry was read on its first line.
    """
    by_id = {str(n["id"]): n for n in (graph.get("nodes") or ())}
    messages = graph.get("messages") or {}
    wanted: list[str] = []
    for one in ids:
        if str(one) in by_id and str(one) not in wanted:
            wanted.append(str(one))
    blocks: list[tuple[int, str, str]] = []
    seen = set(wanted)
    frontier = wanted
    for _ in range(max(1, int(hops or 1))):
        after: list[str] = []
        for node_id in frontier:
            if len(blocks) >= entries:
                break
            node = by_id[node_id]
            attrs = node.get("attrs") or {}
            head = (f"- {node.get('label')} [{node_id}] "
                    f"({attrs.get('type') or node.get('kind') or 'entry'})")
            if cite and (read_at := _cited(graph, node)):
                head += f" [read at {read_at}]"
            for key in sorted(attrs):
                if key != "type" and str(attrs[key]).strip():
                    head += f", {key}: {attrs[key]}"
            lines = [head, *_said_lines(node, messages, most=SAID)]
            for other, reads, _weight in _edges_around(graph, node_id)[:joined]:
                near = by_id.get(other)
                if near is None:
                    continue
                kind = str((near.get("attrs") or {}).get("type")
                           or near.get("kind") or "entry")
                lines.append(f"    {reads} {near.get('label')} [{other}] ({kind})")
                lines += _said_lines(near, messages, most=1, indent="        ")
                if other not in seen:
                    seen.add(other)
                    after.append(other)
            blocks.append((int(node.get("mentions") or 0), node_id, "\n".join(lines)))
        frontier = after
    return _packed(blocks, budget)


def path_between(graph: Mapping[str, Any], start: str, goal: str) -> dict[str, Any]:
    """The chain of entries from one to another, and how it reads."""
    edges = list(graph.get("edges") or ())
    ids = between(edges, start, goal)
    if not ids:
        return {"path": [], "why": "nothing in the graph joins those two"}
    label = {str(n["id"]): str(n.get("label") or "") for n in (graph.get("nodes") or ())}
    steps = shortest_path(edges, start, goal)
    return {"path": ids, "rels": [e.get("rel", "") for e in steps],
            "reads": " → ".join(label.get(i, i) for i in ids)}


def kind_of(node: Mapping[str, Any]) -> str:
    """A node's kind: what it says, else the ``kind:`` prefix of its id, else nothing."""
    kind = str(node.get("kind") or "")
    if kind:
        return kind
    head, sep, _rest = str(node.get("id") or "").partition(":")
    return head if sep else ""


def _singular(word: str) -> tuple[str, ...]:
    """The kind names a word might be asking for: itself, and what it is the plural of."""
    forms = [word]
    if word.endswith("ies"):
        forms.append(word[:-3] + "y")
    if word.endswith("es"):
        forms.append(word[:-2])
    if word.endswith("s"):
        forms.append(word[:-1])
    return tuple(forms)


def list_kind(graph: Mapping[str, Any], kind: str, *, limit: int = LISTED,
              budget: int | None = None) -> dict[str, Any]:
    """Every entry of one kind, most mentioned first — or the kinds there are.

    For the question a search cannot reach: "which companies do people here work for?" is
    answered by every ``org`` in the graph, and no word finds those, because nothing is
    labelled "company". ``kind`` is matched without regard to case or number, so ``Orgs``
    lists ``org``. A kind the graph does not have comes back as ``{"none": ..., "kinds":
    {name: count}}``, so a model that guessed wrong learns the real names on its first miss
    rather than guessing again.

    ``budget`` -- a conversation's `reach` -- replaces ``limit`` rather than joining it: a
    model that reads cheaply wants every organisation there is, and 40 was only ever a
    guess at what a result may cost. With one, as many entries as that many tokens will
    carry go, most mentioned first; ``total`` still says how many there were.
    """
    nodes = [n for n in graph.get("nodes") or () if not (n.get("attrs") or {}).get("hidden")]
    counts: dict[str, int] = {}
    for node in nodes:
        named = kind_of(node)
        if named:
            counts[named] = counts.get(named, 0) + 1
    by_fold = {k.casefold(): k for k in counts}
    want = " ".join(str(kind or "").split()).casefold()
    found = next((by_fold[w] for w in _singular(want) if w in by_fold), None)
    if found is None:
        kinds = dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))
        return {"none": f"no kind {str(kind or '')!r}", "kinds": kinds}
    rows = [n for n in nodes if kind_of(n) == found]
    rows.sort(key=lambda n: (-int(n.get("mentions") or 0), str(n.get("label") or "")))
    entries = [{"id": str(n["id"]), "label": str(n.get("label") or ""),
                "mentions": int(n.get("mentions") or 0)}
               for n in (rows if budget is not None else rows[:limit])]
    if budget is not None:
        entries = _within(entries, budget)
    return {"kind": found, "total": len(rows), "entries": entries}


def _within(entries: Sequence[Mapping[str, Any]], budget: int) -> list[dict[str, Any]]:
    """As many of those entries as ``budget`` tokens will carry, in the order given.

    They arrive most-mentioned first, so this cuts the tail rather than choosing: a listing
    read out of order is not a listing. One always goes, however long it is, because an
    empty result reads to a model as "there are none".
    """
    kept: list[dict[str, Any]] = []
    left = int(budget)
    for row in entries:
        left -= estimate_tokens(json.dumps(row, ensure_ascii=False)) + 1
        if kept and left < 0:
            break
        kept.append(dict(row))
    return kept


def cut(text: str, reach: int | None) -> str:
    """One tool result, cut to what the conversation will carry it in.

    Without a ``reach`` this is the flat `CUT` characters a tool message has always been
    trimmed to -- byte for byte what it was, so nothing measured before moves. With one it
    is that many tokens, found by halving rather than by a characters-per-token guess,
    because the guess is wrong in both directions on the text a graph returns: ids and
    punctuation cost more than prose, and prose costs less.
    """
    if not reach:
        return text[:CUT]
    if estimate_tokens(text) <= reach:
        return text
    low, high = 0, len(text)
    while low < high:
        mid = (low + high + 1) // 2
        if estimate_tokens(text[:mid]) <= reach:
            low = mid
        else:
            high = mid - 1
    return text[:low]


_BRACKETED = re.compile(r"\[([^\[\]\s]{1,120})\]")


def ids_in(text: str) -> list[str]:
    """The bracketed ids a `look_around` result read out, in the order it read them.

    Read back off the text rather than recomputed, so what the answer counts as found is
    exactly what the model was shown -- a neighbourhood the budget cut short did not happen.
    Anything that is not an id in this graph is dropped by the caller.
    """
    out: list[str] = []
    for one in _BRACKETED.findall(text or ""):
        if one not in out:
            out.append(one)
    return out


def tools_for(graph: Mapping[str, Any], *, asking: Asking = ASKING, finder: Any = None,
              terse: bool = False, cite: bool = False
              ) -> list[tuple[dict[str, Any], Any]]:
    """The built-in tools over that graph, as ``(schema, callable)`` pairs.

    Each callable takes the parsed arguments mapping. ``finder`` replaces how look_up looks:
    it takes the text and returns ``[{"id", "label", "kind"}, ...]``. ``asking`` is the
    :class:`~ml_stack.graph.Asking` these are offered under -- ``rich``, ``tight``,
    ``reach``, ``batch``, ``single``, ``few`` and ``summary`` between them decide what each
    schema says and how much one result may carry; `Asking.tools` is the same thing as
    keyword arguments. ``terse`` chooses the short set.

    ``cite`` renders where each entry was read into every `look_at` and `look_around`
    result and offers `quote`, the source's own words behind an entry.
    """
    rich, reach = bool(asking.rich), asking.reach

    def find(args: Mapping[str, Any]) -> Any:
        # one word or several: a staffing question needs a lookup per skill, and doing them
        # one round at a time is what spent every turn a question had
        wanted = [str(x) for x in (args.get("texts") or ()) if str(x).strip()]
        if not wanted and str(args.get("text") or "").strip():
            wanted = [str(args["text"])]
        rows, seen = [], set()
        for text in wanted:
            for r in (finder(text) if finder is not None else look_up(graph, text, rich=rich)):
                if r["id"] not in seen:
                    seen.add(r["id"])
                    rows.append(r)
        if rich:
            rows = enriched(graph, rows)
        # an empty list reads as "try again"; saying nothing matched reads as "move on"
        return rows or {"none": f"Nothing in the graph matches {', '.join(map(repr, wanted))}. "
                                "Try different words, or answer with what you already have."}

    def read(args: Mapping[str, Any]) -> str:
        return look_at(graph, [str(i) for i in (args.get("ids") or ())], budget=reach,
                       cite=cite)

    def around(args: Mapping[str, Any]) -> str:
        try:
            hops = int(args.get("hops") or 1)
        except (TypeError, ValueError):
            hops = 1                  # a model that wrote "one" meant one, not an error
        return look_around(graph, [str(i) for i in (args.get("ids") or ())],
                           hops=max(1, min(hops, 3)), budget=reach, cite=cite)

    def trace(args: Mapping[str, Any]) -> dict[str, Any]:
        return path_between(graph, str(args.get("from_id") or ""), str(args.get("to_id") or ""))

    def listing(args: Mapping[str, Any]) -> dict[str, Any]:
        return list_kind(graph, str(args.get("kind") or ""), budget=reach)

    def light(args: Mapping[str, Any]) -> str:
        # the ids are the whole result; the model is told they arrived so it stops calling it
        return f"selected {len(list(args.get('ids') or ()))} on the graph"

    def glance(args: Mapping[str, Any]) -> str:
        return summarise(graph)

    def said(args: Mapping[str, Any]) -> list[dict[str, Any]]:
        return quotes(graph, [str(i) for i in (args.get("ids") or ())], budget=reach)

    does = {"look_up": find, "look_at": read, "look_around": around, "path_between": trace,
            "list_kind": listing, "show": light, "summarise": glance, "quote": said}
    base: Sequence[Mapping[str, Any]] = TERSE if terse else TOOLS
    if asking.summary:
        base = [*base, SUMMARY_SCHEMA]
    if cite:
        base = [*base, QUOTE_SCHEMA]
    schemas = as_asked(base, asking)
    # in the order the schemas are written, which is the order a model reads them in
    return [(schema, does[str(schema["function"]["name"])]) for schema in schemas]
