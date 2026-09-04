"""A corpus read back against the truth it was written from, and every generated name
through the name detector.

`consistency` reads what `world.emit` wrote through `ml_stack.sources` and holds it against
the graph `world.simulate.run` wrote beside it: every person in the graph speaks, is written
to, or is named in some message; every outcome an arc wrote back as an edge is named by both
its ends in some message; and no sender, recipient or name-shaped phrase in the corpus is a
person the graph does not hold. `privacy` runs every generated person and organisation name
through the lists and the recogniser the pre-commit hook uses (`ml_stack.redact.hook`).
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ml_stack.redact.hook import FLOOR, from_database, permitted, recogniser, shapes
from ml_stack.world import Message
from ml_stack.world.story import ORG_KINDS, OUTCOMES, kind_of

__all__ = ["Report", "consistency", "default_fixtures", "privacy", "truth"]

# two or more capitalised words in a row: what a person's name looks like in prose
PAIR = re.compile(r"(?<![\w'’-])([A-Z][a-z]+(?:[ ][A-Z][a-z]+)+)(?![\w'’-])")
WORD = re.compile(r"[A-Za-z][\w'’-]*")
# what a sentence ends with, so the capital that opens the next one is not read as a name
SENTENCE_END = re.compile(r"(?:^|[.!?:;\n]|--|[\"'“‘(\[])\s*$")
SHOWN = 40


@dataclass
class Report:
    """What one pass counted and what it refused. ``misses`` are inconsistencies, ``hits``
    are names the detector or a list knows; ``ok`` when both are empty."""

    counts: dict[str, Any] = field(default_factory=dict)
    misses: list[str] = field(default_factory=list)
    hits: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.misses and not self.hits


def truth(where: str | Path) -> dict[str, Any]:
    """The graph at ``where``: a ``graph.json``, or a directory holding one."""
    path = Path(where).expanduser()
    if path.is_dir():
        path = path / "graph.json"
    if not path.is_file():
        raise FileNotFoundError(f"no graph.json at {where}")
    graph = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(graph, Mapping) or "nodes" not in graph:
        raise FileNotFoundError(f"{path} is not a graph (no nodes)")
    return dict(graph)


def default_fixtures() -> str:
    """The repository's ``tests/known-fixtures.txt`` when this package runs from a checkout,
    else an empty string."""
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "tests" / "known-fixtures.txt"
        if candidate.is_file():
            return str(candidate)
    return ""


def _labels(graph: Mapping[str, Any]) -> dict[str, tuple[str, str]]:
    """``{id: (label, kind)}`` for every node."""
    return {str(n["id"]): (str(n.get("label") or n["id"]), kind_of(n))
            for n in graph.get("nodes") or () if n.get("id")}


def _given(label: str) -> str:
    return (label.split() or [label])[0]


def _names_first(label: str) -> re.Pattern[str]:
    return re.compile(rf"(?<![\w'’-]){re.escape(_given(label))}(?:['’]s)?(?![\w'’-])")


def _windows(label: str, longest: int = 4) -> set[str]:
    """Every run of two to ``longest`` consecutive words of the label, casefolded."""
    words = [w.casefold() for w in WORD.findall(label)]
    return {" ".join(words[i:i + n]) for n in range(2, longest + 1)
            for i in range(len(words) - n + 1)}


def _read(corpus: Sequence[str | Path], people: Mapping[str, Mapping[str, str]], domain: str,
          ) -> list[tuple[str, Message]]:
    from ml_stack import sources

    out: list[tuple[str, Message]] = []
    for path in corpus:
        where = str(path)
        out.extend((where, m) for m in sources.read(path, people, domain=domain)
                   if m.kind != "reaction")
    return out


def consistency(corpus: Sequence[str | Path], where: str | Path, *,
                domain: str = "example.com") -> Report:
    """Read every path in ``corpus`` back and hold it against the graph at ``where``.

    Counts: corpora, messages, people, ``spoken`` (edges an arc wrote back, each naming
    the message it was said in) and ``spoken_found``, and -- when a ``messages.jsonl``
    sits beside the graph -- ``asserted`` (relations the simulation recorded a message
    stating, where the record is exact) and ``asserted_found``. Misses: a person the corpus
    never carries, an outcome whose message the corpus lacks or whose thread does not name
    both ends, an asserted relation the message's text does not name, a sender or
    recipient the graph does not hold, and a name-shaped phrase that is nobody in the
    graph.
    """
    graph = truth(where)
    labels = _labels(graph)
    people = {i: label for i, (label, kind) in labels.items() if kind == "person"}
    said = _read(corpus, {i: {"label": label} for i, label in people.items()}, domain)
    report = Report(counts={"corpora": len(corpus), "messages": len(said),
                            "people": len(people), "spoken": 0, "spoken_found": 0})
    blob = "\n".join(m.text for _, m in said)
    words = {w for w in WORD.findall(blob)}

    # every person is an author, a recipient, or named
    reached: set[str] = set()
    for _, m in said:
        reached.add(m.sender)
        reached.update(m.recipients)
    for pid, label in people.items():
        if pid in reached or label in blob or _given(label) in words:
            continue
        report.misses.append(f"{label} ({pid}) never speaks, is written to, or is named")

    # every sender and recipient is somebody in the graph
    strangers: dict[str, str] = {}
    for where_, m in said:
        if m.sender not in people:
            name = str(m.attrs.get("sender_name") or "")
            strangers.setdefault(m.sender, f"sender {m.sender!r}"
                                 + (f" ({name})" if name else "")
                                 + f" in {where_} is not in the truth (first {m.id})")
        for r in m.recipients:
            if r not in people:
                strangers.setdefault(r, f"recipient {r!r} in {where_} is not in the truth "
                                        f"(first {m.id})")
    report.misses.extend(strangers.values())

    # every outcome an arc wrote back was said in a message the corpus holds, in a thread
    # that names both ends
    by_id = {m.id: m for _, m in said}
    threads: dict[str, list[str]] = {}
    for _, m in said:
        threads.setdefault(m.thread or m.id, []).append(m.text)

    def names(end: str, text: str) -> bool:
        label, kind = labels.get(end, (end, ""))
        return label in text or (kind == "person" and bool(_names_first(label).search(text)))

    def name_of(end: str) -> str:
        return labels.get(end, (end, ""))[0]

    for edge in graph.get("edges") or ():
        attrs = edge.get("attrs") or {}
        rel = str(edge.get("rel") or edge.get("relation") or "")
        said_in = str(attrs.get("said_in") or "")
        if not said_in and rel not in OUTCOMES:
            continue
        report.counts["spoken"] += 1
        a, b = str(edge.get("source") or ""), str(edge.get("target") or "")
        first = by_id.get(said_in)
        if first is None:
            report.misses.append(f"{rel} {name_of(a)} -> {name_of(b)} was said in "
                                 f"{said_in or 'no message'}, which the corpus does not hold")
            continue
        root = first.thread or first.id
        thread = "\n".join(threads.get(root, ()))
        if all(names(end, thread) for end in (a, b)):
            report.counts["spoken_found"] += 1
        else:
            report.misses.append(f"{rel} {name_of(a)} -> {name_of(b)} (said in {said_in}) is "
                                 "not named by both ends in its thread")

    # every relation a message asserts is named by both ends in that message
    recorded = Path(where).expanduser()
    recorded = recorded / "messages.jsonl" if recorded.is_dir() else recorded.parent / "messages.jsonl"
    report.counts["asserted"] = report.counts["asserted_found"] = 0
    if recorded.is_file():
        from ml_stack.world.cli import read_messages

        for told in read_messages(recorded):
            if not told.attrs.get("asserts_exact", True):
                continue
            for a, rel, b in ((told.attrs.get("asserts") or {}).get("relations") or ()):
                report.counts["asserted"] += 1
                heard = by_id.get(told.id)
                if heard is None:
                    report.misses.append(f"{told.id} asserts {rel} {name_of(a)} -> {name_of(b)} "
                                         "and the corpus does not hold it")
                elif names(a, heard.text) and names(b, heard.text):
                    report.counts["asserted_found"] += 1
                else:
                    report.misses.append(f"{told.id} asserts {rel} {name_of(a)} -> {name_of(b)} "
                                         f"and its text names neither: {heard.text[:80]!r}")

    # no name-shaped phrase is somebody outside the graph
    rules = shapes()
    known = {label.casefold() for label, _ in labels.values()}
    windows: set[str] = set()
    for label, _ in labels.values():
        windows |= _windows(label)
    truth_words = {w.casefold() for label, _ in labels.values() for w in WORD.findall(label)}
    seen: dict[str, str] = {}
    for where_, m in said:
        for found_ in PAIR.finditer(m.text):
            body = found_.group(1)
            key = body.casefold()
            if key in known or key in windows or key in seen:
                continue
            if rules.stood_down(body) or rules.in_context(body, m.text, found_.span(1)):
                continue
            if SENTENCE_END.search(m.text[:found_.start(1)]):
                rest = body.split()[1:]
                if all(w.casefold() in truth_words for w in rest):
                    continue
            seen[key] = f"{body!r} in {where_} ({m.id}) is not in the truth"
    report.misses.extend(seen.values())
    return report


_STORES: dict[int, frozenset[str]] = {}


def _vocabulary(engine: Any) -> Callable[[str], bool]:
    """Whether the detector's language model held a token before this process analysed
    anything; everything, when the engine exposes no string store."""
    try:
        strings = engine.nlp_engine.nlp["en"].vocab.strings
    except (AttributeError, KeyError, TypeError):
        return lambda _: True
    # analysing a text adds its tokens to the store, so the snapshot is taken once, first
    held = _STORES.setdefault(id(engine), frozenset(str(s) for s in strings))
    return lambda word: word in held


def privacy(where: str | Path, *, fixtures: str | Path = "", allow: str | Path = "",
            env: Mapping[str, str] | None = None) -> Report:
    """Every person and organisation name in the graph at ``where``, through the hook's lists
    and its recogniser.

    A name on the machine's own list of people (``NAMES_GRAPH`` / ``NAMES_SCRAPE`` in
    ``env``) is a hit. A name the recogniser reads as a PERSON at or above the hook's floor,
    every token of which its language model already holds, is a hit; a name on the
    ``fixtures`` or ``allow`` list is not. Counts: ``names``, ``detector`` ("presidio" or
    "not installed"), ``read_as_person`` and ``familiar``.
    """
    env = os.environ if env is None else env
    graph = truth(where)
    names = [(label, kind) for label, kind in _labels(graph).values()
             if kind == "person" or kind in ORG_KINDS]
    fixtures = str(fixtures or "")
    allowed = permitted(str(Path(fixtures).parent) if fixtures else os.getcwd(),
                        fixtures or "none", str(allow or ""))
    report = Report(counts={"names": len(names), "detector": "not installed",
                            "read_as_person": 0, "familiar": 0})

    listed = {n.casefold() for n in from_database(env.get("NAMES_GRAPH", ""),
                                                  env.get("NAMES_SCRAPE", ""))}
    for label, kind in names:
        if label.casefold() in listed:
            report.hits.append(f"{label!r} ({kind}) is on the machine's own list of people")

    engine = recogniser()
    if engine is None:
        return report
    report.counts["detector"] = "presidio"
    knows = _vocabulary(engine)
    for label, kind in names:
        if label.casefold() in allowed:
            continue
        score = max((h.score for h in engine.analyze(text=label, language="en")
                     if h.entity_type == "PERSON" and h.score >= FLOOR), default=0.0)
        if not score:
            continue
        report.counts["read_as_person"] += 1
        if not all(knows(w) for w in label.split()):
            continue
        report.counts["familiar"] += 1
        report.hits.append(f"{label!r} ({kind}) reads as a person the detector knows "
                           f"(score {score:.2f})")
    return report


def render(consistent: Report | None, private: Report | None) -> str:
    """Both reports as the lines the command prints."""
    lines: list[str] = []
    if consistent is not None:
        c = consistent.counts
        lines.append(f"consistency: {c['corpora']} corpora, {c['messages']} messages, "
                     f"{c['people']} people, {c['spoken']} outcomes spoken "
                     f"({c['spoken_found']} found), {c['asserted']} relations asserted "
                     f"({c['asserted_found']} found), {len(consistent.misses)} misses")
        lines.extend(f"  {miss}" for miss in consistent.misses[:SHOWN])
        if len(consistent.misses) > SHOWN:
            lines.append(f"  ...and {len(consistent.misses) - SHOWN} more")
    if private is not None:
        p = private.counts
        detector = ("presidio not installed; only the lists were checked"
                    if p["detector"] != "presidio" else
                    f"presidio read {p['read_as_person']} as a person, {p['familiar']} familiar")
        lines.append(f"privacy: {p['names']} names, {detector}, {len(private.hits)} hits")
        lines.extend(f"  {hit}" for hit in private.hits[:SHOWN])
        if len(private.hits) > SHOWN:
            lines.append(f"  ...and {len(private.hits) - SHOWN} more")
    return "\n".join(lines)
