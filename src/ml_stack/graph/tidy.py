"""The hygiene pass over a whole graph store: duplicates merged, inverses folded, the
doubtful flagged, the rest reported.

Two passes touch a knowledge graph, and they are kept apart on purpose. Adam: "hygiene
pass is separate from upsert/additive pass ... you're conflating two things." The *fold*
(``ml_stack.ingest``) is an upsert -- it adds nodes and edges and grows the ones it finds,
never merges, never removes, so a graph can be written into as reads land without ever
losing something good to something broken. This is the other pass, the dedupe: run when a
person asks, dry by default, and what it merges it merges with everything kept on the
survivor.

What it does, in order:

1. **Duplicate nodes**, within a kind and across every source. Two names are the same thing
   when they differ only by case, spacing, hyphens or underscores, or when one is the
   other's plural; the heavier name survives. A name `entities.close` calls one letter off
   another is *reported* as a possible duplicate and not merged: that rule is right for a
   community's typos and wrong for a textbook, where the dry run over a biology shelf
   would have folded "Natrium" into "atrium", "Isobutene" into "isobutane" and
   "Triacylglycerol" into "Diacylglycerol". Those go to the **judge** when one is given
   (`ModelJudge`: a model asked whether the two are one thing, from what it knows first,
   and if it cannot say, from the passages the two names were read from, found through
   their provenance -- Adam: "LLM should re-review the related text if it can't determine
   from internal knowledge"). "Same" merges into the heavier name; "different" is
   written down so the pair is never asked again; only what the model cannot settle after
   reading stays for a person, who hands the decision back as ``written`` (``{name: the
   name it is}``), applied whatever the weights. Every verdict is kept in the store
   (``tidy:decisions``) with its reason, the model, and the units it read. Figures, books
   and runs are never folded. A merge
   moves every edge to the survivor (an edge the survivor already had takes the sum of the
   weights and the union of the provenance), sums mentions, unions provenance, and keeps
   the merged name as an alias.
2. **Relation spellings** folded to the vocabulary the store uses most (`fold_edges`), so
   ``has_part`` and ``haspart`` are one relationship.
3. **Inverse pairs**: ``X part_of Y`` beside ``Y has_part X`` is one fact stored twice. The
   canonical direction (`INVERSES`) is kept, the other's weight and provenance fold into
   it, and the duplicate edge goes -- a fact that is still there the other way round.
4. **Suspect labels**: a clause rather than a name, an over-generic word, a number, a
   single letter. Without a judge they are flagged and left (``attrs.suspect`` says why).
   With one, the label and its passages go to the model, which renames it (through
   `GraphStore.rename`, or a merge when the new name is already a node), drops it -- the
   pass's one removal beyond inverse duplicates and rejected conflict edges -- or keeps it
   and clears the flag.
5. **Verb conflicts**: two edges between the same ends with verbs that are not each
   other's inverse (``X causes Y`` and ``X regulates Y``). Without a judge, reported. With
   one, both edges, the two nodes' definitions and the passages both were read from go to
   the model in one call: ``keep both``, ``keep <verb>`` or ``unsure``. A rejected edge
   goes, its weight and provenance folded into the kept one.
6. **Orphans** reported: nodes with no edge but their source link. Left alone.
7. **Self-loops** reported. Left alone.

A merge keeps both definitions: the longer one that is not a clause fragment becomes
``attrs.definition`` and the other joins ``attrs.definitions_also``; with a judge, and only
when the two say substantially different things, the model picks which is the definition.

A hidden node (``attrs.hidden`` -- an ingest run, a unit) is never touched. Idempotent: a
second run reports nothing to do.

`judge_gold` scores a judge against `ml_stack/data/tidy-gold.json`, a set of invented pairs
whose right answers are known, so a prompt or a model change shows as a number.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ml_stack.entities.fold import ESTABLISHED, fold_edges

__all__ = ["INVERSES", "ModelJudge", "Report", "Scored", "VERDICTS", "absorb",
           "canonical_direction", "excerpts", "gold_file", "judge_gold", "load_gold",
           "plurals", "same_name", "suspect", "tidy", "written_from"]

# The direction a fact is kept in, and the verbs that say the same thing the other way.
INVERSES: dict[str, frozenset[str]] = {
    "part_of": frozenset({"has_part", "contains", "has"}),
    "causes": frozenset({"caused_by"}),
    "precedes": frozenset({"follows", "after"}),
    "produces": frozenset({"produced_by", "made_by"}),
    "created_by": frozenset({"authored", "wrote", "created", "built", "proposed"}),
    "requires": frozenset({"required_by", "enables"}),
}
"""``{canonical verb: the verbs that state it with the ends swapped}``. ``X has_part Y`` is
``Y part_of X``; the pass keeps the left-hand form."""

# What is kept off the "no edge but its source" count: the links that say where a node came
# from rather than what it stands in relation to.
_SOURCE_LINKS = frozenset({"read_from", "read_by", "illustrates"})

_CLAUSE = re.compile(r"\b(that|which|who|aims? to|is an?|are|used to|in order to)\b", re.I)
_TRAILING_PREPOSITION = re.compile(r"\b(of|in|to|for|by|with|on|at|from|into)$", re.I)
_GENERIC = frozenset({"thing", "things", "process", "form of science", "form", "type",
                      "concept", "part", "item", "object", "structure", "substance"})


VERDICTS = ("same", "different", "unsure")
DECISIONS = "tidy:decisions"      # the store document every judged pair is written to
_SECTIONS = ("pairs", "conflicts", "definitions", "suspects")

JUDGE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": list(VERDICTS)},
        "why": {"type": "string", "maxLength": 300},
    },
    "required": ["verdict", "why"],
    "additionalProperties": False,
}

JUDGE_INSTRUCTIONS = (
    "Two names from a knowledge graph built from textbooks may be one thing spelled two "
    "ways, or two different things a letter apart. Say which. `same` only when the two "
    "names denote the very same thing (a typo, a variant spelling, a hyphenation, a "
    "capitalisation, a synonym the field treats as identical). `different` when they name "
    "distinct things, however alike -- an isomer, a numbered form, a subtype, a related "
    "molecule or process. `unsure` when you cannot tell from the names, their definitions "
    "and your own knowledge; you will then be shown the passages they were read from. "
    "`why` is one sentence."
)

JUDGE_READ = (
    "You said you were unsure. Here are passages from the books where each name was read. "
    "Decide from them: `same` or `different`; `unsure` only if the passages do not settle it."
)

CONFLICT_INSTRUCTIONS = (
    "Two edges in a knowledge graph join the same two things with different verbs. Decide "
    "from the passages. `keep both` only when the passages support both relationships as "
    "true. `keep <verb>` when the passages support that one and not the other. `unsure` when "
    "the passages show neither. `why` is one sentence."
)

CONFLICT_READ = "Passages the two edges were read from:"

DEFINITION_INSTRUCTIONS = (
    "Two definitions were written for one thing in a knowledge graph, and the two entries are "
    "being merged into one. Say which should be the definition: `a`, `b`, or `both` when "
    "neither is better and each says something the other does not. Prefer the one that names "
    "what the thing is over one that only says what it does or reads as half a sentence."
)

SUSPECT_INSTRUCTIONS = (
    "A node in a knowledge graph has a label that does not read as the name of a thing -- a "
    "clause, an over-generic word, a number, a single letter. Here is the label and the "
    "passages it was read from. `rename` with `name` set to the thing the passages are "
    "actually about, when there is one. `drop` when the label names nothing the graph should "
    "hold. `keep` when it is a real name after all. `why` is one sentence."
)

CONFLICT_VERDICTS = ("keep both", "unsure")

DEFINITION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "keep": {"type": "string", "enum": ["a", "b", "both"]},
        "why": {"type": "string", "maxLength": 300},
    },
    "required": ["keep"],
    "additionalProperties": False,
}

SUSPECT_VERDICTS = ("rename", "drop", "keep")

SUSPECT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": list(SUSPECT_VERDICTS)},
        "name": {"type": "string", "maxLength": 80},
        "why": {"type": "string", "maxLength": 300},
    },
    "required": ["verdict"],
    "additionalProperties": False,
}


def conflict_schema(verbs: Iterable[str]) -> dict[str, Any]:
    """The verdict schema for one verb conflict: ``keep both``, ``keep <verb>``, ``unsure``."""
    choices = ["keep both", *(f"keep {verb}" for verb in verbs), "unsure"]
    return {
        "type": "object",
        "properties": {
            "verdict": {"type": "string", "enum": choices},
            "why": {"type": "string", "maxLength": 300},
        },
        "required": ["verdict", "why"],
        "additionalProperties": False,
    }


class ModelJudge:
    """A model that says whether two names are one thing -- from knowledge first, from the
    source passages when it cannot.

    ``client`` is a `ml_stack.client.Client` (``extract`` against a small schema, so the
    verdict is one of three words). ``sources`` turns a unit id into the text it was read
    from, for the second look; without it, an unsure verdict stays unsure.
    """

    def __init__(self, client: Any, *, sources: Callable[[str], str] | None = None,
                 model: str = "", excerpt_chars: int = 400, most_units: int = 4,
                 pointers: Callable[[Mapping[str, Any]], list[str]] | None = None) -> None:
        self.client = client
        self.sources = sources
        # where a node's pointers back to its source live: ``provenance`` (the ingest's
        # unit ids) unless the graph keeps them elsewhere -- a community graph keeps
        # message ids under ``data.messages`` -- so a caller says how to find them
        self.pointers = pointers or (lambda node: list(node.get("provenance") or ()))
        self.model = model or str(getattr(client, "model", "") or "")
        self.excerpt_chars = excerpt_chars
        self.most_units = most_units
        self.asked = 0
        self.read = 0
        self.failed = 0

    def _ask(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """One model call, never fatal to the pass: a server that answers 500 for one
        pair (a compute error mid-shelf, 2026-09-03) makes that pair `unsure` and marks
        it ``failed`` so it is not written down as decided, and the pass goes on."""
        try:
            answer = self.client.extract(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001 - one pair's failure is that pair's
            self.failed += 1
            return {"verdict": "unsure", "keep": "", "why": f"the model failed: "
                    f"{type(exc).__name__}: {str(exc)[:160]}", "failed": True}
        return answer if isinstance(answer, dict) else {}

    def decide(self, one: Mapping[str, Any], other: Mapping[str, Any]) -> dict[str, Any]:
        """``{verdict, why, read: [unit ids the second look used]}``."""
        self.asked += 1
        text = self._question(one, other)
        answer = self._ask(text, JUDGE_SCHEMA, instructions=JUDGE_INSTRUCTIONS,
                                     tries=1, n_predict=1024)
        verdict = str((answer or {}).get("verdict") or "unsure")
        why = str((answer or {}).get("why") or "")
        used: list[str] = []
        if (answer or {}).get("failed"):
            return {"verdict": "unsure", "why": why, "read": used, "failed": True}
        if verdict == "unsure" and self.sources is not None:
            passages = self._passages(one) + self._passages(other)
            if passages:
                self.read += 1
                used = [unit for unit, _ in passages]
                shown = "\n\n".join(f"[{unit}] {piece}" for unit, piece in passages)
                answer = self._ask(
                    text + "\n\n" + JUDGE_READ + "\n\n" + shown, JUDGE_SCHEMA,
                    instructions=JUDGE_INSTRUCTIONS, tries=1, n_predict=1024)
                verdict = str((answer or {}).get("verdict") or "unsure")
                why = str((answer or {}).get("why") or why)
        if verdict not in VERDICTS:
            verdict = "unsure"
        return {"verdict": verdict, "why": why, "read": used}

    def decide_conflict(self, one: Mapping[str, Any], other: Mapping[str, Any],
                        edge: Mapping[str, Any], rival: Mapping[str, Any]) -> dict[str, Any]:
        """``{verdict, why, read}`` for two edges between the same ends with different verbs;
        one call, with both nodes' definitions and the passages both edges were read from."""
        self.asked += 1
        verbs = [str(edge.get("rel") or ""), str(rival.get("rel") or "")]
        text = (self._said(one) + "\n\n" + self._said(other) + "\n\nEdge 1: "
                + self._edge_line(one, other, edge) + "\nEdge 2: "
                + self._edge_line(one, other, rival))
        used: list[str] = []
        units = _union(self.pointers(edge), self.pointers(rival))[: self.most_units]
        passages = self._around(units, [one, other])
        if passages:
            self.read += 1
            used = list(dict.fromkeys(unit for unit, _ in passages))
            text += "\n\n" + CONFLICT_READ + "\n\n" + "\n\n".join(
                f"[{unit}] {piece}" for unit, piece in passages)
        answer = self._ask(text, conflict_schema(verbs),
                                     instructions=CONFLICT_INSTRUCTIONS, tries=1, n_predict=1024)
        verdict = str((answer or {}).get("verdict") or "unsure")
        if verdict not in {*CONFLICT_VERDICTS, *(f"keep {verb}" for verb in verbs)}:
            verdict = "unsure"
        return {"verdict": verdict, "why": str((answer or {}).get("why") or ""), "read": used}

    def decide_definition(self, one: Mapping[str, Any], other: Mapping[str, Any],
                          a_said: str, b_said: str) -> dict[str, Any]:
        """``{keep: 'a' | 'b' | 'both', why}`` -- which of two definitions of one thing to keep."""
        self.asked += 1
        text = (f"The thing: {one.get('label')!r} ({one.get('kind')}), being merged with "
                f"{other.get('label')!r}.\n\na. {a_said}\n\nb. {b_said}")
        answer = self._ask(text, DEFINITION_SCHEMA,
                                     instructions=DEFINITION_INSTRUCTIONS, tries=1, n_predict=1024)
        keep = str((answer or {}).get("keep") or "both")
        if keep not in ("a", "b", "both"):
            keep = "both"
        return {"keep": keep, "why": str((answer or {}).get("why") or "")}

    def decide_suspect(self, node: Mapping[str, Any], why: str) -> dict[str, Any]:
        """``{verdict: 'rename' | 'drop' | 'keep', name, why, read}`` for a doubtful label."""
        self.asked += 1
        text = self._said(node) + f"\n\nflagged: {why}"
        used: list[str] = []
        passages = self._around(list(self.pointers(node))[: self.most_units], [node])
        if passages:
            self.read += 1
            used = list(dict.fromkeys(unit for unit, _ in passages))
            text += "\n\nPassages it was read from:\n\n" + "\n\n".join(
                f"[{unit}] {piece}" for unit, piece in passages)
        answer = self._ask(text, SUSPECT_SCHEMA, instructions=SUSPECT_INSTRUCTIONS,
                                     tries=1, n_predict=1024)
        verdict = str((answer or {}).get("verdict") or "keep")
        if verdict not in SUSPECT_VERDICTS:
            verdict = "keep"
        return {"verdict": verdict, "name": str((answer or {}).get("name") or "").strip(),
                "why": str((answer or {}).get("why") or ""), "read": used}

    def _around(self, units: Iterable[str],
                nodes: Iterable[Mapping[str, Any]]) -> list[tuple[str, str]]:
        """One excerpt per unit per label, in order, nothing repeated."""
        if self.sources is None:
            return []
        held = list(nodes)
        out: list[tuple[str, str]] = []
        for unit in units:
            try:
                text = self.sources(str(unit))
            except Exception:  # noqa: BLE001 - a book that cannot be re-read is skipped
                continue
            for node in held:
                for piece in excerpts(text, str(node.get("label") or ""),
                                      chars=self.excerpt_chars, most=1):
                    if (str(unit), piece) not in out:
                        out.append((str(unit), piece))
        return out

    @staticmethod
    def _edge_line(one: Mapping[str, Any], other: Mapping[str, Any],
                   edge: Mapping[str, Any]) -> str:
        names = {str(one.get("id") or ""): str(one.get("label") or ""),
                 str(other.get("id") or ""): str(other.get("label") or "")}
        source = names.get(str(edge.get("source") or ""), str(edge.get("source") or ""))
        target = names.get(str(edge.get("target") or ""), str(edge.get("target") or ""))
        return f"{source} {edge.get('rel')} {target} (seen {int(edge.get('weight') or 0)} time(s))"

    @staticmethod
    def _said(node: Mapping[str, Any]) -> str:
        attrs = node.get("attrs") or {}
        parts = [f"name: {node.get('label')!r}", f"kind: {node.get('kind')}"]
        if attrs.get("definition"):
            parts.append(f"definition: {attrs['definition']}")
        if attrs.get("aliases"):
            parts.append(f"also written: {', '.join(attrs['aliases'])}")
        parts.append(f"mentioned {int(node.get('mentions') or 0)} time(s)")
        return "\n".join(parts)

    @staticmethod
    def _question(one: Mapping[str, Any], other: Mapping[str, Any]) -> str:
        def said(node: Mapping[str, Any]) -> str:
            attrs = node.get("attrs") or {}
            parts = [f"name: {node.get('label')!r}", f"kind: {node.get('kind')}"]
            if attrs.get("definition"):
                parts.append(f"definition: {attrs['definition']}")
            if attrs.get("aliases"):
                parts.append(f"also written: {', '.join(attrs['aliases'])}")
            parts.append(f"mentioned {int(node.get('mentions') or 0)} time(s)")
            return "\n".join(parts)

        return "A.\n" + said(one) + "\n\nB.\n" + said(other)

    def _passages(self, node: Mapping[str, Any]) -> list[tuple[str, str]]:
        assert self.sources is not None
        label = str(node.get("label") or "")
        out: list[tuple[str, str]] = []
        for unit in list(self.pointers(node))[: self.most_units]:
            try:
                text = self.sources(str(unit))
            except Exception:  # noqa: BLE001 - a book that cannot be re-read is skipped, said in why
                continue
            for piece in excerpts(text, label, chars=self.excerpt_chars, most=2):
                out.append((str(unit), piece))
        return out


def excerpts(text: str, label: str, *, chars: int = 400, most: int = 2) -> list[str]:
    """Up to ``most`` windows of ``chars`` characters around where ``label`` appears in
    ``text``, case-insensitively; the first window of the text when it appears nowhere."""
    if not text:
        return []
    low, needle = text.casefold(), str(label or "").casefold()
    out: list[str] = []
    start = 0
    while needle and len(out) < most:
        at = low.find(needle, start)
        if at < 0:
            break
        left = max(0, at - chars // 2)
        right = min(len(text), at + len(needle) + chars // 2)
        out.append(" ".join(text[left:right].split()))
        start = right
    if not out:
        out.append(" ".join(text[:chars].split()))
    return out


@dataclass
class Report:
    """What one pass did or would do; every step's count, and a line per decision."""

    dry_run: bool = True
    findings: list[str] = field(default_factory=list)   # the store's own check, after the writes
    merged_nodes: int = 0
    possible: list[tuple[str, str]] = field(default_factory=list)   # close spellings, unmerged
    judged_same: int = 0
    judged_different: int = 0
    merged_edges: int = 0
    relations_folded: int = 0
    inverses_folded: int = 0
    flagged: int = 0
    refused: int = 0
    conflicts_judged: int = 0
    conflict_edges_dropped: int = 0
    definitions_judged: int = 0
    suspects_resolved: int = 0
    suspects_dropped: int = 0
    conflicts: list[tuple[str, str, str, str]] = field(default_factory=list)
    orphans: list[str] = field(default_factory=list)
    self_loops: list[tuple[str, str]] = field(default_factory=list)
    lines: list[str] = field(default_factory=list)
    # what `absorb` did to an incoming graph, and the graph with its ids rewritten
    graph: dict[str, Any] = field(default_factory=dict)
    mapped_same_name: int = 0
    mapped_plural: int = 0
    left_possible: int = 0

    @property
    def nothing_to_do(self) -> bool:
        return not (self.merged_nodes or self.relations_folded or self.inverses_folded
                    or self.flagged or self.conflict_edges_dropped or self.suspects_dropped)

    @property
    def sound(self) -> bool:
        """Whether the store read back whole after the pass -- a pass that left a store
        that does not read back by id is not a success, whatever it merged."""
        return not self.findings

    def said(self) -> str:
        if self.findings:
            return (f"NOT SOUND: after the pass the store does not read back whole -- "
                    f"{len(self.findings)} finding(s), e.g. {self.findings[0]!r}; the pass "
                    f"is not a success. Rebuild from the reads (ml-stack-ingest fold) and "
                    f"report the store engine version. It would have said: ") + self._said()
        return self._said()

    def _said(self) -> str:
        head = "would merge" if self.dry_run else "merged"
        return (f"{head} {self.merged_nodes} node(s) ({self.merged_edges} edge(s) moved), "
                f"folded {self.relations_folded} relation spelling(s) and "
                f"{self.inverses_folded} inverse pair(s), flagged {self.flagged} label(s); "
                f"judged {self.judged_same} pair(s) the same and {self.judged_different} "
                f"different; {len(self.possible)} possible duplicate(s) by spelling left for a "
                f"person, "
                f"{len(self.conflicts)} verb conflict(s), {len(self.orphans)} orphan(s), "
                f"{len(self.self_loops)} self-loop(s) reported; "
                f"put {self.conflicts_judged} conflicting verb pair(s) to the judge "
                f"({self.conflict_edges_dropped} edge(s) dropped) and "
                f"{self.definitions_judged} definition(s); resolved {self.suspects_resolved} "
                f"suspect label(s) ({self.suspects_dropped} node(s) dropped)")

    def absorbed(self) -> str:
        """What `absorb` did to an incoming graph."""
        return (f"{len(self.graph.get('nodes') or ())} incoming node(s): "
                f"{self.mapped_same_name} onto the same name, {self.mapped_plural} onto a "
                f"singular, {self.judged_same} judged the same, {self.judged_different} judged "
                f"different, {self.left_possible} close spelling(s) left new")


def written_from(path: str | Path | None) -> dict[str, str]:
    """The duplicates a person settled, from a JSON file ``{name: the name it is}``; an
    empty map when no file is named. A file that is not that shape is an error said plainly."""
    import json

    if not path:
        return {}
    held = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    if not isinstance(held, dict) or not all(isinstance(v, str) for v in held.values()):
        raise ValueError(f"{path}: expected a JSON object of name -> name")
    return {str(k): v for k, v in held.items()}


@dataclass
class Scored:
    """What a judge got right on the gold set, per class, and what it cost."""

    total: int = 0
    right: int = 0
    read: int = 0
    seconds: float = 0.0
    model: str = ""
    per_class: dict[str, list[int]] = field(default_factory=dict)
    wrong: list[tuple[str, str, str, str]] = field(default_factory=list)

    @property
    def accuracy(self) -> float:
        return self.right / self.total if self.total else 0.0

    def said(self) -> str:
        classes = "; ".join(f"{name} {right}/{total}"
                            for name, (right, total) in sorted(self.per_class.items()))
        on = f" on {self.model}" if self.model else ""
        return (f"{self.total} pair(s), {self.right} right ({self.accuracy:.0%}){on} -- "
                f"{classes} -- {self.read} needed the passages, {self.seconds:.1f}s")


def gold_file() -> Path:
    """The gold set that ships: invented pairs whose right verdicts are known."""
    return Path(__file__).resolve().parent.parent / "data" / "tidy-gold.json"


def load_gold(path: str | Path | None = None) -> list[dict[str, Any]]:
    """The pairs of a gold set: ``{class, verdict, a, b, passages}`` each."""
    import json

    where = Path(path).expanduser() if path else gold_file()
    held = json.loads(where.read_text(encoding="utf-8"))
    pairs = held.get("pairs") if isinstance(held, dict) else held
    if not isinstance(pairs, list) or not pairs:
        raise ValueError(f"{where}: expected a JSON object with a non-empty 'pairs' list")
    return [dict(pair) for pair in pairs]


def judge_gold(client: Any, gold: str | Path | list[dict[str, Any]] | None = None, *,
               model: str = "", log: Callable[[str], None] | None = None) -> Scored:
    """Score a judge against a gold set: accuracy overall and per class, how many pairs
    needed the passages, and the seconds it took."""
    import time

    pairs = gold if isinstance(gold, list) else load_gold(gold)
    say = log or (lambda _line: None)
    scored = Scored(model=model or str(getattr(client, "model", "") or ""))
    started = time.monotonic()
    for pair in pairs:
        one, other = _gold_node(pair.get("a") or {}), _gold_node(pair.get("b") or {})
        passages = dict(pair.get("passages") or {})
        judge = ModelJudge(client, sources=lambda unit, held=passages: held.get(unit, ""),
                           model=scored.model)
        answer = judge.decide(one, other)
        got = str(answer.get("verdict") or "unsure")
        wanted = str(pair.get("verdict") or "")
        name = str(pair.get("class") or wanted)
        tally = scored.per_class.setdefault(name, [0, 0])
        tally[1] += 1
        scored.total += 1
        scored.read += 1 if judge.read else 0
        if got == wanted:
            tally[0] += 1
            scored.right += 1
        else:
            scored.wrong.append((str(one.get("label")), str(other.get("label")), wanted, got))
            say(f"wrong ({name}): {one.get('label')!r} | {other.get('label')!r} -- "
                f"wanted {wanted}, said {got}: {answer.get('why', '')}")
    scored.seconds = time.monotonic() - started
    say(scored.said())
    return scored


def _gold_node(said: Mapping[str, Any]) -> dict[str, Any]:
    """One side of a gold pair as the node shape the judge reads."""
    attrs = dict(said.get("attrs") or {})
    if said.get("definition"):
        attrs["definition"] = said["definition"]
    return {"id": str(said.get("id") or said.get("label") or ""),
            "label": str(said.get("label") or ""), "kind": str(said.get("kind") or "concept"),
            "mentions": int(said.get("mentions") or 0), "attrs": attrs,
            "provenance": list(said.get("provenance") or ())}


def plurals(names: Iterable[str]) -> dict[str, str]:
    """``{plural (casefolded): singular}`` for every name whose singular is also a name."""
    held = {str(name).casefold(): str(name) for name in names}
    out: dict[str, str] = {}
    for low, _name in held.items():
        for ending, singular in (("ies", "y"), ("es", ""), ("s", "")):
            if low.endswith(ending) and len(low) > len(ending) + 2:
                stem = low[: -len(ending)] + singular
                if stem in held and stem != low:
                    out[low] = held[stem]
                    break
    return out


def suspect(label: str) -> str:
    """Why a label is doubtful as a *name*, or ``""`` when it reads as one."""
    text = " ".join(str(label or "").split())
    if not text:
        return "empty"
    words = text.split()
    if len(words) > 6:
        return f"a clause of {len(words)} words, not a name"
    if _CLAUSE.search(text):
        return "reads as a clause"
    if _TRAILING_PREPOSITION.search(text):
        return "ends in a preposition"
    if text.casefold() in _GENERIC:
        return "too generic to be one thing"
    if re.fullmatch(r"[\d.,%-]+", text):
        return "a number"
    if len(text) == 1:
        return "a single letter"
    return ""


def canonical_direction(rel: str) -> tuple[str, bool]:
    """``(canonical verb, flipped)``: the verb a fact is kept under, and whether the ends
    must be swapped to get there."""
    if rel in INVERSES:
        return rel, False
    for keep, others in INVERSES.items():
        if rel in others:
            return keep, True
    return rel, False


_NEVER_FOLDED = frozenset({"figure", "book", "run", "unit"})


def same_name(label: str) -> str:
    """The form under which two labels are one name: casefolded, with spaces, hyphens and
    underscores collapsed -- "T-cell", "t cell" and "T_cell" are one; "atrium" and
    "Natrium" are not."""
    return re.sub(r"[\s_\-]+", "", str(label or "").casefold())


def tidy(store: Any, *, dry_run: bool = True, established: int = ESTABLISHED,
         written: Mapping[str, str] | None = None, judge: Any = None,
         log: Callable[[str], None] | None = None) -> Report:
    """The pass, over a `GraphStore` or the path to one. Dry by default. ``written`` is
    the map of duplicates a person settled (``{name: the name it is}``), applied as given.
    ``judge`` (a `ModelJudge`) decides the pairs a spelling apart, and with one the pass
    is automated: it applies everything it decides and writes every verdict to the store
    with its reason (Adam: "record the mergers but don't defer them to a human"). A pair
    once judged different is not asked again; a pair the judge cannot settle even after
    reading the source is the one thing left for a person, as ``written``."""
    from ml_stack.entities.spelling import close
    from ml_stack.graph.store import GraphStore

    if judge is not None:
        dry_run = False
    if isinstance(store, (str, Path)):
        with GraphStore(store, read_only=dry_run) as opened:
            return tidy(opened, dry_run=dry_run, established=established, written=written,
                        judge=judge, log=log)
    report = Report(dry_run=dry_run)
    say = log or (lambda _line: None)

    def note(line: str) -> None:
        report.lines.append(line)
        say(line)

    nodes = {n["id"]: n for n in store.nodes() if not _hidden(n)}
    edges = [e for e in store.edges() if e["source"] in nodes and e["target"] in nodes]
    decisions = _decisions(store)

    # 1. duplicate nodes, within a kind
    by_kind: dict[str, list[dict[str, Any]]] = {}
    for node in nodes.values():
        by_kind.setdefault(str(node.get("kind") or ""), []).append(node)
    merged_into: dict[str, str] = {}
    decided = {str(k).casefold(): str(v) for k, v in (written or {}).items()}
    for kind, held in by_kind.items():
        if kind in _NEVER_FOLDED:
            continue
        # the same name under two ids -- case, spacing, hyphens -- is one node; the
        # heavier survives
        by_same: dict[str, dict[str, Any]] = {}
        for node in sorted(held, key=lambda n: -int(n.get("mentions") or 0)):
            key = same_name(node.get("label"))
            if key in by_same:
                merged_into[node["id"]] = by_same[key]["id"]
            else:
                by_same[key] = node
        survivors = list(by_same.values())
        by_label = {str(n.get("label") or ""): n for n in survivors}
        weight = {label: int(n.get("mentions") or 0) for label, n in by_label.items()}
        # plurals and written decisions join whatever the weights; nothing else does
        joins = plurals(weight)
        for low, into in decided.items():
            source = next((lbl for lbl in by_label if lbl.casefold() == low), None)
            target = next((lbl for lbl in by_label if lbl.casefold() == into.casefold()), None)
            if source and target and source != target:
                joins[source.casefold()] = target
        for low, into in joins.items():
            name = next((lbl for lbl in by_label if lbl.casefold() == low), None)
            if name and into in by_label and name != into:
                merged_into[by_label[name]["id"]] = by_label[into]["id"]
        # close spellings: the judge's call when there is one, else a person's
        labels = [lbl for lbl in by_label if by_label[lbl]["id"] not in merged_into]
        for i, one in enumerate(labels):
            for other in labels[i + 1:]:
                if not close(same_name(one), same_name(other)):
                    continue
                a, b = by_label[one], by_label[other]
                key = _pair_key(a["id"], b["id"])
                held = decisions["pairs"].get(key)
                if held is None and judge is not None:
                    held = judge.decide(a, b)
                    held = {**held, "a": a["id"], "b": b["id"], "kind": kind,
                            "model": getattr(judge, "model", ""),
                            "when": _now()}
                    if not held.get("failed"):
                        _keep_decision(store, decisions, "pairs", key, held)
                verdict = str((held or {}).get("verdict") or "unsure")
                if verdict == "same":
                    heavier, lighter = ((a, b) if int(a.get("mentions") or 0)
                                        >= int(b.get("mentions") or 0) else (b, a))
                    if lighter["id"] not in merged_into:
                        merged_into[lighter["id"]] = heavier["id"]
                    report.judged_same += 1
                    note(f"judged same ({kind}): {lighter['label']!r} -> {heavier['label']!r}"
                         f" -- {held.get('why', '')}"
                         + (f" (read {len(held.get('read') or ())} passage(s))"
                            if held.get("read") else ""))
                elif verdict == "different":
                    report.judged_different += 1
                    note(f"judged different ({kind}): {one!r} | {other!r} -- {held.get('why', '')}")
                else:
                    report.possible.append((one, other))
                    note(f"possible ({kind}): {one!r} ~ {other!r} -- a spelling apart"
                         + ("; the judge could not settle it" if held else "")
                         + "; hand it back as written if they are one")
    # follow chains a -> b -> c to their end
    for remove in list(merged_into):
        keep = merged_into[remove]
        seen = {remove}
        while keep in merged_into and keep not in seen:
            seen.add(keep)
            keep = merged_into[keep]
        merged_into[remove] = keep
    for remove, keep in merged_into.items():
        if remove == keep or keep not in nodes or remove not in nodes:
            continue
        gone_label, gone_kind = nodes[remove]["label"], nodes[remove]["kind"]
        moved = _merge(store, nodes, edges, keep, remove, dry_run=dry_run, judge=judge,
                       decisions=decisions, report=report, note=note)
        report.merged_nodes += 1
        report.merged_edges += moved
        note(f"merge: {gone_label!r} -> {nodes[keep]['label']!r} "
             f"({gone_kind}, {moved} edge(s) moved)")
    if merged_into:
        edges = [e for e in edges if e["source"] in nodes and e["target"] in nodes]

    # 2. relation spellings
    keyed = {(e["source"], e["rel"], e["target"]): dict(e) for e in edges
             if e["rel"] not in _SOURCE_LINKS}
    folded, records = fold_edges(keyed, label="relations", provenance="provenance",
                                 settles="the spelling the store uses more is right")
    for record in records:
        report.relations_folded += 1
        note(f"relation: {record.get('from')} -> {record.get('into')}")
    if records and not dry_run:
        for key in keyed:
            if key not in folded:
                store.remove_edge(key[0], key[2], key[1])
        for key, edge in folded.items():
            store.upsert_edge({**edge, "source": key[0], "rel": key[1], "target": key[2]})
    edges = [e for e in edges if e["rel"] in _SOURCE_LINKS] + list(folded.values())

    # 3. inverse pairs
    by_triple = {(e["source"], e["rel"], e["target"]): e for e in edges}
    for triple, edge in list(by_triple.items()):
        source, rel, target = triple
        keep_rel, flipped = canonical_direction(rel)
        if not flipped:
            continue
        canonical_triple = (target, keep_rel, source)
        other = by_triple.get(canonical_triple)
        merged = {"source": target, "rel": keep_rel, "target": source,
                  "weight": int(edge.get("weight") or 0) + int((other or {}).get("weight") or 0),
                  "provenance": _union((other or {}).get("provenance"), edge.get("provenance"))}
        report.inverses_folded += 1
        note(f"inverse: {_label(nodes, source)} {rel} {_label(nodes, target)} -> "
             f"{_label(nodes, target)} {keep_rel} {_label(nodes, source)}")
        if not dry_run:
            store.remove_edge(source, target, rel)
            store.upsert_edge(merged)
        by_triple.pop(triple)
        by_triple[canonical_triple] = merged
    edges = list(by_triple.values())

    # 4. suspect labels
    for node in list(nodes.values()):
        why = suspect(str(node.get("label") or ""))
        if not why:
            continue
        if judge is not None:
            _resolve_suspect(store, nodes, edges, node, why, judge=judge, decisions=decisions,
                             report=report, note=note)
        elif not (node.get("attrs") or {}).get("suspect"):
            report.flagged += 1
            note(f"suspect: {node['label']!r} -- {why}")
            if not dry_run:
                store.set_attribute(node["id"], "suspect", why)

    # 5. verb conflicts, 6. orphans, 7. self-loops
    by_ends: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}
    touched: dict[str, int] = {}
    for edge in edges:
        if edge["rel"] in _SOURCE_LINKS:
            continue
        ends = (min(edge["source"], edge["target"]), max(edge["source"], edge["target"]))
        by_ends.setdefault(ends, {})[edge["rel"]] = edge
        touched[edge["source"]] = touched.get(edge["source"], 0) + 1
        touched[edge["target"]] = touched.get(edge["target"], 0) + 1
        if edge["source"] == edge["target"]:
            report.self_loops.append((edge["source"], edge["rel"]))
            note(f"self-loop: {_label(nodes, edge['source'])} {edge['rel']} itself")
    for (a, b), group in by_ends.items():
        rels = sorted(group)
        if len(rels) < 2:
            continue
        if judge is not None and a in nodes and b in nodes:
            rels = _resolve_conflict(store, nodes, edges, (a, b), group, rels, judge=judge,
                                     decisions=decisions, report=report, note=note)
        if len(rels) > 1:
            report.conflicts.append((a, b, rels[0], rels[1]))
            note(f"conflict: {_label(nodes, a)} and {_label(nodes, b)} are joined by "
                 + " and ".join(rels))
    for node_id, node in nodes.items():
        if node_id not in touched and str(node.get("kind") or "") not in ("book", "figure"):
            report.orphans.append(node_id)
    if report.orphans:
        note(f"orphans: {len(report.orphans)} node(s) with no relation, e.g. "
             + ", ".join(_label(nodes, n) for n in report.orphans[:5]))
    if not dry_run and hasattr(store, "check"):
        # 2026-09-03: a store engine blanked other nodes' strings on a delete, and the
        # pass reported success over a store that no longer read back by id. Never again:
        # the store's own check runs after the writes and the report carries what it found
        report.findings = list(store.check())
    note(report.said())
    return report


def absorb(store: Any, graph: Mapping[str, Any], *, judge: Any = None,
           sources: Callable[[str], str] | None = None,
           log: Callable[[str], None] | None = None) -> Report:
    """Reconcile an incoming ``{nodes, edges}`` graph against what a store already holds,
    on the way in, and return it with its ids rewritten (``report.graph``).

    Every writer of learned knowledge calls this before it upserts -- the ingest's fold, a
    community extraction, a conversation kept in the graph -- because the same concepts come
    back as more is read. The whole-store pass (`tidy`) is for what was written before it
    existed.

    An incoming node whose name is an existing node's under case, spacing and hyphens
    (`same_name`), or its plural, is rewritten onto that node and its edges with it. A close
    spelling goes to ``judge``, with the incoming node's own passage (``attrs.passage``, or
    ``sources`` over its provenance) beside the existing node's: ``same`` rewrites it onto
    the existing node, ``different`` is written to ``tidy:decisions`` and the node stays new,
    ``unsure`` stays new and is reported. Nothing in the store is written but that document.
    """
    from ml_stack.graph.store import GraphStore

    if isinstance(store, (str, Path)):
        with GraphStore(store, read_only=judge is None) as opened:
            return absorb(opened, graph, judge=judge, sources=sources, log=log)
    report = Report(dry_run=True)
    say = log or (lambda _line: None)

    def note(line: str) -> None:
        report.lines.append(line)
        say(line)

    incoming = [dict(node) for node in (graph.get("nodes") or ())]
    by_key: dict[str, dict[str, dict[str, Any]]] = {}
    by_length: dict[str, dict[int, list[str]]] = {}
    by_id: dict[str, dict[str, Any]] = {}
    for node in store.nodes():
        kind = str(node.get("kind") or "")
        if _hidden(node) or kind in _NEVER_FOLDED:
            continue
        key = same_name(node.get("label"))
        by_id[str(node["id"])] = node
        if key not in by_key.setdefault(kind, {}):
            by_key[kind][key] = node
            by_length.setdefault(kind, {}).setdefault(len(key), []).append(key)

    decisions = _decisions(store)
    carried: dict[str, str] = {}
    original = getattr(judge, "sources", None)
    if judge is not None:
        def _text(unit: str) -> str:
            if unit in carried:
                return carried[unit]
            if sources is not None:
                held = sources(unit)
                if held:
                    return held
            return original(unit) if original is not None else ""

        judge.sources = _text
    mapping: dict[str, str] = {}
    try:
        for node in incoming:
            kind = str(node.get("kind") or "")
            held = by_key.get(kind)
            if _hidden(node) or kind in _NEVER_FOLDED or not held:
                continue
            node_id = str(node.get("id") or "")
            key = same_name(node.get("label"))
            found = held.get(key)
            if found is not None and str(found["id"]) == node_id:
                continue
            how = "the same name"
            if found is None:
                for kin in _kin(key):
                    found = held.get(kin)
                    if found is not None:
                        how = "a plural"
                        break
            if found is not None:
                mapping[node_id] = str(found["id"])
                if how == "the same name":
                    report.mapped_same_name += 1
                else:
                    report.mapped_plural += 1
                note(f"absorb: {node.get('label')!r} is {found['label']!r}, already held "
                     f"({how})")
                continue
            for other_key in _near(by_length.get(kind) or {}, key):
                other = held[other_key]
                if str(other["id"]) == node_id:
                    continue
                pair = _pair_key(node_id, str(other["id"]))
                verdict = "unsure"
                answer: Mapping[str, Any] = decisions["pairs"].get(pair) or {}
                if answer:
                    verdict = str(answer.get("verdict") or "unsure")
                elif judge is not None:
                    carried.clear()
                    asked = dict(node)
                    passage = (node.get("attrs") or {}).get("passage")
                    if passage:
                        carried[f"incoming:{node_id}"] = str(passage)
                        asked["provenance"] = [f"incoming:{node_id}",
                                               *(node.get("provenance") or ())]
                    answer = judge.decide(asked, other)
                    verdict = str(answer.get("verdict") or "unsure")
                    _keep_decision(store, decisions, "pairs", pair,
                                   {**answer, "a": node_id, "b": str(other["id"]), "kind": kind,
                                    "model": getattr(judge, "model", ""), "when": _now(),
                                    "absorbed": True})
                if verdict == "same":
                    mapping[node_id] = str(other["id"])
                    report.judged_same += 1
                    note(f"absorb: {node.get('label')!r} judged the same as "
                         f"{other['label']!r} -- {answer.get('why', '')}")
                    break
                if verdict == "different":
                    report.judged_different += 1
                    note(f"absorb: {node.get('label')!r} judged different from "
                         f"{other['label']!r} -- {answer.get('why', '')}")
                    continue
                report.left_possible += 1
                report.possible.append((str(node.get("label") or ""),
                                        str(other.get("label") or "")))
                note(f"absorb: {node.get('label')!r} ~ {other['label']!r} -- a spelling apart, "
                     "left as a new node")
    finally:
        if judge is not None:
            judge.sources = original

    out: dict[str, dict[str, Any]] = {}
    for node in incoming:
        node_id = str(node.get("id") or "")
        final = mapping.get(node_id, node_id)
        if final in out:
            out[final] = _fold_node(out[final], node)
        elif final != node_id:
            out[final] = _fold_node(by_id[final], node)
        else:
            out[final] = node
    edged: dict[tuple[str, str, str], dict[str, Any]] = {}
    for edge in (graph.get("edges") or ()):
        source = mapping.get(str(edge.get("source") or ""), str(edge.get("source") or ""))
        target = mapping.get(str(edge.get("target") or ""), str(edge.get("target") or ""))
        rel = str(edge.get("rel") or "")
        if source == target and rel not in _SOURCE_LINKS:
            continue
        merged = {**edge, "source": source, "target": target}
        already = edged.get((source, rel, target))
        if already is not None:
            merged["weight"] = int(already.get("weight") or 0) + int(edge.get("weight") or 0)
            merged["provenance"] = _union(already.get("provenance"), edge.get("provenance"))
        edged[(source, rel, target)] = merged
    report.graph = {**{k: v for k, v in graph.items() if k not in ("nodes", "edges")},
                    "nodes": list(out.values()), "edges": list(edged.values())}
    note(report.absorbed())
    return report


def _kin(key: str) -> list[str]:
    """The singulars and plurals of one `same_name` key."""
    out: list[str] = []
    for ending, singular in (("ies", "y"), ("es", ""), ("s", "")):
        if key.endswith(ending) and len(key) > len(ending) + 2:
            out.append(key[: -len(ending)] + singular)
    if key.endswith("y") and len(key) > 3:
        out.append(key[:-1] + "ies")
    out += [key + "s", key + "es"]
    return [kin for kin in dict.fromkeys(out) if kin and kin != key]


def _near(by_length: Mapping[int, list[str]], key: str) -> list[str]:
    """The store's names a spelling away from one incoming name; only lengths that could be."""
    from ml_stack.entities.spelling import close

    out: list[str] = []
    for length in range(len(key) - 2, len(key) + 3):
        for other in by_length.get(length) or ():
            if other != key and close(key, other):
                out.append(other)
    return out


def _fold_node(kept: Mapping[str, Any], gone: Mapping[str, Any]) -> dict[str, Any]:
    """An incoming node onto the node it turned out to be: mentions summed, provenance and
    aliases unioned, both definitions kept."""
    node = dict(kept)
    attrs = dict(node.get("attrs") or {})
    aliases = list(attrs.get("aliases") or [])
    for alias in [str(gone.get("label") or ""),
                  *((gone.get("attrs") or {}).get("aliases") or ())]:
        if alias and alias != node.get("label") and alias not in aliases:
            aliases.append(alias)
    attrs["aliases"] = aliases
    for key, value in (gone.get("attrs") or {}).items():
        if (key not in ("aliases", "hidden", "suspect", "passage", "definition",
                        "definitions_also") and value and not attrs.get(key)):
            attrs[key] = value
    _merge_definitions(node, gone, attrs, judge=None, decisions=None, store=None, report=None,
                       note=None)
    node["attrs"] = attrs
    node["mentions"] = int(node.get("mentions") or 0) + int(gone.get("mentions") or 0)
    node["provenance"] = _union(node.get("provenance"), gone.get("provenance"))
    return node


def _pair_key(a: str, b: str) -> str:
    return "|".join(sorted((a, b)))


def _now() -> str:
    import time

    return time.strftime("%FT%T")


def _decisions(store: Any) -> dict[str, dict[str, Any]]:
    """Every verdict the store remembers, by section: ``pairs`` (two names), ``conflicts``
    (two verbs between the same ends), ``definitions``, ``suspects``."""
    held = store.get_doc(DECISIONS) if hasattr(store, "get_doc") else None
    held = held if isinstance(held, dict) else {}
    return {name: (dict(held[name]) if isinstance(held.get(name), dict) else {})
            for name in _SECTIONS}


def _keep_decision(store: Any, decisions: dict[str, dict[str, Any]], section: str, key: str,
                   decision: Mapping[str, Any]) -> None:
    """One verdict into the store's decisions document, at once: a judge run killed
    halfway keeps what it paid for."""
    decisions.setdefault(section, {})[key] = dict(decision)
    if not hasattr(store, "put_doc") or getattr(store, "read_only", False):
        return
    store.put_doc(DECISIONS, {**{name: decisions.get(name) or {} for name in _SECTIONS},
                              "hidden": True})


def _conflict_key(ends: tuple[str, str], one: str, other: str) -> str:
    return "|".join(ends) + "::" + "|".join(sorted((one, other)))


def _resolve_conflict(store: Any, nodes: dict[str, dict[str, Any]], edges: list[dict[str, Any]],
                      ends: tuple[str, str], group: dict[str, dict[str, Any]], rels: list[str],
                      *, judge: Any, decisions: dict[str, dict[str, Any]], report: Report,
                      note: Callable[[str], None]) -> list[str]:
    """The verbs left between two nodes after the judge has been asked about each pair."""
    one, other = nodes[ends[0]], nodes[ends[1]]
    left = list(rels)
    at = 0
    while at < len(left) - 1:
        a_rel, b_rel = left[at], left[at + 1]
        key = _conflict_key(ends, a_rel, b_rel)
        held = decisions["conflicts"].get(key)
        if held is None:
            held = judge.decide_conflict(one, other, group[a_rel], group[b_rel])
            held = {**held, "ends": list(ends), "verbs": sorted((a_rel, b_rel)),
                    "model": getattr(judge, "model", ""), "when": _now()}
            if not held.get("failed"):
                _keep_decision(store, decisions, "conflicts", key, held)
        report.conflicts_judged += 1
        verdict = str(held.get("verdict") or "unsure")
        drop = b_rel if verdict == f"keep {a_rel}" else a_rel if verdict == f"keep {b_rel}" else ""
        if not drop:
            note(f"conflict judged ({verdict}): {one['label']!r} and {other['label']!r} joined "
                 f"by {a_rel} and {b_rel} -- {held.get('why', '')}")
            at += 1
            continue
        kept_rel = a_rel if drop == b_rel else b_rel
        _drop_edge(store, edges, group[kept_rel], group.pop(drop))
        left.remove(drop)
        report.conflict_edges_dropped += 1
        note(f"conflict judged: {one['label']!r} and {other['label']!r} keep {kept_rel}; "
             f"{drop} dropped, its weight and provenance folded in -- {held.get('why', '')}")
    return left


def _drop_edge(store: Any, edges: list[dict[str, Any]], keep: dict[str, Any],
               gone: dict[str, Any]) -> None:
    """``gone`` out of the store and the working list, its weight and provenance into ``keep``."""
    keep["weight"] = int(keep.get("weight") or 0) + int(gone.get("weight") or 0)
    keep["provenance"] = _union(keep.get("provenance"), gone.get("provenance"))
    store.remove_edge(gone["source"], gone["target"], gone["rel"])
    store.upsert_edge(keep)
    for at, edge in enumerate(edges):
        if edge is gone:
            edges.pop(at)
            break


def _resolve_suspect(store: Any, nodes: dict[str, dict[str, Any]], edges: list[dict[str, Any]],
                     node: dict[str, Any], why: str, *, judge: Any,
                     decisions: dict[str, dict[str, Any]], report: Report,
                     note: Callable[[str], None]) -> None:
    """One doubtful label put to the judge: renamed, dropped, or kept with the flag cleared."""
    key = node["id"]
    held = decisions["suspects"].get(key)
    if held is None:
        held = judge.decide_suspect(node, why)
        held = {**held, "label": node.get("label"), "flagged": why,
                "model": getattr(judge, "model", ""), "when": _now()}
        if not held.get("failed"):
            _keep_decision(store, decisions, "suspects", key, held)
    report.suspects_resolved += 1
    verdict = str(held.get("verdict") or "keep")
    label = str(node.get("label") or "")
    name = str(held.get("name") or "").strip()
    if verdict == "rename" and name and name != label:
        into = next((n for n in nodes.values()
                     if n["id"] != node["id"] and str(n.get("label") or "") == name
                     and n.get("kind") == node.get("kind")), None)
        if into is not None:
            moved = _merge(store, nodes, edges, into["id"], node["id"], dry_run=False,
                           judge=judge, decisions=decisions, report=report, note=note)
            report.merged_nodes += 1
            report.merged_edges += moved
            note(f"suspect: {label!r} is {name!r}, which is already a node -- merged into it "
                 f"({moved} edge(s) moved)")
            return
        node["label"] = name
        store.rename(node["id"], name)
        store.set_attribute(node["id"], "suspect", "")
        (node.setdefault("attrs", {}))["suspect"] = ""
        note(f"suspect: {label!r} renamed to {name!r} -- {held.get('why', '')}")
    elif verdict == "drop":
        gone = [e for e in edges if node["id"] in (e["source"], e["target"])]
        store.drop([node["id"]])
        nodes.pop(node["id"], None)
        edges[:] = [e for e in edges if node["id"] not in (e["source"], e["target"])]
        report.suspects_dropped += 1
        note(f"suspect: {label!r} dropped with {len(gone)} edge(s) -- {held.get('why', '') or why}")
    else:
        store.set_attribute(node["id"], "suspect", "")
        (node.setdefault("attrs", {}))["suspect"] = ""
        note(f"suspect: {label!r} kept, the flag cleared -- {held.get('why', '') or why}")


_FRAGMENT = re.compile(r"^(that|which|who|and|or|but|of|in|to|for|with|by)\b", re.I)


def _is_fragment(text: str) -> bool:
    """Whether a definition reads as half a sentence rather than a definition."""
    words = text.split()
    return (len(words) < 3 or bool(_FRAGMENT.match(text))
            or bool(_TRAILING_PREPOSITION.search(text.rstrip("."))))


def _better_definition(one: str, other: str) -> tuple[str, str]:
    """``(the definition, the other)``: the longer that is not a clause fragment."""
    if _is_fragment(one) != _is_fragment(other):
        return (other, one) if _is_fragment(one) else (one, other)
    return (one, other) if len(one) >= len(other) else (other, one)


def _substantially(one: str, other: str) -> bool:
    """Whether two definitions say different things -- neither one the start of the other."""
    a, b = " ".join(one.split()).casefold(), " ".join(other.split()).casefold()
    return not (a.startswith(b) or b.startswith(a))


def _merge_definitions(kept: Mapping[str, Any], gone: Mapping[str, Any],
                       attrs: dict[str, Any], *, judge: Any,
                       decisions: dict[str, dict[str, Any]] | None, store: Any,
                       report: Report | None, note: Callable[[str], None] | None) -> None:
    """The definition of the survivor, with the other kept in ``definitions_also``."""
    a_said = str((kept.get("attrs") or {}).get("definition") or "").strip()
    b_said = str((gone.get("attrs") or {}).get("definition") or "").strip()
    also = [d for d in (attrs.get("definitions_also") or []) if d]
    for held in ((gone.get("attrs") or {}).get("definitions_also") or []):
        if held and held not in also:
            also.append(held)
    if a_said and b_said:
        better, spare = _better_definition(a_said, b_said)
        differ = _substantially(a_said, b_said)
        if judge is not None and differ:
            answer = judge.decide_definition(kept, gone, a_said, b_said)
            pick = str((answer or {}).get("keep") or "both")
            if pick == "a":
                better, spare = a_said, b_said
            elif pick == "b":
                better, spare = b_said, a_said
            if report is not None:
                report.definitions_judged += 1
            if decisions is not None:
                _keep_decision(store, decisions, "definitions",
                               _pair_key(str(kept.get("id")), str(gone.get("id"))),
                               {**answer, "a": a_said, "b": b_said,
                                "model": getattr(judge, "model", ""), "when": _now()})
            if note is not None:
                note(f"definition ({kept.get('label')!r}): kept {pick} -- "
                     f"{(answer or {}).get('why', '')}")
        attrs["definition"] = better
        if differ and spare not in also:
            also.append(spare)
    elif b_said and not a_said:
        attrs["definition"] = b_said
    if also:
        attrs["definitions_also"] = [d for d in also if d != attrs.get("definition")]


def _merge(store: Any, nodes: dict[str, dict[str, Any]], edges: list[dict[str, Any]],
           keep: str, remove: str, *, dry_run: bool, judge: Any = None,
           decisions: dict[str, dict[str, Any]] | None = None, report: Report | None = None,
           note: Callable[[str], None] | None = None) -> int:
    """``remove`` into ``keep`` with everything kept: edges moved (summed where the
    survivor had the same one), mentions summed, provenance unioned, the name an alias."""
    kept, gone = nodes[keep], nodes[remove]
    moved = 0
    by_triple = {(e["source"], e["rel"], e["target"]): e for e in edges}
    for edge in [e for e in edges if remove in (e["source"], e["target"])]:
        source = keep if edge["source"] == remove else edge["source"]
        target = keep if edge["target"] == remove else edge["target"]
        if source == target and edge["rel"] not in _SOURCE_LINKS:
            # a relation between the two names being joined says nothing once they are one
            by_triple.pop((edge["source"], edge["rel"], edge["target"]), None)
            if not dry_run:
                store.remove_edge(edge["source"], edge["target"], edge["rel"])
            continue
        other = by_triple.get((source, edge["rel"], target))
        merged = {"source": source, "rel": edge["rel"], "target": target,
                  "weight": int(edge.get("weight") or 0) + int((other or {}).get("weight") or 0),
                  "provenance": _union((other or {}).get("provenance"), edge.get("provenance"))}
        by_triple.pop((edge["source"], edge["rel"], edge["target"]), None)
        by_triple[(source, edge["rel"], target)] = merged
        moved += 1
        if not dry_run:
            store.remove_edge(edge["source"], edge["target"], edge["rel"])
            store.upsert_edge(merged)
    attrs = dict(kept.get("attrs") or {})
    aliases = list(attrs.get("aliases") or [])
    for alias in [str(gone.get("label") or ""), *((gone.get("attrs") or {}).get("aliases") or [])]:
        if alias and alias != kept.get("label") and alias not in aliases:
            aliases.append(alias)
    attrs["aliases"] = aliases
    for key, value in (gone.get("attrs") or {}).items():
        if (key not in ("aliases", "hidden", "suspect", "definition", "definitions_also")
                and value and not attrs.get(key)):
            attrs[key] = value
    _merge_definitions(kept, gone, attrs, judge=judge, decisions=decisions, store=store,
                       report=report, note=note)
    kept["attrs"] = attrs
    kept["mentions"] = int(kept.get("mentions") or 0) + int(gone.get("mentions") or 0)
    kept["provenance"] = _union(kept.get("provenance"), gone.get("provenance"))
    if not dry_run:
        store.upsert_node(kept)
        store.drop([remove])
    edges[:] = list(by_triple.values())
    nodes.pop(remove, None)
    return moved


def _union(*lists: Any) -> list[str]:
    out: list[str] = []
    for held in lists:
        for item in held or ():
            if item not in out:
                out.append(item)
    return out


def _hidden(node: Mapping[str, Any]) -> bool:
    return bool((node.get("attrs") or {}).get("hidden"))


def _label(nodes: Mapping[str, Mapping[str, Any]], node_id: str) -> str:
    return str((nodes.get(node_id) or {}).get("label") or node_id)
