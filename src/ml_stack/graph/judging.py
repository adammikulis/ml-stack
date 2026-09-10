"""The judge: a model asked whether two names are one thing, which of two verbs between the
same ends the passages support, which of two definitions to keep, and what a doubtful label
is really about -- from what it knows first, and from the source passages when it cannot
say. `judge_gold` scores one against the gold set that ships.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ml_stack.graph.hygiene import union

__all__ = [
    "CONFLICT_VERDICTS",
    "JUDGE_SCHEMA",
    "SUSPECT_VERDICTS",
    "VERDICTS",
    "ModelJudge",
    "Scored",
    "conflict_schema",
    "described",
    "excerpts",
    "gold_file",
    "judge_gold",
    "load_gold",
]

VERDICTS = ("same", "different", "unsure")

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
    "Two names from a knowledge graph built from documents may be one thing spelled two "
    "ways, or two different things a letter apart. Say which. `same` only when the two "
    "names denote the very same thing (a typo, a variant spelling, a hyphenation, a "
    "capitalisation, a synonym the field treats as identical). `different` when they name "
    "distinct things, however alike -- an isomer, a numbered form, a subtype, a related "
    "molecule or process. `unsure` when you cannot tell from the names, their definitions "
    "and your own knowledge; you will then be shown the passages they were read from. "
    "`why` is one sentence."
)

JUDGE_READ = (
    "You said you were unsure. Here are passages from the sources where each name was "
    "read. "
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


def described(node: Mapping[str, Any]) -> str:
    """One node as the lines a judge is shown: name, kind, definition, aliases, mentions."""
    attrs = node.get("attrs") or {}
    parts = [f"name: {node.get('label')!r}", f"kind: {node.get('kind')}"]
    if attrs.get("definition"):
        parts.append(f"definition: {attrs['definition']}")
    if attrs.get("aliases"):
        parts.append(f"also written: {', '.join(attrs['aliases'])}")
    parts.append(f"mentioned {int(node.get('mentions') or 0)} time(s)")
    return "\n".join(parts)


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
        pair (a compute error mid-run, 2026-09-03) makes that pair `unsure` and marks
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
        text = (described(one) + "\n\n" + described(other) + "\n\nEdge 1: "
                + self._edge_line(one, other, edge) + "\nEdge 2: "
                + self._edge_line(one, other, rival))
        used: list[str] = []
        units = union(self.pointers(edge), self.pointers(rival))[: self.most_units]
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
        text = described(node) + f"\n\nflagged: {why}"
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
        snapshot = list(nodes)
        out: list[tuple[str, str]] = []
        for unit in units:
            try:
                text = self.sources(str(unit))
            except Exception:  # noqa: BLE001 - a source that cannot be re-read is skipped
                continue
            for node in snapshot:
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
    def _question(one: Mapping[str, Any], other: Mapping[str, Any]) -> str:
        return "A.\n" + described(one) + "\n\nB.\n" + described(other)

    def _passages(self, node: Mapping[str, Any]) -> list[tuple[str, str]]:
        assert self.sources is not None
        label = str(node.get("label") or "")
        out: list[tuple[str, str]] = []
        for unit in list(self.pointers(node))[: self.most_units]:
            try:
                text = self.sources(str(unit))
            except Exception:  # noqa: BLE001 - a source that cannot be re-read is skipped
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
    where = Path(path).expanduser() if path else gold_file()
    data = json.loads(where.read_text(encoding="utf-8"))
    pairs = data.get("pairs") if isinstance(data, dict) else data
    if not isinstance(pairs, list) or not pairs:
        raise ValueError(f"{where}: expected a JSON object with a non-empty 'pairs' list")
    return [dict(pair) for pair in pairs]



def judge_gold(client: Any, gold: str | Path | list[dict[str, Any]] | None = None, *,
               model: str = "", log: Callable[[str], None] | None = None) -> Scored:
    """Score a judge against a gold set: accuracy overall and per class, how many pairs
    needed the passages, and the seconds it took."""
    pairs = gold if isinstance(gold, list) else load_gold(gold)
    say = log or (lambda _line: None)
    scored = Scored(model=model or str(getattr(client, "model", "") or ""))
    started = time.monotonic()
    for pair in pairs:
        one, other = _gold_node(pair.get("a") or {}), _gold_node(pair.get("b") or {})
        passages = dict(pair.get("passages") or {})
        judge = ModelJudge(client, sources=lambda unit, text_of=passages: text_of.get(unit, ""),
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
