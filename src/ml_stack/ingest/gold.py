"""Whether the extraction does a good job: passages with known triples through the
same path, scored."""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ml_stack.ingest.extract import PER_SECTION
from ml_stack.ingest.reads import _slug

__all__ = ["INVERSES", "Scored", "gold_lines", "gold_score", "read_gold"]


INVERSES: dict[str, frozenset[str]] = {
    "created_by": frozenset({"authored", "wrote", "author_of", "created", "built", "proposed",
                             "discovered", "invented", "founded"}),
    "adopted_by": frozenset({"adopted", "enacted", "ratified"}),
    "has_part": frozenset({"part_of"}),
    "part_of": frozenset({"has_part", "contains", "has"}),
    "causes": frozenset({"caused_by"}),
    "produces": frozenset({"produced_by", "made_by"}),
    "precedes": frozenset({"follows", "after"}),
    "requires": frozenset({"required_by", "enables"}),
}
"""What each verb says when the ends are swapped: `X created_by Y` is `Y authored X`.

A gold set written by hand names facts in whichever direction the sentence did, and a
closed vocabulary says each one in one direction only. Matching the flipped triple through
this map is how the gate tells 'the vocabulary has no word for it' from 'it has the word,
pointing the other way' -- the first gold run counted 'the author authored the charter'
as unsayable when the model had said `charter created_by the author`, the same fact."""


@dataclass
class Scored:
    """What a gold run found: the rates, and every triple that was missed."""

    passages: int = 0
    wanted: int = 0
    found: int = 0
    matched: int = 0
    seconds: float = 0.0
    misses: list[dict[str, str]] = field(default_factory=list)
    spurious: list[dict[str, str]] = field(default_factory=list)
    # Gold triples whose predicate -- and none of its aliases -- is a word the schema's
    # closed vocabulary has. A constrained decode cannot say it, so the triple can never be
    # matched and the recall it drags down is a fact about the gold set, not the model.
    # Said out loud rather than quietly subtracted: which of the two to change is a
    # decision, and it is not this function's.
    unsayable: list[dict[str, str]] = field(default_factory=list)

    @property
    def recall(self) -> float:
        return round(self.matched / self.wanted, 4) if self.wanted else 0.0

    @property
    def precision(self) -> float:
        return round(self.matched / self.found, 4) if self.found else 0.0

    @property
    def f1(self) -> float:
        r, p = self.recall, self.precision
        return round(2 * r * p / (r + p), 4) if r + p else 0.0


def read_gold(path: str | Path) -> list[dict[str, Any]]:
    """A gold set: ``{"passages": [{"passage_id", "source", "text", "triples": [...]}]}``.

    A triple is ``{"subject", "predicate", "object"}`` with optional ``*_aliases`` -- the
    other names a right answer is allowed to use. The aliases are the whole point: an
    extractor that says "mitochondrion" where the gold says "mitochondria" is right, and a
    scorer with no aliases would call it wrong and send somebody tuning a prompt for a week.
    """
    held = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    passages = held.get("passages") if isinstance(held, Mapping) else held
    if not isinstance(passages, list) or not passages:
        raise ValueError(f"{path}: no passages")
    return [dict(p) for p in passages if isinstance(p, Mapping)]


def _names(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    return [str(v) for v in (value or ()) if str(v).strip()]


def _same(said: str, wanted: str, aliases: Sequence[str]) -> bool:
    """Whether ``said`` names the same thing as ``wanted`` or one of its aliases.

    Exact first (casefolded, underscores and spaces the same thing), then containment either
    way -- "the mitochondria" names mitochondria -- then `entities.close`, which is what
    forgives a plural or one letter.
    """
    from ml_stack.entities import close

    def flat(text: str) -> str:
        return re.sub(r"[^a-z0-9]+", " ", str(text).casefold()).strip()

    here = flat(said)
    if not here:
        return False
    for other in [wanted, *aliases]:
        there = flat(other)
        if not there:
            continue
        if here == there:
            return True
        if f" {there} " in f" {here} " or f" {here} " in f" {there} ":
            return True
        if close(here.replace(" ", ""), there.replace(" ", "")):
            return True
    return False


def gold_score(client: Any, passages: Sequence[Mapping[str, Any]], shape: Mapping[str, Any],
               *, per_section: float = PER_SECTION,
               log: Callable[[str], None] | None = None) -> Scored:
    """Every gold passage through the same extraction, scored triple by triple.

    A gold triple is found when some extracted relation matches on all three: subject and
    object through their aliases and `entities.close`, predicate through its own aliases.
    Precision is against the same matching, so a relation the passage does state but the
    gold set does not list counts against it -- which is why a gold passage is written
    short, with everything it states written down.
    """
    from ml_stack import ingest

    out = Scored(passages=len(passages))
    vocabulary = {str(v) for v in
                  ((shape.get("properties") or {}).get("relations") or {})
                  .get("items", {}).get("properties", {}).get("rel", {}).get("enum") or ()}
    for passage in passages:
        text = str(passage.get("text") or "")
        unit = _passage_unit(passage)
        began = time.time()
        row = ingest.extract_unit(client, unit, shape, per_section=per_section)
        out.seconds += time.time() - began
        said = [r for r in (row.extracted.get("relations") or ()) if isinstance(r, Mapping)]
        wanted = [t for t in (passage.get("triples") or ()) if isinstance(t, Mapping)]
        out.wanted += len(wanted)
        out.found += len(said)
        for triple in wanted:
            words = {str(triple.get("predicate") or ""), *_names(triple.get("predicate_aliases"))}
            flipped = {verb for verb, other_way in INVERSES.items() if words & other_way}
            if vocabulary and not ((words | flipped) & vocabulary):
                out.unsayable.append({"passage": str(passage.get("passage_id") or ""),
                                      "predicate": str(triple.get("predicate") or "")})
        taken: set[int] = set()
        for triple in wanted:
            hit = next((i for i, r in enumerate(said) if i not in taken and _matches(r, triple)),
                       None)
            if hit is None:
                out.misses.append({"passage": str(passage.get("passage_id") or ""),
                                   "triple": f"{triple.get('subject')} "
                                             f"{triple.get('predicate')} {triple.get('object')}"})
            else:
                taken.add(hit)
                out.matched += 1
        for index, relation in enumerate(said):
            if index not in taken:
                out.spurious.append({"passage": str(passage.get("passage_id") or ""),
                                     "triple": f"{relation.get('from')} {relation.get('rel')} "
                                               f"{relation.get('to')}"})
        if log:
            log(f"  {row.seconds:5.1f}s  {len(said):>2} said, {len(wanted):>2} wanted  "
                f"{str(passage.get('passage_id') or text[:40])}"
                + (f"  {row.error}" if row.error else ""))
    return out


def _matches(said: Mapping[str, Any], triple: Mapping[str, Any]) -> bool:
    """The extracted relation says the gold triple -- as written, or the other way round
    through `INVERSES` (`charter created_by orlan vesk` says `orlan vesk authored charter`)."""
    subject, subject_aliases = str(triple.get("subject") or ""), _names(triple.get("subject_aliases"))
    obj, obj_aliases = str(triple.get("object") or ""), _names(triple.get("object_aliases"))
    rel = str(said.get("rel") or "")
    if (_same(str(said.get("from") or ""), subject, subject_aliases)
            and _same(rel, str(triple.get("predicate") or ""),
                      _names(triple.get("predicate_aliases")))
            and _same(str(said.get("to") or ""), obj, obj_aliases)):
        return True
    words = {str(triple.get("predicate") or ""), *_names(triple.get("predicate_aliases"))}
    if not (words & INVERSES.get(rel, frozenset())):
        return False
    return (_same(str(said.get("from") or ""), obj, obj_aliases)
            and _same(str(said.get("to") or ""), subject, subject_aliases))


def _passage_unit(passage: Mapping[str, Any]) -> Any:
    """A gold passage dressed as a `Unit`, so it goes through the run's own extraction.

    Scoring a path other than the one that runs is scoring nothing: the prompt, the schema,
    the sampling and the parsing are all the ones the shelf will be read with.
    """
    from ml_stack.sources.pdf import Unit

    source = str(passage.get("source") or "gold")
    return Unit(book=_slug(source), book_title=source, chapter="", chapter_title="",
                section="", section_title=str(passage.get("passage_id") or "passage"),
                first_page=0, last_page=0, text=str(passage.get("text") or ""))


def gold_lines(scored: Scored, *, most: int = 20) -> list[str]:
    """The gold report as lines: the rates, then what was missed."""
    out = [f"gold: {scored.matched} of {scored.wanted} triples over {scored.passages} passages "
           f"-- recall {scored.recall:.0%}, precision {scored.precision:.0%}, "
           f"F1 {scored.f1:.0%} ({scored.seconds:.0f}s)"]
    if scored.misses:
        out.append(f"  missed ({len(scored.misses)}):")
        out += [f"    {m['passage']}: {m['triple']}" for m in scored.misses[:most]]
        if len(scored.misses) > most:
            out.append(f"    ... and {len(scored.misses) - most} more")
    if scored.unsayable:
        words = sorted({m["predicate"] for m in scored.unsayable})
        out.append(f"  {len(scored.unsayable)} of {scored.wanted} gold triples ask for a "
                   f"predicate the schema has no word for ({', '.join(words)}); no "
                   f"constrained answer can match them, so the recall above is a floor -- "
                   f"widen the vocabulary or give those triples an alias inside it")
    if scored.spurious:
        out.append(f"  said but not in the gold ({len(scored.spurious)}):")
        out += [f"    {m['passage']}: {m['triple']}" for m in scored.spurious[:most]]
        if len(scored.spurious) > most:
            out.append(f"    ... and {len(scored.spurious) - most} more")
    return out
