"""A nodes/edges CSV pair from another extractor, brought into a store as one source.

The pair is turned into this library's own reads -- one per section, in the document
schema's shape -- and folded in by `ingest.fold_into`, so an imported source is the same
thing in the store as a read one: the same node and edge shape, the same unit ids in its
provenance, the same commands over it.

`RELATIONS` is the table between the two vocabularies. This library sets eighteen verbs
itself -- `fold.CORE`; an extractor with an open vocabulary writes thousands. Each entry
maps one predicate onto one of the eighteen, swapping subject and object where the natural
reading is the inverse (``includes`` is ``has_part``; ``defines`` is ``defined_by`` the
other way round), or onto nothing where the predicate has no counterpart among them.

Every predicate is written either way: one with a counterpart under that verb, one without
as it stands, its edges carrying ``extension``. ``core_only`` writes the first and leaves
the second. What each predicate became is counted by name and kept in the store as
``ingest:predicates:<source>``.

`VAGUE` is the part of an open vocabulary that is not a vocabulary. ``related_to``,
``describes``, ``supports`` and their kind say the two things were named near one another
and nothing about how they stand; an edge under one carries ``vague`` as well as
``extension``, so a question put to the graph can leave them out.
"""

from __future__ import annotations

import csv
import json
import platform
import sys
import time
from collections import Counter
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ml_stack.ingest.reads import Read, _keep_reads, _slug
from ml_stack.sources.pdf import Unit

__all__ = ["CONFIDENCE", "KINDS", "RELATIONS", "VAGUE", "Imported", "bring", "imported",
           "lines", "named", "vague", "verb_for"]


CONFIDENCE = ("low", "medium", "high")
"""The confidence a row may carry, least sure first."""


KINDS = {
    "concept": "concept", "entity": "concept", "condition": "concept", "disease": "concept",
    "symptom": "concept", "sign": "concept", "state": "concept", "property": "concept",
    "trait": "concept", "function": "concept", "discipline": "concept", "field": "concept",
    "domain": "concept", "category": "concept", "group": "concept", "phenomenon": "concept",
    "outcome": "concept", "result": "concept", "observation": "concept", "form": "concept",
    "identifier": "concept", "alias": "concept", "synonym": "concept", "expression": "concept",
    "section": "concept", "document": "concept", "table": "concept", "course": "concept",
    "award": "concept", "energy": "concept", "emotion": "concept", "phase": "concept",
    "process": "process", "event": "process", "movement": "process", "activity": "process",
    "action": "process", "behavior": "process", "behaviour": "process", "cycle": "process",
    "reflex": "process", "procedure": "process", "intervention": "process",
    "application": "process", "imaging": "process", "lifestage": "process",
    "structure": "structure", "organ": "structure", "organelle": "structure",
    "cell": "structure", "tissue": "structure", "vessel": "structure", "system": "structure",
    "complex": "structure", "instrument": "structure", "tool": "structure",
    "organism": "structure", "population": "structure", "injury": "structure",
    "anatomical region": "structure", "diagram": "concept", "figure": "concept",
    "substance": "substance", "compound": "substance", "element": "substance",
    "isotope": "substance", "ion": "substance", "protein": "substance", "enzyme": "substance",
    "hormone": "substance", "neurotransmitter": "substance", "vitamin": "substance",
    "polymer": "substance", "amino acid": "substance", "functional group": "substance",
    "drug": "substance", "medication": "substance", "therapeutic agent": "substance",
    "ligand": "substance", "intermediate": "substance", "product": "substance",
    "resource": "substance", "physical factor": "substance",
    "location": "place", "place": "place", "direction": "place", "region": "place",
    "unit": "unit", "measurement": "unit", "physiological parameter": "unit",
    "timeframe": "unit", "diagnostic test": "method",
    "method": "method", "technique": "method",
    "law": "law", "person": "person", "profession": "person",
    "organisation": "organisation", "organization": "organisation",
}
"""``category (casefolded) -> the kind this library names it``; anything else is a concept."""

_KIND = "concept"


RELATIONS: dict[str, tuple[str, bool] | None] = {
    # part and whole
    "part_of": ("part_of", False), "is_part_of": ("part_of", False),
    "component_of": ("part_of", False), "belongs_to": ("part_of", False),
    "subfield_of": ("part_of", False), "subdiscipline_of": ("part_of", False),
    "branch_of": ("part_of", False), "contained_in": ("part_of", False),
    "subset_of": ("part_of", False),
    "has_part": ("has_part", False), "includes": ("has_part", False),
    "contains": ("has_part", False), "composed_of": ("has_part", False),
    "is_composed_of": ("has_part", False), "consists_of": ("has_part", False),
    "comprises": ("has_part", False), "made_of": ("has_part", False),
    "made_up_of": ("has_part", False), "has_component": ("has_part", False),
    "has_division": ("has_part", False), "subdivided_into": ("has_part", False),
    "divided_into": ("has_part", False),
    # kind and instance
    "is_a": ("is_a", False), "type_of": ("is_a", False), "is_type_of": ("is_a", False),
    "a_type_of": ("is_a", False), "kind_of": ("is_a", False), "subtype_of": ("is_a", False),
    "instance_of": ("is_a", False), "classified_as": ("is_a", False),
    "categorized_as": ("is_a", False), "categorised_as": ("is_a", False),
    "has_type": ("is_a", True), "has_subtype": ("is_a", True),
    "example_of": ("example_of", False), "is_example_of": ("example_of", False),
    "has_example": ("example_of", True),
    # cause
    "causes": ("causes", False), "results_in": ("causes", False),
    "leads_to": ("causes", False), "gives_rise_to": ("causes", False),
    "triggers": ("causes", False), "induces": ("causes", False),
    "can_cause": ("causes", False), "contributes_to": ("causes", False),
    "brings_about": ("causes", False), "drives": ("causes", False),
    "initiates": ("causes", False),
    "caused_by": ("causes", True), "results_from": ("causes", True),
    "due_to": ("causes", True), "triggered_by": ("causes", True),
    "induced_by": ("causes", True), "arises_from": ("causes", True),
    # making and unmaking
    "produces": ("produces", False), "forms": ("produces", False),
    "creates": ("produces", False), "generates": ("produces", False),
    "synthesizes": ("produces", False), "synthesises": ("produces", False),
    "secretes": ("produces", False), "releases": ("produces", False),
    "yields": ("produces", False), "makes": ("produces", False),
    "produced_by": ("produces", True), "formed_by": ("produces", True),
    "created_from": ("produces", True), "derived_from": ("produces", True),
    "secreted_by": ("produces", True), "released_by": ("produces", True),
    "synthesized_by": ("produces", True), "generated_by": ("produces", True),
    "consumes": ("consumes", False), "uses": ("consumes", False),
    "utilizes": ("consumes", False), "utilises": ("consumes", False),
    "used_by": ("consumes", True),
    "converts": ("converts", False), "converts_to": ("converts", False),
    "converted_to": ("converts", False), "transforms_into": ("converts", False),
    "reduced_to": ("converts", False), "oxidized_to": ("converts", False),
    "converted_from": ("converts", True), "oxidized_by": ("converts", True),
    # control
    "regulates": ("regulates", False), "inhibits": ("regulates", False),
    "stimulates": ("regulates", False), "activates": ("regulates", False),
    "modulates": ("regulates", False), "controls": ("regulates", False),
    "suppresses": ("regulates", False), "increases": ("regulates", False),
    "decreases": ("regulates", False), "reduces": ("regulates", False),
    "raises": ("regulates", False), "lowers": ("regulates", False),
    "enhances": ("regulates", False), "promotes": ("regulates", False),
    "prevents": ("regulates", False), "maintains": ("regulates", False),
    "regulated_by": ("regulates", True), "inhibited_by": ("regulates", True),
    "controlled_by": ("regulates", True), "activated_by": ("regulates", True),
    "stimulated_by": ("regulates", True), "modulated_by": ("regulates", True),
    "increased_by": ("regulates", True), "decreased_by": ("regulates", True),
    "maintained_by": ("regulates", True),
    # place
    "located_in": ("located_in", False), "occurs_in": ("located_in", False),
    "found_in": ("located_in", False), "present_in": ("located_in", False),
    "occurs_at": ("located_in", False), "located_at": ("located_in", False),
    "situated_in": ("located_in", False), "resides_in": ("located_in", False),
    "occurs_within": ("located_in", False), "found_within": ("located_in", False),
    # measure and definition
    "measured_in": ("measured_in", False), "expressed_in": ("measured_in", False),
    "has_unit": ("measured_in", False),
    "defined_by": ("defined_by", False), "characterized_by": ("defined_by", False),
    "characterised_by": ("defined_by", False), "defines": ("defined_by", True),
    "illustrates": ("illustrates", False), "depicts": ("illustrates", False),
    "illustrated_by": ("illustrates", True), "shown_in": ("illustrates", True),
    "contrasts_with": ("contrasts_with", False), "differs_from": ("contrasts_with", False),
    "contrasted_with": ("contrasts_with", False), "compared_to": ("contrasts_with", False),
    "distinguished_from": ("contrasts_with", False), "opposes": ("contrasts_with", False),
    # order
    "precedes": ("precedes", False), "followed_by": ("precedes", False),
    "occurs_before": ("precedes", False),
    "follows": ("precedes", True), "preceded_by": ("precedes", True),
    "occurs_after": ("precedes", True), "after": ("precedes", True),
    # need
    "requires": ("requires", False), "depends_on": ("requires", False),
    "needs": ("requires", False), "relies_on": ("requires", False),
    "required_for": ("requires", True), "essential_for": ("requires", True),
    "necessary_for": ("requires", True), "required_by": ("requires", True),
    "enables": ("requires", True), "allows": ("requires", True),
    # who made it, who took it up
    "created_by": ("created_by", False), "discovered_by": ("created_by", False),
    "invented_by": ("created_by", False), "proposed_by": ("created_by", False),
    "developed_by": ("created_by", False), "authored_by": ("created_by", False),
    "written_by": ("created_by", False), "performed_by": ("created_by", False),
    "discovered": ("created_by", True), "invented": ("created_by", True),
    "proposed": ("created_by", True), "developed": ("created_by", True),
    "authored": ("created_by", True), "created": ("created_by", True),
    "founded": ("created_by", True), "built": ("created_by", True),
    "adopted_by": ("adopted_by", False), "enacted_by": ("adopted_by", False),
    "ratified_by": ("adopted_by", False), "adopted": ("adopted_by", True),
    "member_of": ("member_of", False), "is_member_of": ("member_of", False),
    "belongs_to_class": ("member_of", False), "has_member": ("member_of", True),
    # the same things said again, in the words this material happens to use
    "are_part_of": ("part_of", False), "may_include": ("has_part", False),
    "is_a_type_of": ("is_a", False), "example_is": ("example_of", True),
    "may_cause": ("causes", False), "can_result_in": ("causes", False),
    "may_result_in": ("causes", False), "may_lead_to": ("causes", False),
    "result_of": ("causes", True), "can_be_caused_by": ("causes", True),
    "can_be_due_to": ("causes", True), "driven_by": ("causes", True),
    "differentiates_into": ("converts", False), "develops_into": ("converts", False),
    "matures_into": ("converts", False),
    "breaks_down": ("consumes", False), "absorbs": ("consumes", False),
    "located_on": ("located_in", False), "stored_in": ("located_in", False),
    "released_into": ("located_in", False), "site_of": ("located_in", True),
    "catalyzes": ("regulates", False), "catalyses": ("regulates", False),
    "impairs": ("regulates", False), "helps_maintain": ("regulates", False),
    "requires_knowledge_of": ("requires", False),
    "requires_understanding_of": ("requires", False),
    # named, and with no counterpart among the eighteen: each is written as it stands
    "related_to": None, "relates_to": None, "associated_with": None, "describes": None,
    "described_by": None, "shows": None,
    "supports": None, "involves": None, "involved_in": None, "participates_in": None,
    "affects": None, "influences": None, "determines": None, "performs": None,
    "provides": None, "provided_by": None, "has_property": None, "has_characteristic": None,
    "has_function": None, "has_measurement": None, "indicates": None, "covers": None,
    "lists": None, "explains": None, "means": None, "applies_to": None, "acts_on": None,
    "acts_as": None, "functions_as": None, "exhibits": None, "same_as": None,
    "synonym_of": None, "studies": None, "used_for": None, "used_in": None,
    "connects": None, "targets": None, "receives": None, "carries": None, "delivers": None,
    "supplies": None, "protects": None, "transports": None, "binds": None, "binds_to": None,
    "attaches_to": None, "surrounds": None, "separates": None, "lines": None, "moves": None,
    "integrates": None, "coordinates": None, "improves": None, "detects": None,
    "detected_by": None, "treats": None, "facilitates": None, "mediates": None,
    "encodes": None, "innervates": None, "increases_risk_of": None, "undergoes": None,
    "originates_from": None, "located_between": None, "compares": None, "characterizes": None,
    "stores": None, "interacts_with": None, "is": None, "are": None, "has": None,
    "can_be": None, "may_be": None, "or": None, "and": None,
    "measures": None, "lacks": None, "correlates_with": None, "equals": None,
    "also_known_as": None, "mediated_by": None, "affected_by": None,
    "is_affected_by": None, "supported_by": None, "determined_by": None,
    "has_shape": None, "has_feature": None, "excludes": None, "connects_to": None,
    "reflects": None, "identifies": None, "assesses": None, "evaluates": None,
    "demonstrates": None, "summarizes": None, "classifies": None,
    "expresses": None, "conveys": None, "quantifies": None, "hosts": None, "records": None,
    "occurs_when": None, "occurs_during": None, "travels_through": None,
    "passes_through": None, "drains_into": None, "experiences": None, "sensitive_to": None,
    "applies": None, "underlies": None, "powers": None, "articulates_with": None,
    "processes": None, "removes": None, "outlines": None, "states": None,
}
"""``predicate -> (one of the eighteen, whether the ends swap)``, or None for a predicate
the table names and has no counterpart for."""


VAGUE = frozenset({
    "related_to", "relates_to", "associated_with", "describes", "described_by",
    "supports", "supported_by", "involves", "involved_in", "affects", "affected_by",
    "is_affected_by", "influences", "determines", "determined_by", "provides",
    "provided_by", "performs", "experiences", "interacts_with", "connects_to",
    "sensitive_to", "underlies", "applies", "applies_to", "acts_as", "acts_on",
    "functions_as", "exhibits", "has_property", "has_characteristic", "has_function",
    "has_measurement", "has_feature", "has_shape", "indicates", "reflects", "covers",
    "lists", "explains", "means", "measures", "quantifies", "assesses", "evaluates",
    "identifies", "classifies", "compares", "characterizes", "demonstrates", "shows",
    "summarizes", "records", "outlines", "states", "correlates_with", "equals", "lacks",
    "excludes", "same_as", "synonym_of", "also_known_as", "is", "are", "has", "can_be",
    "may_be", "or", "and",
})
"""Predicates that say the two things were named together and not how they stand.

An edge under one of these carries ``vague``. Counted over the anatomy pair: 3,861 of its
21,922 relations under 92 predicates -- ``supports`` 615, ``related_to`` 608, ``describes``
601. They are not wrong, they are unspecific, and a graph asked a question cannot use
them."""


def _word(predicate: str) -> str:
    """One predicate as the table keys it."""
    return "_".join(str(predicate or "").casefold().replace("-", " ").replace("_", " ").split())


def _keys(predicate: str) -> Iterator[str]:
    """The table keys one predicate is looked up under: as written, then with the third
    person 's' put on or taken off the verb it starts with -- a source writes both
    ``include`` and ``includes``, both ``contribute_to`` and ``contributes_to``."""
    word = _word(predicate)
    yield word
    verb, _, rest = word.partition("_")
    other = verb[:-1] if verb.endswith("s") else verb + "s"
    yield f"{other}_{rest}" if rest else other


def named(predicate: str) -> bool:
    """Whether `RELATIONS` names this predicate at all."""
    return any(key in RELATIONS for key in _keys(predicate))


def vague(predicate: str) -> bool:
    """Whether this predicate says only that the two things were named together."""
    return any(key in VAGUE for key in _keys(predicate))


def verb_for(predicate: str) -> tuple[str, bool] | None:
    """``(verb, whether the ends swap)`` for one predicate, or None when it maps to nothing
    or the table does not name it."""
    for key in _keys(predicate):
        if key in RELATIONS:
            return RELATIONS[key]
    return None


@dataclass
class Imported:
    """One CSV pair as reads, with what the mapping did to it."""

    slug: str
    title: str
    path: str = ""
    rows: list[dict[str, Any]] = field(default_factory=list)
    units: dict[str, Any] = field(default_factory=dict)
    node_rows: int = 0
    edge_rows: int = 0
    concepts: int = 0
    relations: int = 0
    table: list[tuple[str, str, bool, int]] = field(default_factory=list)
    extensions: Counter = field(default_factory=Counter)     # written as they stand
    unnamed: Counter = field(default_factory=Counter)        # and not in the table at all
    vague: Counter = field(default_factory=Counter)          # and `VAGUE` names them
    left: Counter = field(default_factory=Counter)           # core_only left these
    dropped: Counter = field(default_factory=Counter)
    models: list[str] = field(default_factory=list)
    runs: list[str] = field(default_factory=list)
    written_at: str = ""      # when the extractor wrote the pair, earliest row first

    @property
    def core(self) -> int:
        """Relations written under one of the eighteen."""
        return self.relations - sum(self.extensions.values())

    @property
    def specific(self) -> int:
        """Relations written as they stand that say something a graph can be asked."""
        return sum(self.extensions.values()) - sum(self.vague.values())


def _json(text: str, where: str, column: str) -> Any:
    """One JSON column, or a ValueError naming the row and the column."""
    text = (text or "").strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except ValueError as why:
        raise ValueError(f"{where}: {column} is not JSON ({why})") from None


def _rows_of(path: Path, wanted: Sequence[str]) -> Iterator[tuple[str, dict[str, str]]]:
    """Every row of a CSV as ``(where it is, the row)``, refusing one missing a column."""
    csv.field_size_limit(min(sys.maxsize, 2**31 - 1))
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = [name for name in wanted if name not in (reader.fieldnames or ())]
        if missing:
            raise ValueError(f"{path}: no {', '.join(missing)} column(s); "
                             f"it has {', '.join(reader.fieldnames or ()) or 'none'}")
        for row in reader:
            where = f"{path}:{reader.line_num}"
            for name in wanted:
                if not (row.get(name) or "").strip():
                    raise ValueError(f"{where}: {name} is empty")
            yield where, {k: (v or "") for k, v in row.items() if isinstance(k, str)}


def _confident(row: Mapping[str, str], floor: str) -> bool:
    said = (row.get("confidence") or "").strip().casefold()
    rank = CONFIDENCE.index(said) if said in CONFIDENCE else -1
    return rank >= CONFIDENCE.index(floor)


def _provisional(row: Mapping[str, str], where: str) -> bool:
    held = _json(row.get("metadata", ""), where, "metadata")
    if not isinstance(held, Mapping):
        return False
    return bool(held.get("provisional")) or \
        str(held.get("validation_status") or "").casefold() == "provisional"


def _where_of(row: Mapping[str, str], slug: str, where: str) -> Unit:
    """The unit one row was read from, in the shape a read source's units have."""
    context = _json(row.get("source_context", ""), where, "source_context") or {}
    context = context if isinstance(context, Mapping) else {}
    meta = _json(row.get("source_metadata", ""), where, "source_metadata") or {}
    meta = meta if isinstance(meta, Mapping) else {}
    chapter = context.get("chapter_number")
    chapter = "" if chapter in (None, "") else str(chapter)
    section = context.get("section_number")
    section = "" if section in (None, "") else str(section)
    title = str(context.get("section_title") or "").strip()
    if not (chapter or section or title):
        locator = str(row.get("source_locator") or "").strip()
        title = locator or "untitled"
    pages = [int(meta.get("page_start") or 0), int(meta.get("page_end") or 0)]
    return Unit(source=slug, book_title="", chapter=chapter,
                chapter_title=str(context.get("chapter_title") or "").strip(),
                section=section, section_title=title or "untitled",
                first_page=min(pages), last_page=max(pages), text="")


def _titled(uri: str) -> tuple[str, str]:
    """``(slug, title)`` from the file a source was read out of."""
    stem = Path(str(uri or "").strip()).name
    stem = stem[: -len(Path(stem).suffix)] if Path(stem).suffix else stem
    title = " ".join(stem.replace("_", " ").replace("-", " ").split()) or "untitled"
    return _slug(stem), title


def imported(nodes: str | Path, edges: str | Path, *, slug: str = "",
             confidence: str = "medium", provisional: bool = True,
             core_only: bool = False) -> Imported:
    """One nodes/edges CSV pair as a source's reads, and what the predicate table did to it.

    Every row under ``confidence`` is left, and every provisional row when ``provisional``
    is false. A predicate `RELATIONS` maps onto one of the eighteen is written as that verb;
    every other predicate is written as it stands, counted by name in ``extensions``, and
    left only under ``core_only``.
    """
    if confidence not in CONFIDENCE:
        raise ValueError(f"confidence is one of {', '.join(CONFIDENCE)}, not {confidence!r}")
    nodes, edges = Path(nodes).expanduser(), Path(edges).expanduser()
    got = Imported(slug=slug, title="")
    by_id: dict[str, dict[str, Any]] = {}
    units: dict[str, Unit] = {}
    per_unit: dict[str, dict[str, list]] = {}
    models: Counter = Counter()
    runs: Counter = Counter()

    def unit_for(row: Mapping[str, str], where: str) -> Unit:
        unit = _where_of(row, got.slug, where)
        return units.setdefault(unit.id, unit)

    for where, row in _rows_of(nodes, ("node_id", "label", "source_uri")):
        got.node_rows += 1
        if not got.slug:
            got.slug, got.title = _titled(row["source_uri"])
            got.path = row["source_uri"]
        elif not got.title:
            got.title = _titled(row["source_uri"])[1]
            got.path = row["source_uri"]
        if not _confident(row, confidence):
            got.dropped[f"a concept under {confidence} confidence"] += 1
            continue
        if not provisional and _provisional(row, where):
            got.dropped["a provisional concept"] += 1
            continue
        unit = unit_for(row, where)
        aliases = _json(row.get("aliases", ""), where, "aliases")
        label = " ".join(row["label"].split())
        concept = {"name": label,
                   "kind": KINDS.get((row.get("category") or "").strip().casefold(), _KIND),
                   "definition": " ".join((row.get("definition") or "").split())[:400],
                   "aliases": [str(a) for a in aliases or () if str(a).strip()]}
        by_id[row["node_id"]] = concept
        held = per_unit.setdefault(unit.id, {"concepts": [], "relations": []})
        held["concepts"].append(concept)
        got.concepts += 1
        models[row.get("model") or ""] += 1
        runs[row.get("run_id") or ""] += 1
        when = (row.get("created_at") or "").strip()
        if when and (not got.written_at or when < got.written_at):
            got.written_at = when

    counted: Counter = Counter()
    verbs: dict[str, tuple[str, bool]] = {}
    for where, row in _rows_of(edges, ("subject_id", "predicate", "object_id")):
        got.edge_rows += 1
        if not _confident(row, confidence):
            got.dropped[f"a relation under {confidence} confidence"] += 1
            continue
        if not provisional and _provisional(row, where):
            got.dropped["a provisional relation"] += 1
            continue
        source = by_id.get(row["subject_id"])
        target = by_id.get(row["object_id"])
        if source is None or target is None:
            got.dropped["a relation whose ends the nodes file does not hold"] += 1
            continue
        word = _word(row["predicate"])
        found, said = verb_for(word), {}
        if found is None:
            if not named(word):
                got.unnamed[word] += 1
            if core_only:
                got.left[word] += 1
                continue
            got.extensions[word] += 1
            found = (word, False)
            if vague(word):
                got.vague[word] += 1
                said = {"vague": True}
        verb, flipped = found
        unit = unit_for(row, where)
        one, other = (target, source) if flipped else (source, target)
        held = per_unit.setdefault(unit.id, {"concepts": [], "relations": []})
        held["relations"].append({"from": one["name"], "rel": verb, "to": other["name"],
                                  **said})
        # the unit a relation was read from names both its ends
        held["concepts"].extend([source, target])
        got.relations += 1
        counted[word] += 1
        verbs[word] = (verb, flipped)
        models[row.get("model") or ""] += 1
        runs[row.get("run_id") or ""] += 1

    got.table = sorted(((word, verbs[word][0], verbs[word][1], count)
                        for word, count in counted.items()),
                       key=lambda r: (-r[3], r[0]))
    got.models = [m for m, _ in models.most_common() if m]
    got.runs = [r for r, _ in runs.most_common() if r]
    got.units = units
    got.rows = [_a_read(units[unit_id], per_unit.get(unit_id) or {})
                for unit_id in sorted(per_unit)]
    return got


def _a_read(unit: Unit, held: Mapping[str, list]) -> dict[str, Any]:
    """One unit's rows as a read of that unit."""
    concepts: dict[str, dict[str, Any]] = {}
    for concept in held.get("concepts") or ():
        kept = concepts.setdefault(concept["name"], dict(concept))
        for alias in concept.get("aliases") or ():
            if alias not in kept["aliases"]:
                kept["aliases"].append(alias)
        if concept.get("definition") and not kept.get("definition"):
            kept["definition"] = concept["definition"]
    relations = list(held.get("relations") or ())
    row = Read(unit=unit.id, source=unit.source, chapter=unit.chapter,
               section=unit.section,
               title=unit.section_title, pages=[unit.first_page, unit.last_page],
               concepts=len(concepts), relations=len(relations),
               extracted={"concepts": list(concepts.values()), "relations": relations,
                          "figures": [], "key_terms": []})
    return {**asdict(row), "chapter_title": unit.chapter_title}


def _run_of(out: str | Path, got: Imported, *, files: Sequence[str]) -> str:
    """The run node an import hangs its units on: which extractor wrote the pair, and when."""
    from importlib import metadata

    from ml_stack.ingest.judge import write_run

    try:
        version = metadata.version("ml-stack")
    except metadata.PackageNotFoundError:  # pragma: no cover - a checkout without install
        version = "unknown"
    stamp = time.strftime("%Y%m%dT%H%M%S")
    return write_run(out, {
        "id": f"run:import:{got.slug}:{stamp}", "label": f"import {got.slug} {stamp}",
        "model": ", ".join(got.models), "serving": "imported", "imported_from": list(files),
        "source_runs": got.runs[:8], "ml_stack": version, "host": platform.node(),
        "started": got.written_at or time.strftime("%FT%T"),
        "imported": time.strftime("%FT%T"), "argv": list(sys.argv[1:])})


def lines(got: Imported, *, most: int = 12) -> list[str]:
    """What the table did, as a person reads it."""
    core = [row for row in got.table if row[1] in _core()]
    out = [f"{got.slug} ({got.title}): {got.node_rows} node row(s), {got.edge_rows} edge "
           f"row(s) -> {len(got.rows)} unit(s), {got.concepts} concept(s), "
           f"{got.relations} relation(s)",
           f"  onto the verbs this library sets ({len(core)} predicate(s), "
           f"{got.core} relation(s))"]
    for word, verb, flipped, count in core[:most]:
        out.append(f"    {word:<28} -> {verb:<16} {count:>6}"
                   + ("   (ends swapped)" if flipped else ""))
    if len(core) > most:
        out.append(f"    ... and {len(core) - most} more, "
                   f"{sum(c for *_, c in core[most:])} relation(s)")
    kept = sum(got.extensions.values())
    specific = {w: c for w, c in got.extensions.items() if w not in got.vague}
    out.append(f"  written as they stand ({len(specific)} predicate(s), {got.specific} "
               f"relation(s)) -- every edge of one carries `extension`")
    for word, count in Counter(specific).most_common(most):
        out.append(f"    {word:<28} {count:>6}"
                   + ("   (the table does not name it)" if word in got.unnamed else ""))
    if len(specific) > most:
        out.append(f"    ... and {len(specific) - most} more, "
                   f"{got.specific - sum(c for _, c in Counter(specific).most_common(most))} "
                   f"relation(s)")
    said = sum(got.vague.values())
    out.append(f"  vague ({len(got.vague)} predicate(s), {said} relation(s)) -- they say the "
               f"two were named together, not how they stand; their edges carry `vague`")
    for word, count in got.vague.most_common(most):
        out.append(f"    {word:<28} {count:>6}")
    if len(got.vague) > most:
        out.append(f"    ... and {len(got.vague) - most} more, "
                   f"{said - sum(c for _, c in got.vague.most_common(most))} relation(s)")
    if got.left:
        out.append(f"  left by --core-only ({len(got.left)} predicate(s), "
                   f"{sum(got.left.values())} relation(s))")
        for word, count in got.left.most_common(most):
            out.append(f"    {word:<28} {count:>6}")
        if len(got.left) > most:
            out.append(f"    ... and {len(got.left) - most} more")
    for what, count in sorted(got.dropped.items()):
        out.append(f"  not taken: {count} row(s) -- {what}")
    return out


def _core() -> frozenset[str]:
    from ml_stack.ingest.fold import CORE

    return CORE


def bring(out: str | Path, paths: Sequence[str], *, slug: str = "", confidence: str = "medium",
          provisional: bool = True, core_only: bool = False, dry_run: bool = False,
          say: Callable[[str], None] = print) -> int:
    """``ml-stack-ingest import NODES.csv EDGES.csv --out STORE``: a pair another extractor
    wrote, into this store as one source.

    A directory holding ``nodes.csv`` and ``edges.csv`` may be named instead of the two
    files. ``--dry-run`` says what would be written -- the source, its units, the predicates
    onto this library's verbs and the ones written as they stand -- and writes nothing.
    """
    from ml_stack.ingest.fold import fold_into
    from ml_stack.ingest.progress import Progress

    try:
        nodes, edges = _pair(paths)
    except ValueError as why:
        say(f"error: {why}")
        return 2
    try:
        got = imported(nodes, edges, slug=slug, confidence=confidence,
                       provisional=provisional, core_only=core_only)
    except (OSError, ValueError) as why:
        say(f"error: {why}")
        return 2
    if not got.rows:
        say(f"nothing to import from {nodes} and {edges}: no row survived the filters")
        return 1
    say(f"{nodes} + {edges}")
    for line in lines(got):
        say(line)
    if dry_run:
        say(f"  nothing written -- source:{got.slug} would be {len(got.rows)} unit(s) "
            f"into {out}")
        return 0
    progress = Progress(Progress.beside(out))
    held = progress.source(got.slug, title=got.title, path=got.path,
                           sections=len(got.rows))
    when = got.written_at or time.strftime("%FT%T")
    for row in got.rows:
        held["done"][str(row["unit"])] = {
            "seconds": 0.0, "concepts": int(row.get("concepts") or 0),
            "relations": int(row.get("relations") or 0), "figures": 0, "images": 0,
            "error": "", "attempts": 1, "at": when}
    progress.save()
    run = _run_of(out, got, files=[str(nodes), str(edges)])
    rows = [dict(row, run=run) for row in got.rows]
    _keep_reads(out, got.slug, rows)
    result = fold_into(out, got.slug, title=got.title, reads=rows, progress=progress)
    _keep_predicates(out, got)
    landed = result.get("absorbed") or {}
    say(f"  into {out}: {result['nodes']} node(s), {result['edges']} edge(s) in "
        f"{result['seconds']:.1f}s; {result['new_nodes']} node(s) and "
        f"{result['new_edges']} edge(s) the store did not hold"
        + (f"; {landed['same_name'] + landed['plural']} name(s) landed on existing nodes"
           if landed else ""))
    return 0


def _keep_predicates(out: str | Path, got: Imported) -> None:
    """What each of this source's predicates became, in the store beside it."""
    from ml_stack.graph.store import GraphStore

    core = _core()
    with GraphStore(out) as store:
        store.put_doc(f"ingest:predicates:{got.slug}", {
            "source": got.slug,
            "core": {word: {"verb": verb, "flipped": flipped, "relations": count}
                     for word, verb, flipped, count in got.table if verb in core},
            "extensions": dict(got.extensions.most_common()),
            "vague": dict(got.vague.most_common()),
            "unnamed": dict(got.unnamed.most_common()),
            "left": dict(got.left.most_common())})


def _pair(paths: Sequence[str]) -> tuple[Path, Path]:
    """``(nodes.csv, edges.csv)`` from a directory holding both, or from the two named."""
    held = [Path(str(p)).expanduser() for p in paths]
    if len(held) == 1 and held[0].is_dir():
        held = [held[0] / "nodes.csv", held[0] / "edges.csv"]
    if len(held) != 2:
        raise ValueError("import takes a directory holding nodes.csv and edges.csv, "
                         "or the two files: `import NODES.csv EDGES.csv --out STORE`")
    for path in held:
        if not path.is_file():
            raise ValueError(f"no file at {path}")
    return held[0], held[1]
