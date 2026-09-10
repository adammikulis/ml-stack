"""What both hygiene passes share: what they leave alone, the folding every merge does, and
the report they hand back."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ml_stack.entities.fold import merged_spans

__all__ = ["NEVER_FOLDED", "Report", "SOURCE_LINKS", "hidden", "kept_spans", "label_of",
           "union", "written_from"]

# What is kept off the "no edge but its source" count: the links that say where a node came
# from rather than what it stands in relation to.
SOURCE_LINKS = frozenset({"read_from", "read_by", "illustrates"})

# the kinds a hygiene pass never folds one of into another
NEVER_FOLDED = frozenset({"figure", "run", "source", "unit"})


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
    rejudged: int = 0
    replayed: int = 0      # verdicts the store remembered, applied without asking again
    stale: int = 0         # verdicts naming a node the store no longer holds
    unjudged: int = 0      # verb pairs met with no verdict and no judge to ask
    conflicts: list[tuple[str, str, str, str]] = field(default_factory=list)
    cycles: list[tuple[str, list[str]]] = field(default_factory=list)
    orphans: list[str] = field(default_factory=list)
    self_loops: list[tuple[str, str]] = field(default_factory=list)
    lines: list[str] = field(default_factory=list)
    # what `absorb` did to an incoming graph, and the graph with its ids rewritten
    graph: dict[str, Any] = field(default_factory=dict)
    mapping: dict[str, str] = field(default_factory=dict)   # incoming id -> the store's
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
                f"{len(self.conflicts)} verb conflict(s), {len(self.cycles)} hierarchy "
                f"cycle(s), {len(self.orphans)} orphan(s), "
                f"{len(self.self_loops)} self-loop(s) reported; "
                f"put {self.conflicts_judged} conflicting verb pair(s) to the judge "
                f"({self.conflict_edges_dropped} edge(s) dropped) and "
                f"{self.definitions_judged} definition(s); resolved {self.suspects_resolved} "
                f"suspect label(s) ({self.suspects_dropped} node(s) dropped)"
                + (f"; replayed {self.replayed} verdict(s) the store remembered"
                   if self.replayed else "")
                + (f"; {self.stale} remembered verdict(s) name a node the store no longer has"
                   if self.stale else "")
                + (f"; {self.unjudged} verb pair(s) no verdict covers"
                   if self.unjudged else "")
                + (f"; asked the judge again about {self.rejudged} remembered verdict(s)"
                   if self.rejudged else ""))

    def absorbed(self) -> str:
        """What `absorb` did to an incoming graph."""
        return (f"{len(self.graph.get('nodes') or ())} incoming node(s): "
                f"{self.mapped_same_name} onto the same name, {self.mapped_plural} onto a "
                f"singular, {self.judged_same} judged the same, {self.judged_different} judged "
                f"different, {self.left_possible} close spelling(s) left new")


def written_from(path: str | Path | None) -> dict[str, str]:
    """The duplicates a person settled, from a JSON file ``{name: the name it is}``; an
    empty map when no file is named. A file that is not that shape is an error said plainly."""
    if not path:
        return {}
    data = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not all(isinstance(v, str) for v in data.values()):
        raise ValueError(f"{path}: expected a JSON object of name -> name")
    return {str(k): v for k, v in data.items()}


def kept_spans(into: dict[str, Any], *things: Any) -> None:
    """Union those spans onto ``into``, in place, leaving it alone when there are none."""
    found = merged_spans(*things)
    if found:
        into["spans"] = found



def union(*lists: Any) -> list[str]:
    out: list[str] = []
    for one in lists:
        for item in one or ():
            if item not in out:
                out.append(item)
    return out


def hidden(node: Mapping[str, Any]) -> bool:
    return bool((node.get("attrs") or {}).get("hidden"))



def label_of(nodes: Mapping[str, Mapping[str, Any]], node_id: str) -> str:
    return str((nodes.get(node_id) or {}).get("label") or node_id)
