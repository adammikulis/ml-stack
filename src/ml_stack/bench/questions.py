"""The questions a run asks, and the short set that still covers every kind.

`read_questions` reads them a line at a time, `filed` groups them by the rarest kind of
answer each asks for, `mix` counts those groups, `sample` draws a short set from them, and
`_how_many` reads off the command line how many to ask.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ml_stack.bench.keep import SHORT, SMOKE


def _how_many(args: Any) -> int:
    """How many questions to ask: --sample wins, then --short, then all of them."""
    asked = int(getattr(args, "sample", 0) or 0)
    if getattr(args, "smoke", False):
        return SMOKE
    return asked or (SHORT if getattr(args, "short", False) else 0)


def filed(questions: Sequence[Mapping[str, Any]],
          graph: Mapping[str, Any] | None = None) -> dict[str, list[dict[str, Any]]]:
    """The questions grouped by the kind of answer each asks for.

    A question is filed under the *rarest* kind it names, so one that asks for an event and
    a person counts towards events -- the kind that has few questions -- rather than towards
    people, who have most of them. That is what stops a kind with three questions from being
    crowded out of a short run by a kind with fifty. `sample` draws from these groups and
    `mix` counts them, so what a short run covers and what the mix reports are one rule.

    ``graph`` says what kind each id is; without one, the invented community's.
    """
    scored = [dict(q) for q in questions]
    if graph is None:
        from ml_stack.graph.community import graph as invented

        graph = invented()
    kind = {str(node.get("id")): str(node.get("kind") or "") for node in
            (graph.get("nodes") or ())}

    def kinds_of(q: Mapping[str, Any]) -> set[str]:
        return {kind.get(str(e), "?") for e in (q.get("expect") or ())} or {"nobody"}

    asked = {k: sum(1 for q in scored if k in kinds_of(q))
             for q in scored for k in kinds_of(q)}
    grouped: dict[str, list[dict[str, Any]]] = {}
    for q in scored:
        grouped.setdefault(min(kinds_of(q), key=asked.__getitem__), []).append(q)
    return grouped


def mix(questions: Sequence[Mapping[str, Any]],
        graph: Mapping[str, Any] | None = None) -> dict[str, int]:
    """How many questions ask for each kind of answer, commonest first.

    The one number that says whether a question set still measures the whole page or has
    drifted into being about people: a set is grown a handful at a time, and the kind each
    addition lands under is not the kind whoever wrote it had in mind. `ml-stack-bench
    prepare --mix` prints it.
    """
    grouped = filed(questions, graph)
    return dict(sorted(((k, len(v)) for k, v in grouped.items()),
                       key=lambda kv: (-kv[1], kv[0])))


def sample(questions: Sequence[Mapping[str, Any]], n: int,
           graph: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    """``n`` questions that still cover every kind of answer, or all of them.

    One of each kind first -- person, org, place, topic, opportunity, event, and the
    questions whose right answer is nobody -- then in proportion, evenly within each kind.
    Deterministic, so two short runs of the same set ask the same questions.
    """
    scored = [dict(q) for q in questions]
    if n <= 0 or n >= len(scored):
        return scored

    # a question is filed under the rarest kind it asks for, so a kind that appears in
    # only one question is never crowded out by one that appears in twenty
    grouped = filed(scored, graph)

    taken: list[dict[str, Any]] = []
    order = sorted(grouped, key=lambda k: len(grouped[k]))
    for group in order:                       # one of every kind, rarest first
        if len(taken) < n:
            taken.append(grouped[group][0])
    for group in order:                       # then in proportion, evenly within each
        rest = grouped[group][1:]
        share = max(0, round((n - len(order)) * len(grouped[group]) / len(scored)))
        step = (len(rest) / share) if share else 0
        for i in range(min(share, len(rest))):
            if len(taken) < n:
                taken.append(rest[int(i * step)])
    for q in scored:                          # and top up in order if rounding left room
        if len(taken) >= n:
            break
        if q not in taken:
            taken.append(q)
    return [q for q in scored if q in taken][:n]


def read_questions(path: str | Path) -> list[dict[str, Any]]:
    """One question per line: a bare string, or ``{"q": ..., "expect": [ids]}``."""
    out = []
    for line in Path(path).expanduser().read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            row = json.loads(line)
        except ValueError:
            row = {"q": line}
        out.append(row if isinstance(row, dict) else {"q": str(row)})
    return out
