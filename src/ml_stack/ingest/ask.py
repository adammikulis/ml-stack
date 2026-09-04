"""A store asked questions: its graph through `graph.ask`, and a set of
questions scored the way the bench scores one."""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

__all__ = ["ask", "asked_f1", "asked_lines", "graph_of", "read_asked", "score_asked",
           "spent_line"]


def graph_of(out: str | Path) -> dict[str, Any]:
    """The store's graph in the shape `graph.ask` takes: ``{"nodes": [...], "edges": [...]}``.

    The hidden nodes -- the run each unit was read by -- and the edges that touch them are
    left out, as `graph.page.shown` leaves them out of a page: they are the record of the
    reading, not the thing being asked about.
    """
    from ml_stack.graph.page import shown
    from ml_stack.graph.store import GraphStore

    with GraphStore(out, read_only=True) as store:
        return shown({"nodes": store.nodes(), "edges": store.edges()})


def spent_line(spent: Any) -> str:
    """What one answer cost, as one line."""
    return (f"Spent: {spent.calls} call(s) in {spent.seconds:.1f}s, "
            f"{spent.prompt_tokens} prompt token(s) ({spent.cached_tokens} cached) and "
            f"{spent.completion_tokens} completion; {spent.tool_calls} tool call(s)"
            + (f"; {spent.model}" if spent.model else ""))


def ask(graph: Mapping[str, Any], question: str, client: Any, *,
        say: Callable[[str], None] = print, **asking: Any) -> Any:
    """One question of a store's graph, through `graph.ask.converse`; the answer, printed.

    ``asking`` goes to `converse` -- ``profile`` above all, so a model is asked the way it
    measured best. Returns the `Answer`.
    """
    from ml_stack.graph.ask import converse

    answer = converse(question, graph, client, **asking)
    say(answer.content or "(no answer)")
    say("  tools: " + (answer.why or "none called"))
    if answer.show or answer.ids:
        say("  about: " + ", ".join(answer.show or answer.ids))
    say("  " + spent_line(answer.spent))
    return answer


def read_asked(path: str | Path) -> list[dict[str, Any]]:
    """A gold set of questions: ``[{"question", "expected": [ids or labels]}]``.

    A list, or ``{"questions": [...]}``. Each entry may carry a ``label`` for the report.
    ``expected`` names nodes by id or by the label the source gave them, because a person
    writing a gold set for a textbook knows "vault current" and not
    ``concept:vault-current``; `score_asked` resolves a label against the graph.
    """
    held = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    asked = held.get("questions") if isinstance(held, Mapping) else held
    if not isinstance(asked, list) or not asked:
        raise ValueError(f"{path}: no questions")
    return [dict(one) for one in asked if isinstance(one, Mapping)]


def _ids_for(graph: Mapping[str, Any], wanted: Iterable[Any]) -> list[str]:
    """Those node ids, each label among them resolved to the id of the node it names."""
    ids = {str(n["id"]) for n in graph.get("nodes") or ()}
    by_label: dict[str, str] = {}
    for node in graph.get("nodes") or ():
        by_label.setdefault(" ".join(str(node.get("label") or "").split()).casefold(),
                            str(node["id"]))
    out = []
    for name in wanted:
        text = str(name)
        out.append(text if text in ids
                   else by_label.get(" ".join(text.split()).casefold(), text))
    return out


def score_asked(graph: Mapping[str, Any], client: Any, asked: Sequence[Mapping[str, Any]], *,
                log: Callable[[str], None] | None = None, **asking: Any) -> list[Any]:
    """Every question through `converse`, scored as the bench scores one: a `Row` each.

    Recall and precision are over the ids the answer selected against the ids the set
    expected, by `graph.bench.score` and not by a second scorer of this command's own -- a
    number measured two ways is two numbers.
    """
    from ml_stack.graph.bench.score import Row

    rows: list[Any] = []
    for index, one in enumerate(asked, start=1):
        question = str(one.get("question") or "")
        began = time.time()
        answer = ask(graph, question, client, say=lambda _: None, **asking)
        row = Row(label=str(one.get("label") or f"q{index}"), question=question,
                  seconds=round(time.time() - began, 2), calls=answer.spent.calls,
                  prompt_tokens=answer.spent.prompt_tokens,
                  completion_tokens=answer.spent.completion_tokens,
                  steps=answer.why, answer_chars=len(answer.content),
                  shown=list(answer.show or answer.ids),
                  expected=_ids_for(graph, one.get("expected") or ()))
        rows.append(row)
        if log:
            log(f"  {row.seconds:5.1f}s  recall {row.recall:.0%}  precision {row.precision:.0%}"
                f"  F1 {row.hit:.0%}  {row.label}")
    return rows


def asked_lines(rows: Sequence[Any], *, most: int = 20) -> list[str]:
    """The score over a set of questions, and what each answer missed or added."""
    scored = [r for r in rows if r.expected]
    if not scored:
        return [f"asked: {len(rows)} question(s), none of them expecting anything"]
    recall = sum(r.recall for r in scored) / len(scored)
    precision = sum(r.precision for r in scored) / len(scored)
    f1 = sum(r.hit for r in scored) / len(scored)
    seconds = sum(r.seconds for r in rows)
    out = [f"asked: {len(scored)} question(s) -- recall {recall:.0%}, "
           f"precision {precision:.0%}, F1 {f1:.0%} ({seconds:.0f}s, "
           f"{sum(r.calls for r in rows)} calls)"]
    for row in scored[:most]:
        missed = [i for i in row.expected if i not in set(row.shown)]
        extra = [i for i in row.shown if i not in set(row.expected)]
        if not (missed or extra):
            continue
        out.append(f"  {row.label}: F1 {row.hit:.0%}"
                   + (f"; missed {', '.join(missed)}" if missed else "")
                   + (f"; also showed {', '.join(extra)}" if extra else ""))
    if len(scored) > most:
        out.append(f"  ... and {len(scored) - most} more")
    return out


def asked_f1(rows: Sequence[Any]) -> float:
    """The mean F1 over the questions that expected something; 0 when none did."""
    scored = [r for r in rows if r.expected]
    return sum(r.hit for r in scored) / len(scored) if scored else 0.0
