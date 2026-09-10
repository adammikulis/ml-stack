"""Training rows out of what a model actually did: the traced questions a bench run kept,
one example per model turn."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from typing import Any

from ml_stack.train.tools.dataset import side_of
from ml_stack.train.tools.schemas import CHAT, _fn, schemas_of


def _scored(row: Mapping[str, Any]) -> float:
    """F1 for one kept bench row, as the bench scores it."""
    from ml_stack.bench.score import _hit

    return float(_hit(row))


def _good_rows(kept: Sequence[Mapping[str, Any]], model: str,
               min_f1: float) -> Iterator[tuple[Mapping[str, Any], Mapping[str, Any]]]:
    """``(run, row)`` for every scored question in a matching run that scored well enough.

    ``model`` is a substring of the run's label or of the model file the server was
    holding, because a run is named for how it was asked (``e4b-shortlist``) and the file
    is what was actually loaded; either identifies it. A question with nothing expected is
    left out -- it was never scored, so "above the threshold" means nothing about it.
    """
    for one in kept:
        server = one.get("server") or {}
        named = f"{one.get('label', '')} {server.get('model', '')}"
        if model and model.lower() not in named.lower():
            continue
        for row in one.get("rows") or ():
            if not row.get("expected") or row.get("timed_out"):
                continue
            if _scored(row) + 1e-9 < min_f1:
                continue
            yield one, row


def traced_rows(kept: Sequence[Mapping[str, Any]], *, model: str = "",
                min_f1: float = 0.8) -> list[dict[str, Any]]:
    """Every scored question in these runs that kept its transcript and scored well enough."""
    return [{**row, "run": one.get("key", ""), "run_label": one.get("label", "")}
            for one, row in _good_rows(kept, model, min_f1) if row.get("trace")]


def would_yield(kept: Sequence[Mapping[str, Any]], *, model: str = "",
                min_f1: float = 0.8) -> dict[str, int]:
    """What these runs *would* have yielded had they been traced: questions, and turns.

    One training example per model turn, and a question's turns are its calls, so the count
    a run of untraced rows would have given is the sum of their ``calls``. The first answer
    this command gives on a store filled before tracing existed is zero, and zero is worth
    nothing without the number beside it.
    """
    questions = turns = traced = 0
    for _one, row in _good_rows(kept, model, min_f1):
        questions += 1
        turns += int(row.get("calls") or 0)
        traced += 1 if row.get("trace") else 0
    return {"questions": questions, "turns": turns, "traced": traced}


def _as_message(entry: Mapping[str, Any]) -> dict[str, Any] | None:
    """One trace entry as the chat message a recipe renders, or None for what is not one."""
    role = str(entry.get("role") or "")
    if role in ("system", "user"):
        return {"role": role, "content": str(entry.get("content") or "")}
    if role == "tool":
        return {"role": "tool", "name": str(entry.get("name") or ""),
                "content": str(entry.get("content") or "")}
    return None


def _calls_made(calls: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """A turn's calls in the ``tool_calls`` shape a chat template renders."""
    return [{"id": f"call_{i}", "type": "function",
             "function": {"name": str(c.get("name") or ""),
                          "arguments": dict(c.get("args") or {})}}
            for i, c in enumerate(calls)]


def _target(entry: Mapping[str, Any]) -> tuple[dict[str, Any], str] | None:
    """The assistant turn a trace entry teaches, and the tool it calls; None for no lesson.

    A turn that called nothing and wrote nothing teaches nothing, and a turn the ceiling
    cut off (``finish`` of ``length``) teaches a truncated call -- the one thing a tool
    caller must never learn. Both are dropped rather than trained on.
    """
    calls = list(entry.get("tool_calls") or ())
    text = str(entry.get("content") or "")
    if str(entry.get("finish") or "") == "length" or entry.get("cut"):
        return None
    if calls:
        return ({"role": "assistant", "content": None,
                 "tool_calls": _calls_made(calls)}, str(calls[0].get("name") or ""))
    if text.strip():
        return {"role": "assistant", "content": text}, CHAT
    return None


def examples_from(row: Mapping[str, Any], *, system: str = "") -> list[dict[str, Any]]:
    """One traced question as one training example per model turn.

    The conversation up to a turn is the input -- the system prompt, the question, and
    every tool result the model had already been given -- and the turn itself is the
    target. A question of four calls is four examples, each one a decision made with
    strictly more evidence than the last.

    The tools each example carries are the ones that were actually offered on that call:
    `graph.conversation` takes tools away as a question goes on, and an example that offers
    a tool the model was not offered teaches it to reach for what will not be there.
    """
    schemas: list[dict[str, Any]] = []
    conversation: list[dict[str, Any]] = []
    out: list[dict[str, Any]] = []
    question = str(row.get("question") or "")
    for entry in row.get("trace") or ():
        if str(entry.get("role") or "") == "tools":
            schemas = schemas_of(entry.get("tools") or [])
            continue
        message = _as_message(entry)
        if message is not None:
            conversation.append(message)
            continue
        if str(entry.get("role") or "") != "assistant":
            continue
        offered = [str(n) for n in (entry.get("offered") or ())]
        taught = _target(entry)
        if taught is not None:
            assistant, tool = taught
            usable = [s for s in schemas if not offered or _fn(s)["name"] in offered]
            out.append({"messages": [*conversation, assistant], "tools": usable,
                        "tool": tool, "from": question, "call": int(entry.get("call") or 0),
                        "run": str(row.get("run") or ""),
                        "model": str(entry.get("model") or ""),
                        "split": side_of(question)})
        # whether or not it was taught, the model said it, and the next turn saw it
        calls = list(entry.get("tool_calls") or ())
        conversation.append({"role": "assistant",
                             "content": str(entry.get("content") or "") or None,
                             **({"tool_calls": _calls_made(calls)} if calls else {})})
    if system:
        for one in out:
            if not one["messages"] or one["messages"][0].get("role") != "system":
                one["messages"] = [{"role": "system", "content": system}, *one["messages"]]
    return out


def from_bench(kept: Sequence[Mapping[str, Any]], *, model: str = "", min_f1: float = 0.8,
               system: str = "") -> list[dict[str, Any]]:
    """Training rows from kept bench runs: every good traced question, turn by turn.

    The rows are the shape `synthesise` writes and the ``tool-calls`` recipe reads, so the
    two sources mix in one directory: synthetic conversations teach the shape of a call,
    and these teach the calls that actually scored, on a real graph, with real ids.
    """
    return [example for row in traced_rows(kept, model=model, min_f1=min_f1)
            for example in examples_from(row, system=system)]
