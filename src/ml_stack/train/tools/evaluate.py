"""Held-out tool-call accuracy: what a served model does with the held-out conversations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from ml_stack.agent.schema import parse_arguments
from ml_stack.train.tools.schemas import CHAT, schemas_of

__all__ = ["evaluate", "score"]

METRICS = ("tool_name_accuracy", "required_arg_fill", "valid_json_rate", "arguments_match",
           "end_to_end")


def _required(row: Mapping[str, Any], name: str) -> list[str]:
    for schema in schemas_of(row.get("tools") or []):
        if schema["function"]["name"] == name:
            return list((schema["function"].get("parameters") or {}).get("required") or [])
    return []


def _expected_arguments(row: Mapping[str, Any]) -> dict[str, Any]:
    calls = row["messages"][-1].get("tool_calls") or [{}]
    return dict((calls[0].get("function") or {}).get("arguments") or {})


def _one(row: Mapping[str, Any], call: Mapping[str, Any] | None) -> dict[str, float]:
    """The five measures for one held-out row and the first call the model made, if any."""
    want = str(row["tool"])
    if want == CHAT:
        right = float(call is None)
        return dict.fromkeys(METRICS, right)
    fn = (call or {}).get("function") or {}
    args, why = parse_arguments(fn.get("arguments")) if call else (None, "no call")
    named = float(fn.get("name") == want)
    need = _required(row, want)
    given = args or {}
    fill = sum(k in given for k in need) / len(need) if need else 1.0
    valid = float(args is not None)
    return {"tool_name_accuracy": named, "required_arg_fill": fill if args is not None else 0.0,
            "valid_json_rate": valid,
            "arguments_match": float(named and given == _expected_arguments(row)),
            "end_to_end": float(named and valid and fill == 1.0 and not why)}


def score(rows: Sequence[Mapping[str, Any]], calls: Sequence[Mapping[str, Any] | None]
          ) -> dict[str, Any]:
    """Mean of each measure over the rows, and the same per expected tool.

    ``calls[i]`` is the first tool call the model made for ``rows[i]`` as an OpenAI
    ``{"function": {"name", "arguments"}}`` dict, or None when it made none. A row whose
    expected tool is ``chat`` scores 1 only when the model made no call.
    """
    if len(rows) != len(calls):
        raise ValueError(f"{len(rows)} rows but {len(calls)} answers")
    per = [_one(r, c) for r, c in zip(rows, calls, strict=True)]

    def mean(picked: list[dict[str, float]]) -> dict[str, float]:
        return {m: round(sum(p[m] for p in picked) / len(picked), 4) for m in METRICS}

    by_tool: dict[str, list[dict[str, float]]] = {}
    for row, got in zip(rows, per, strict=True):
        by_tool.setdefault(str(row["tool"]), []).append(got)
    return {"n": len(rows), **(mean(per) if per else dict.fromkeys(METRICS, 0.0)),
            "per_tool": {t: {"n": len(v), **mean(v)} for t, v in sorted(by_tool.items())}}


def evaluate(client: Any, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """`score` over what ``client`` (an `ml_stack.client.Client`) calls for each row's
    conversation up to the assistant turn, offered that row's tools."""
    calls = []
    for row in rows:
        reply = client.chat(list(row["messages"][:-1]), tools=list(row["tools"]))
        calls.append((reply.tool_calls or [None])[0])
    return score(rows, calls)
