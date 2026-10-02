"""Reading a model's tool call into the shape the interventions check."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from ml_stack.interventions import Call

__all__ = ["parse_call"]


def parse_call(call: Mapping[str, Any]) -> Call:
    """An OpenAI-style tool call as a :class:`Call`; arguments that are not a JSON object come
    out as ``None`` for the policy to refuse."""
    fn = call.get("function") or {}
    raw = fn.get("arguments") or "{}"
    arguments: Any
    if isinstance(raw, Mapping):
        arguments = dict(raw)
    else:
        try:
            arguments = json.loads(raw)
        except (TypeError, ValueError):
            arguments = None
    if not isinstance(arguments, dict):
        arguments = None
    return Call(str(fn.get("name") or ""), arguments, str(call.get("id") or ""),
                raw if isinstance(raw, str) else json.dumps(raw))
