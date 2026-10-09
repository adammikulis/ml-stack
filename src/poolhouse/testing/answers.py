"""The answers a fake llama-server process gives, read from a file at every request."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

__all__ = ["answers_in"]


def answers_in(where: Path) -> Callable[[dict[str, Any]], str]:
    """The answer of a fake server process, read from ``answers.json`` in ``where`` at every
    request, so a test changes how the model behaves while it runs: ``{"default": text,
    "contains": {phrase in the last user message: text}}``."""
    def answer(body: dict[str, Any]) -> str:
        try:
            spec = json.loads((where / "answers.json").read_text())
        except (OSError, ValueError):
            return "hello"
        said = [str(m.get("content") or "") for m in body.get("messages") or []
                if m.get("role") == "user"]
        prompt = said[-1] if said else str(body.get("prompt") or "")
        for phrase, text in (spec.get("contains") or {}).items():
            if phrase in prompt:
                return str(text)
        return str(spec.get("default", "hello"))
    return answer
