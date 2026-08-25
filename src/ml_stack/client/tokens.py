"""Token accounting for a prompt assembler."""

from __future__ import annotations

import re
from collections.abc import Callable

_CHARS_PER_TOKEN = 3.3

_PUNCT_RUN = re.compile(r"[^\w\s]{2,}")
_WORD = re.compile(r"\w+")


def _heuristic(text: str) -> int:
    if not text:
        return 0
    base = len(text) / _CHARS_PER_TOKEN
    base += text.count("\n") * 0.5
    base += sum(len(m.group(0)) * 0.4 for m in _PUNCT_RUN.finditer(text))
    base += sum(1 for m in _WORD.finditer(text) if len(m.group(0)) > 12)
    return int(base) + 1


_counter: Callable[[str], int] = _heuristic


def set_token_counter(fn: Callable[[str], int] | None) -> None:
    """Install an exact tokenizer. ``None`` restores the heuristic."""
    global _counter
    _counter = fn if fn is not None else _heuristic


def estimate_tokens(text: str) -> int:
    return _counter(text)


def heuristic_tokens(text: str) -> int:
    """The heuristic regardless of what is installed. Used by tests."""
    return _heuristic(text)
