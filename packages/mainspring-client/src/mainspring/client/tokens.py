"""Token accounting for a prompt assembler.

A prompt assembler needs a token count *before* a request exists, so it cannot ask the
server. The default is a deterministic, deliberately **conservative** heuristic: it
over-counts slightly, so a prompt the assembler believes fits always fits.

If an exact counter is available (``llama-server /tokenize``), install it with
``set_token_counter``. The heuristic stays the fallback and the test baseline -- tests
must not depend on a running server.
"""

from __future__ import annotations

import re
from collections.abc import Callable

# Qwen-family BPE averages ~3.6 chars/token on English prose and noticeably fewer on
# markup, handles and punctuation soup. 3.3 is the conservative divisor: it over-counts
# prose by ~10% and lands close on markup.
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
    """Install an exact tokenizer. ``None`` restores the heuristic.

    Whatever is installed must be pure and deterministic: an assembler's drop decisions
    are part of its reproducibility guarantee.
    """
    global _counter
    _counter = fn if fn is not None else _heuristic


def estimate_tokens(text: str) -> int:
    return _counter(text)


def heuristic_tokens(text: str) -> int:
    """The heuristic regardless of what is installed. Used by tests."""
    return _heuristic(text)
