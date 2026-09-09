"""Where an extraction's own words sit in the unit it was read from: character spans."""

from __future__ import annotations

import re
from collections.abc import Mapping
from difflib import SequenceMatcher
from typing import Any

__all__ = ["LEAST", "locate", "sentence_span", "spans_for"]

LEAST = 0.9
"""How much of a quote must match for a fuzzy hit to count."""

_WORD = re.compile("\\w+(?:['\u2019-]\\w+)*")
_SENTENCE = re.compile(r"[^.!?\n]+[.!?]*", re.MULTILINE)
_FOLD = str.maketrans({"\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
                       "\u2013": "-", "\u2014": "-", "\u2012": "-", "\u2212": "-"})


def _folded(text: str) -> str:
    return text.translate(_FOLD).casefold()


def _words(text: str) -> list[tuple[str, int, int]]:
    """Each word of ``text`` folded, with the offsets it sits at."""
    return [(_folded(m.group()), m.start(), m.end()) for m in _WORD.finditer(text)]


def _window(text_words: list[tuple[str, int, int]], quote_words: list[str],
            least: float) -> tuple[int, int] | None:
    """The best run of ``text_words`` matching ``quote_words``, when it matches enough."""
    held = [w for w, _s, _e in text_words]
    matcher = SequenceMatcher(None, held, quote_words, autojunk=False)
    anchor = matcher.find_longest_match(0, len(held), 0, len(quote_words))
    if not anchor.size:
        return None
    reach = max(1, len(quote_words) // 10)
    best: tuple[float, int, int] = (0.0, 0, 0)
    for slack in (0, reach):
        end = min(len(held), max(0, anchor.a - anchor.b) + len(quote_words) + slack)
        start = max(0, end - len(quote_words) - 2 * slack)
        score = SequenceMatcher(None, held[start:end], quote_words, autojunk=False).ratio()
        if score > best[0]:
            best = (score, start, end)
    if best[0] < least or best[1] >= best[2]:
        return None
    return (text_words[best[1]][1], text_words[best[2] - 1][2])


def locate(text: str, quote: str, *, least: float = LEAST) -> tuple[int, int] | None:
    """Where ``quote`` sits in ``text`` as ``(start, end)``, or None when it is not there."""
    if not text or not quote:
        return None
    at = text.find(quote)
    if at >= 0:
        return (at, at + len(quote))
    quote_words = [w for w, _s, _e in _words(quote)]
    text_words = _words(text)
    if not quote_words or not text_words:
        return None
    return _window(text_words, quote_words, least)


def sentence_span(text: str, *names: str) -> tuple[int, int] | None:
    """The first sentence of ``text`` carrying every one of those names, as ``(start, end)``."""
    wanted = [_folded(n) for n in names if str(n).strip()]
    if not text or not wanted:
        return None
    for match in _SENTENCE.finditer(text):
        said = _folded(match.group())
        if all(name in said for name in wanted):
            start = match.start() + (len(match.group()) - len(match.group().lstrip()))
            return (start, match.end())
    return None


def spans_for(extraction: Mapping[str, Any], unit: Any, *, text: str = ""
               ) -> dict[str, tuple[int, int] | None]:
    """``{name: span}`` for every defined concept of an extraction; None where the
    definition is not in the unit's text, and empty when the unit has no text."""
    text = str(text or getattr(unit, "text", "") or "")
    out: dict[str, tuple[int, int] | None] = {}
    if not text:
        return out
    said = [(c.get("name"), c.get("definition")) for c in extraction.get("concepts") or ()
            if isinstance(c, Mapping)]
    said += [(t.get("term"), t.get("definition")) for t in extraction.get("key_terms") or ()
             if isinstance(t, Mapping)]
    for name, definition in said:
        clean = " ".join(str(name or "").split())
        words = " ".join(str(definition or "").split())
        if not (clean and words) or clean in out:
            continue
        out[clean] = locate(text, words)
    return out
