"""What a model asks to be sampled with: its card's prose, and its GGUF's own metadata."""

from __future__ import annotations

import re
from pathlib import Path

from ml_stack.hub.modelfile import Limits, NotAModelFile, scan_gguf
from ml_stack.hub.remote import text_file

# What a card calls a sampler setting, and what llama.cpp calls it. A card writes prose, so
# `temperature=1.0`, `temperature: 1.0`, `"temperature": 1.0` and `--temp 1.0` all appear.
_KNOBS = {"temperature": "temperature", "temp": "temperature", "top_p": "top_p",
          "top-p": "top_p", "top_k": "top_k", "top-k": "top_k", "min_p": "min_p",
          "min-p": "min_p", "repeat_penalty": "repeat_penalty",
          "repetition_penalty": "repeat_penalty"}
_NAMES = "|".join(sorted(_KNOBS, key=len, reverse=True))
# `temperature=1.0`, `temperature: 1.0`, `"temperature": 1.0`, and `--temp 1.0`. The last
# form has no separator, so a card that only shows a llama.cpp command line still reads.
_SETTING = re.compile(
    r"(?:--)?[`\"\'*]*\b(" + _NAMES + r")\b[`\"\'*]*\s*(?:[=:]\s*|(?<=--\w)\s+)"
    r"[`\"\'*]*([0-9]*\.?[0-9]+)", re.IGNORECASE)
_FLAG = re.compile(r"--(" + _NAMES + r")\s+([0-9]*\.?[0-9]+)", re.IGNORECASE)

# Where a card puts what it recommends. Searched first, so a document that opens by warning
# against a setting is not read as recommending it -- "first mention wins" is only safe
# inside the section that is doing the recommending.
_ADVISING = re.compile(
    r"^[ \t]*(#{1,6})[ \t]*[^\n]*\b"
    r"(?:sampling|best practice|recommend|inference|usage|parameters)\b",
    re.IGNORECASE | re.MULTILINE)


def in_gguf(path: str | Path) -> dict[str, float]:
    """The sampler settings written into a GGUF's own metadata, or {}.

    Better than the card by every measure: it is in the file being served rather than in
    prose beside it, it cannot drift from the weights, and it needs no parsing. Qwen3.8
    carries `general.sampling.temp`, `.top_k` and `.top_p`; many models carry none, which is
    why the card is still read when this is empty.
    """
    want = {"temp": "temperature", "temperature": "temperature", "top_k": "top_k",
            "top_p": "top_p", "min_p": "min_p"}
    try:
        got = scan_gguf(path, want=lambda key: key.startswith("general.sampling."), limits=Limits(keep_array=0))
    except (OSError, NotAModelFile):
        return {}
    return {want[name.rsplit(".", 1)[-1]]: float(value) for name, value in got.values.items()
            if name.rsplit(".", 1)[-1] in want and isinstance(value, (int, float))}


def card(repo: str) -> str:
    """The repository's README, which is where a publisher writes down what it wants."""
    return text_file(repo, "README.md")


def advice(text: str) -> dict[str, float]:
    """The sampler settings a card names, as llama.cpp spells them; {} when it names none.

    A card's recommending section is read first when it has one -- a heading naming
    sampling, best practices, recommendations, inference or usage -- and within it the
    first mention of a setting wins. An empty answer means the card was silent, which is
    the difference between a publisher choosing a default and nobody having chosen one.
    """
    body = text or ""
    for where in (_advising(body), body):
        found: dict[str, float] = {}
        for pattern in (_SETTING, _FLAG):
            for match in pattern.finditer(where):
                name = _KNOBS[match.group(1).lower()]
                found.setdefault(name, float(match.group(2)))
        if found:
            return found
    return {}


def _advising(text: str) -> str:
    """The part of a card that is making a recommendation, or '' when none is marked."""
    first = _ADVISING.search(text)
    if first is None:
        return ""
    after = text[first.start():]
    # Search past this section's own heading line, not past one character of it: a heading
    # may be indented, and skipping a single space leaves the hashes to match themselves.
    line_end = after.find("\n")
    if line_end == -1:
        return after
    # to the next heading of the same depth or shallower, so a section keeps its subsections
    depth = len(first.group(1))
    nxt = re.search(rf"^[ \t]*#{{1,{depth}}}[ \t]", after[line_end:], re.MULTILINE)
    return after[:line_end + nxt.start()] if nxt else after
