"""Reading what a model said back: the planning a reply opens with, a `show` call written
out as prose, and how many entries one turn read."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import Any

# Openings that mean the model is planning rather than answering. gpt-oss puts its analysis
# in a channel of its own, but when a turn spends its whole budget deciding what to do the
# reply that comes back IS the analysis — not empty, so nothing caught it, and the reader
# was shown "We need to answer: ... Search for X again maybe missing." as the answer.
WORKING = ("we need to", "i need to", "need to", "need ", "let me", "let's", "lets ", "from data:",
           "the user asks", "the user wants", "first, i", "i should", "we should", "okay,",
           "ok,", "so the question", "search for", "we must", "i'll produce", "we can use",
           "we need", "look up ", "maybe ")


# A sentence end, including the one with no space after it: the leak arrives welded to the
# answer as "…tool search.Grace Hopper brings…", and splitting on ". " alone keeps them
# as one sentence, so cutting the notes cuts the answer with it.
_ENDS = re.compile(r"(?<=[.!?])(?=\s|[A-Z\u201c\u2018\"'])")


def _sentences(said: str) -> list[str]:
    return [x.strip() for x in _ENDS.split((said or "").replace("\n", " ")) if x.strip()]


def _is_note(one: str) -> bool:
    head = " ".join((one or "").strip().split()).casefold()
    return any(head.startswith(x) for x in WORKING)


def without_notes(said: str) -> str:
    """``said`` with the planning it opens with removed.

    Measured against a real model: a good answer arrives with one sentence of the analysis
    channel stuck to the front — "We need to use tool search.Grace Hopper brings decades
    of healthcare experience…". Throwing the whole reply away over that opening loses a real
    answer and shows the reader "the model did not finish an answer", which is worse than
    the leak. So the leak is cut off and the answer kept.
    """
    parts = _sentences(said)
    while parts and _is_note(parts[0]):
        parts.pop(0)
    return " ".join(parts).strip()


# A tool call written out as prose. On the last turn the tools are taken away so the model
# answers in words, and a model that wanted to call `show` writes the call instead, at the end
# of an answer or inside it, with or without brackets: `show({"ids": [...]})`,
# `show {"ids": [...]}`, sometimes inside a `<|tool_call|>` wrapper. The call is cut and its
# ids kept.
_CALL = re.compile(r"\s*(?:<[^>]*>)?\s*\bshow\s*(\()?\s*(?=\{)", re.IGNORECASE)
_CLOSE = re.compile(r"\s*\)")
_WRAP = re.compile(r"\s*<[^>]*>")


def spoken_show(said: str) -> tuple[str, list[str]]:
    """``said`` without any written-out ``show`` call, and the ids those calls named."""
    text, ids, at = said or "", [], 0
    decoder = json.JSONDecoder()
    while found := _CALL.search(text, at):
        try:
            args, end = decoder.raw_decode(text, found.end())
        except ValueError:
            at = found.end()
            continue
        if not isinstance(args, Mapping):
            at = found.end()
            continue
        if found.group(1) and (closed := _CLOSE.match(text, end)):
            end = closed.end()
        if wrapped := _WRAP.match(text, end):
            end = wrapped.end()
        ids += [str(i) for i in (args.get("ids") or ())]
        tail = text[end:].lstrip()
        text = text[:found.start()].rstrip() + (" " if tail[:1].isalnum() else "") + tail
        at = found.start()
    return text.strip(), ids


def is_working(said: str) -> bool:
    """Whether a reply is notes and nothing else.

    Length is not the test — "Nobody in the graph does both." is a short answer, not a note.
    What decides it is whether anything survives taking the planning off the front.
    """
    return bool((said or "").strip()) and not without_notes(said)


def all_at_once(reply: Any) -> bool:
    """Whether that reply read more than one entry in a single reading call.

    The shape `single` exists to stop, and the mirror of `one_by_one`: one `look_at`
    carrying three ids is the fat result the thread gets lost in, however few calls the
    turn made.
    """
    for call in (getattr(reply, "tool_calls", None) or []):
        fn = call.get("function") or {}
        if str(fn.get("name") or "") not in ("look_at", "look_around"):
            continue
        try:
            args = json.loads(fn.get("arguments") or "{}")
        except ValueError:
            continue
        if isinstance(args, Mapping) and len(list(args.get("ids") or ())) > 1:
            return True
    return False


def one_by_one(reply: Any) -> bool:
    """Whether that reply read exactly one entry, in exactly one reading call.

    The shape `batch` exists to stop: one `look_at` with one id, when a `look_up` has
    already handed over several. Two calls in a turn is already the habit being asked for,
    however few ids each carried.
    """
    reading = [call for call in (getattr(reply, "tool_calls", None) or [])
               if str((call.get("function") or {}).get("name") or "")
               in ("look_at", "look_around")]
    if len(reading) != 1:
        return False
    try:
        args = json.loads((reading[0].get("function") or {}).get("arguments") or "{}")
    except ValueError:
        return False
    return isinstance(args, Mapping) and len(list(args.get("ids") or ())) == 1
