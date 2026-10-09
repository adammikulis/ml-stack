"""What a route sends back: an answer as the page reads it, one server-sent frame, and a
thread request read out of a path."""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from typing import Any

__all__ = ["PAYLOAD", "answer_payload", "drained", "sse", "thread_request"]

# what an answer carries to the page, in the order the page reads them: `content` is the
# words, `ids` everything the tools touched, then the four ways they touched it, then the
# working as one line (`why`) and one step at a time (`steps`)
PAYLOAD = ("content", "ids", "found", "read", "path", "show", "why", "steps")


def sse(wfile: Any, event: Mapping[str, Any]) -> None:
    """Write one server-sent event frame -- ``data: {json}\\n\\n`` -- and flush it.

    One frame per event, JSON in the data line and nothing else. Flushing is the point: a
    frame held in a buffer until the answer is complete is an answer that did not stream.
    """
    wfile.write(f"data: {json.dumps(dict(event), ensure_ascii=False)}\n\n".encode())
    wfile.flush()


def answer_payload(out: Any, **extra: Any) -> dict[str, Any]:
    """The shape both ask routes return: an ``Answer``, or a mapping already in that shape.

    ``content`` is what to say; ``ids`` what to light; ``found``, ``read``, ``path`` and
    ``show`` how each entry was touched; ``steps`` the working one step at a time, and
    ``why`` the same steps joined. A mapping keeps every key it came with, and only ``why``
    and ``steps`` are filled in from each other when one is missing. ``extra`` rides on top.
    """
    if isinstance(out, Mapping):
        payload = dict(out)
    else:
        payload = {k: getattr(out, k) for k in PAYLOAD if hasattr(out, k)}
        spent = getattr(out, "spent", None)
        if spent is not None and hasattr(spent, "public") and getattr(spent, "calls", 0):
            # which model answered and what it cost: calls, seconds, tokens read, written,
            # cached and drafted -- for the page's footer and for anyone testing an answer.
            # An answer no model was asked for (a greeting, a cached one) carries neither.
            payload["model"] = spent.model
            payload["spent"] = spent.public()
    payload.setdefault("content", "")
    steps = payload.get("steps")
    if steps is not None:
        payload["steps"] = [str(s) for s in steps]
    if payload.get("why") is None and steps is not None:
        payload["why"] = "; ".join(payload["steps"])
    if steps is None and payload.get("why"):
        payload["steps"] = [s for s in str(payload["why"]).split("; ") if s]
    payload.setdefault("steps", [])
    payload.setdefault("why", "")
    for key in ("ids", "found", "read", "path", "show"):
        if key in payload and payload[key] is not None:
            payload[key] = [str(i) for i in payload[key]]
    payload.update(extra)
    return payload


def thread_request(path: str) -> tuple[str, bool] | None:
    """``/thread/<name>[?working=1]`` read out of a request path; None for any other path.

    The name is capped at 64 characters. ``working`` asks for what each turn drew on, which
    is what lets an old answer light the graph again.
    """
    if not path.startswith("/thread/"):
        return None
    rest = path[len("/thread/"):]
    name, _, query = rest.partition("?")
    return name[:64], "working=1" in query


def drained(events: Iterator[Any], emit: Any) -> Any:
    """Relay what an iterator of events yields; the answer is what it returns.

    One that returns nothing is taken at its last ``done`` event, minus the ``event`` key.
    """
    last: Any = None
    while True:
        try:
            event = next(events)
        except StopIteration as stop:
            if stop.value is not None:
                return stop.value
            break
        if isinstance(event, Mapping) and event.get("event") == "done":
            last = {k: v for k, v in event.items() if k != "event"}
        elif emit is not None:
            emit(event)
    return last or {}
