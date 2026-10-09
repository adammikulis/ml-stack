"""Showing a request in the desktop's one dialog and answering it from the button pressed."""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping

from poolhouse import person, requests
from poolhouse.requests.model import Choice, Request

__all__ = ["BUTTONS_MOST", "ask", "buttons", "text"]

logger = logging.getLogger("poolhouse.requests")
BUTTONS_MOST = 3
Choose = Callable[[str, str, tuple[str, ...]], str]


def buttons(request: Request) -> list[Choice]:
    """The choices the dialog can show: at most three, the ones that refuse first (so Escape is
    safe) and the approving one last."""
    refuse = [c for c in request.choices if not c.approving]
    approve = [c for c in request.choices if c.approving]
    room = BUTTONS_MOST - min(len(approve), 1)
    return [*refuse[:room], *approve[:BUTTONS_MOST - min(len(refuse), room)]]


def label_of(choice: Choice) -> str:
    return choice.label.capitalize()


def text(request: Request, shown: list[Choice]) -> tuple[str, str]:
    """The title and body: the request's own words, then what each button does."""
    effects = " ".join(f"{label_of(c)}: {c.effect}" for c in shown)
    return request.subject, f"{request.reason} {effects}".strip()


def ask(request: Request, choose: Choose, *, env: Mapping[str, str] | None = None,
        inbox: requests.Inbox | None = None) -> str:
    """Put ``request`` in the dialog through ``choose`` and answer it with the button pressed;
    returns the label pressed. An answer that is refused (already resolved, changed, an agent's
    process) changes nothing."""
    shown = buttons(request)
    title, body = text(request, shown)
    pressed = choose(title, body, tuple(label_of(c) for c in shown))
    picked = next((c for c in shown if label_of(c) == pressed), None)
    if picked is not None:
        try:
            requests.answer(request.id, picked.id, request.fingerprint, "dialog",
                            requests.Context(env=env, inbox=inbox))
        except (requests.Refused, requests.Unavailable, person.HumanRequired) as exc:
            logger.info("dialog answer not taken: %s", exc)
    return pressed
