"""Launch-secret plumbing for tests that drive a real `Serving` daemon the way the app does."""

from __future__ import annotations

import contextlib
import urllib.parse
from collections.abc import Iterator

from ml_stack.fleet.launch_secret import HEADER, LaunchSecret

ARMED: dict[int, object] = {}
"""Test daemons by port, for the pages that open them."""

_hand_typed = False


def arm(served) -> list[dict]:
    """Give the served UI a launch secret and an audit sink; the sink's rows are returned."""
    rows: list[dict] = []
    served.ui.launch = LaunchSecret(served.files.parent, served.port)
    served.ui.audit = lambda event, **fields: rows.append({"event": event, **fields})
    ARMED[served.port] = served
    return rows


def disarm(served) -> None:
    """Forget a closed test daemon."""
    ARMED.pop(served.port, None)


def ticket(served, secret: str | None = None) -> tuple[int, object, dict]:
    """Ask the daemon for a launch ticket with ``secret`` (the real one when not given)."""
    sent = served.ui.launch.value if secret is None else secret
    return served.call("/ui/launch/ticket", method="POST", headers={HEADER: sent})


def redeem(served, value: str) -> tuple[int, object, dict]:
    """Present a ticket to the sign-in route, as the page does."""
    return served.call("/ui/session", method="POST", body={"ticket": value})


def sign_in(served) -> str:
    """A cookie for a session opened by a freshly issued, spent launch ticket."""
    status, got, _ = ticket(served)
    assert status == 200, got
    status, body, headers = redeem(served, got["ticket"])
    assert status == 200, body
    return headers["Set-Cookie"].split(";", 1)[0]


def browser(served) -> dict[str, str]:
    """The headers the page's own requests carry."""
    return {"Origin": f"http://127.0.0.1:{served.port}", "Sec-Fetch-Site": "same-origin"}


def signed(served):
    """``served`` with every call carrying a launch session's cookie unless the call names its own."""
    cookie = sign_in(served)
    call = served.call

    def with_cookie(path, **options):
        options.setdefault("cookie", cookie)
        return call(path, **options)

    served.call = with_cookie
    served.cookie = cookie
    return served


@contextlib.contextmanager
def hand_typed() -> Iterator[None]:
    """Pages opened inside the block carry no ticket, as when a person types the address."""
    global _hand_typed
    _hand_typed = True
    try:
        yield
    finally:
        _hand_typed = False


def with_ticket(url: str) -> str:
    """``url`` with a fresh launch ticket when it is a page of an armed test daemon."""
    parts = urllib.parse.urlsplit(url)
    served = ARMED.get(parts.port or 0)
    if (_hand_typed or served is None or parts.hostname != "127.0.0.1"
            or not (parts.path == "/ui" or parts.path.startswith("/ui/"))
            or "launch_ticket" in parts.query):
        return url
    status, got, _ = ticket(served)
    if status != 200:
        return url
    query = (parts.query + "&" if parts.query else "") + "launch_ticket=" + got["ticket"]
    return urllib.parse.urlunsplit((parts.scheme, parts.netloc, parts.path, query, parts.fragment))
