"""Every request that needs a person: raised by a component, shown anywhere, answered once.

    handle = requests.raise_request(requests.Ask("tool_call", "run_shell(...)", "why",
                                                 ("allow-once", "deny")))
    outcome = handle.wait(timeout=60)      # outcome.approved is true only for an approving answer

A terminal, the UI and the desktop dialog are three ways to answer the same request: the first
answer wins and a later one is refused as already resolved. An answer is bound to the
fingerprint of the words shown, and only a person answers (`inbox.answer`).
"""

from __future__ import annotations

from poolhouse.requests.inbox import (
    Context,
    Handle,
    Outcome,
    answer,
    default,
    get,
    list_requests,
    oldest_pending,
    pending_count,
    raise_request,
    subscribe,
    summary,
)
from poolhouse.requests.model import CHOICES, KINDS, Choice, Origin, Request
from poolhouse.requests.store import Ask, Inbox, Refused, Unavailable

__all__ = ["CHOICES", "KINDS", "Ask", "Choice", "Context", "Handle", "Inbox", "Origin", "Outcome", "Refused", "Request",
           "Unavailable", "answer", "default", "get", "list_requests", "oldest_pending", "pending_count",
           "raise_request", "subscribe", "summary"]
