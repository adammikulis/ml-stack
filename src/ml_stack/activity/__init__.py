"""One per-user, encrypted, tamper-evident log of what the stack did, with a read-only viewer.

    from ml_stack import activity

    activity.record("net.download", subject="host:example.org", outcome="ok", meta={"size": 4096})

Bodies are never stored; `record` never raises. `ml-stack-log` reads the log back.
"""

from __future__ import annotations

from ml_stack.activity.feeds import attach, mirror_requests
from ml_stack.activity.schema import BODY_KEYS, KINDS, Entry
from ml_stack.activity.writer import bind_session, drops, log, record, session

__all__ = ["BODY_KEYS", "KINDS", "Entry", "attach", "bind_session", "drops", "log", "mirror_requests", "record", "session"]
