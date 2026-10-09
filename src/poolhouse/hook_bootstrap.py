"""Hook failure output before workspace and authorization imports."""

import hashlib
import json
import os
import sys
import threading
import time
from pathlib import Path

from poolhouse.hook_diagnostics import record
from poolhouse.worktreerules import checkout_metadata

POST_SECONDS = 8.0
STARTED = time.monotonic()
STAGE = "bootstrap"
TIMINGS: dict[str, float] = {}
TIMER = None
METADATA: dict[str, str] = {}


def stage(name: str) -> None:
    """Mark the elapsed start of a hook stage."""
    global STAGE
    STAGE = name
    TIMINGS[name] = round(time.monotonic() - STARTED, 3)


def metadata(payload: dict) -> None:
    """Capture checkout and hashed correlation identifiers without tool input."""
    METADATA.update(checkout_metadata(payload.get("cwd") or str(Path.cwd())))
    for name in ("session_id", "tool_use_id", "call_id"):
        value = payload.get(name)
        if isinstance(value, str) and value:
            METADATA[name] = hashlib.sha256(value.encode()).hexdigest()[:24]


def timings() -> dict:
    """Return elapsed stage and checkout metadata for this invocation."""
    return {"elapsed_seconds": round(time.monotonic() - STARTED, 3),
            "stage_started_seconds": TIMINGS.copy(), "checkout": METADATA.copy()}


def remaining() -> float:
    """Return the remaining notification budget."""
    if TIMER is None:
        return POST_SECONDS
    return max(0.05, POST_SECONDS - (time.monotonic() - STARTED) - 0.5)


def arm(event: str) -> None:
    """Bound notification hooks without changing authorization event budgets."""
    global TIMER
    if event != "post":
        return
    def expired():
        code = failure(TimeoutError(f"post notification exceeded {POST_SECONDS:g}s deadline"), "post", STAGE)
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(code)
    TIMER = threading.Timer(POST_SECONDS, expired)
    TIMER.daemon = True
    TIMER.start()


def finish() -> None:
    """Cancel a completed notification deadline."""
    if TIMER is not None:
        TIMER.cancel()


def failure(error: BaseException, event: str, stage: str, stdout=None) -> int:
    """Emit a diagnostic reference and preserve event-specific failure admission."""
    notification = event == "post"
    outcome = "notification unavailable" if notification else "call blocked"
    message = f"poolhouse hook failed, {outcome}: {record(error, event, stage, metadata=timings())}"
    sys.stderr.write(message + "\n")
    if notification:
        result = {"hookSpecificOutput": {"hookEventName": "PostToolUse", "additionalContext": message}}
    elif event == "stop":
        result = {"decision": "block", "reason": message}
    else:
        result = {"hookSpecificOutput": {"hookEventName": "PreToolUse", "permissionDecision": "deny",
                                         "permissionDecisionReason": message}}
    (stdout or sys.stdout).write(json.dumps(result, sort_keys=True) + "\n")
    return 0 if notification else 2


def block(_kind, value: BaseException, _trace) -> None:
    """Report an unhandled hook exception and exit with its admission result."""
    code = failure(value, next(iter(sys.argv[1:2]), "pre"), "bootstrap")
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)
