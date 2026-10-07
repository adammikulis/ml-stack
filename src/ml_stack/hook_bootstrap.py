"""Hook failure output before workspace and authorization imports."""

import json
import os
import sys

from ml_stack.hook_diagnostics import record


def failure(error: BaseException, event: str, stage: str, stdout=None) -> int:
    """Emit a diagnostic reference and preserve event-specific failure admission."""
    notification = event == "post"
    outcome = "notification unavailable" if notification else "call blocked"
    message = f"ml-stack hook failed, {outcome}: {record(error, event, stage)}"
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
