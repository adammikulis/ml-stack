"""Capture in-process command output and exit status."""

import contextlib
import io
from collections.abc import Callable
from typing import Any


def captured(fn: Callable[[], int]) -> dict[str, Any]:
    """Run a command's ``main`` in-process and hand back what it printed and its exit."""
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try:
            code = int(fn() or 0)
        except SystemExit as left:
            code = int(left.code or 0) if isinstance(left.code, int) else 1
    return {"exit": code, "output": out.getvalue(), "errors": err.getvalue()}
