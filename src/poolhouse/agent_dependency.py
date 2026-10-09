"""Readiness of the local Agents SDK runtime."""
import json
import platform
import sys
from importlib import import_module
from pathlib import Path

from poolhouse.log import say


def report():
    """Return the running immutable artifact's agent readiness and provenance."""
    error = problem()
    marker = Path(__file__).parent / "fleet" / "built-from"
    commit = marker.read_text().strip() if marker.is_file() else ""
    say(json.dumps({"commit": commit, "platform": sys.platform,
                      "machine": platform.machine(), "runtime_ready": not error,
                      "runtime_error": error}))
    return 1 if error else 0


def problem():
    """Return a dependency error when the local chat runtime cannot import."""
    try:
        runtime = import_module("agents")
        for name in ("Agent", "ModelSettings", "OpenAIChatCompletionsModel", "RunConfig", "Runner"):
            getattr(runtime, name)
    except (ImportError, AttributeError) as exc:
        return f"Local chat and Qwen agents require the agent runtime: pip install 'poolhouse[agents]'. {exc}"
    return ""
