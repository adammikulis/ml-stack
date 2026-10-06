"""Readiness of the local Agents SDK runtime."""
from importlib import import_module


def problem():
    """Return a dependency error when the local chat runtime cannot import."""
    try:
        runtime = import_module("agents")
        for name in ("Agent", "ModelSettings", "OpenAIChatCompletionsModel", "RunConfig", "Runner"):
            getattr(runtime, name)
    except (ImportError, AttributeError) as exc:
        return f"Local chat and Qwen agents require the agent runtime: pip install 'ml-stack[agents]'. {exc}"
    return ""
