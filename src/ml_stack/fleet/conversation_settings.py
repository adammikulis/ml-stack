"""Validated settings saved with one conversation."""
from __future__ import annotations

import math
from typing import Any

from ml_stack import roles

DEFAULTS: dict[str, Any] = {"mode": "chat", "temperature": None, "project": "", "role": "",
                            "effort": "off", "max_effort": "medium"}
LEVELS = ("off", "low", "medium", "high")


def checked(settings: dict | None = None) -> dict[str, Any]:
    """Return complete conversation settings, rejecting unknown fields and invalid values."""
    if settings is not None and not isinstance(settings, dict):
        raise ValueError("conversation settings must be an object")
    supplied = settings or {}
    if set(supplied) - set(DEFAULTS):
        raise ValueError("unknown conversation setting")
    result = {**DEFAULTS, **supplied}
    if result["mode"] not in ("chat", "coding"):
        raise ValueError("mode is chat or coding")
    for key in ("project", "role", "effort", "max_effort"):
        if not isinstance(result[key], str):
            raise ValueError(f"{key} must be text")
    if result["role"]:
        roles.get(result["role"])
    if result["effort"] not in (*LEVELS, "auto") or result["max_effort"] not in LEVELS:
        raise ValueError("invalid effort level")
    if len(result["project"]) > 4096:
        raise ValueError("project path is too long")
    temperature = result["temperature"]
    if temperature is not None:
        if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
            raise ValueError("temperature must be a number or null")
        if not math.isfinite(temperature) or not 0 <= temperature <= 2:
            raise ValueError("temperature must be between 0 and 2")
    return result
