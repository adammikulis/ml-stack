"""Windows-owned login startup for the WSL daemon."""

import json
import os
import subprocess
import sys
from collections.abc import Callable
from typing import Any


def guest() -> bool:
    return sys.platform.startswith("linux") and bool(os.environ.get("WSL_DISTRO_NAME"))


def call(action: str, **values: Any) -> dict[str, Any]:
    executable = os.environ.get("ML_STACK_WINDOWS_PYTHON", "")
    if not executable:
        return {"installed": False, "mode": "manual", "paths": [],
                "note": "Open ml-stack from Windows to configure login startup."}
    try:
        done = subprocess.run([executable, "-m", "ml_stack.fleet.autostart", "windows-owner"],
                              input=json.dumps({"action": action, **values}), capture_output=True,
                              text=True, timeout=60, check=True)
        return json.loads(done.stdout)
    except (OSError, subprocess.SubprocessError, ValueError):
        return {"installed": False, "mode": "manual", "paths": [],
                "note": "Windows login startup could not be configured."}


def validate(request: Any) -> dict[str, Any]:
    if not isinstance(request, dict) or request.get("action") not in {"status", "configure"}:
        raise ValueError("invalid startup action")
    if request["action"] == "status":
        return request
    if request.get("mode") not in {"login", "manual"}:
        raise ValueError("Windows-owned WSL startup supports login or manual")
    if type(request.get("slots")) is not int or not 1 <= request["slots"] <= 64:
        raise ValueError("invalid slots")
    labels, report = request.get("labels"), request.get("report")
    if not isinstance(labels, list) or len(labels) > 32 or not isinstance(report, str):
        raise ValueError("invalid startup arguments")
    for value in [*labels, report, sys.executable]:
        if not isinstance(value, str) or len(value) > 1024 or any(c in value for c in '"&|<>^%!\r\n'):
            raise ValueError("unsafe Windows startup argument")
    return request


def answer(status: Callable, configure: Callable) -> int:
    if sys.platform != "win32":
        return 2
    try:
        raw = sys.stdin.read(8193)
        if len(raw) > 8192:
            raise ValueError("startup request too large")
        request = validate(json.loads(raw))
        result = status() if request["action"] == "status" else configure(request)
        print(json.dumps(result))
        return 0
    except (ValueError, OSError, subprocess.SubprocessError):
        return 2
