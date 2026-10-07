"""Compute capabilities of the installed managed job interpreter."""

import json
import subprocess
import threading
import time

from ml_stack.credentials import child_environment

_LOCK = threading.Lock()
_PROBE = """
import json, sys
from importlib.metadata import version
from ml_stack.fleet.measuring import installed_commit
from ml_stack.train.accelerator import report
result = report()
result['compute_runtime'] = {'python': sys.executable, 'version': version('ml-stack'),
                             'commit': installed_commit()}
print(json.dumps(result))
"""


def process_environment():
    """Return a credential-redacted environment for an external installed Python."""
    child = child_environment()
    for name in ("PYTHONHOME", "PYTHONPATH", "DYLD_LIBRARY_PATH"):
        child.pop(name, None)
    if "LD_LIBRARY_PATH_ORIG" in child:
        child["LD_LIBRARY_PATH"] = child.pop("LD_LIBRARY_PATH_ORIG")
    else:
        child.pop("LD_LIBRARY_PATH", None)
    return child


def report(environment):
    """Return cached capabilities verified by the managed job interpreter."""
    if not environment.exists:
        return {}
    if hasattr(environment, "require_current_runtime"):
        try:
            environment.require_current_runtime()
        except (OSError, subprocess.SubprocessError) as exc:
            return {"backends": [], "accelerator": False, "cuda": False, "rocm": False,
                    "compute_runtime": {"python": str(environment.python), "ready": False,
                                        "detail": str(exc)}}
    with _LOCK:
        now = time.monotonic()
        cached = environment._cache.get("compute_report")
        if cached and now - cached[0] < 60:
            return dict(cached[1])
        try:
            done = subprocess.run([str(environment.python), "-I", "-c", _PROBE],
                                  capture_output=True, text=True, timeout=20, env=process_environment())
            result = json.loads(done.stdout) if done.returncode == 0 else {}
            if not isinstance(result, dict) or not isinstance(result.get("compute_runtime"), dict):
                result = {}
        except (OSError, subprocess.SubprocessError, ValueError):
            result = {}
        environment._cache["compute_report"] = (now, result)
        return dict(result)
