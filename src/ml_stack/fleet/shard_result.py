"""What a shard reports: per-file counts, failing node ids with messages, durations and the platform."""

from __future__ import annotations

import os
import platform
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

MOST_FAILURES = 200
MOST_MESSAGE = 2000
MOST_TAIL = 8000
MOST_JUNIT = 32 << 20
MOST_TESTS = 3000
OUTCOMES = ("failure", "error", "skipped")


def stamp() -> dict:
    """Where this result was produced: ``sys.platform``, the Python version, the CPU count and whether it is WSL."""
    wsl = sys.platform == "linux" and "microsoft" in platform.release().lower()
    return {"system": sys.platform, "python": platform.python_version(),
            "machine": platform.machine(), "cpus": os.cpu_count() or 1, "wsl": wsl}


def file_of(classname: str, files: list[str]) -> str:
    """The shard file a junit classname (``tests.test_x.TestY``) belongs to, or ''.

    With no file list (a whole tier) the file is whatever ``tests/NAME.py`` the class names."""
    parts = classname.split(".")
    if not files:
        return f"tests/{parts[1]}.py" if len(parts) > 1 and parts[0] == "tests" else ""
    for end in range(len(parts), 0, -1):
        candidate = "/".join(parts[:end]) + ".py"
        if candidate in files:
            return candidate
    return ""


def outcome(case: ET.Element) -> tuple[str, str]:
    """``passed``/``failed``/``error``/``skipped`` and the message of one testcase."""
    for kind in OUTCOMES:
        found = case.find(kind)
        if found is not None:
            text = (found.get("message") or found.text or "")[:MOST_MESSAGE]
            return {"failure": "failed"}.get(kind, kind), text
    return "passed", ""


def read_junit(path: Path, files: list[str]) -> dict:
    """Per-file counts, durations and failures out of the junit file at ``path``."""
    counts = {name: {"passed": 0, "failed": 0, "error": 0, "skipped": 0, "wall_s": 0.0} for name in files}
    failures: list[dict] = []
    tests: list[list] = []
    if not path.is_file() or path.stat().st_size > MOST_JUNIT:
        return {"files": counts, "failures": failures, "tests": tests}
    try:
        cases = list(ET.parse(path).getroot().iter("testcase"))  # noqa: S314
    except ET.ParseError:
        return {"files": counts, "failures": failures, "tests": tests}
    for case in cases:
        name = file_of(case.get("classname", ""), files)
        state, message = outcome(case)
        seconds = round(float(case.get("time") or 0.0), 4)
        if name:
            counts.setdefault(name, {"passed": 0, "failed": 0, "error": 0, "skipped": 0, "wall_s": 0.0})
            counts[name][state] += 1
            counts[name]["wall_s"] = round(counts[name]["wall_s"] + seconds, 4)
        nodeid = f"{name}::{case.get('classname', '').split('.')[-1]}::{case.get('name', '')}"
        tests.append([nodeid, seconds])
        if state in ("failed", "error") and len(failures) < MOST_FAILURES:
            failures.append({"nodeid": nodeid, "state": state, "message": message})
    tests.sort(key=lambda row: -row[1])
    return {"files": counts, "failures": failures, "tests": tests[:MOST_TESTS]}


def build(shard: dict, junit: Path, exit_code: int, took: tuple[float, float], tail: str) -> dict:
    """The result record of one finished shard."""
    summary = read_junit(junit, shard["files"])
    wall_s, cpu_s = took
    return {"id": shard["id"], "state": "done" if exit_code == 0 else "failed", "exit": exit_code,
            "platform": stamp(), "wall_s": round(wall_s, 2), "cpu_s": round(cpu_s, 2),
            "tree_sha256": shard["tree_sha256"], "output_tail": tail[-MOST_TAIL:], **summary}
