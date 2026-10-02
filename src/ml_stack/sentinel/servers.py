"""Servers and processes: listeners ml-stack did not start, and servers running a binary other
than the one they were started from. The process table is read by the caller and passed in."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

from ml_stack.sentinel.events import Severity
from ml_stack.sentinel.findings import HEURISTIC, HIGH, Finding, finding

__all__ = ["exe_mismatches", "unmanaged_findings"]


def unmanaged_findings(found: Iterable[Mapping[str, Any]]) -> list[Finding]:
    """One notice per server process whose port the lease registry has no record of. These
    are reported and never acted on; sentinel does not stop a process it did not start."""
    return [finding("server.unmanaged", Severity.NOTICE, ("server", f"port:{one['port']}"), HEURISTIC,
                    {"port": int(one["port"]), "pid": int(one.get("pid") or 0),
                     "exe": Path(str(one.get("exe") or "")).name})
            for one in found]


def exe_mismatches(records: Mapping[Any, Mapping[str, Any]],
                   exe_of: Callable[[int], str]) -> list[Finding]:
    """A finding for each recorded server whose pid now runs an executable other than the one
    in its record (``exe``), or whose pinned binary digest differs from the record's."""
    out = []
    for port, record in records.items():
        want, pid = str(record.get("exe") or ""), int(record.get("pid") or 0)
        if not want or not pid:
            continue
        have = exe_of(pid)
        if have and Path(have).resolve() != Path(want).resolve():
            out.append(finding("server.exe_changed", Severity.CRITICAL, ("server", f"port:{port}"), HIGH,
                               {"pid": pid, "recorded": Path(want).name,
                                "running": Path(have).name}))
    return out
