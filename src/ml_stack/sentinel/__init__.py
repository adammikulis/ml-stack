"""Detecting attacks on this node and its fleet, and quarantining what they touched.

    from ml_stack import sentinel

    s = sentinel.default()
    s.manifest.pin(model_path, "model", source="hf://org/repo@rev")
    s.verify_before_load(model_path)
    s.screen(tool_result, "tool:web", session="s1")
    s.bus.subscribe(print)

Releasing and purging need a `HumanGrant`, minted by `human.mint` at a terminal.
"""

from __future__ import annotations

import threading

from ml_stack.sentinel.core import ENV, Screened, Sentinel
from ml_stack.sentinel.events import Bus, Event, EventLog, Severity
from ml_stack.sentinel.findings import HEURISTIC, HIGH, Finding
from ml_stack.sentinel.human import HumanGrant, HumanRequired, agent_may, mint
from ml_stack.sentinel.policy import Mode
from ml_stack.sentinel.store import KINDS, Limits, Record, State, Store, sentinel_dir
from ml_stack.sentinel.watch import arm_scan, ensure_honey, opt_out

__all__ = ["ENV", "HEURISTIC", "HIGH", "KINDS", "Bus", "Event", "EventLog", "Finding",
           "HumanGrant", "HumanRequired", "Limits", "Mode", "Record", "Screened", "Sentinel",
           "Severity", "State", "Store", "agent_may", "arm_scan", "armed", "default", "mint", "opt_out"]

def armed() -> Sentinel:
    """`default`, with the decoy files planted under the state root: what a long-running or
    first-run ml-stack process asks for. A sentinel in mode ``off`` plants nothing."""
    node = default()
    if node.mode != Mode.OFF:
        ensure_honey(node)
    return node


_LOCK = threading.Lock()
_DEFAULT: dict[str, Sentinel] = {}


def default() -> Sentinel:
    """The sentinel of the state root in force now, built on first use."""
    with _LOCK:
        where = str(sentinel_dir())
        if where not in _DEFAULT:
            _DEFAULT[where] = Sentinel()
        return _DEFAULT[where]
