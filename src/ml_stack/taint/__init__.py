"""Taint tracking for an agent loop: untrusted content cannot reach the arguments of a privileged
tool unless the person confirms it or the value is one the system can vouch for.

    from ml_stack import taint

    rail = taint.TaintRail()            # an intervention; `ml_stack.guard.default()` includes one
    taint.subscribe(print)              # TaintEvent for each refusal or question

`Ledger` is the record a run keeps (in ``Context.notes``), `Sinks` classifies tools by what they can
do, and `extract` runs a quarantined model that may return only validated values. Standard library
only.
"""

from __future__ import annotations

from ml_stack.taint.cache import LabelCache
from ml_stack.taint.events import TaintEvent, emit, subscribe
from ml_stack.taint.extract import ExtractionError, check_schema, extract, quarantined, validate
from ml_stack.taint.judge import Finding, judge
from ml_stack.taint.labels import Label, Labelled, Level, join
from ml_stack.taint.ledger import SUMMARY_PREFIX, Ledger, ledger_of
from ml_stack.taint.rail import TaintRail
from ml_stack.taint.sinks import (
    HARD,
    Arg,
    Capability,
    Sink,
    Sinks,
    claude_code,
    ml_stack_tools,
    sinks_from_mcp,
)

__all__ = [
    "HARD",
    "SUMMARY_PREFIX",
    "Arg",
    "Capability",
    "ExtractionError",
    "Finding",
    "Label",
    "LabelCache",
    "Labelled",
    "Ledger",
    "Level",
    "Sink",
    "Sinks",
    "TaintEvent",
    "TaintRail",
    "check_schema",
    "claude_code",
    "emit",
    "extract",
    "join",
    "judge",
    "ledger_of",
    "ml_stack_tools",
    "quarantined",
    "sinks_from_mcp",
    "subscribe",
    "validate",
]
