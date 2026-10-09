"""Taint tracking for an agent loop: untrusted content cannot reach the arguments of a privileged
tool unless the person confirms it or the value is one the system can vouch for.

    from poolhouse import taint

    rail = taint.TaintRail()            # an intervention; `poolhouse.guard.default()` includes one
    taint.subscribe(print)              # TaintEvent for each refusal or question

`Ledger` is the record a run keeps (in ``Context.notes``), `Sinks` classifies tools by what they can
do, and `extract` runs a quarantined model that may return only validated values. Standard library
only.
"""

from __future__ import annotations

from poolhouse.taint.cache import LabelCache
from poolhouse.taint.events import TaintEvent, emit, subscribe
from poolhouse.taint.extract import ExtractionError, check_schema, extract, quarantined, validate
from poolhouse.taint.judge import Finding, judge
from poolhouse.taint.labels import Label, Labelled, Level, join
from poolhouse.taint.ledger import SUMMARY_PREFIX, Ledger, ledger_of
from poolhouse.taint.rail import TaintRail
from poolhouse.taint.sinks import (
    HARD,
    Arg,
    Capability,
    Sink,
    Sinks,
    claude_code,
    poolhouse_tools,
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
    "poolhouse_tools",
    "quarantined",
    "sinks_from_mcp",
    "subscribe",
    "validate",
]
