"""Which findings act on their own, by mode."""

from __future__ import annotations

from enum import StrEnum

from ml_stack.sentinel.findings import HIGH, Finding

__all__ = ["NEVER_ACT", "Mode", "decide"]


class Mode(StrEnum):
    OFF = "off"
    OBSERVE = "observe"
    GUARDED = "guarded"
    ENFORCE = "enforce"


NEVER_ACT = frozenset({
    "server.unmanaged", "tools.mix_shift", "peer.flapping", "peer.version_mismatch",
    "peer.binary_mismatch", "guard.tainted", "honey.file_read", "honey.file_changed", "honey.file_gone",
})
"""Event kinds that are reported and watched but never quarantine anything, in any mode."""


def decide(found: Finding, mode: Mode) -> str:
    """``ignore`` (off), ``watch`` (record and watch) or ``act`` (quarantine) for a finding."""
    if mode == Mode.OFF:
        return "ignore"
    if mode == Mode.OBSERVE:
        return "watch"
    if found.event.kind in NEVER_ACT:
        return "watch"
    if found.confidence == HIGH:
        return "act"
    return "act" if mode == Mode.ENFORCE else "watch"
