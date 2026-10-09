"""A workspace sender's standing in the reputation ledger."""

from __future__ import annotations

from poolhouse.sentinel import observers
from poolhouse.workspace.identity import AGENT, HUMAN, LEAD, Identity

__all__ = ["KIND", "ledger_installed", "record_injection", "sender_standing"]

KIND = "peer"
"""The ledger kind a sender is recorded under."""
LIVE_ROLES = (AGENT, HUMAN, LEAD)


def _key(identity: Identity) -> str:
    return f"workspace:{identity.id}"


def ledger_installed() -> bool:
    """Whether a reputation ledger is installed in this process."""
    return observers.installed() is not None


def sender_standing(identity: Identity | None) -> str:
    """``good``, ``watch`` or ``bad`` for a live token holder (``good`` when the ledger holds
    nothing against it or is not installed), ``unknown`` for anyone else. Never raises."""
    if identity is None or not identity.id or identity.role not in LIVE_ROLES:
        return "unknown"
    gate = observers.gate(KIND, _key(identity))
    if gate is None:
        return "good"
    return gate.state if gate.state in ("watch", "bad") else "good"


def record_injection(identity: Identity) -> None:
    """Note one held hard-marker message against ``identity``. Never raises."""
    observers.observe(KIND, _key(identity), "injection_flagged")
