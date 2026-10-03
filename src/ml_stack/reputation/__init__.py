"""How every source ml-stack deals with has behaved, on two timescales: an encrypted per-user
graph, a score, a notice when a reliable source changes, and the calls that tighten what is
asked of it. ``docs/reputation.md`` has the rules."""

from ml_stack.reputation.model import EVENTS, KINDS, State, canonical

__all__ = ["EVENTS", "KINDS", "State", "canonical"]
