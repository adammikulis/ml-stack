"""The errors of the public API (docs/api.md): one base class, so a caller catches `poolhouse.Error` and nothing else.

The internal failures (`NodeError`, `ShardError`, `Refusal`, `NodeUnavailable`) derive from these, so a caller of
`import poolhouse as ph` never needs to import an internal module to handle one.
"""

from __future__ import annotations

__all__ = ["Conflict", "Denied", "Error", "NotRunning"]


class Error(Exception):
    """Something the public API could not do; the message says why and what to change."""


class NotRunning(Error):
    """No node answers on this device; start it with `poolhouse` (or `python -m poolhouse.device_setup --yes`)."""


class Denied(Error):
    """The node refused: the session may not do that, or does not hold the grant for it."""


class Conflict(Denied):
    """Another session holds the claim that was asked for."""
