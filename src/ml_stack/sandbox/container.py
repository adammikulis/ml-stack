"""The container backend: not implemented. docs/sandbox.md holds the plan."""

from __future__ import annotations

from collections.abc import Sequence

from ml_stack.sandbox.backend import Availability, Wrapped
from ml_stack.sandbox.policy import Policy

__all__ = ["Container"]


class Container:
    """A Linux container or lightweight VM that would replace Seatbelt for CPU-only work."""

    name = "container"

    def available(self) -> Availability:
        return Availability(False, "the container backend is planned, not written "
                                   "(docs/sandbox.md, 'Replacement path')")

    def wrap(self, argv: Sequence[str], policy: Policy) -> Wrapped:
        raise NotImplementedError("the container backend is planned, not written; see "
                                  "docs/sandbox.md, 'Replacement path'")

    def denials(self, tag: str, since: float) -> list[dict[str, str]]:
        return []
