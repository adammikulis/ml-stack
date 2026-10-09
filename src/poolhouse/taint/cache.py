"""A cache whose entries keep the label of what they hold."""

from __future__ import annotations

from collections.abc import Callable

from poolhouse.taint.labels import Label, Labelled
from poolhouse.taint.ledger import Ledger

__all__ = ["LabelCache"]


class LabelCache:
    """Text stored by key with the label it was read at. A hit is admitted to the ledger at that
    label, so a cached page stays untrusted however it is served later."""

    def __init__(self, ledger: Ledger) -> None:
        self.ledger = ledger
        self.entries: dict[str, Labelled[str]] = {}

    def put(self, key: str, text: str, label: Label) -> None:
        self.entries[key] = Labelled(text, label)

    def get(self, key: str) -> str | None:
        """The text under ``key``, admitted to the ledger first; None on a miss."""
        got = self.entries.get(key)
        if got is None:
            return None
        self.ledger.admit_labelled(got)
        return got.value

    def through(self, key: str, make: Callable[[], str], label: Label) -> str:
        """The text under ``key``, made by ``make`` and stored at ``label`` on a miss."""
        got = self.get(key)
        if got is not None:
            return got
        text = make()
        self.put(key, text, label)
        self.ledger.admit(text, label)
        return text
