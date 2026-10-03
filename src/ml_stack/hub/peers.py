"""Paired devices are asked for a model file before the internet is.

The hub's side: what is wanted (`Wanted`), what a session does (`Session`), the on/off switches and
how the fleet's provider is found. The shared data type is `hub.peerbook`. The provider
(``ml_stack.fleet.onboard.peerfirst``) is loaded by module path, because the hub sits below the fleet
(tests/test_layers.py) and a lower layer cannot be handed higher code without an import or an
import-time registration, which the hard rules forbid; but only when a pull is about to start, peers
are on and the book has a row. It owns trust: the digest comes from `Wanted`, never from a peer.

Off for one pull with ``peers=False`` (``--no-peers``), for the shell with ``ML_STACK_NO_PEERS=1``,
and for the machine with ``ml-stack fleet peers off``.
"""

from __future__ import annotations

import importlib
import logging
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from ml_stack.hub.peerbook import PeerBook

__all__ = ["ENV", "PROVIDER", "Session", "Stopped", "Wanted", "enabled", "session"]

logger = logging.getLogger(__name__)

ENV = "ML_STACK_NO_PEERS"
PROVIDER = "ml_stack.fleet.onboard.peerfirst"


class Stopped(Exception):
    """The caller cancelled while peers were being used; what was verified stays on disk."""


@dataclass(frozen=True, slots=True)
class Wanted:
    """One file as the Hub's listing describes it: ``sha256`` is the pin (empty when the
    listing carries none) and ``size`` the byte count the listing gives."""

    repo: str
    path: str
    size: int
    sha256: str = ""

    @property
    def name(self) -> str:
        return self.path.rsplit("/", 1)[-1]


class Session(Protocol):
    """Peers for the length of one pull. A peer that sent bad bytes stays out of it."""

    def fetch(self, wanted: Wanted, final: Path, *, cancelled: Callable[[], bool],
              progress: Callable[[int], None], phase: Callable[[str], None]) -> bool:
        """True when ``final`` now holds the file, checked against ``wanted.sha256`` and put
        through the same scan and hold as an internet download; False when no peer could give
        it (the caller goes to the Hub). `Stopped` when ``cancelled()`` turned true."""
        ...


def enabled(flag: bool | None = None) -> bool:
    """Whether peers are asked: ``flag`` when given, else not when ``ML_STACK_NO_PEERS`` is set."""
    if flag is not None:
        return flag
    return os.environ.get(ENV, "").strip().lower() in ("", "0", "false", "no", "off")


def session(flag: bool | None = None) -> Session | None:
    """The peer session for a pull, or None when peers are off or there is nobody to ask."""
    if not enabled(flag):
        return None
    book = PeerBook()
    if not book.enabled or not book.rows():
        return None
    try:
        return importlib.import_module(PROVIDER).session()  # type: ignore[no-any-return]
    except (ImportError, OSError, ValueError) as exc:
        logger.info("no peers to ask: %s", exc)
        return None
