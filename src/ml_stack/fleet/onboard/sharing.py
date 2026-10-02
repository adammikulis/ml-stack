"""Who may be handed which file: the three levels of `Entry.sharing`, and the licence record.

``open``   any paired peer may have it (programs, ungated models).
``owner``  a licence-restricted or gated model: only a device the owner marked as THEIRS at
           pairing, and only once the owner's acceptance of that licence is on record (who,
           when, which licence, its URL). A device belonging to another person never gets it,
           even inside the same fleet.
``never``  the licence forbids copies: every device downloads it itself from the source.

When the licence status is unknown, the answer is ``owner`` and the owner is asked; it is
``open`` only for a file known not to be gated. Credentials (a Hub token, an API key) are not
files in a manifest and never travel between devices.
"""

from __future__ import annotations

import getpass
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ml_stack.files import read_json, write_json
from ml_stack.platform import private_file

from .human import HumanGrant
from .manifest import SHARING_LEVELS, Entry

__all__ = ["NEVER", "OPEN", "OWNER", "SHARING", "Access", "Licences", "classify", "decide"]

OPEN, OWNER, NEVER = SHARING_LEVELS
SHARING = SHARING_LEVELS


def classify(*, gated: bool | None, forbids_copies: bool | None) -> str:
    """The level for a file from what is known about it; ``None`` means not known."""
    if forbids_copies:
        return NEVER
    if gated is False and forbids_copies is False:
        return OPEN
    return OWNER


@dataclass(frozen=True, slots=True)
class Access:
    """Who is asking, as far as the server can tell: the paired device whose own secret signed
    the request (``device``, its certificate fingerprint) and whether the owner marked it
    theirs. A request signed with only the cluster key has neither."""

    device: str = ""
    mine: bool = False


class Licences:
    """The owner's record of licences accepted, in a private file."""

    def __init__(self, path: Path, clock: Callable[[], float] = time.time) -> None:
        self.path, self.clock = Path(path), clock

    def _rows(self) -> list[dict[str, Any]]:
        doc = read_json(self.path, {})
        rows = doc.get("accepted", []) if isinstance(doc, dict) else []
        return [r for r in rows if isinstance(r, dict)]

    def record(self, grant: HumanGrant, entry: Entry, *, who: str | None = None) -> dict[str, Any]:
        """Write down that the owner accepted ``entry``'s licence, after a person confirmed."""
        grant.check("accept-licence", entry.licence)
        row = {"licence": entry.licence, "url": entry.licence_url, "model": entry.name,
               "by": who or getpass.getuser(), "at": self.clock()}
        write_json(self.path, {"schema_version": 1, "accepted": [*self._rows(), row]})
        private_file(self.path)
        return row

    def accepted(self, entry: Entry) -> dict[str, Any] | None:
        for row in self._rows():
            if row.get("licence") == entry.licence and row.get("url") == entry.licence_url \
                    and row.get("by") and row.get("at"):
                return row
        return None


def decide(entry: Entry, access: Access, licences: Licences | None) -> str:
    """An empty string when ``access`` may have ``entry``, else the reason it may not."""
    if entry.sharing == OPEN:
        return ""
    if entry.sharing == NEVER:
        return ("the licence forbids copies; download it from "
                f"{entry.source or 'its source'} on each device")
    if not access.mine:
        return "this file goes only to the owner's own devices, and this one is not marked theirs"
    if not entry.licence or licences is None or licences.accepted(entry) is None:
        return "the owner's acceptance of this file's licence is not on record"
    return ""
