"""One device's mesh state: its journals, the peers it syncs with and what each has acknowledged."""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ml_stack import home
from ml_stack.fleet.onboard.signing import SigningKeys
from ml_stack.workspace import coordinator_client, plain
from ml_stack.workspace.journal import Journals, Signing, Table
from ml_stack.workspace.journal_merge import row_id

__all__ = ["Mesh", "device_signer", "paired_fingerprints"]

MAX_REJECTED = 200
SIGNING_ERRORS = (RuntimeError, OSError, ValueError)


def device_signer() -> Signing:
    """The signing key of this device."""
    return SigningKeys(home.state("onboard"))


def paired_fingerprints() -> list[str]:
    """The fingerprints of the active devices paired with this one."""
    return [str(row["fingerprint"]) for row in coordinator_client._paired_rows()]


class Mesh:
    """The journals of a workspace and the status of what was posted into them. With no paired
    device every row is synced the moment it is written."""

    def __init__(self, base: Path, clock: Callable[[], float] = time.time,
                 signer: Callable[[], Signing] | None = None,
                 roster: Callable[[], list[str]] | None = None) -> None:
        directory = Path(base) / "mesh"
        self.journals = Journals(directory / "journal", signer or device_signer, clock)
        self.roster = roster or paired_fingerprints
        names = ("peers", "acks", "applied", "labels", "rejected")
        self.peers, self.acks, self.applied, self.labels, self.rejected = (
            Table(directory / f"{name}.json") for name in names)

    @property
    def origin(self) -> str:
        """This device's journal origin."""
        return self.journals.origin

    def record(self, kind: str, actor: str, body: dict[str, Any], idem: str = "") -> dict[str, Any]:
        """Write a row to this device's journal; returns it."""
        return self.journals.append(kind, actor, body, idem)

    @staticmethod
    def jid(row: dict[str, Any]) -> str:
        """The id of a journal row."""
        return row_id(row)

    def status(self, jid: str) -> str:
        """`local` for a post that was never journaled, `provisional` until every paired device
        holds the row, `synced` after."""
        origin, _, seq = jid.rpartition(":")
        if not seq.isdigit() or origin != self.origin:
            return "local"
        acks = self.acks.all()
        if any(int(acks.get(fingerprint, 0)) < int(seq) for fingerprint in self.roster()):
            return "provisional"
        return "synced"

    def seal(self) -> bool:
        """Sign the journal head; False when this device cannot sign."""
        try:
            self.journals.seal()
        except SIGNING_ERRORS:
            return False
        return True

    def bind_peer(self, fingerprint: str, origin: str) -> bool:
        """Tie a paired device to the origin it writes the first time it is seen; False when the
        device already writes a different one."""
        return self.peers.update(lambda entries: entries.setdefault(fingerprint, origin)) == origin

    def owns(self, fingerprint: str, origin: str) -> bool:
        """Whether the paired device ``fingerprint`` is the writer of ``origin``."""
        return bool(fingerprint) and self.peers.get(fingerprint) == origin

    def acknowledge(self, fingerprint: str, vector: object) -> bool:
        """Record how much of this device's journal the paired device holds, when its hash for
        that row is the one held here; True when it holds less than it claimed before."""
        entry = vector.get(self.origin) if isinstance(vector, dict) else None
        rows = self.journals.rows(self.origin)
        seq = entry.get("seq") if isinstance(entry, dict) else 0
        if type(seq) is not int or not 0 < seq <= len(rows) or rows[seq - 1]["hash"] != entry.get("hash"):
            seq = 0
        before = int(self.acks.get(fingerprint, 0))
        if seq > before:
            self.acks.put(fingerprint, seq)
        return seq < before

    def label(self, origin: str) -> str:
        """The local name of a device's origin, `d1`, `d2`, in the order they were first seen."""
        return self.labels.update(lambda entries: entries.setdefault(origin, f"d{len(entries) + 1}"))

    def reject(self, jid: str, reason: str) -> None:
        """Record a foreign row that was not folded, keeping the last few."""
        def keep(entries: dict[str, Any]) -> None:
            entries[jid] = plain.line(reason, 200)
            for old in list(entries)[:-MAX_REJECTED]:
                del entries[old]
        self.rejected.update(keep)
