"""One device's mesh state: its journals, the peers it syncs with and what each has acknowledged."""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ml_stack import home
from ml_stack.fleet.onboard.signing import SigningKeys
from ml_stack.workspace import coordinator_client
from ml_stack.workspace.journal import Journals, Signing, Table
from ml_stack.workspace.journal_merge import row_id

__all__ = ["Mesh", "device_signer", "paired_fingerprints"]

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
        self.peers, self.acks, self.applied = (Table(directory / f"{name}.json") for name in ("peers", "acks", "applied"))

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
        peers = self.peers.all()
        for fingerprint in self.roster():
            held = acks.get(peers.get(fingerprint, ""), {})
            if int(held.get(self.origin, 0)) < int(seq):
                return "provisional"
        return "synced"

    def seal(self) -> bool:
        """Sign the journal head; False when this device cannot sign."""
        try:
            self.journals.seal()
        except SIGNING_ERRORS:
            return False
        return True

    def note_peer(self, fingerprint: str, origin: str) -> None:
        """Remember which origin a paired device writes."""
        if fingerprint and self.peers.get(fingerprint) != origin:
            self.peers.put(fingerprint, origin)

    def acknowledge(self, peer_origin: str, vector: dict[str, Any]) -> list[str]:
        """Record what a peer holds; returns the origins where it holds less than it did before."""
        before = self.acks.get(peer_origin, {})
        now = {origin: int(entry["seq"]) for origin, entry in vector.items()}
        self.acks.put(peer_origin, {o: max(now.get(o, 0), int(before.get(o, 0))) for o in {*now, *before}})
        return sorted(o for o in before if now.get(o, 0) < int(before[o]))
