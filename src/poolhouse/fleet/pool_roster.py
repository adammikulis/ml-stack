"""Every cluster this machine is in, each with its record of devices (`membership`).

A daemon asks one `Pool` who may call it, a client asks it whom to trust, and the membership
commands change it. The clusters come from the memberships file (`discovery.memberships`), so
leaving a cluster takes its devices out of the answer at once.
"""

from __future__ import annotations

import socket
from collections.abc import Callable
from pathlib import Path
from typing import Any

from . import tls
from .discovery import Membership, clusters_path, memberships
from .membership import Device, Roster, fingerprint_of
from .tls import Identity

__all__ = ["Pool"]


class Pool:
    """The devices of all the clusters whose keys are in the file at ``cluster_key_path``."""

    def __init__(self, cluster_key_path: Path | str | None = None) -> None:
        self.path = cluster_key_path

    def rosters(self) -> list[tuple[Membership, Roster]]:
        return [(member, Roster(member.key)) for member in memberships(self.path)]

    def roster(self, group: str) -> Roster | None:
        return next((r for m, r in self.rosters() if m.group == group), None)

    def stamp(self) -> tuple[Any, ...]:
        """What changes when a cluster is joined or left or a device is let in or put out."""
        try:
            listed = clusters_path(self.path).stat()
            held = (listed.st_mtime_ns, listed.st_size, listed.st_ino)
        except OSError:
            held = (0, 0, 0)
        return (held, *(r.stamp() for _, r in self.rosters()))

    def groups_of(self, fingerprint: str) -> frozenset[str]:
        """The clusters in which ``fingerprint`` is an active device."""
        return frozenset(m.group for m, r in self.rosters() if r.is_active(fingerprint))

    def is_active(self, fingerprint: str) -> bool:
        return bool(fingerprint) and bool(self.groups_of(fingerprint))

    def certs_pem(self) -> str:
        """The certificates of every active device of every cluster, which a server trusts a client to be."""
        return "".join(r.certs_pem() for _, r in self.rosters())

    def enrol(self, group: str, cert: str, name: str, by: str) -> Device:
        roster = self.roster(group)
        if roster is None:
            raise ValueError(f"this machine is not in a cluster called {group!r}")
        return roster.enrol(cert, name, by)

    def revoke(self, fingerprint: str, by: str, group: str | None = None) -> list[str]:
        """Put a device out of one cluster, or of every cluster this machine is in; the clusters it left."""
        done = []
        for member, roster in self.rosters():
            if group in (None, member.group) and roster.get(fingerprint) is not None:
                roster.revoke(fingerprint, by)
                done.append(member.group)
        return done

    def export(self, group: str) -> list[dict[str, Any]]:
        roster = self.roster(group)
        return roster.export() if roster else []

    def merge(self, group: str, rows: list[Any], by: str) -> int:
        roster = self.roster(group)
        return roster.merge(rows, by) if roster else 0

    def ensure_self(self, identity: Callable[[], Identity], name: str) -> int:
        """List this machine's own certificate (made only if there is a cluster to list it in) in
        every cluster it is in; how many it was missing from."""
        rosters, added = self.rosters(), 0
        for _, roster in rosters:
            cert = identity().beacon
            if roster.get(fingerprint_of(cert)) is None:
                roster.enrol(cert, name, "self")
                added += 1
        return added

    def joined(self, group: str, host_cert: str, host_name: str, *, by: str = "pairing") -> None:
        """Record a join this machine just made: itself, and the machine that took it in, whose
        certificate the join flow authenticated. The rest of the cluster follows by sync."""
        roster = self.roster(group)
        if roster is None:
            raise ValueError(f"this machine is not in a cluster called {group!r}")
        roster.enrol(tls.local().beacon, socket.gethostname()[:40], "self")
        if host_cert:
            roster.enrol(host_cert, host_name, by)

    def alone(self) -> list[str]:
        """The clusters whose record lists no device but this machine's own: the ones a machine
        that was in a cluster before devices had certificates of their own still has to re-pair."""
        return [m.group for m, r in self.rosters() if len(r.devices()) <= 1]
