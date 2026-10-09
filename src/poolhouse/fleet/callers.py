"""Who is on the other end of a connection to the daemon, and whether that is enough.

A request that arrives over TLS from another machine carries the certificate its sender showed
in the handshake. It is answered only when that certificate is a member's (`pool_roster`) and,
when the request is signed with a cluster's secret, a member of that same cluster. A sender that
showed no certificate (a phone, a browser) may use only the credential of its own device.
The check runs on every request, so a device put out of the cluster is refused at its next one
even on a connection it opened before.
"""

from __future__ import annotations

import hashlib
import ssl
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from poolhouse.macauth import Authenticator, Verdict

from .discovery import derive_token, memberships
from .pool_roster import Pool

__all__ = ["ANONYMOUS", "LOCAL", "MEMBER", "STRANGER", "Screen", "Screened", "cluster_of", "identify"]

LOCAL, ANONYMOUS, MEMBER, STRANGER = "local", "anonymous", "member", "stranger"


def identify(connection: Any, pool: Pool) -> tuple[str, str]:
    """``(kind, fingerprint)``: ``local`` for a plain connection from this machine; over TLS,
    ``anonymous`` for a caller with no certificate, ``member`` for one showing a current member's,
    ``stranger`` for any other (a device put out since its handshake)."""
    if not isinstance(connection, ssl.SSLSocket):
        return LOCAL, ""
    der = connection.getpeercert(binary_form=True)
    if not der:
        return ANONYMOUS, ""
    fingerprint = hashlib.sha256(der).hexdigest()
    return (MEMBER if pool.is_active(fingerprint) else STRANGER), fingerprint


@dataclass(frozen=True, slots=True)
class Screened:
    """What the daemon made of a request: who sent it, the verdict on its signature, and the
    status and reason to answer with when it may not be served."""

    kind: str
    peer: str
    verdict: Verdict | None = None
    refusal: tuple[int, str] | None = None


class Screen:
    """The checks a request must pass, which differ by who sent it and by route."""

    def __init__(self, secrets_now: Callable[[], Iterable[str]], device_secrets: Callable[[], Iterable[str]],
                 pool: Pool, cluster_key_path: Path | str | None, watch: Callable[[Any], Any]) -> None:
        self.pool, self.cluster_key_path = pool, cluster_key_path
        self.cluster = watch(Authenticator(secrets_now))
        self.workspace = Authenticator(lambda: [*secrets_now(), *device_secrets()])
        self.enrolled = Authenticator(device_secrets)

    def check(self, handler: Any, body: bytes | None) -> Screened:
        """Screen the request ``handler`` (a request handler) is serving, whose body was ``body``."""
        kind, peer = identify(handler.connection, self.pool)
        workspace = handler.path.split("?")[0].startswith("/workspace/v1/")
        if kind == STRANGER:
            return Screened(kind, peer, refusal=(403, "this device is not a member of the cluster"))
        if kind == ANONYMOUS and not workspace:
            return Screened(kind, peer, refusal=(401, "a request from another machine presents its device certificate"))
        checking = self.enrolled if kind == ANONYMOUS else self.workspace if workspace else self.cluster
        verdict = checking.check(handler.command, handler.path, handler.headers, body, handler.client_address[0])
        if verdict.ok and kind == MEMBER and (group := cluster_of(verdict.secret, self.cluster_key_path)) is not None \
                and group not in self.pool.groups_of(peer):
            return Screened(kind, peer, verdict, (403, "this device is not a member of that cluster"))
        return Screened(kind, peer, verdict)


def cluster_of(secret: str, cluster_key_path: Path | str | None) -> str | None:
    """The cluster whose key a request secret was derived from; None for any other secret."""
    return next((m.group for m in memberships(cluster_key_path) if derive_token(m.key) == secret), None)
