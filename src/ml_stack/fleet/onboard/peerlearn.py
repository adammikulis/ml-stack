"""Pairing fills the peer book: each side keeps how to reach the other's share, authenticated by
the pairing itself (docs/onboarding.md).

What is stored comes from two places, both of which the exchange authenticated: the accepting
machine's `Grant` (its share port, certificate, manifest signing key and the key this machine
signs requests with) and the asking machine's `Offer` (the same four things the other way,
sealed under the exchange's key). The address is the one the pairing connection used. A
certificate that is not the one the exchange bound to the code is not stored, so a machine cannot
be made to pin some other certificate by a pairing it took part in. A machine that shares nothing
has a share port of 0 and gets no row. ``fleet peers add`` is still there for a device paired
some other way.
"""

from __future__ import annotations

import base64
import binascii
import os
from pathlib import Path

from ml_stack.hub.peerbook import PeerBook

from .pairing import Grant, Offer, fingerprint_of
from .requests import Device, Devices, Request

__all__ = ["forget", "learn_from_grant", "learn_from_offer", "new_secret", "share_url"]


def new_secret() -> str:
    """A fresh request key, urlsafe base64 of 32 random bytes (as `Grant.device_secret`)."""
    return base64.urlsafe_b64encode(os.urandom(32)).decode()


def share_url(host: str, port: int) -> str:
    return f"https://[{host}]:{port}" if ":" in host and not host.startswith("[") \
        else f"https://{host}:{port}"


def _bound(certificate: str, fingerprint: str) -> bool:
    """Whether the beacon ``certificate`` is the certificate with ``fingerprint``."""
    try:
        return bool(fingerprint) and fingerprint_of(base64.b64decode(certificate, validate=True)) \
            == fingerprint
    except (ValueError, binascii.Error):
        return False


def _name(book: PeerBook, name: str, fingerprint: str) -> str:
    """``name``, or ``name-<fingerprint>`` when another device already goes by it."""
    clash = next((r for r in book.rows() if r["name"] == name
                  and r.get("fingerprint") != fingerprint), None)
    return f"{name}-{fingerprint[:6]}" if clash else name


def _key_ok(signing_key: str) -> bool:
    try:
        return len(base64.b64decode(signing_key, validate=True)) == 32
    except (ValueError, binascii.Error):
        return False


def learn_from_grant(directory: Path, grant: Grant, *, host: str, server_fingerprint: str,  # noqa: PLR0913 - all keywords
                     my_secret: str, mine: bool = False) -> dict[str, object] | None:
    """The asking side: remember the machine that accepted (so its requests to our share are
    recognised: ``my_secret`` is what we offered it) and, when it shares, add it to the peer
    book. Returns the row, or None when there is nothing to ask of it."""
    if not _bound(grant.certificate, server_fingerprint):
        return None
    Devices(directory / "devices.json").add_accepter(
        server_fingerprint, grant.name or host, host, secret=my_secret, mine=mine,
        shared_cluster_key=bool(grant.key))
    if not (grant.share_port and grant.device_secret and _key_ok(grant.signing_key)):
        return None
    book = PeerBook(directory / "peers.json")
    row = {"name": _name(book, grant.name or host, server_fingerprint),
           "url": share_url(host, grant.share_port), "certificate": grant.certificate,
           "signing_key": grant.signing_key, "device_secret": grant.device_secret,
           "min_serial": 0, "fingerprint": server_fingerprint, "source": "pairing"}
    book.add(row)
    return row


def learn_from_offer(directory: Path, request: Request, offer: Offer) -> dict[str, object] | None:
    """The accepting side: add the machine that asked to the peer book, from its authenticated
    offer. Returns the row, or None when the offer carries no share (or a certificate that is
    not the one the exchange bound)."""
    if not (offer.share_port and offer.device_secret and _key_ok(offer.signing_key)
            and _bound(offer.certificate, request.fingerprint)):
        return None
    book = PeerBook(directory / "peers.json")
    row = {"name": _name(book, request.name, request.fingerprint),
           "url": share_url(request.address, offer.share_port), "certificate": offer.certificate,
           "signing_key": offer.signing_key, "device_secret": offer.device_secret,
           "min_serial": 0, "fingerprint": request.fingerprint, "source": "pairing"}
    book.add(row)
    return row


def forget(directory: Path, device: Device) -> list[str]:
    """A revoked device is no longer asked for files: drop its peer-book rows."""
    return PeerBook(directory / "peers.json").remove_device(device.fingerprint)
