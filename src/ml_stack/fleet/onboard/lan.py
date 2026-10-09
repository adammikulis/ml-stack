"""Onboarding talks to machines on this network and to nothing on the public internet.

Pairing, fetching a manifest and fetching a file from a peer open their own connections (the
peer's certificate is pinned, not looked up in a trust store), so they do not go through
`ml_stack.net`. What keeps them from being a way to the internet is this check, made at every
connection: the address must not be a public one (loopback, private, link-local and
carrier-grade NAT, which covers tailnets, all pass). `tests/test_net_no_bypass.py` allows these
modules to import ``http.client`` only because of it.
"""

from __future__ import annotations

import ipaddress
import socket
from urllib.parse import urlsplit

from ml_stack.httpguard import allowed_address

__all__ = ["NotLocal", "in_tailnet", "loopback_host", "require_local", "require_local_url",
           "secure_scheme"]

TAILNET_V4 = ipaddress.ip_network("100.64.0.0/10")
TAILNET_V6 = ipaddress.ip_network("fd7a:115c:a1e0::/48")


class NotLocal(OSError):
    """The address is on the public internet; onboarding does not go there."""


def in_tailnet(address: str) -> bool:
    """Whether ``address`` is in the range Tailscale assigns (100.64.0.0/10, fd7a:115c:a1e0::/48)."""
    try:
        ip = ipaddress.ip_address(address.strip("[]").split("%")[0])
    except ValueError:
        return False
    return ip in (TAILNET_V4 if ip.version == 4 else TAILNET_V6)


def loopback_host(host: str) -> bool:
    """Whether ``host`` is this machine by name or loopback address."""
    host = host.strip("[]").lower()
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host.split("%")[0]).is_loopback
    except ValueError:
        return False


def secure_scheme(url: str) -> bool:
    """Whether ``url`` is https, or plain http to this machine: the only two forms a pool link
    takes. Plain http to anything else is not a way to reach a LAN machine."""
    parts = urlsplit(url)
    return parts.scheme == "https" or (parts.scheme == "http" and loopback_host(parts.hostname or ""))


def require_local(host: str, port: int = 0) -> None:
    """Return when every address ``host`` names is on this machine or its network; `NotLocal`
    when one is public or the name does not resolve."""
    try:
        addresses = [str(ipaddress.ip_address(host.strip("[]")))]
    except ValueError:
        try:
            addresses = sorted({info[4][0] for info in socket.getaddrinfo(
                host, port or None, type=socket.SOCK_STREAM)})
        except socket.gaierror as exc:
            raise NotLocal(f"cannot resolve {host!r}: {exc}") from None
    if not addresses:
        raise NotLocal(f"{host!r} resolves to nothing")
    for address in addresses:
        if allowed_address(address):
            raise NotLocal(f"{host} is {address.split('%')[0]}, on the public internet; "
                           "onboarding is with machines on this network")


def require_local_url(url: str) -> None:
    parts = urlsplit(url)
    require_local(parts.hostname or "", parts.port or 0)
