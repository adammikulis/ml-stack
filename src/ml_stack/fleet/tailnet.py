"""Finding a Tailscale client on this machine and reading what it reports about the tailnet.

Read-only: the only command ever run is ``tailscale status --json``. Nothing is installed,
started, logged in or changed. The output is bounded in size and time, the child gets the
environment less anything that looks like a secret, and only whitelisted fields are copied out
(no auth URL, key or token field is ever returned). Addresses outside the tailnet ranges are
dropped; names are reduced to printable text.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

if os.name == "posix":
    import resource
else:
    resource = None

from ml_stack.credentials.environment import child_environment

from .onboard.lan import in_tailnet
from .onboard.requests import clean

__all__ = ["MAC_APP_CLI", "READ_ONLY", "Tailnet", "TailnetPeer", "detect", "find_cli", "parse"]

logger = logging.getLogger("ml_stack.tailnet")

MAC_APP_CLI = "/Applications/Tailscale.app/Contents/MacOS/Tailscale"
READ_ONLY: frozenset[tuple[str, ...]] = frozenset({("status", "--json")})
MOST_OUTPUT = 1 << 20
MOST_PEERS = 256
MOST_ADDRESSES = 8
LABEL = re.compile(r"[^\w.\- ]")
DNS_NAME = re.compile(r"[a-z0-9]([a-z0-9-]{0,62}[a-z0-9])?(\.[a-z0-9]([a-z0-9-]{0,62}[a-z0-9])?)*")


@dataclass(frozen=True, slots=True)
class TailnetPeer:
    name: str
    dns_name: str
    addresses: tuple[str, ...]
    online: bool

    def public(self) -> dict[str, object]:
        return {"name": self.name, "dns_name": self.dns_name,
                "addresses": list(self.addresses), "online": self.online}


@dataclass(frozen=True, slots=True)
class Tailnet:
    """What the local Tailscale client reports. ``installed`` is False when there is no client;
    ``up`` is True when it is connected."""

    installed: bool = False
    up: bool = False
    state: str = ""
    name: str = ""
    addresses: tuple[str, ...] = ()
    peers: tuple[TailnetPeer, ...] = ()

    def peer_named(self, name: str) -> TailnetPeer | None:
        """The peer ``status`` reported under ``name`` (host name or MagicDNS name); nothing is
        looked up anywhere else."""
        wanted = name.strip().rstrip(".").lower()
        if not self.up or not wanted:
            return None
        return next((p for p in self.peers
                     if wanted in (p.name.lower(), p.dns_name, p.dns_name.split(".")[0])), None)

    def public(self) -> dict[str, object]:
        return {"installed": self.installed, "up": self.up, "state": self.state,
                "name": self.name, "addresses": list(self.addresses),
                "peers": [p.public() for p in self.peers]}


def _label(value: Any) -> str:
    return LABEL.sub("_", clean(value))


def _dns(value: Any) -> str:
    text = value.strip().rstrip(".").lower() if isinstance(value, str) else ""
    return text if len(text) <= 253 and DNS_NAME.fullmatch(text) else ""


def _addresses(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    kept = [a for a in value[:MOST_ADDRESSES * 2] if isinstance(a, str) and in_tailnet(a)]
    return tuple(dict.fromkeys(kept))[:MOST_ADDRESSES]


def _peer(row: Any) -> TailnetPeer | None:
    if not isinstance(row, dict):
        return None
    addresses = _addresses(row.get("TailscaleIPs"))
    if not addresses:
        return None
    return TailnetPeer(name=_label(row.get("HostName")), dns_name=_dns(row.get("DNSName")),
                       addresses=addresses, online=row.get("Online") is True)


def parse(raw: bytes) -> Tailnet:
    """The tailnet ``tailscale status --json`` described; unreadable when it is not that JSON."""
    try:
        doc = json.loads(raw)
    except (ValueError, RecursionError):
        return Tailnet(installed=True, state="unreadable")
    if not isinstance(doc, dict):
        return Tailnet(installed=True, state="unreadable")
    state = _label(doc.get("BackendState"))[:32]
    me = doc.get("Self") if isinstance(doc.get("Self"), dict) else {}
    mine = _addresses(doc.get("TailscaleIPs")) or _addresses(me.get("TailscaleIPs"))
    table = doc.get("Peer") if isinstance(doc.get("Peer"), dict) else {}
    peers = tuple(p for p in (_peer(row) for row in list(table.values())[:MOST_PEERS]) if p)
    up = state == "Running" and bool(mine)
    return Tailnet(installed=True, up=up, state=state, name=_label(me.get("HostName")),
                   addresses=mine if up else (), peers=peers if up else ())


def find_cli(which: Callable[[str], str | None] = shutil.which,
             app: str | None = None) -> str | None:
    """The path of the ``tailscale`` command line, or None."""
    found = which("tailscale")
    if found:
        return found
    app = app or MAC_APP_CLI
    return app if Path(app).is_file() and os.access(app, os.X_OK) else None


def _limit(size: int) -> Callable[[], None]:
    if resource is None:
        raise RuntimeError("process output resource limits require a POSIX platform")
    def apply() -> None:
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        resource.setrlimit(resource.RLIMIT_FSIZE, (size, size))
    return apply


def _run(cli: str, args: tuple[str, ...], *, timeout_s: float, most: int) -> bytes | None:
    """Standard output of ``cli args``, or None on a failure, a timeout or output over ``most``."""
    if args not in READ_ONLY:
        raise ValueError(f"tailscale {' '.join(args)} is not a read-only command")
    if resource is None:
        logger.warning("tailscale status requires POSIX output resource limits; command was not run")
        return None
    with tempfile.TemporaryFile() as out:
        try:
            done = subprocess.run(
                [cli, *args], stdin=subprocess.DEVNULL, stdout=out, stderr=subprocess.DEVNULL,
                env=child_environment(), timeout=timeout_s, check=False,
                start_new_session=True, preexec_fn=_limit(most + 1))
        except subprocess.TimeoutExpired:
            logger.warning("tailscale status timed out after %.1f s", timeout_s)
            return None
        except OSError as exc:
            logger.warning("tailscale could not be run: %s", type(exc).__name__)
            return None
        size = out.seek(0, os.SEEK_END)
        if size > most:
            logger.warning("tailscale status wrote more than %d bytes; ignored", most)
            return None
        out.seek(0)
        data = out.read(most)
    return data if done.returncode == 0 else None


def detect(*, cli: str | None = None, timeout_s: float = 3.0, most: int = MOST_OUTPUT) -> Tailnet:
    """The local tailnet: not installed, installed but down, or up with its peers."""
    path = cli or find_cli()
    if not path:
        return Tailnet()
    raw = _run(path, ("status", "--json"), timeout_s=timeout_s, most=most)
    if raw is None:
        return Tailnet(installed=True, state="unavailable")
    return parse(raw)
