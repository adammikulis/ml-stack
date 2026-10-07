"""Find the box with the card, without being told where it is."""

from __future__ import annotations

import base64
import contextlib
import hashlib
import hmac
import json
import os
import secrets
import socket
import ssl
import struct
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypedDict, Unpack

from ml_stack import home, http, macauth, sealing
from ml_stack.files import write_json
from ml_stack.log import warn
from ml_stack.platform import private_file

from . import tls, wsl_network
from .cluster_modes import validate

#: Link-local scope in the administratively-scoped block. TTL 1 keeps it there.
DEFAULT_GROUP = "239.255.77.70"
#: One above traind's HTTP port. UDP, where the daemon's own port is TCP -- so a firewall
#: that filters inbound (Windows Defender does, by default, on every profile) needs two
#: rules: TCP 8770 for the daemon and UDP 8771 for the beacons. `windows_firewall_rules`.
DEFAULT_PORT = 8771
#: traind's own port, repeated here so the firewall rule can name it without importing
#: the daemon.
DEFAULT_HTTP_PORT = 8770
#: What a cluster is called when nobody names one.
DEFAULT_CLUSTER = "ml-stack"


def default_group() -> str:
    """``$ML_STACK_DISCOVERY_GROUP`` if set. Read at call time, not import"""
    return os.environ.get("ML_STACK_DISCOVERY_GROUP") or DEFAULT_GROUP


def default_port() -> int:
    raw = os.environ.get("ML_STACK_DISCOVERY_PORT")
    return int(raw) if raw else DEFAULT_PORT

PROTOCOL = 3
"""3 sealed every datagram under a key derived from the cluster key. A peer speaking 2 is not heard."""
MAX_SKEW_S = 60.0
MAGIC = b"MLD3"
#: The most one UDP datagram carries.
MAX_DATAGRAM = 65507
#: What a beacon body may take. macOS refuses a datagram over ``net.inet.udp.maxdgram``
#: -- 9216 by default -- with EMSGSIZE, well below what IP allows.
BEACON_BUDGET = 8000


class DiscoveryError(RuntimeError):
    pass


# -- the key -------------------------------------------------------------
def require_name(value: object) -> str:
    """A nonempty cluster name without control characters, at most 64 characters."""
    if not isinstance(value, str) or not value.strip():
        raise DiscoveryError("cluster name is required")
    return check_name(value)


def key_path(path: Path | str | None = None) -> Path:
    """Where the cluster key lives. ``$ML_STACK_CLUSTER_KEY`` wins if set."""
    if path is not None:
        return home.expand(path)
    env = os.environ.get("ML_STACK_CLUSTER_KEY")
    return home.expand(env) if env else home.state("cluster.key")


def mint_cluster(group: str, path: Path | str | None = None, *, join: str = "", mode: str = "dev",
                 selection: str = "manual") -> Membership:
    """Make a cluster of a fresh random 256-bit key, replacing one of the same name."""
    group = require_name(group)
    key = base64.urlsafe_b64encode(secrets.token_bytes(32)).rstrip(b"=")
    return adopt(Membership(group=group, key=key, join=join, mode=mode, selection=selection), path)


def create_cluster_key(path: Path | str | None = None, *,
                       overwrite: bool = False, group: str = "") -> str:
    """Mint a cluster key that no passphrase protects, or return the one here."""
    joined = memberships(path)
    if joined and not overwrite:
        return joined[0].key.decode()
    return mint_cluster(require_name(group), path).key.decode()


def adopt(member: Membership, path: Path | str | None = None) -> Membership:
    """Record ``member`` as a cluster this machine is in, replacing one of the same name."""
    require_name(member.group)
    _write_memberships([member, *[m for m in memberships(path) if m.group != member.group]], path)
    return member


# -- the passphrase ------------------------------------------------------
MIN_PASSPHRASE = 5
"""Shortest passphrase accepted. It only ever goes through the join handshake, which a
listener cannot test a guess against and which locks out a source that keeps failing."""


def check_length(passphrase: str) -> str:
    """``passphrase`` stripped, or `DiscoveryError` when it is too short."""
    passphrase = passphrase.strip()
    if len(passphrase) < MIN_PASSPHRASE:
        raise DiscoveryError(f"The passphrase needs at least {MIN_PASSPHRASE} characters.")
    return passphrase


MOST_GROUP_NAME = 64
REFUSED_IN_NAME = "/\\"


def check_name(name: str) -> str:
    """``name`` trimmed, or `DiscoveryError` saying which rule it breaks."""
    name = name.strip()
    if not name:
        raise DiscoveryError("A cluster name cannot be empty.")
    if len(name) > MOST_GROUP_NAME:
        raise DiscoveryError(f"A cluster name is at most {MOST_GROUP_NAME} characters; this one is {len(name)}.")
    bad = sorted({c for c in name if c in REFUSED_IN_NAME or not c.isprintable()})
    if bad:
        shown = ", ".join(repr(c) for c in bad)
        raise DiscoveryError(f"A cluster name cannot contain {shown}: the join handshake uses "
                                  "slashes as separators and the memberships file cannot carry "
                                  "control characters.")
    return name


def cluster_group(path: Path | str | None = None) -> str | None:
    """The cluster this machine answers as, or None."""
    rows = memberships(path)
    return rows[0].group if rows else None


@dataclass(frozen=True, slots=True)
class Membership:
    """One cluster this machine belongs to."""

    group: str
    key: bytes
    """The cluster's random 256-bit key, urlsafe base64."""
    join: str = ""
    """What the join handshake takes as the passphrase (`onboard.joining.join_secret`); empty
    on a machine that does not know the passphrase."""

    mode: str = "dev"
    selection: str = "automatic"

    def __post_init__(self) -> None:
        require_name(self.group)
        validate(self.mode)
        if self.selection not in {"automatic", "manual"}:
            raise ValueError("cluster selection must be automatic or manual")
        if not isinstance(self.key, bytes) or len(self.key) != 43 or len(base64.b64decode(
                self.key + b"=" * (-len(self.key) % 4), altchars=b"-_", validate=True)) != 32:
            raise ValueError("cluster key must contain 256 bits")
        if not isinstance(self.join, str):
            raise ValueError("cluster join secret must be text")

    def public(self) -> dict[str, Any]:
        return {"group": self.group, "mode": self.mode, "selection": self.selection}


def clusters_path(path: Path | str | None = None) -> Path:
    """Where the clusters this machine belongs to are recorded.

    Named after the path it is given rather than fixed, so two callers pointed at
    different files do not share one store.
    """
    return key_path(path).with_suffix(".json")


def memberships(path: Path | str | None = None) -> list[Membership]:
    """Every cluster this machine is in, the first being the one it answers as."""
    out: list[Membership] = []
    seen: set[str] = set()
    try:
        raw = json.loads(clusters_path(path).read_text())
    except (OSError, ValueError):
        return []
    for row in raw if isinstance(raw, list) else []:
        try:
            group, key, join = row["group"], row["key"].encode("ascii"), row.get("join", "")
            member = Membership(group=group, key=key, join=join, mode=row["mode"],
                                selection=row.get("selection", "automatic"))
        except (KeyError, TypeError, AttributeError, ValueError, DiscoveryError):
            continue
        if key and group not in seen:
            seen.add(group)
            out.append(member)
    return out


def _write_memberships(rows: list[Membership],
                       path: Path | str | None = None) -> None:
    """Record the list this machine belongs to."""
    listed = clusters_path(path)
    write_json(listed, [{"group": m.group, "key": m.key.decode(), "join": m.join, "mode": m.mode,
                          "selection": m.selection} for m in rows])
    private_file(listed)


def leave(group: str, path: Path | str | None = None) -> list[Membership]:
    """Drop a cluster. The machine stops answering to it at once."""
    rows = [m for m in memberships(path) if m.group != group]
    _write_memberships(rows, path)
    return rows


def in_cluster(path: Path | str | None = None) -> bool:
    """Whether this machine has joined one. The question a setup wizard opens with."""
    return load_cluster_key(path) is not None


def load_cluster_key(path: Path | str | None = None) -> bytes | None:
    """The key this machine answers as, or None if it is in no cluster."""
    rows = memberships(path)
    return rows[0].key if rows else None


def derive_token(key: bytes) -> str:
    """The secret requests are signed with, which both ends compute independently."""
    return macauth.derive(key)


# -- the wire ------------------------------------------------------------
def _canonical(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()


def fit_beacon(body: dict[str, Any], budget: int = BEACON_BUDGET) -> dict[str, Any]:
    """The beacon body shortened until one datagram carries it.

    ``device["models_total"]`` is how many models the machine holds whenever the list on
    the beacon is shorter; a peer reads the rest from the daemon's ``/models``.
    """
    def over(device: dict[str, Any]) -> bool:
        return len(_canonical({**body, "device": device})) > budget

    device = dict(body.get("device") or {})
    if not over(device):
        return body
    models = list(device.get("models") or [])
    device["models_total"] = int(device.get("models_total") or len(models))
    keep = len(models)
    while keep and over({**device, "models": models[:keep]}):
        keep //= 2
    device["models"] = models[:keep]
    if over(device):
        device = {k: v for k, v in device.items() if not isinstance(v, (list, dict))}
    return {**body, "device": device}


def _box_key(key: bytes) -> bytes:
    return sealing.box_key(macauth.derive(key))


def _pack(key: bytes, payload: dict[str, Any]) -> bytes:
    """``payload`` as a datagram: sealed under the key derived from the cluster key, with the
    kind as associated data."""
    return MAGIC + sealing.seal(_box_key(key), _canonical(payload),
                                _data(str(payload.get("kind", ""))))


def _data(kind: str) -> bytes:
    return f"ml-stack-discovery/{kind}".encode()


def _verify(key: bytes, raw: bytes, *, kind: str,
            nonce: str | None = None) -> dict[str, Any] | None:
    """Open and authenticate a datagram, or return None."""
    if not raw.startswith(MAGIC):
        return None
    try:
        msg = json.loads(sealing.open_(_box_key(key), raw[len(MAGIC):], _data(kind)))
    except (sealing.SealError, ValueError):
        return None
    if not isinstance(msg, dict) or msg.get("v") != PROTOCOL or msg.get("kind") != kind:
        return None
    ts = msg.get("t")
    if not isinstance(ts, (int, float)) or abs(time.time() - ts) > MAX_SKEW_S:
        return None
    if nonce is not None and not hmac.compare_digest(str(msg.get("nonce", "")), nonce):
        return None
    return msg


# -- what gets advertised ------------------------------------------------
@dataclass
class Beacon:
    """One daemon, as seen on the network."""

    name: str
    port: int = 8770
    device: dict[str, Any] = field(default_factory=dict)
    busy: bool = False
    queued: int = 0
    slots: int = 1
    free: int = 1
    host: str = ""
    hostname: str = ""
    instance: str = ""
    """Minted per advertiser: tells one daemon's answers from another's in one listen."""
    machine: str = ""
    """`home.machine_id` of the machine it runs on: kept across restarts."""
    cert: str = ""
    """The daemon's certificate, base64 DER, which peers pin; empty when it serves signed-only."""

    @property
    def base_url(self) -> str:
        return f"{'https' if self.cert else 'http'}://{self.host or self.hostname}:{self.port}"

    def public(self) -> dict[str, Any]:
        return {"name": self.name, "port": self.port, "device": self.device,
                "busy": self.busy, "queued": self.queued,
                "slots": self.slots, "free": self.free,
                "hostname": self.hostname, "instance": self.instance,
                "machine": self.machine, "cert": self.cert}

    @property
    def identity(self) -> str:
        """What distinguishes one daemon from another, address aside."""
        return self.instance or f"{self.hostname}:{self.name}:{self.port}"


def named_apart(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """``rows``, each ``name`` two machines share followed by ``#`` and the shortest prefix
    of four or more characters of its ``machine`` that tells them apart."""
    by_name: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_name.setdefault(str(row.get("name") or ""), []).append(row)
    for name, same in by_name.items():
        machines = {str(row.get("machine") or "") for row in same}
        if len(machines) < 2:
            continue
        width = 4
        while len({m[:width] for m in machines}) < len(machines):
            width += 1
        for row in same:
            row["name"] = f"{name}#{str(row.get('machine') or '')[:width] or '?'}"
    return rows


def _prefer(existing: Beacon, candidate: Beacon) -> Beacon:
    """Merge a repeat answer from one daemon: the later state, the better address."""
    if not (candidate.host.startswith("127.") or existing.host.startswith("127.")):
        return candidate
    candidate.host = (candidate.host if candidate.host.startswith("127.")
                      else existing.host)
    return candidate


def primary_ip() -> str:
    """This machine's address on the interface that reaches the LAN."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except OSError:
        return ""
    finally:
        s.close()


LOOPBACK = "127.0.0.1"


def _destinations(group: str, port: int) -> list[tuple[tuple[str, int], str]]:
    """Every way to say "anyone out there?" on this link, as ``(address, interface)``: a
    multicast with an interface goes out of that one, and every other out of the LAN's."""
    return [((group, port), ""), ((group, port), LOOPBACK),
            (("255.255.255.255", port), ""), ((LOOPBACK, port), "")]


def _say(sock: socket.socket | wsl_network.DiscoverySocket, data: bytes, group: str,
         port: int) -> list[tuple[tuple[str, int], OSError]]:
    """Send ``data`` every way `_destinations` names; returns the ones refused."""
    lan = primary_ip()
    anywhere = struct.pack("!I", socket.INADDR_ANY)
    refused = []
    for dest, via in _destinations(group, port):
        out = via or lan
        try:
            sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_IF,
                            socket.inet_aton(out) if out else anywhere)
            sock.sendto(data, dest)
        except OSError as exc:
            refused.append((dest, exc))
    return refused


# -- what a firewall has to let in ---------------------------------------
FIREWALL_RULE_HTTP = "ml-stack traind"
FIREWALL_RULE_DISCOVERY = "ml-stack discovery"


def windows_firewall_rules(http_port: int = DEFAULT_HTTP_PORT,
                           port: int | None = None) -> list[tuple[str, str]]:
    """The two inbound rules Windows Defender Firewall needs, as ``(name, netsh line)``.

    Beacons are UDP to the multicast group, to the broadcast address and to loopback on
    ``port``; peers then talk to the daemon over TCP on ``http_port``. Windows blocks
    inbound on both by default -- on the Private profile too, unless the first-run prompt
    was answered for this exact executable, which a console script's ``python.exe`` never
    is -- so a daemon there is invisible to ``ml-stack-peers ls`` until these exist. Each
    line needs an administrator's shell.
    """
    port = port if port is not None else default_port()
    return [
        (FIREWALL_RULE_HTTP,
         f'netsh advfirewall firewall add rule name="{FIREWALL_RULE_HTTP}" dir=in '
         f'action=allow protocol=TCP localport={http_port}'),
        (FIREWALL_RULE_DISCOVERY,
         f'netsh advfirewall firewall add rule name="{FIREWALL_RULE_DISCOVERY}" dir=in '
         f'action=allow protocol=UDP localport={port}'),
    ]


def windows_firewall_line(http_port: int = DEFAULT_HTTP_PORT,
                          port: int | None = None) -> str:
    """Both rules as one line a person can paste into an administrator's prompt."""
    return " && ".join(line for _, line in windows_firewall_rules(http_port, port))


def _socket(*, broadcast: bool = False, bind: tuple[str, int] | None = None,
            group: str | None = None) -> socket.socket | wsl_network.DiscoverySocket:
    if wsl_network.wsl_registration.configuration() is not None:
        return wsl_network.DiscoverySocket({"broadcast": broadcast, "bind": bind, "group": group})
    return _native_socket(broadcast=broadcast, bind=bind, group=group)


def _native_socket(*, broadcast: bool = False, bind: tuple[str, int] | None = None,
                   group: str | None = None) -> socket.socket:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    if os.name == "nt":
        wsl_network.disable_udp_reset(s)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    if hasattr(socket, "SO_REUSEPORT"):
        with contextlib.suppress(OSError):
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
    if broadcast:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    # TTL 1: this is a LAN facility. Never let it escape the local segment.
    s.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 1)
    s.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_LOOP, 1)
    if bind is not None:
        s.bind(bind)
    if group is not None:
        mreq = struct.pack("4sl", socket.inet_aton(group), socket.INADDR_ANY)
        s.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
        # Linux's lo carries no MULTICAST flag, so this join is refused there
        with contextlib.suppress(OSError):
            s.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, struct.pack(
                "4s4s", socket.inet_aton(group), socket.inet_aton(LOOPBACK)))
    return s


class AdvertiserOptions(TypedDict, total=False):
    group: str | None
    port: int | None
    interval_s: float
    cluster: str
    refresh: Callable[[Beacon], None] | None


class Advertiser:
    """Answers 'who is out there' on behalf of one daemon."""

    def __init__(self, beacon: Beacon, key: bytes, **options: Unpack[AdvertiserOptions]) -> None:
        if extra := options.keys() - AdvertiserOptions.__annotations__.keys():
            raise TypeError(f"unknown advertiser options: {', '.join(sorted(extra))}")
        beacon.instance = beacon.instance or secrets.token_hex(8)
        self.beacon = beacon
        self.key = key
        self.refresh = options.get("refresh")
        self.group = options.get("group") or default_group()
        port = options.get("port")
        self.port = port if port is not None else default_port()
        self.interval_s = options.get("interval_s", 10.0)
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._sock: socket.socket | wsl_network.DiscoverySocket | None = None
        self._ready = threading.Event()
        self._error: BaseException | None = None
        self._asked = threading.Event()
        self._state: dict[str, Any] = {}
        self.undelivered = 0
        self.last_error = ""
        self._said: set[str] = set()
        self.cluster = options.get("cluster", "")
        self.joinable = False
        self.mode = "dev"
        """The cluster's name; a machine that asks to join it is told where to shake hands."""

    # -- lifecycle --
    def start(self, *, wait_s: float = 2.0) -> Advertiser:
        for target in (self._serve, self._announce_loop, self._sample_loop):
            t = threading.Thread(target=target, daemon=True,
                                 name=f"advertiser-{target.__name__.strip('_')}")
            t.start()
            self._threads.append(t)
        if not self._ready.wait(wait_s):
            raise DiscoveryError("advertiser did not bind in time")
        if self._error is not None:
            raise DiscoveryError(f"advertiser failed to bind: {self._error}")
        # Say so at once. The loop's first unsolicited beacon is `interval_s` away, and a
        # daemon that has just come up should appear in the next `ml-stack-peers ls`
        # rather than ten seconds later.
        with contextlib.suppress(OSError):
            self.announce()
        return self

    def stop(self) -> None:
        self._stop.set()
        self._asked.set()
        if self._sock is not None:
            with contextlib.suppress(OSError):
                self._sock.close()
        for t in self._threads:
            t.join(timeout=2.0)

    def __enter__(self) -> Advertiser:
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.stop()

    # -- internals --
    def _body(self) -> dict[str, Any]:
        b = self.beacon.public()
        b["hostname"] = b["hostname"] or socket.gethostname()
        return fit_beacon(b)

    def _undelivered(self, addr: tuple[str, int], exc: OSError, size: int) -> None:
        """Record a beacon that did not go out, and say so once per reason."""
        reason = f"{type(exc).__name__}: {exc}"
        self.undelivered += 1
        self.last_error = reason
        if reason in self._said:
            return
        self._said.add(reason)
        warn(f"a {size}-byte beacon from {self.beacon.name} could not be sent to "
             f"{addr[0]}:{addr[1]} ({reason}). Other discovery routes may still work.")

    def _sample(self) -> None:
        """Run ``refresh`` and keep what it produced as the beacon to send."""
        if self.refresh is not None:
            with contextlib.suppress(Exception):
                self.refresh(self.beacon)
        self._state = self._body()

    def _sample_loop(self) -> None:
        while not self._stop.is_set():
            self._sample()
            self._asked.wait(self.interval_s)
            self._asked.clear()

    def _payload(self, kind: str, nonce: str = "") -> bytes:
        return _pack(self.key, {"v": PROTOCOL, "kind": kind, "t": time.time(),
                                "nonce": nonce, "beacon": self._state or self._body()})

    def _serve(self) -> None:
        try:
            sock = _socket(broadcast=True, bind=("", self.port), group=self.group)
        except OSError as exc:
            self._error = exc
            self._ready.set()
            return
        self._sock = sock
        self._ready.set()
        sock.settimeout(0.5)
        while not self._stop.is_set():
            try:
                raw, addr = sock.recvfrom(65535)
            except TimeoutError:
                continue
            except OSError:
                break
            if self.cluster and (nonce := _join_nonce(raw, self.cluster)) is not None:
                self._tell_join(sock, nonce, addr)
                continue
            msg = _verify(self.key, raw, kind="who")
            if msg is None:
                continue
            reply = self._payload("beacon", str(msg.get("nonce", "")))
            try:
                sock.sendto(reply, addr)
            except OSError as exc:
                self._undelivered(addr, exc, len(reply))
                continue
            self._asked.set()

    def _tell_join(self, sock: socket.socket | wsl_network.DiscoverySocket,
                   nonce: str, addr: tuple[str, int]) -> None:
        """Answer a machine asking to join this cluster: the port and scheme to shake hands on."""
        reply = _canonical({"v": PROTOCOL, "kind": "join", "group": self.cluster, "nonce": nonce,
                            "name": self.beacon.name, "port": self.beacon.port, "tls": bool(self.beacon.cert),
                            "method": "automatic" if self.mode == "dev" else "passphrase" if self.joinable else "recovery",
                            "mode": self.mode, "cluster_id": hashlib.sha256(self.key).hexdigest(),
                            "fingerprint": hashlib.sha256(base64.b64decode(self.beacon.cert)).hexdigest() if self.beacon.cert else ""})
        with contextlib.suppress(OSError):
            sock.sendto(reply, addr)

    def _announce_loop(self) -> None:
        while not self._stop.wait(self.interval_s):
            self.announce()

    def announce(self) -> None:
        """Push an unsolicited beacon. Safe to call at any time."""
        data = self._payload("beacon")
        with _socket(broadcast=True) as s:
            refused = _say(s, data, self.group, self.port)
        if len(refused) == len(_destinations(self.group, self.port)):
            self._undelivered(*refused[0], len(data))


def _join_nonce(raw: bytes, group: str) -> str | None:
    """The nonce of a plain datagram asking to join ``group``, or None for anything else."""
    if len(raw) > 2048:
        return None
    try:
        msg = json.loads(raw)
    except (ValueError, UnicodeDecodeError):
        return None
    if isinstance(msg, dict) and msg.get("v") == PROTOCOL and msg.get("kind") == "join?" \
            and msg.get("group") in (group, ""):
        nonce = msg.get("nonce")
        return nonce if isinstance(nonce, str) and 1 <= len(nonce) <= 64 else None
    return None


def discover(key: bytes, *, timeout_s: float = 2.0, group: str | None = None,
             port: int | None = None, retry_s: float = 0.3) -> list[Beacon]:
    """Ask the LAN who is running a daemon, and return everyone who proves it."""
    group = group or default_group()
    port = port if port is not None else default_port()
    nonce = secrets.token_hex(16)
    query = _pack(key, {"v": PROTOCOL, "kind": "who", "t": time.time(), "nonce": nonce})
    found: dict[str, Beacon] = {}
    with _socket(broadcast=True, bind=("", 0)) as sock:
        _say(sock, query, group, port)
        deadline = time.time() + timeout_s
        next_query = time.time() + retry_s
        while True:
            now = time.time()
            if now >= deadline:
                break
            if now >= next_query:
                _say(sock, query, group, port)
                next_query = now + retry_s
            sock.settimeout(max(0.0, min(deadline, next_query) - time.time()))
            try:
                raw, addr = sock.recvfrom(65535)
            except TimeoutError:
                continue
            except OSError:
                break
            msg = _verify(key, raw, kind="beacon", nonce=nonce)
            if msg is None:
                continue
            body = msg.get("beacon")
            if not isinstance(body, dict):
                continue
            try:
                beacon = Beacon(name=str(body.get("name", "")),
                                port=int(body.get("port", 8770)),
                                device=dict(body.get("device") or {}),
                                busy=bool(body.get("busy")),
                                queued=int(body.get("queued") or 0),
                                slots=int(body.get("slots") or 1),
                                free=int(body["free"]) if "free" in body
                                else (0 if body.get("busy") else 1),
                                host=addr[0],
                                hostname=str(body.get("hostname", "")),
                                instance=str(body.get("instance", "")),
                                machine=str(body.get("machine", "")),
                                cert=str(body.get("cert", "")))
            except (TypeError, ValueError):
                continue
            if not _trusted(beacon):
                continue
            key_id = beacon.identity
            prior = found.get(key_id)
            found[key_id] = beacon if prior is None else _prefer(prior, beacon)
    return sorted(found.values(), key=lambda b: (b.name, b.host))


_WARNED: set[str] = set()


def _trusted(beacon: Beacon) -> bool:
    """Whether this machine may be talked to, and if it offers a certificate, pin it.

    A beacon with no certificate is a daemon serving signed-only, which is ignored unless
    this machine has opted into that (`tls.disabled`) or it is on this machine."""
    if beacon.cert:
        try:
            context = tls.pinned_context(beacon.cert)
        except (ValueError, ssl.SSLError):
            return False
        for name in {beacon.host, beacon.hostname}:
            if name:
                http.pin(f"{name}:{beacon.port}", context)
        return True
    if tls.disabled() or beacon.host.startswith("127."):
        return True
    if beacon.identity not in _WARNED:
        _WARNED.add(beacon.identity)
        warn(f"{beacon.name} at {beacon.host} offers no TLS, so it is ignored; "
             f"{tls.ENV}=off here talks to it with signed requests only")
    return False
