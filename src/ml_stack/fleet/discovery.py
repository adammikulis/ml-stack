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
import struct
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import Any

from ml_stack import home
from ml_stack.files import write_json
from ml_stack.log import warn
from ml_stack.platform import private_file

#: Link-local scope in the administratively-scoped block. TTL 1 keeps it there.
DEFAULT_GROUP = "239.255.77.70"
#: One above traind's HTTP port. UDP, where the daemon's own port is TCP -- so a firewall
#: that filters inbound (Windows Defender does, by default, on every profile) needs two
#: rules: TCP 8770 for the daemon and UDP 8771 for the beacons. `windows_firewall_rules`.
DEFAULT_PORT = 8771
#: traind's own port, repeated here so the firewall rule can name it without importing
#: the daemon.
DEFAULT_HTTP_PORT = 8770
#: What a cluster is called when nobody names one. Two machines that type the same
#: passphrase derive the same key only if they also agree on this.
DEFAULT_CLUSTER = "ml-stack"


def default_group() -> str:
    """``$ML_STACK_DISCOVERY_GROUP`` if set. Read at call time, not import"""
    return os.environ.get("ML_STACK_DISCOVERY_GROUP") or DEFAULT_GROUP


def default_port() -> int:
    raw = os.environ.get("ML_STACK_DISCOVERY_PORT")
    return int(raw) if raw else DEFAULT_PORT

PROTOCOL = 1
MAX_SKEW_S = 60.0
#: The most one UDP datagram carries.
MAX_DATAGRAM = 65507
#: What a beacon body may take. macOS refuses a datagram over ``net.inet.udp.maxdgram``
#: -- 9216 by default -- with EMSGSIZE, well below what IP allows.
BEACON_BUDGET = 8000
_TOKEN_INFO = b"ml-stack-traind-api-token-v1"


class DiscoveryError(RuntimeError):
    pass


# -- the key -------------------------------------------------------------
def key_path(path: Path | str | None = None) -> Path:
    """Where the cluster key lives. ``$ML_STACK_CLUSTER_KEY`` wins if set."""
    if path is not None:
        return home.expand(path)
    env = os.environ.get("ML_STACK_CLUSTER_KEY")
    return home.expand(env) if env else home.state("cluster.key")


def create_cluster_key(path: Path | str | None = None, *,
                       overwrite: bool = False) -> str:
    """Mint a cluster key with no passphrase behind it, or return the one here."""
    joined = memberships(path)
    if joined and not overwrite:
        return joined[0].key.decode()
    key = base64.urlsafe_b64encode(secrets.token_bytes(32)).decode().rstrip("=")
    rows = [Membership(group=DEFAULT_CLUSTER, key=key.encode())]
    rows += [m for m in joined if m.group != DEFAULT_CLUSTER]
    _write_memberships(rows, path)
    return key


# -- joining by password -------------------------------------------------
MIN_PASSPHRASE = 5
"""Shortest passphrase accepted. Low, because a refusal people work around by typing"""

SCRYPT_N = 1 << 16
SCRYPT_R = 8
SCRYPT_P = 1
"""~80ms and 64MB on a laptop, a second or two on a Pi. Paid once, at join time: the"""


def _salt_for(group: str) -> bytes:
    """Deterministic, because both machines have to derive the same key from the same"""
    return sha256(b"ml-stack-cluster-v1:" + group.encode()).digest()


def key_from_passphrase(passphrase: str, *, group: str = DEFAULT_CLUSTER) -> bytes:
    """The cluster key two machines derive from the same words."""
    passphrase = passphrase.strip()
    if len(passphrase) < MIN_PASSPHRASE:
        raise DiscoveryError(
            f"passphrase must be at least {MIN_PASSPHRASE} characters -- everyone on "
            "this network can hear the beacons and grind guesses against them offline")
    raw = hashlib.scrypt(passphrase.encode("utf-8"), salt=_salt_for(group),
                         n=SCRYPT_N, r=SCRYPT_R, p=SCRYPT_P, dklen=32,
                         maxmem=2 * 128 * SCRYPT_N * SCRYPT_R)
    return base64.urlsafe_b64encode(raw).rstrip(b"=")


def group_path(path: Path | str | None = None) -> Path:
    """Where the group name is recorded, beside the key."""
    return key_path(path).with_suffix(".group")


def cluster_group(path: Path | str | None = None) -> str | None:
    """The cluster this machine answers as, or None."""
    rows = memberships(path)
    return rows[0].group if rows else None


def join_cluster(passphrase: str, *, group: str = DEFAULT_CLUSTER,
                 path: Path | str | None = None, overwrite: bool = True) -> bytes:
    """Join, and answer as this cluster from now on. Returns the key."""
    key = key_from_passphrase(passphrase, group=group)
    joined = memberships(path)
    if joined and not overwrite:
        return joined[0].key
    rows = [Membership(group=group, key=key)]
    rows += [m for m in joined if m.group != group]
    _write_memberships(rows, path)
    return key


def check_passphrase(passphrase: str, *, group: str | None = None,
                     path: Path | str | None = None) -> bool:
    """Whether these words derive the key this machine already holds."""
    key = load_cluster_key(path)
    if key is None:
        return False
    group = group if group is not None else (cluster_group(path) or DEFAULT_CLUSTER)
    try:
        candidate = key_from_passphrase(passphrase, group=group)
    except DiscoveryError:
        return False
    return hmac.compare_digest(candidate, key)


@dataclass(frozen=True, slots=True)
class Membership:
    """One cluster this machine belongs to."""

    group: str
    key: bytes

    def public(self) -> dict[str, Any]:
        return {"group": self.group}


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
        return _adopt_single(path)
    for row in raw if isinstance(raw, list) else []:
        try:
            group, key = str(row["group"]), str(row["key"]).encode()
        except (KeyError, TypeError, AttributeError):
            continue
        if key and group not in seen:
            seen.add(group)
            out.append(Membership(group=group, key=key))
    return out


def _adopt_single(path: Path | str | None = None) -> list[Membership]:
    """The one cluster written before the list existed, moved into the list."""
    try:
        key = key_path(path).read_text().strip()
    except OSError:
        return []
    if not key:
        return []
    try:
        group = group_path(path).read_text().strip()
    except OSError:
        group = ""
    rows = [Membership(group=group or DEFAULT_CLUSTER, key=key.encode())]
    with contextlib.suppress(OSError):
        _write_memberships(rows, path)
    return rows


def _write_memberships(rows: list[Membership],
                       path: Path | str | None = None) -> None:
    """Record the list this machine belongs to."""
    listed = clusters_path(path)
    write_json(listed, [{"group": m.group, "key": m.key.decode()} for m in rows])
    private_file(listed)


def join(passphrase: str, *, group: str = "", path: Path | str | None = None
         ) -> list[Membership]:
    """Add a cluster. Joining one already joined replaces its key."""
    group = group.strip() or DEFAULT_CLUSTER
    key = key_from_passphrase(passphrase, group=group)
    rows = [m for m in memberships(path) if m.group != group]
    rows.append(Membership(group=group, key=key))
    _write_memberships(rows, path)
    return rows


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
    """The traind bearer token both ends compute independently."""
    mac = hmac.new(key, _TOKEN_INFO, sha256).digest()
    return base64.urlsafe_b64encode(mac).decode().rstrip("=")


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


def _sign(key: bytes, payload: dict[str, Any]) -> bytes:
    body = {k: v for k, v in payload.items() if k != "mac"}
    mac = hmac.new(key, _canonical(body), sha256).hexdigest()
    return _canonical({**body, "mac": mac})


def _verify(key: bytes, raw: bytes, *, kind: str,
            nonce: str | None = None) -> dict[str, Any] | None:
    """Parse and authenticate a packet, or return None."""
    try:
        msg = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None
    if not isinstance(msg, dict) or msg.get("v") != PROTOCOL or msg.get("kind") != kind:
        return None
    got = msg.get("mac")
    if not isinstance(got, str):
        return None
    body = {k: v for k, v in msg.items() if k != "mac"}
    want = hmac.new(key, _canonical(body), sha256).hexdigest()
    if not hmac.compare_digest(got, want):
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

    @property
    def base_url(self) -> str:
        return f"http://{self.host or self.hostname}:{self.port}"

    def public(self) -> dict[str, Any]:
        return {"name": self.name, "port": self.port, "device": self.device,
                "busy": self.busy, "queued": self.queued,
                "slots": self.slots, "free": self.free,
                "hostname": self.hostname, "instance": self.instance,
                "machine": self.machine}

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


def _say(sock: socket.socket, data: bytes, group: str,
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
            group: str | None = None) -> socket.socket:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
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


class Advertiser:
    """Answers 'who is out there' on behalf of one daemon."""

    def __init__(self, beacon: Beacon, key: bytes, *,
                 group: str | None = None, port: int | None = None,
                 interval_s: float = 10.0,
                 refresh: Callable[[Beacon], None] | None = None) -> None:
        beacon.instance = beacon.instance or secrets.token_hex(8)
        self.beacon = beacon
        self.key = key
        self.refresh = refresh
        self.group = group or default_group()
        self.port = port if port is not None else default_port()
        self.interval_s = interval_s
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._sock: socket.socket | None = None
        self._ready = threading.Event()
        self._error: BaseException | None = None
        self._asked = threading.Event()
        self._state: dict[str, Any] = {}
        self.undelivered = 0
        self.last_error = ""
        self._said: set[str] = set()

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
        warn(f"a {size}-byte beacon did not reach {addr[0]}:{addr[1]} ({reason}). "
             f"Until one does, {self.beacon.name} is not in the fleet: the other "
             "machines do not list it.")

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
        return _sign(self.key, {"v": PROTOCOL, "kind": kind, "t": time.time(),
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


def discover(key: bytes, *, timeout_s: float = 2.0, group: str | None = None,
             port: int | None = None, retry_s: float = 0.3) -> list[Beacon]:
    """Ask the LAN who is running a daemon, and return everyone who proves it."""
    group = group or default_group()
    port = port if port is not None else default_port()
    nonce = secrets.token_hex(16)
    query = _sign(key, {"v": PROTOCOL, "kind": "who", "t": time.time(),
                        "nonce": nonce})
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
                                machine=str(body.get("machine", "")))
            except (TypeError, ValueError):
                continue
            key_id = beacon.identity
            prior = found.get(key_id)
            found[key_id] = beacon if prior is None else _prefer(prior, beacon)
    return sorted(found.values(), key=lambda b: (b.name, b.host))
