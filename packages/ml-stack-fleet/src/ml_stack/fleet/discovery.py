"""Find the box with the card, without being told where it is."""

from __future__ import annotations

import base64
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

#: Link-local scope in the administratively-scoped block. TTL 1 keeps it there.
DEFAULT_GROUP = "239.255.77.70"
#: One above traind's HTTP port, so a single firewall rule covers both.
DEFAULT_PORT = 8771
DEFAULT_KEY_PATH = Path("~/.ml-stack/cluster.key")


def default_group() -> str:
    """``$ML_STACK_DISCOVERY_GROUP`` if set. Read at call time, not import"""
    return os.environ.get("ML_STACK_DISCOVERY_GROUP") or DEFAULT_GROUP


def default_port() -> int:
    raw = os.environ.get("ML_STACK_DISCOVERY_PORT")
    return int(raw) if raw else DEFAULT_PORT

PROTOCOL = 1
MAX_SKEW_S = 60.0
_TOKEN_INFO = b"ml-stack-traind-api-token-v1"


class DiscoveryError(RuntimeError):
    pass


# -- the key -------------------------------------------------------------
def key_path(path: Path | str | None = None) -> Path:
    """Where the cluster key lives. ``$ML_STACK_CLUSTER_KEY`` wins if set."""
    if path is not None:
        return Path(path).expanduser()
    env = os.environ.get("ML_STACK_CLUSTER_KEY")
    return Path(env).expanduser() if env else DEFAULT_KEY_PATH.expanduser()


def create_cluster_key(path: Path | str | None = None, *,
                       overwrite: bool = False) -> str:
    """Mint a cluster key, or return the existing one."""
    p = key_path(path)
    if p.exists() and not overwrite:
        return p.read_text().strip()
    p.parent.mkdir(parents=True, exist_ok=True)
    key = base64.urlsafe_b64encode(secrets.token_bytes(32)).decode().rstrip("=")
    p.write_text(key + "\n")
    p.chmod(0o600)
    return key


# -- joining by password -------------------------------------------------
MIN_PASSPHRASE = 8
"""Shortest passphrase accepted. Low, because a refusal people work around by typing"""

SCRYPT_N = 1 << 16
SCRYPT_R = 8
SCRYPT_P = 1
"""~80ms and 64MB on a laptop, a second or two on a Pi. Paid once, at join time: the"""


def _salt_for(group: str) -> bytes:
    """Deterministic, because both machines have to derive the same key from the same"""
    return sha256(b"ml-stack-cluster-v1:" + group.encode()).digest()


def key_from_passphrase(passphrase: str, *, group: str = "ml-stack") -> bytes:
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
    """Which cluster this machine joined, or None."""
    p = group_path(path)
    if not p.exists():
        return None
    return p.read_text().strip() or None


def join_cluster(passphrase: str, *, group: str = "ml-stack",
                 path: Path | str | None = None, overwrite: bool = True) -> bytes:
    """Derive the key from a passphrase and write it here. Returns the key."""
    key = key_from_passphrase(passphrase, group=group)
    p = key_path(path)
    if p.exists() and not overwrite:
        return load_cluster_key(p) or key
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(key.decode() + "\n")
    p.chmod(0o600)
    gp = group_path(path)
    gp.write_text(group + "\n")
    gp.chmod(0o644)
    return key


def check_passphrase(passphrase: str, *, group: str | None = None,
                     path: Path | str | None = None) -> bool:
    """Whether these words derive the key this machine already holds."""
    key = load_cluster_key(path)
    if key is None:
        return False
    group = group if group is not None else (cluster_group(path) or "ml-stack")
    try:
        candidate = key_from_passphrase(passphrase, group=group)
    except DiscoveryError:
        return False
    return hmac.compare_digest(candidate, key)


def in_cluster(path: Path | str | None = None) -> bool:
    """Whether this machine has joined one. The question a setup wizard opens with."""
    return load_cluster_key(path) is not None


def load_cluster_key(path: Path | str | None = None) -> bytes | None:
    """The key as bytes, or None if this machine is not in a cluster."""
    p = key_path(path)
    if not p.exists():
        return None
    text = p.read_text().strip()
    return text.encode() if text else None


def derive_token(key: bytes) -> str:
    """The traind bearer token both ends compute independently."""
    mac = hmac.new(key, _TOKEN_INFO, sha256).digest()
    return base64.urlsafe_b64encode(mac).decode().rstrip("=")


# -- the wire ------------------------------------------------------------
def _canonical(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()


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

    @property
    def base_url(self) -> str:
        return f"http://{self.host or self.hostname}:{self.port}"

    def public(self) -> dict[str, Any]:
        return {"name": self.name, "port": self.port, "device": self.device,
                "busy": self.busy, "queued": self.queued,
                "slots": self.slots, "free": self.free,
                "hostname": self.hostname, "instance": self.instance}

    @property
    def identity(self) -> str:
        """What distinguishes one daemon from another, address aside."""
        return self.instance or f"{self.hostname}:{self.name}:{self.port}"


def _prefer(existing: Beacon, candidate: Beacon) -> Beacon:
    """Pick which address to keep for a daemon that answered more than once."""
    if candidate.host.startswith("127.") and not existing.host.startswith("127."):
        return candidate
    return existing


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


def _destinations(group: str, port: int) -> list[tuple[str, int]]:
    """Every way to say "anyone out there?" on this link."""
    return [(group, port), ("255.255.255.255", port), ("127.0.0.1", port)]


def _socket(*, broadcast: bool = False, bind: tuple[str, int] | None = None,
            group: str | None = None) -> socket.socket:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    if hasattr(socket, "SO_REUSEPORT"):
        try:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        except OSError:
            pass
    if broadcast:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    # TTL 1: this is a LAN facility. Never let it escape the local segment.
    s.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 1)
    s.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_LOOP, 1)
    ip = primary_ip()
    if ip:
        try:
            s.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_IF,
                         socket.inet_aton(ip))
        except OSError:
            pass
    if bind is not None:
        s.bind(bind)
    if group is not None:
        mreq = struct.pack("4sl", socket.inet_aton(group), socket.INADDR_ANY)
        s.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
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

    # -- lifecycle --
    def start(self, *, wait_s: float = 2.0) -> "Advertiser":
        for target in (self._serve, self._announce_loop):
            t = threading.Thread(target=target, daemon=True,
                                 name=f"advertiser-{target.__name__.strip('_')}")
            t.start()
            self._threads.append(t)
        if not self._ready.wait(wait_s):
            raise DiscoveryError("advertiser did not bind in time")
        if self._error is not None:
            raise DiscoveryError(f"advertiser failed to bind: {self._error}")
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
        for t in self._threads:
            t.join(timeout=2.0)

    def __enter__(self) -> "Advertiser":
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.stop()

    # -- internals --
    def _payload(self, kind: str, nonce: str = "") -> bytes:
        if self.refresh is not None:
            try:
                self.refresh(self.beacon)
            except Exception:                         # noqa: BLE001
                pass
        b = self.beacon.public()
        b["hostname"] = b["hostname"] or socket.gethostname()
        return _sign(self.key, {"v": PROTOCOL, "kind": kind, "t": time.time(),
                                "nonce": nonce, "beacon": b})

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
            except socket.timeout:
                continue
            except OSError:
                break
            msg = _verify(self.key, raw, kind="who")
            if msg is None:
                continue
            try:
                sock.sendto(self._payload("beacon", str(msg.get("nonce", ""))), addr)
            except OSError:
                continue

    def _announce_loop(self) -> None:
        while not self._stop.wait(self.interval_s):
            self.announce()

    def announce(self) -> None:
        """Push an unsolicited beacon. Safe to call at any time."""
        data = self._payload("beacon")
        with _socket(broadcast=True) as s:
            for dest in _destinations(self.group, self.port):
                try:
                    s.sendto(data, dest)
                except OSError:
                    continue


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
        for dest in _destinations(group, port):
            try:
                sock.sendto(query, dest)
            except OSError:
                continue
        deadline = time.time() + timeout_s
        next_query = time.time() + retry_s
        while True:
            now = time.time()
            if now >= deadline:
                break
            if now >= next_query:
                for dest in _destinations(group, port):
                    try:
                        sock.sendto(query, dest)
                    except OSError:
                        continue
                next_query = now + retry_s
            sock.settimeout(max(0.0, min(deadline, next_query) - time.time()))
            try:
                raw, addr = sock.recvfrom(65535)
            except socket.timeout:
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
                                instance=str(body.get("instance", "")))
            except (TypeError, ValueError):
                continue
            key_id = beacon.identity
            prior = found.get(key_id)
            found[key_id] = beacon if prior is None else _prefer(prior, beacon)
    return sorted(found.values(), key=lambda b: (b.name, b.host))
