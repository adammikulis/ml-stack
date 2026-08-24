"""Find the box with the card, without being told where it is.

    # once, on either machine
    key = create_cluster_key()          # writes ~/.ml-stack/cluster.key

    # on the GPU box (traind does this for you)
    Advertiser(Beacon(name="rtx", port=8770), key).start()

    # anywhere else on the LAN
    for peer in discover(key):
        print(peer.name, peer.base_url, peer.device)

Stdlib only, so this stays importable on a machine that has no torch.

WHY A SHARED KEY AND NOT PLAIN mDNS. The thing being advertised executes
commands it is sent. Announcing it to an open LAN means any device on the
network -- a guest phone, a smart plug, a laptop someone brought home -- can
learn there is a job-runner here, and can equally well *pretend to be one*.
The second half is the dangerous one: a client that trusts an unsigned beacon
can be pointed at an attacker's box and will happily push its dataset there and
hand over its token.

So every packet carries an HMAC over its contents keyed by a secret both ends
already hold. Without the key you cannot be found and you cannot be
impersonated, and the exchange is challenge-response -- the querier picks a
nonce, the reply signs it back -- so a recorded packet cannot be replayed later.

The API bearer token is DERIVED from the same key rather than transmitted (see
``derive_token``). That is what makes this zero-config: two machines holding the
same key independently compute the same token, so nothing secret is ever on the
wire, and there is no token to copy after the key is in place.

TRUST BOUNDARY: possession of the cluster key is full authority over every
daemon in the cluster. It is exactly as sensitive as an ssh private key, and
the file is written 0600 for the same reason.

ONE MORE THING WORTH KNOWING: a peer's address comes from the UDP source
address of its reply, never from a self-reported field in the payload. A signed
beacon claiming "connect to 10.0.0.9" is still a redirect if the signer is
confused or the packet is relayed; the source address is where the thing that
proved it holds the key actually is.
"""

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
    """``$ML_STACK_DISCOVERY_GROUP`` if set. Read at call time, not import
    time, so a test or a systemd unit can set it without re-importing."""
    return os.environ.get("ML_STACK_DISCOVERY_GROUP") or DEFAULT_GROUP


def default_port() -> int:
    raw = os.environ.get("ML_STACK_DISCOVERY_PORT")
    return int(raw) if raw else DEFAULT_PORT

PROTOCOL = 1
#: Replies older than this are refused. Generous enough for clock skew between
#: machines that have never spoken to an NTP server, tight enough to matter.
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
    """Mint a cluster key, or return the existing one.

    Not overwriting by default is the whole safety property: re-running the
    setup command on a machine that is already in the cluster must not silently
    evict every other machine from it.
    """
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
"""Shortest passphrase accepted. Low, because a refusal people work around by typing
"password1" is worse than the length it enforced -- the real defence is the KDF cost
below and the fact that this is a LAN, not the internet."""

SCRYPT_N = 1 << 16
SCRYPT_R = 8
SCRYPT_P = 1
"""~80ms and 64MB on a laptop, a second or two on a Pi. Paid once, at join time: the
daemon derives its bearer token from the *key*, not the passphrase, so nothing on the
hot path pays this again.

The cost is the whole point. A passphrase someone can type is far weaker than 32 random
bytes, and a beacon on the wire is a verifiable target -- anyone on this LAN can capture
one and grind guesses against it offline. scrypt makes each guess expensive in time and
in memory, which is what turns "a word and two digits" from instantly broken into merely
weak."""


def _salt_for(group: str) -> bytes:
    """Deterministic, because both machines have to derive the same key from the same
    words, and they have nothing to exchange first. That is the trade a random salt
    would break, and it is why the group name matters: it is the only thing separating
    two households that both chose "letmein"."""
    return sha256(b"ml-stack-cluster-v1:" + group.encode()).digest()


def key_from_passphrase(passphrase: str, *, group: str = "ml-stack") -> bytes:
    """The cluster key two machines derive from the same words.

    This is what makes joining a cluster something a person can do: type the same
    passphrase on each box and they find each other. Different passphrase, or different
    group, means a different key -- so several clusters share one LAN without seeing
    each other, and no configuration says so. Beacons are signed with this key and a
    beacon that will not verify is simply not answered, so the isolation is the same
    mechanism that keeps a stranger out.
    """
    passphrase = passphrase.strip()
    if len(passphrase) < MIN_PASSPHRASE:
        raise DiscoveryError(
            f"passphrase must be at least {MIN_PASSPHRASE} characters -- everyone on "
            "this network can hear the beacons and grind guesses against them offline")
    # maxmem is not a tuning knob: OpenSSL refuses above 32MB by default and these
    # parameters need exactly that, so leaving it out fails on some builds and not
    # others -- which would look like a corrupt passphrase rather than a library limit.
    raw = hashlib.scrypt(passphrase.encode("utf-8"), salt=_salt_for(group),
                         n=SCRYPT_N, r=SCRYPT_R, p=SCRYPT_P, dklen=32,
                         maxmem=2 * 128 * SCRYPT_N * SCRYPT_R)
    return base64.urlsafe_b64encode(raw).rstrip(b"=")


def join_cluster(passphrase: str, *, group: str = "ml-stack",
                 path: Path | str | None = None, overwrite: bool = True) -> bytes:
    """Derive the key from a passphrase and write it here. Returns the key.

    ``overwrite`` defaults True, unlike ``create_cluster_key``: someone typing a
    passphrase has said which cluster they mean, and silently keeping an old key would
    leave the box in a cluster the person just told it to leave.
    """
    key = key_from_passphrase(passphrase, group=group)
    p = key_path(path)
    if p.exists() and not overwrite:
        return load_cluster_key(p) or key
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(key.decode() + "\n")
    p.chmod(0o600)
    return key


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
    """The traind bearer token both ends compute independently.

    Derived, not transmitted. Two machines holding the key agree on the token
    without it ever crossing the network, which is why adding a peer needs no
    copy-paste of a secret beyond the key itself.
    """
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
    """Parse and authenticate a packet, or return None.

    Returns None rather than raising for every rejection. A discovery socket
    receives whatever else is on the multicast group, and one malformed packet
    from an unrelated service must not take the listener down.
    """
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
    # The reply must sign back the nonce WE chose, so a recording of an earlier
    # exchange cannot be replayed at us as a fresh answer.
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
    #: How many jobs this daemon will run at once, and how many of those are free.
    #: ``busy`` stays on the wire beside them: it is what an older peer reads, and
    #: ``free`` is what a newer one reads, so a mixed-version fleet degrades to the
    #: old meaning rather than to an error. The HMAC covers the whole body, so the
    #: extra keys verify either way.
    slots: int = 1
    free: int = 1
    #: Filled in by the receiver from the packet source address, never trusted
    #: from the payload. Empty on the advertising side.
    host: str = ""
    #: The sender's own idea of its hostname. Display only -- it may be a name
    #: that does not resolve from here, which is half the reason this exists.
    hostname: str = ""
    #: Stable for the life of one daemon process. This, not the address, is
    #: what makes a peer one peer: a machine with several interfaces answers
    #: the same query from each of them, and without an identity to collapse
    #: on, one daemon reachable four ways reads as four GPUs.
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
    """Pick which address to keep for a daemon that answered more than once.

    Loopback wins when it is offered: hearing it means the daemon is on this
    very machine, and the loopback route is both the shortest one and the one
    a host firewall cannot be persuaded to drop.
    """
    if candidate.host.startswith("127.") and not existing.host.startswith("127."):
        return candidate
    return existing


def primary_ip() -> str:
    """This machine's address on the interface that reaches the LAN.

    The connect() is to a UDP socket and sends nothing -- it only asks the
    routing table which local address would be used. That matters on a laptop
    with a VPN up (this one has four utun interfaces): without pinning the
    multicast interface, an announcement can leave down the tunnel and never
    touch the LAN it was meant for.
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except OSError:
        return ""
    finally:
        s.close()


def _destinations(group: str, port: int) -> list[tuple[str, int]]:
    """Every way to say "anyone out there?" on this link.

    All three are needed and none is redundant:

    - **multicast**: the well-behaved path, and the only one that reaches a
      peer whose subnet broadcast is filtered.
    - **limited broadcast**: consumer access points routinely drop multicast
      (IGMP snooping with no querier on the segment eats it), and you find out
      only when discovery mysteriously returns nothing.
    - **loopback**: a daemon on *this* machine. It is also the only path that
      survives a host firewall that drops inbound UDP on real interfaces --
      the macOS Application Firewall does exactly that, and it is on by
      default.
    """
    return [(group, port), ("255.255.255.255", port), ("127.0.0.1", port)]


def _socket(*, broadcast: bool = False, bind: tuple[str, int] | None = None,
            group: str | None = None) -> socket.socket:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    # REUSEPORT so a listener and a querier can coexist on one machine, which
    # is the normal case when you test the cluster from the box that runs it.
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
    """Answers 'who is out there' on behalf of one daemon.

    Both halves are needed. The periodic announcement lets a passive listener
    build a picture without asking, and the reply-to-query half means a client
    that starts up gets an answer in milliseconds instead of waiting out an
    announcement interval.
    """

    def __init__(self, beacon: Beacon, key: bytes, *,
                 group: str | None = None, port: int | None = None,
                 interval_s: float = 10.0,
                 refresh: Callable[[Beacon], None] | None = None) -> None:
        beacon.instance = beacon.instance or secrets.token_hex(8)
        self.beacon = beacon
        self.key = key
        #: Called just before each announcement and each reply, to bring the beacon's
        #: mutable half up to date. Without it ``busy``, ``queued`` and the free-VRAM
        #: reading are whatever they were when the daemon booted -- so a box that has
        #: been training for six hours still advertises itself as idle, and anything
        #: choosing a peer on that basis is reading a constant.
        self.refresh = refresh
        self.group = group or default_group()
        self.port = port if port is not None else default_port()
        self.interval_s = interval_s
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._sock: socket.socket | None = None
        #: Set once the listener is actually bound, so start() can report a
        #: bind failure instead of returning a dead advertiser.
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
                # A probe that raises must not silence the beacon. Advertising a stale
                # answer keeps the box discoverable; advertising nothing removes it from
                # the fleet, which is strictly worse.
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
                # Unicast straight back to the querier's ephemeral port. This is
                # why discover() never has to share this port with us.
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
    """Ask the LAN who is running a daemon, and return everyone who proves it.

    The query is re-sent every ``retry_s`` for the whole window rather than
    asked once. Two things make a single query wrong, and both are ordinary
    rather than exotic: UDP drops packets, so one lost query is a discovery
    that returns nothing at all; and a daemon that finishes binding 50ms into
    the window never hears a question asked before it was listening, which is
    exactly what happens when you start the daemon and look for it in the same
    breath.

    The nonce is held constant across retries, so every reply -- to whichever
    copy of the query arrived -- still proves freshness against the challenge
    this call issued.
    """
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
                # A quiet interval, not the end of the window: go round again
                # and re-ask. Returning here is what made one lost packet look
                # like an empty network.
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
                                # An old beacon carries no "free"; its "busy" flag is
                                # the whole truth it has, so read it as one slot used.
                                free=int(body["free"]) if "free" in body
                                else (0 if body.get("busy") else 1),
                                host=addr[0],
                                hostname=str(body.get("hostname", "")),
                                instance=str(body.get("instance", "")))
            except (TypeError, ValueError):
                continue
            # Keyed on the daemon, not on the route to it. The same daemon
            # answers over multicast, over broadcast, and once per interface it
            # holds; all of that is one peer.
            key_id = beacon.identity
            prior = found.get(key_id)
            found[key_id] = beacon if prior is None else _prefer(prior, beacon)
    return sorted(found.values(), key=lambda b: (b.name, b.host))
