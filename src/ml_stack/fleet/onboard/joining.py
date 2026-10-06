"""Authenticated passphrase exchange for existing clusters and explicit cluster creation."""

from __future__ import annotations

import base64
import hashlib
import hmac
import http.client
import json
import re
import secrets
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ml_stack import keystore, macauth, sealing
from ml_stack.log import warn

from .. import discovery as disc, recovery
from ..discovery import DiscoveryError, Membership
from . import pake
from .lan import require_local
from .pairing import fingerprint_of, unverified_context

__all__ = [
    "API",
    "Declined",
    "Joiner",
    "Joining",
    "Offer",
    "cluster_action",
    "create_by_passphrase",
    "find_clusters",
    "find_joiners",
    "join_by_passphrase",
    "join_existing",
    "join_secret",
    "matches",
]

API = "/join/v1"
CLIENT = "joiner"
"""The identity the new machine gives the exchange; it has no certificate."""
PLAIN = "plain"
"""The daemon's identity when it serves without TLS."""
MOST_PENDING = 16
PENDING_S = 30.0
MOST_BODY = 4096


def join_secret(passphrase: str, group: str) -> str:
    """The password the join handshake runs on: scrypt of the passphrase under the cluster's name."""
    salt = hashlib.sha256(b"ml-stack-join-v1/" + group.encode()).digest()
    return base64.urlsafe_b64encode(keystore.scrypt_key(passphrase.strip(), salt)).decode().rstrip("=")


def matches(passphrase: str, group: str = "", path: Path | str | None = None) -> bool:
    """Whether ``passphrase`` is the one cluster ``group`` (default: the first) was joined with."""
    rows = disc.memberships(path)
    held = next((m for m in rows if not group or m.group == group), None)
    if held is None or not held.join:
        return False
    return hmac.compare_digest(join_secret(passphrase, held.group), held.join)


class Refusal(Exception):
    """A status and a sentence for the machine that asked."""

    def __init__(self, status: int, reason: str) -> None:
        super().__init__(reason)
        self.status, self.reason = status, reason


@dataclass(slots=True)
class _Pending:
    session: pake.Session
    group: str
    context: bytes
    source: str
    expires: float


def _context(group: str, nonce: str) -> bytes:
    return f"ml-stack-join/{group}/{nonce}".encode()


class Joining:
    """What a daemon in a cluster answers a machine that asks to join, from the clusters it holds
    (``groups()``) and its own certificate's fingerprint."""

    def __init__(self, groups: Callable[[], list[Membership]],
                 fingerprint: Callable[[], str] = lambda: PLAIN,
                 log: Callable[[str], None] = warn) -> None:
        self.groups, self.fingerprint, self.log = groups, fingerprint, log
        self.lockout = macauth.Lockout(failures=5, window_s=600.0, lock_s=600.0)
        self.everyone = macauth.Lockout(failures=30, window_s=600.0, lock_s=600.0)
        self.attempts: deque[tuple[float, str, str, str]] = deque(maxlen=200)
        """``(time, source, group, outcome)`` of every attempt, newest last."""
        self._pending: dict[str, _Pending] = {}
        self._lock = threading.Lock()

    def handle(self, path: str, body: dict[str, Any], source: str) -> tuple[int, dict[str, Any]]:
        """The status and body answering ``POST`` to ``path`` under `API`."""
        try:
            if path == f"{API}/automatic":
                return 200, self._automatic(body, source)
            if path == f"{API}/start":
                return 200, self._start(body, source)
            if path == f"{API}/finish":
                return 200, self._finish(body, source)
        except Refusal as why:
            return why.status, {"error": why.reason}
        return 404, {"error": "no such route"}

    def _automatic(self, body: dict[str, Any], source: str) -> dict[str, Any]:
        group = disc.require_name(body.get("group"))
        member = next((row for row in self.groups() if row.group == group), None)
        if member is None or member.mode != "dev":
            raise Refusal(403, "this cluster requires explicit admission")
        if self.fingerprint() == PLAIN:
            raise Refusal(403, "automatic cluster admission requires TLS")
        if self.lockout.locked(source) or self.everyone.locked("automatic"):
            raise Refusal(429, "too many automatic joins; retry shortly")
        nonce = body.get("nonce")
        if not isinstance(nonce, str) or not re.fullmatch("[0-9a-f]{32}", nonce):
            raise Refusal(400, "automatic join needs a request nonce")
        self.lockout.failed(source)
        self.everyone.failed("automatic")
        self._note(source, group, "joined development cluster")
        return {"group": group, "key": member.key.decode(), "mode": member.mode,
                "cluster_id": hashlib.sha256(member.key).hexdigest(), "nonce": nonce}

    def _note(self, source: str, group: str, outcome: str) -> None:
        self.attempts.append((time.time(), source, group, outcome))
        self.log(f"join: {outcome} from {source} for cluster '{group}'")

    def _start(self, body: dict[str, Any], source: str) -> dict[str, Any]:
        try:
            group = disc.require_name(body.get("group"))
        except DiscoveryError as exc:
            raise Refusal(400, str(exc)) from None
        if self.lockout.locked(source) or self.everyone.locked("*"):
            self._note(source, group, "refused, too many attempts")
            raise Refusal(429, "too many attempts; try again later")
        held = next((m for m in self.groups() if m.group == group), None)
        words = held.join if held is not None and held.join else None
        if words is None:
            self._note(source, group, "refused, this machine cannot take machines in")
            raise Refusal(404, "this machine cannot take a machine into that cluster")
        self.lockout.failed(source)
        self.everyone.failed("*")
        sid, context = secrets.token_hex(16), _context(group, str(body.get("nonce"))[:64])
        try:
            session = pake.start_responder(words, context=context,
                                           mine=self.fingerprint(), theirs=CLIENT)
            session.receive(str(body.get("message")))
        except pake.Bad as exc:
            self._note(source, group, "refused, bad message")
            raise Refusal(400, f"bad message: {exc}") from None
        now = time.monotonic()
        with self._lock:
            for old in [k for k, v in self._pending.items() if v.expires < now]:
                del self._pending[old]
            if len(self._pending) >= MOST_PENDING:
                raise Refusal(429, "too many joins at once; try again in a moment")
            self._pending[sid] = _Pending(session, group, context, source, now + PENDING_S)
        self._note(source, group, "started")
        return {"id": sid, "message": session.message}

    def _finish(self, body: dict[str, Any], source: str) -> dict[str, Any]:
        with self._lock:
            held = self._pending.pop(str(body.get("id")), None)
        if held is None or held.expires < time.monotonic() or held.source != source:
            raise Refusal(409, "no join is open under that id")
        if not held.session.check(str(body.get("confirmation"))):
            self._note(source, held.group, "refused, wrong passphrase")
            raise Refusal(403, "that passphrase was not accepted")
        member = next((m for m in self.groups() if m.group == held.group), None)
        if member is None:
            raise Refusal(404, "this machine left that cluster")
        self.lockout.passed(source)
        payload = json.dumps({"group": member.group, "key": member.key.decode()}).encode()
        sealed = sealing.seal(held.session.key("join"), payload, held.context)
        self._note(source, held.group, "joined")
        return {"confirmation": held.session.confirmation(), "sealed": sealed.hex()}


# -- the new machine -------------------------------------------------------------
class Declined(DiscoveryError):
    """A daemon answered and would not take this machine in; ``status`` says how."""

    def __init__(self, status: int, reason: str) -> None:
        super().__init__(reason)
        self.status = status


@dataclass(frozen=True, slots=True)
class Joiner:
    """A daemon that answered a request to join, and where to shake hands with it."""

    host: str
    port: int
    tls: bool


def _ask_join(group: str, *, timeout_s: float, port: int | None, most: int
              ) -> list[tuple[str, dict[str, Any]]]:
    """``(address, answer)`` for each daemon that answers a ``join?`` for ``group`` (every cluster when empty)."""
    nonce = secrets.token_hex(16)
    ask = json.dumps({"v": disc.PROTOCOL, "kind": "join?", "group": group, "nonce": nonce}).encode()
    found: dict[tuple[str, int, str], tuple[str, dict[str, Any]]] = {}
    where, port_ = disc.default_group(), port if port is not None else disc.default_port()
    with disc._socket(broadcast=True, bind=("", 0)) as sock:
        disc._say(sock, ask, where, port_)
        deadline = time.time() + timeout_s
        resent = time.time() + 0.3
        while time.time() < deadline and len(found) < most:
            if time.time() >= resent:
                disc._say(sock, ask, where, port_)
                resent = time.time() + 0.3
            sock.settimeout(max(0.01, min(deadline, resent) - time.time()))
            try:
                raw, addr = sock.recvfrom(65535)
                if len(raw) > 2048:
                    continue
                said = json.loads(raw)
                if not isinstance(said, dict) or said.get("v") != disc.PROTOCOL:
                    continue
                theirs = said["port"]
                if type(theirs) is not int or not 1 <= theirs <= 65535:
                    continue
                disc.require_name(said.get("group"))
            except (TimeoutError, KeyError, TypeError, ValueError, DiscoveryError):
                continue
            except OSError:
                break
            if isinstance(said, dict) and said.get("kind") == "join" and said.get("nonce") == nonce \
                    and (not group or said.get("group") == group):
                found.setdefault((addr[0], theirs, str(said.get("group"))), (addr[0], said))
    return list(found.values())


def find_joiners(group: str, *, timeout_s: float = 1.5, port: int | None = None,
                 most: int = 8) -> list[Joiner]:
    """Daemons on this network in cluster ``group`` that offer to take a machine in."""
    return [Joiner(host, int(said["port"]), bool(said.get("tls")))
            for host, said in _ask_join(group, timeout_s=timeout_s, port=port, most=most)]


@dataclass(frozen=True, slots=True)
class Offer:
    """A daemon that says it takes machines into the cluster ``group``."""

    group: str
    machine: str
    host: str
    method: str = "passphrase"


def find_clusters(*, timeout_s: float = 1.5, port: int | None = None, most: int = 64) -> list[Offer]:
    """The clusters daemons on this network offer to take a machine into, one `Offer` per daemon and cluster."""
    offers = (Offer(str(said.get("group") or ""), str(said.get("name") or host), host,
                    "recovery" if said.get("method") == "recovery" else "passphrase")
              for host, said in _ask_join("", timeout_s=timeout_s, port=port, most=most)
              if said.get("group"))
    return list({(o.group, o.machine): o for o in offers}.values())


class _Call:
    """One daemon's join routes. The certificate it presents is what the exchange binds, so it
    is read once and every later call must see the same one."""

    def __init__(self, joiner: Joiner, timeout: float) -> None:
        self.joiner, self.timeout = joiner, timeout
        self.seen = ""

    def _connect(self) -> http.client.HTTPConnection:
        j = self.joiner
        require_local(j.host, j.port)
        conn: http.client.HTTPConnection = (
            http.client.HTTPSConnection(j.host, j.port, context=unverified_context(), timeout=self.timeout)
            if j.tls else http.client.HTTPConnection(j.host, j.port, timeout=self.timeout))
        conn.connect()
        seen = fingerprint_of(conn.sock.getpeercert(binary_form=True) or b"") if j.tls else PLAIN  # type: ignore[union-attr]
        if self.seen and seen != self.seen:
            conn.close()
            raise DiscoveryError("the machine changed its certificate during the join")
        self.seen = seen
        return conn

    def fingerprint(self) -> str:
        try:
            self._connect().close()
        except (OSError, http.client.HTTPException) as exc:
            raise DiscoveryError(f"cannot reach {self.joiner.host}:{self.joiner.port}: {exc}") from None
        return self.seen

    def post(self, step: str, body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        try:
            conn = self._connect()
            try:
                conn.request("POST", f"{API}/{step}", body=json.dumps(body).encode(),
                             headers={"Content-Type": "application/json"})
                response = conn.getresponse()
                data = response.read(MOST_BODY)
            finally:
                conn.close()
        except (OSError, http.client.HTTPException) as exc:
            raise DiscoveryError(f"cannot reach {self.joiner.host}:{self.joiner.port}: {exc}") from None
        try:
            parsed = json.loads(data) if data else {}
        except ValueError:
            parsed = {}
        return response.status, parsed if isinstance(parsed, dict) else {}


def _shake(joiner: Joiner, group: str, words: str, timeout: float) -> bytes:
    """The cluster key ``joiner`` gives a machine that knows ``words``; `Declined` when it will not."""
    call = _Call(joiner, timeout)
    context = _context(group, secrets.token_hex(16))
    try:
        session = pake.start_initiator(words, context=context, mine=CLIENT, theirs=call.fingerprint())
    except pake.PakeUnavailable as exc:
        raise DiscoveryError(str(exc)) from None
    status, got = call.post("start", {"group": group, "message": session.message,
                                      "nonce": context.rsplit(b"/", 1)[1].decode()})
    if status != 200:
        raise Declined(status, str(got.get("error", f"status {status}")))
    try:
        session.receive(str(got.get("message")))
    except pake.Bad as exc:
        raise Declined(400, f"bad message from the machine: {exc}") from None
    status, done = call.post("finish", {"id": got.get("id"), "confirmation": session.confirmation()})
    if status != 200:
        raise Declined(status, str(done.get("error", f"status {status}")))
    if not session.check(str(done.get("confirmation"))):
        raise Declined(403, "the machine did not prove it knew the passphrase")
    try:
        held = recovery.parse_recovery(sealing.open_(
            session.key("join"), bytes.fromhex(str(done.get("sealed"))), context).decode())
        if held.group != group:
            raise DiscoveryError("the machine answered for a different cluster")
        key = held.key
    except (sealing.SealError, ValueError, KeyError, TypeError, DiscoveryError):
        raise Declined(400, "the machine's answer did not authenticate") from None
    return key


def create_by_passphrase(passphrase: str, group: str,
                         path: Path | str | None = None, *, port: int | None = None) -> Membership:
    """Create a named cluster with a passphrase."""
    group = disc.require_name(group)
    if any(member.group == group for member in disc.memberships(path)):
        raise DiscoveryError(f"This machine already belongs to '{group}'. Leave it before creating another.")
    words = disc.check_length(passphrase)
    if find_joiners(group, port=port):
        raise DiscoveryError(f"A cluster named '{group}' is on this network. Join it instead.")
    return disc.mint_cluster(group, path, join=join_secret(words, group))


def cluster_action(mode: str, passphrase: str, group: str,
                   path: Path | str | None = None, *, port: int | None = None) -> Membership:
    """Join an existing cluster or create a new one."""
    if mode == "join":
        return join_existing(passphrase, group, path, port=port)
    if mode == "create":
        return create_by_passphrase(passphrase, group, path, port=port)
    raise DiscoveryError("Choose Join existing cluster or Create new cluster.")


def join_by_passphrase(passphrase: str, group: str,
                       path: Path | str | None = None, *, timeout_s: float = 1.5,
                       port: int | None = None) -> Membership:
    """Join a named cluster or create it when no daemon answers."""
    group = disc.require_name(group)
    secret = join_secret(disc.check_length(passphrase), group)
    joiners = find_joiners(group, timeout_s=timeout_s, port=port)
    if not joiners:
        held = next((m for m in disc.memberships(path) if m.group == group), None)
        return held or disc.mint_cluster(group, path, join=secret)
    return _accept(joiners, group, secret, path)


def join_existing(passphrase: str, group: str, path: Path | str | None = None, *,
                  timeout_s: float = 1.5, port: int | None = None) -> Membership:
    """Join a live cluster without creating a replacement when it disappears."""
    group = disc.require_name(group)
    secret = join_secret(disc.check_length(passphrase), group)
    joiners = find_joiners(group, timeout_s=timeout_s, port=port)
    if not joiners:
        raise DiscoveryError(f"No machine in '{group}' answered on this network; "
                             "cluster is no longer available; refresh nearby clusters.")
    return _accept(joiners, group, secret, path)


def _accept(joiners: list[Joiner], group: str, secret: str,
            path: Path | str | None) -> Membership:
    refused: list[Declined] = []
    for one in joiners:
        try:
            return disc.adopt(Membership(group=group, key=_shake(one, group, secret, 10.0), join=secret), path)
        except Declined as why:
            refused.append(why)
        except DiscoveryError as why:
            refused.append(Declined(0, str(why)))
    if any(w.status == 403 for w in refused):
        raise DiscoveryError(f"{len(joiners)} cluster(s) on this network answered and none "
                             "accepts that passphrase")
    reason = next((why.args[0] for why in refused if why.status),
                  refused[0].args[0] if refused else "no machine took this one in")
    raise DiscoveryError(reason)
