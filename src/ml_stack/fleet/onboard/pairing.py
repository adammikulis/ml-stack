"""Pairing over the wire: the accepting machine's listener and the asking machine's client.

All over TLS to the accepting machine's self-signed certificate: ``POST /onboard/v1/requests``
(who I am; the owner is told), ``GET .../<id>`` (poll until the owner says yes, which is when
a code first exists, on the owner's screen), then ``POST .../exchange`` and ``.../confirm``
(SPAKE2 from the `spake2` package, both certificate fingerprints as its identities, three tries). The TLS handshake
is deliberately unverified, since the asker has nothing to check it against yet; what
authenticates the certificate is the exchange, which binds the one actually presented to the
code only the two people know. The accepting side reveals nothing until the asker's
confirmation checks out. Design and threat model: docs/onboarding.md.
"""

from __future__ import annotations

import base64
import hashlib
import http.client
import json
import re
import secrets
import ssl
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ml_stack import macauth
from ml_stack.fleet import tls
from ml_stack.fleet.framing import Malformed

from . import pake
from .events import BUS, Bus
from .lan import require_local
from .requests import Refused, Request, Requests, State
from .web import Call, Listener, Reply, json_reply

__all__ = ["DEFAULT_PORT", "DEFAULT_SHARE_PORT", "Grant", "Hooks", "Offer", "PairError",
           "PairingClient", "PairingServer", "context_for", "fingerprint_of"]

DEFAULT_PORT = 8772
DEFAULT_SHARE_PORT = 8773
"""Where ``fleet share`` listens unless told otherwise, and what pairing records for a device that shares."""
MOST_BODY = 16 * 1024
MOST_EXCHANGES = 32
API = "/onboard/v1/requests"
ID = re.compile(r"[0-9a-f]{32}")


class PairError(RuntimeError):
    """Pairing did not complete; ``tries_left`` is how many more the code allows, if known."""

    def __init__(self, message: str, *, status: int = 0, tries_left: int | None = None) -> None:
        super().__init__(message)
        self.status, self.tries_left = status, tries_left


def fingerprint_of(der: bytes) -> str:
    return hashlib.sha256(der).hexdigest()


def context_for(request_id: str, nonce: str) -> bytes:
    """What ties one exchange to one request: neither can be replayed into another."""
    return f"ml-stack-pair/{request_id}/{nonce}".encode()


@dataclass(frozen=True, slots=True)
class Grant:
    """What an accepted machine receives: the cluster it joins (``group``, ``key``, ``salt``
    from `discovery`) and the pinned certificate of the machine that took it in. A grant
    with no ``key`` is a device pairing only, with no cluster membership."""

    group: str = ""
    key: str = ""
    salt: str = ""
    certificate: str = ""
    signing_key: str = ""
    device_secret: str = ""
    """What this device signs its file requests with, so the owner's machine knows which device
    asks (urlsafe base64 of 32 random bytes)."""
    share_port: int = 0
    """The port this machine's ``fleet share`` listens on (0: it shares nothing, so there is no
    share address to record)."""
    name: str = ""
    """What this machine calls itself, for the peer book."""

    def encode(self) -> bytes:
        return json.dumps({"v": 1, "group": self.group, "key": self.key, "salt": self.salt,
                           "certificate": self.certificate, "signing_key": self.signing_key,
                           "device_secret": self.device_secret, "share_port": self.share_port,
                           "name": self.name},
                          sort_keys=True, separators=(",", ":")).encode()

    @classmethod
    def decode(cls, raw: bytes) -> Grant:
        data = json.loads(raw)
        if not isinstance(data, dict) or data.get("v") != 1:
            raise PairError("the grant is in a format this version does not know")
        return cls(**{k: str(data.get(k, "")) for k in
                      ("group", "key", "salt", "certificate", "signing_key", "device_secret",
                       "name")}, share_port=_port(data.get("share_port")))


def _port(value: Any) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) \
        and 0 < value < 65536 else 0


@dataclass(frozen=True, slots=True)
class Offer:
    """What the asking machine tells the accepting one once the code has been proved: how to
    reach its ``fleet share`` and what to check it with. Sent sealed under the exchange's key
    (`pake.Session.seal`), so it is as authentic as the pairing; an offer whose tag does not
    verify is refused and nothing is recorded."""

    share_port: int = 0
    certificate: str = ""
    signing_key: str = ""
    device_secret: str = ""
    """What the accepting machine signs its file requests to this one with."""

    def encode(self) -> bytes:
        return b"offer:" + json.dumps(
            {"v": 1, "share_port": self.share_port, "certificate": self.certificate,
             "signing_key": self.signing_key, "device_secret": self.device_secret},
            sort_keys=True, separators=(",", ":")).encode()

    @classmethod
    def decode(cls, raw: bytes) -> Offer:
        if not raw.startswith(b"offer:"):
            raise PairError("not an offer")
        data = json.loads(raw[len(b"offer:"):])
        if not isinstance(data, dict) or data.get("v") != 1:
            raise PairError("the offer is in a format this version does not know")
        return cls(_port(data.get("share_port")),
                   *(str(data.get(k, "")) for k in ("certificate", "signing_key", "device_secret")))


@dataclass(slots=True)
class Hooks:
    """What the owner's side supplies: the grant for an accepted request, and how to tell
    the owner of a new one (a notifier that raises is reported and the request kept)."""

    grant: Callable[[Request], Grant]
    notify: Callable[[Request], None] | None = None
    learned: Callable[[Request, Offer], None] | None = None
    """Called with the asking machine's `Offer` after it paired, when it sent one that verified."""


class PairingServer:
    """The listener a machine runs while the owner has pairing open."""

    def __init__(self, requests: Requests, ident: tls.Identity, hooks: Hooks, *,
                 address: tuple[str, int] = ("127.0.0.1", DEFAULT_PORT), bus: Bus = BUS) -> None:
        self.requests, self.ident, self.hooks, self.bus = requests, ident, hooks, bus
        self.lockout = macauth.Lockout(failures=20, window_s=60.0, lock_s=120.0)
        self._sessions: dict[str, pake.Session] = {}
        pake.require()
        self._lock = threading.Lock()
        self.listener = Listener(self.dispatch, address, tls.server_context(ident))

    @property
    def port(self) -> int:
        return self.listener.port

    def start(self) -> PairingServer:
        self.listener.start()
        return self

    def stop(self) -> None:
        self.listener.stop()

    def __enter__(self) -> PairingServer:
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.stop()

    # -- routes --
    def dispatch(self, call: Call) -> Reply:
        if not call.secure:
            return json_reply(403, {"error": "pairing needs TLS"})
        if self.lockout.locked(call.client):
            return json_reply(429, {"error": "too many bad requests from this address"})
        try:
            status, body = self._route(call)
        except Refused as why:
            if why.status in (400, 404, 409):
                self.lockout.failed(call.client)
            return json_reply(why.status, {"error": why.reason})
        except (Malformed, ValueError) as bad:
            self.lockout.failed(call.client)
            return json_reply(getattr(bad, "status", 400), {"error": getattr(
                bad, "message", "the body is not JSON")})
        return json_reply(status, body)

    def _route(self, call: Call) -> tuple[int, dict[str, Any]]:
        path = call.path.split("?")[0].rstrip("/")
        if call.method == "POST" and path == API:
            return 202, self.submit(call.json(MOST_BODY), call.client)
        tail = path[len(API) + 1:] if path.startswith(API + "/") else ""
        request_id, _, action = tail.partition("/")
        if not ID.fullmatch(request_id):
            raise Refused(404, "no such request")
        if call.method == "GET" and not action:
            request = self.requests.get(request_id)
            if request is None:
                raise Refused(404, "no such request")
            return 200, {"state": request.state.value}
        if call.method == "POST" and action == "exchange":
            return 200, {"message": self.exchange(request_id,
                                                  str(call.json(MOST_BODY).get("message")))}
        if call.method == "POST" and action == "confirm":
            body = call.json(MOST_BODY)
            sealed = {"offer": str(body["offer"]), "offer_tag": str(body.get("offer_tag") or "")} \
                if body.get("offer") else {}
            return 200, self.confirm(request_id, str(body.get("confirmation")), **sealed)
        raise Refused(404, "no such request")

    def submit(self, info: dict[str, Any], client: str) -> dict[str, Any]:
        request = self.requests.submit(info, client)
        if self.hooks.notify is not None:
            try:
                self.hooks.notify(request)
            except (OSError, RuntimeError, ValueError):   # a broken toast must not drop the ask
                self.bus.emit("onboard.notify.failed", "notice", f"request:{request.id}")
        return {"id": request.id, "state": request.state.value,
                "server": self.ident.fingerprint,
                "expires_in": self.requests.limits.pending_ttl_s}

    def exchange(self, request_id: str, message: str) -> str:
        request = self.requests.take_attempt(request_id)
        try:
            session = pake.start_responder(
                request.code, context=context_for(request.id, request.nonce),
                mine=self.ident.fingerprint, theirs=request.fingerprint)
            session.receive(message)
        except pake.Bad as exc:
            self.requests.wrong(request_id)
            raise Refused(400, f"bad message: {exc}") from None
        with self._lock:
            while len(self._sessions) >= MOST_EXCHANGES:
                self._sessions.pop(next(iter(self._sessions)))
            self._sessions[request_id] = session
        return session.message

    def _offer(self, raw: str, tag: str, session: pake.Session) -> Offer:
        """The asker's offer, if it is sealed under this exchange's key; `Refused` otherwise."""
        try:
            payload = base64.b64decode(raw, validate=True)
            if not session.open(payload, tag):
                raise Refused(403, "the offer does not carry the exchange's tag")
            return Offer.decode(payload)
        except (ValueError, PairError):
            raise Refused(400, "the offer is malformed") from None

    def confirm(self, request_id: str, confirmation: str, offer: str = "",
                offer_tag: str = "") -> dict[str, Any]:
        with self._lock:
            session = self._sessions.pop(request_id, None)
        if session is None:
            raise Refused(409, "no exchange is open for this request")
        request = self.requests.get(request_id)
        if request is None or request.state is not State.ACCEPTED:
            raise Refused(409, "the request is not waiting for a code")
        if not session.check(confirmation):
            left = self.requests.wrong(request_id)
            raise Refused(403, f"wrong code; {left} tries left" if left else
                          "wrong code; the request is closed")
        asker = self._offer(offer, offer_tag, session) if offer else None
        grant = self.hooks.grant(request)
        payload = grant.encode()
        self.requests.paired(request_id, shared_cluster_key=bool(grant.key),
                             secret=grant.device_secret)
        if asker is not None and self.hooks.learned is not None:
            self.hooks.learned(request, asker)
        return {"confirmation": session.confirmation(),
                "grant": base64.b64encode(payload).decode(), "tag": session.seal(payload)}


def unverified_context() -> ssl.SSLContext:
    """A client context that accepts any certificate and so authenticates nothing; used only
    to learn which certificate the other end presents, which the exchange then binds."""
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    ctx.minimum_version = ssl.TLSVersion.TLSv1_2
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


class PairingClient:
    """The asking machine's side. ``fingerprint`` is the certificate its own daemon serves."""

    def __init__(self, host: str, port: int = DEFAULT_PORT, *, fingerprint: str,
                 timeout: float = 10.0) -> None:
        self.host, self.port, self.fingerprint, self.timeout = host, port, fingerprint, timeout
        self.server_fingerprint = ""
        self.request_id = ""
        self.nonce = ""

    def _call(self, method: str, path: str, body: dict[str, Any] | None = None
              ) -> tuple[int, dict[str, Any]]:
        try:
            require_local(self.host, self.port)
        except OSError as exc:
            raise PairError(str(exc)) from None
        conn = http.client.HTTPSConnection(self.host, self.port, context=unverified_context(),
                                           timeout=self.timeout)
        try:
            raw = json.dumps(body).encode() if body is not None else None
            conn.connect()
            seen = fingerprint_of(conn.sock.getpeercert(binary_form=True) or b"")  # type: ignore[union-attr]
            conn.request(method, path, body=raw, headers={"Content-Type": "application/json"})
            response = conn.getresponse()
            data = response.read(MOST_BODY + 1)
        except (OSError, http.client.HTTPException) as exc:
            raise PairError(f"cannot reach {self.host}:{self.port}: {exc}") from None
        finally:
            conn.close()
        if self.server_fingerprint and seen != self.server_fingerprint:
            raise PairError("the machine now presents a different certificate than at the "
                            "start; stopping")
        self.server_fingerprint = self.server_fingerprint or seen
        try:
            parsed = json.loads(data) if data else {}
        except ValueError:
            raise PairError("the machine answered with something that is not JSON") from None
        return response.status, parsed if isinstance(parsed, dict) else {}

    def ask(self, *, name: str, hostname: str, model: str) -> str:
        """Send the request; returns its id."""
        self.nonce = base64.b16encode(secrets.token_bytes(16)).decode().lower()
        status, body = self._call("POST", API, {"name": name, "hostname": hostname,
                                                "model": model, "fingerprint": self.fingerprint,
                                                "nonce": self.nonce})
        if status != 202:
            raise PairError(str(body.get("error", f"refused with status {status}")),
                            status=status)
        self.request_id = str(body["id"])
        if body.get("server") != self.server_fingerprint:
            raise PairError("the machine's certificate is not the one it names")
        return self.request_id

    def state(self) -> str:
        status, body = self._call("GET", f"{API}/{self.request_id}")
        if status != 200:
            raise PairError(str(body.get("error", f"status {status}")), status=status)
        return str(body.get("state", ""))

    def wait(self, timeout_s: float, poll_s: float = 1.0) -> str:
        """Poll until the request is no longer pending; returns its state."""
        end = time.monotonic() + timeout_s
        while True:
            state = self.state()
            if state != State.PENDING.value or time.monotonic() >= end:
                return state
            time.sleep(poll_s)

    def finish(self, code: str, offer: Offer | None = None) -> Grant:
        """Prove the code and receive the grant, sending ``offer`` (how to reach this machine's
        share) sealed under the exchange. `PairError` for a wrong code, with ``tries_left``, or
        for a machine that fails to prove it knew the code too."""
        code = re.sub(r"\s", "", code)
        try:
            session = pake.start_initiator(
                code, context=context_for(self.request_id, self.nonce), mine=self.fingerprint,
                theirs=self.server_fingerprint)
        except pake.PakeUnavailable as exc:
            raise PairError(str(exc)) from None
        status, body = self._call("POST", f"{API}/{self.request_id}/exchange",
                                  {"message": session.message})
        if status != 200:
            raise PairError(str(body.get("error", f"status {status}")), status=status)
        try:
            session.receive(str(body.get("message")))
        except pake.Bad as exc:
            raise PairError(f"bad message from the machine: {exc}") from None
        sent: dict[str, Any] = {"confirmation": session.confirmation()}
        if offer is not None:
            raw = offer.encode()
            sent |= {"offer": base64.b64encode(raw).decode(), "offer_tag": session.seal(raw)}
        status, body = self._call("POST", f"{API}/{self.request_id}/confirm", sent)
        if status != 200:
            message = str(body.get("error", f"status {status}"))
            left = re.search(r"(\d+) tries left", message)
            raise PairError(message, status=status, tries_left=int(left[1]) if left else 0)
        try:
            payload = base64.b64decode(str(body.get("grant")), validate=True)
        except ValueError:
            raise PairError("the grant is not base64") from None
        if not session.check(str(body.get("confirmation"))):
            raise PairError("the machine did not prove it knew the code")
        if not session.open(payload, str(body.get("tag"))):
            raise PairError("the grant does not carry the exchange's tag")
        return Grant.decode(payload)
