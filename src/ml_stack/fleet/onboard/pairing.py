"""Pairing over the wire: the accepting machine's listener and the asking machine's client.

The conversation, all over TLS to the accepting machine's self-signed certificate:

1. ``POST /onboard/v1/requests`` -- "this is who I am" (name, hostname, model, the
   fingerprint of the certificate my own daemon will serve). Answered 202 with an id. The
   owner is told; nothing secret exists yet.
2. ``GET /onboard/v1/requests/<id>`` -- poll until the owner has said yes. Only then does a
   code exist, on the owner's screen, and the asking machine's person types it in.
3. ``POST .../exchange`` and ``POST .../confirm`` -- SPAKE2 (`spake.py`) with both
   certificate fingerprints in the transcript. A wrong code, or a machine in the middle
   presenting its own certificate, fails the confirmation. The accepting side reveals nothing
   until the asking side's confirmation checks out, and each exchange spends one of three
   tries.
4. The reply to a good confirmation carries the accepting side's own confirmation and the
   grant (see `Grant`), tagged under the exchange key.

The TLS handshake here is deliberately unverified: the asking machine has nothing to verify
the certificate against yet. What authenticates it is step 3, which binds the fingerprint
that was actually presented to the code only the two people know.
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
from http.server import BaseHTTPRequestHandler
from typing import Any

from ml_stack import macauth
from ml_stack.fleet import tls
from ml_stack.fleet.framing import Limited, LimitedServer, Malformed, read_body

from . import spake
from .events import BUS, Bus
from .requests import Refused, Request, Requests, State

__all__ = ["DEFAULT_PORT", "Grant", "PairError", "PairingClient", "PairingServer",
           "context_for", "fingerprint_of"]

DEFAULT_PORT = 8772
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

    def encode(self) -> bytes:
        return json.dumps({"v": 1, "group": self.group, "key": self.key, "salt": self.salt,
                           "certificate": self.certificate, "signing_key": self.signing_key},
                          sort_keys=True, separators=(",", ":")).encode()

    @classmethod
    def decode(cls, raw: bytes) -> Grant:
        data = json.loads(raw)
        if not isinstance(data, dict) or data.get("v") != 1:
            raise PairError("the grant is in a format this version does not know")
        return cls(**{k: str(data.get(k, "")) for k in
                      ("group", "key", "salt", "certificate", "signing_key")})


class PairingServer:
    """The listener a machine runs while the owner has pairing open."""

    def __init__(self, requests: Requests, ident: tls.Identity, *,
                 grant: Callable[[Request], Grant], notify: Callable[[Request], None] | None = None,
                 host: str = "127.0.0.1", port: int = DEFAULT_PORT, bus: Bus = BUS,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.requests, self.ident, self.grant, self.notify = requests, ident, grant, notify
        self.bus = bus
        self.lockout = macauth.Lockout(failures=20, window_s=60.0, lock_s=120.0, clock=clock)
        self._sessions: dict[str, spake.Session] = {}
        self._lock = threading.Lock()
        self.httpd = LimitedServer((host, port), _handler(self), tls=tls.server_context(ident))
        self._thread: threading.Thread | None = None

    @property
    def port(self) -> int:
        return int(self.httpd.server_address[1])

    def start(self) -> PairingServer:
        self._thread = threading.Thread(target=self.httpd.serve_forever, daemon=True,
                                        name="onboard-pairing")
        self._thread.start()
        return self

    def stop(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()

    def __enter__(self) -> PairingServer:
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.stop()

    # -- what the handler asks of the server --
    def exchange(self, request_id: str, message: str) -> str:
        request = self.requests.take_attempt(request_id)
        try:
            session = spake.start_responder(
                request.code, context=context_for(request.id, request.nonce),
                mine=self.ident.fingerprint, theirs=request.fingerprint)
            session.receive(message)
        except spake.Bad as exc:
            self.requests.wrong(request_id)
            raise Refused(400, f"bad message: {exc}") from None
        with self._lock:
            while len(self._sessions) >= MOST_EXCHANGES:
                self._sessions.pop(next(iter(self._sessions)))
            self._sessions[request_id] = session
        return session.message

    def confirm(self, request_id: str, confirmation: str) -> dict[str, Any]:
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
        grant = self.grant(request)
        payload = grant.encode()
        self.requests.paired(request_id, shared_cluster_key=bool(grant.key))
        return {"confirmation": session.confirmation(),
                "grant": base64.b64encode(payload).decode(), "tag": session.seal(payload)}


def _handler(server: PairingServer) -> type[BaseHTTPRequestHandler]:
    class Handler(Limited, BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *_args: object) -> None:
            return

        def _reply(self, status: int, body: dict[str, Any]) -> None:
            raw = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(raw)

        def _json(self) -> dict[str, Any]:
            data = json.loads(read_body(self, MOST_BODY) or b"{}")
            if not isinstance(data, dict):
                raise Malformed(400, "the body is a JSON object")
            return data

        def _route(self, method: str) -> None:
            who = self.client_address[0]
            if not isinstance(self.connection, ssl.SSLSocket):
                self._reply(403, {"error": "pairing needs TLS"})
                return
            if server.lockout.locked(who):
                self._reply(429, {"error": "too many bad requests from this address"})
                return
            parts = self.path.split("?")[0].rstrip("/")
            try:
                status, body = self._dispatch(method, parts, who)
            except Refused as why:
                if why.status in (400, 404, 409):
                    server.lockout.failed(who)
                self._reply(why.status, {"error": why.reason})
                return
            except Malformed as bad:
                server.lockout.failed(who)
                self._reply(bad.status, {"error": bad.message})
                return
            except (ValueError, json.JSONDecodeError):
                server.lockout.failed(who)
                self._reply(400, {"error": "the body is not JSON"})
                return
            self._reply(status, body)

        def _dispatch(self, method: str, path: str, who: str) -> tuple[int, dict[str, Any]]:
            if method == "POST" and path == API:
                request = server.requests.submit(self._json(), who)
                if server.notify is not None:
                    try:
                        server.notify(request)
                    except Exception:  # noqa: BLE001  # a broken toast must not drop the ask
                        server.bus.emit("onboard.notify.failed", "notice",
                                        f"request:{request.id}")
                return 202, {"id": request.id, "state": request.state.value,
                             "server": server.ident.fingerprint,
                             "expires_in": server.requests.limits.pending_ttl_s}
            tail = path[len(API) + 1:] if path.startswith(API + "/") else ""
            request_id, _, action = tail.partition("/")
            if not ID.fullmatch(request_id):
                raise Refused(404, "no such request")
            if method == "GET" and not action:
                request = server.requests.get(request_id)
                if request is None:
                    raise Refused(404, "no such request")
                return 200, {"state": request.state.value}
            if method == "POST" and action == "exchange":
                message = self._json().get("message")
                return 200, {"message": server.exchange(request_id, str(message))}
            if method == "POST" and action == "confirm":
                return 200, server.confirm(request_id, str(self._json().get("confirmation")))
            raise Refused(404, "no such request")

        def do_GET(self) -> None:
            self._route("GET")

        def do_POST(self) -> None:
            self._route("POST")

    return Handler


def unverified_context() -> ssl.SSLContext:
    """A client context that accepts any certificate and so authenticates nothing; used only
    to learn which certificate the other end presents, which the exchange then binds."""
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    ctx.minimum_version = ssl.TLSVersion.TLSv1_2
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE  # noqa: S504
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

    def finish(self, code: str) -> Grant:
        """Prove the code and receive the grant. `PairError` for a wrong code, with
        ``tries_left``, or for a machine that fails to prove it knew the code too."""
        code = re.sub(r"\s", "", code)
        session = spake.start_initiator(
            code, context=context_for(self.request_id, self.nonce), mine=self.fingerprint,
            theirs=self.server_fingerprint)
        status, body = self._call("POST", f"{API}/{self.request_id}/exchange",
                                  {"message": session.message})
        if status != 200:
            raise PairError(str(body.get("error", f"status {status}")), status=status)
        try:
            session.receive(str(body.get("message")))
        except spake.Bad as exc:
            raise PairError(f"bad message from the machine: {exc}") from None
        status, body = self._call("POST", f"{API}/{self.request_id}/confirm",
                                  {"confirmation": session.confirmation()})
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
