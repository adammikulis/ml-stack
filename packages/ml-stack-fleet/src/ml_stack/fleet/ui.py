"""The fleet's web interface: routes, and the guard on first-run setup.

Served by the daemon itself, from every box, because a browser cannot speak UDP and so
cannot discover anything on its own -- whichever daemon you open becomes the front door
and does the discovering for you. It is also one less thing to install and start.

The assets under ``web/`` are data, not code. That is the same call ``contracts/`` makes,
and it is what keeps this package device tier: no framework, no build step, nothing
imported that the standard library does not already have.

**The one genuinely dangerous route is first-run setup.** A daemon that has not joined a
cluster has no credential to check, and the setup route can *create* one -- so whoever
reaches it first owns the machine. Three things close that:

* it is refused from anywhere but loopback until the box has joined, so being on the LAN
  is not enough; you have to be on the machine, or use ssh and the CLI, which already
  does the whole job;
* a ``Host`` allowlist, because loopback-only is not loopback-safe on its own -- a page
  in the owner's browser can post to ``127.0.0.1`` from anywhere on the internet, and
  the ``Host`` header is what tells the two apart;
* a header no cross-origin form can set without a preflight the daemon refuses.

Once the box *has* joined, setup requires a session like everything else. Changing which
cluster a machine belongs to is a thing you may do, but not anonymously.
"""

from __future__ import annotations

import json
import mimetypes
import time
import urllib.parse
from pathlib import Path
from typing import Any

from .discovery import (
    DiscoveryError,
    check_passphrase,
    cluster_group,
    derive_token,
    discover,
    in_cluster,
    join_cluster,
    load_cluster_key,
)
from .session import Sessions, Throttle, parse_cookie

__all__ = ["ASSETS", "UI", "asset_bytes"]

ASSETS = Path(__file__).parent / "web"

UI_HEADER = "X-ML-Stack-UI"
"""Required on every UI request. A cross-origin form, image or link cannot set a custom
header without a preflight, and the daemon answers no preflight -- so this alone stops a
malicious page driving the fleet through a logged-in browser."""

LOOPBACK = {"127.0.0.1", "::1", "localhost"}
DISCOVER_CACHE_S = 3.0
"""Long enough that three panels refreshing do not each fire a multicast sweep, short
enough that a machine appearing shows up while someone is still watching for it."""


def asset_bytes(name: str) -> tuple[bytes, str] | None:
    """One file from ``web/``, by exact name.

    An allowlist built from the directory rather than a path join: there is no traversal
    to get wrong if no caller-supplied string ever reaches the filesystem.
    """
    allowed = {p.name: p for p in ASSETS.iterdir() if p.is_file()} if ASSETS.is_dir() else {}
    path = allowed.get(name)
    if path is None:
        return None
    kind, _ = mimetypes.guess_type(name)
    return path.read_bytes(), kind or "application/octet-stream"


class UI:
    """Route handling for ``/ui/*``. Holds the session store and the login throttle."""

    def __init__(self, *, name: str = "", cluster_key_path: Path | str | None = None,
                 peer_port: int = 8770, setup_token: str = "",
                 on_join: "Any | None" = None) -> None:
        self.name = name
        #: Called after a successful join. A daemon that started before the box was in a
        #: cluster is not advertising -- it had no key to sign a beacon with -- so
        #: without this the machine stays invisible until someone restarts it, and
        #: "set it up and then restart it" is not a setup wizard.
        self.on_join = on_join
        self.cluster_key_path = cluster_key_path
        self.peer_port = peer_port
        self.setup_token = setup_token
        self.sessions = Sessions()
        self.throttle = Throttle()
        self._peers: tuple[float, list[dict[str, Any]]] = (0.0, [])

    # -- guards ----------------------------------------------------------
    def host_ok(self, host_header: str) -> bool:
        host = (host_header or "").rsplit(":", 1)[0].strip("[]").lower()
        return host in LOOPBACK or host == "" or not _looks_like_dns(host)

    def may_setup(self, client_ip: str, host_header: str, token: str) -> str:
        """Empty when first-run setup is allowed from here, else the reason it is not."""
        if not self.host_ok(host_header):
            return ("refused: this request arrived addressed to a hostname rather than "
                    "to the machine itself, which is how a web page tries to reach your "
                    "loopback. Open the address bar and type it yourself.")
        if client_ip in LOOPBACK:
            return ""
        if self.setup_token and token and _same(token, self.setup_token):
            return ""
        return ("refused: set this machine up on the machine itself, or over ssh with "
                "'ml-stack-peers setup'. A daemon that has not joined a cluster has no "
                "password to check, so the first person to reach this page would own "
                "the box. Start it with --setup-from-lan to allow this deliberately.")

    def authed(self, cookie_header: str) -> bool:
        return self.sessions.get(parse_cookie(cookie_header)) is not None

    # -- state -----------------------------------------------------------
    def state(self) -> dict[str, Any]:
        joined = in_cluster(self.cluster_key_path)
        return {"in_cluster": joined, "name": self.name,
                "group": cluster_group(self.cluster_key_path) if joined else None,
                "needs_setup": not joined}

    def peers(self, *, force: bool = False) -> list[dict[str, Any]]:
        """Everyone on the LAN, cached briefly. The browser cannot do this itself."""
        age, cached = self._peers
        if not force and time.time() - age < DISCOVER_CACHE_S:
            return cached
        key = load_cluster_key(self.cluster_key_path)
        if key is None:
            return []
        found = []
        for beacon in discover(key, timeout_s=1.5):
            row = beacon.public()
            row["host"] = beacon.host
            row["base_url"] = beacon.base_url
            row["is_self"] = beacon.name == self.name
            found.append(row)
        found.sort(key=lambda r: (not r["is_self"], r["name"]))
        self._peers = (time.time(), found)
        return found

    # -- actions ---------------------------------------------------------
    def join(self, passphrase: str, group: str, source: str) -> dict[str, Any]:
        """Join a cluster. Costs one scrypt, so it goes through the throttle."""
        held = self.throttle.blocked_for(source)
        if held:
            raise DiscoveryError(f"too many attempts -- wait {held:.0f}s")
        if not self.throttle.acquire():
            raise DiscoveryError("busy deriving another key; try again in a moment")
        try:
            join_cluster(passphrase, group=group or "ml-stack",
                         path=self.cluster_key_path)
        finally:
            self.throttle.release()
        self.throttle.succeeded(source)
        self._peers = (0.0, [])
        if self.on_join is not None:
            try:
                self.on_join()
            except Exception:                         # noqa: BLE001
                # Joining worked; announcing is best effort. A box that is in the
                # cluster but silent can still be reached by address, and will announce
                # on its next start.
                pass
        return self.state()

    def login(self, source: str, *, passphrase: str = "", group: str = "",
              token: str = "", ticket: str = "") -> str | None:
        """A session id, or None. Raises ``DiscoveryError`` when held off or overloaded."""
        if ticket:
            return self.sessions.open("ticket").sid if self.sessions.spend_ticket(ticket) else None

        key = load_cluster_key(self.cluster_key_path)
        if key is None:
            return None

        if token:
            # No scrypt, so no throttle slot needed -- but a wrong token still counts
            # toward backoff, or the token path becomes the way around it.
            if _same(token, derive_token(key)):
                self.throttle.succeeded(source)
                return self.sessions.open("token").sid
            self.throttle.failed(source)
            return None

        held = self.throttle.blocked_for(source)
        if held:
            raise DiscoveryError(f"too many attempts -- wait {held:.0f}s")
        if not self.throttle.acquire():
            raise DiscoveryError("busy checking another passphrase; try again")
        try:
            ok = check_passphrase(passphrase, group=group or None,
                                  path=self.cluster_key_path)
        finally:
            self.throttle.release()
        if not ok:
            self.throttle.failed(source)
            return None
        self.throttle.succeeded(source)
        return self.sessions.open("passphrase").sid


def _same(a: str, b: str) -> bool:
    import hmac
    return hmac.compare_digest(a.strip(), b.strip())


def _looks_like_dns(host: str) -> bool:
    """Whether a Host header names something that had to be resolved.

    An IP literal or a bare machine name is fine. A dotted name is the rebinding shape:
    a domain whose A record points at 127.0.0.1, so a page anywhere can post to a daemon
    it has no business reaching.
    """
    if not host or host.replace(".", "").isdigit():
        return False
    return "." in host and not host.endswith(".local")


def routes(ui: UI, handler: Any) -> bool:
    """Handle one ``/ui/*`` request. Returns True when it did.

    Takes the handler rather than living inside it so the logic above stays testable
    without a socket, which is most of why this module exists at all.
    """
    parsed = urllib.parse.urlparse(handler.path)
    path = parsed.path
    if not path.startswith("/ui"):
        return False

    client_ip = handler.client_address[0]
    host_header = handler.headers.get("Host", "")
    has_ui_header = bool(handler.headers.get(UI_HEADER))
    cookie = handler.headers.get("Cookie", "")
    method = handler.command

    def send(code: int, payload: Any, extra: dict[str, str] | None = None) -> None:
        handler._send(code, payload, headers=extra)

    def body() -> dict[str, Any]:
        length = int(handler.headers.get("Content-Length", "0"))
        try:
            return json.loads(handler.rfile.read(length) or b"{}")
        except ValueError:
            return {}

    # -- static ----------------------------------------------------------
    if path in ("/ui", "/ui/"):
        asset = asset_bytes("index.html")
        if asset is None:
            send(500, {"error": "the UI assets are missing from this install"})
            return True
        raw, kind = asset
        handler._send(200, None, raw=raw, content_type=kind)
        return True

    if path.startswith("/ui/static/"):
        asset = asset_bytes(path[len("/ui/static/"):])
        if asset is None:
            send(404, {"error": "no such asset"})
            return True
        raw, kind = asset
        handler._send(200, None, raw=raw, content_type=kind,
                      headers={"Cache-Control": "no-cache"})
        return True

    # Everything below is API, and none of it is reachable from a cross-origin page.
    if not has_ui_header:
        send(403, {"error": f"{UI_HEADER} header required"})
        return True

    # -- setup -----------------------------------------------------------
    if path == "/ui/setup" and method == "GET":
        send(200, ui.state())
        return True

    if path == "/ui/setup/join" and method == "POST":
        joined = in_cluster(ui.cluster_key_path)
        if joined and not ui.authed(cookie):
            send(401, {"error": "this machine is already in a cluster -- sign in to "
                                "move it to another one"})
            return True
        if not joined:
            why = ui.may_setup(client_ip, host_header,
                               handler.headers.get("X-ML-Stack-Setup", ""))
            if why:
                send(403, {"error": why})
                return True
        req = body()
        try:
            state = ui.join(str(req.get("passphrase") or ""),
                            str(req.get("group") or ""), client_ip)
        except DiscoveryError as exc:
            send(429 if "attempts" in str(exc) or "busy" in str(exc) else 400,
                 {"error": str(exc)})
            return True
        send(200, state)
        return True

    if path == "/ui/setup/peers" and method == "GET":
        send(200, {"peers": ui.peers(force=True)})
        return True

    # -- session ---------------------------------------------------------
    if path == "/ui/session":
        if method == "GET":
            session = ui.sessions.get(parse_cookie(cookie))
            send(200, {"signed_in": session is not None,
                       "expires_at": session.expires_at if session else None})
            return True
        if method == "DELETE":
            ui.sessions.close(parse_cookie(cookie))
            send(200, {"signed_in": False},
                 {"Set-Cookie": ui.sessions.clear_header()})
            return True
        if method == "POST":
            req = body()
            try:
                sid = ui.login(client_ip,
                               passphrase=str(req.get("passphrase") or ""),
                               group=str(req.get("group") or ""),
                               token=str(req.get("token") or ""),
                               ticket=str(req.get("ticket") or ""))
            except DiscoveryError as exc:
                send(429, {"error": str(exc)})
                return True
            if sid is None:
                send(401, {"error": "that passphrase does not match this cluster"})
                return True
            session = ui.sessions.get(sid)
            send(200, {"signed_in": True, "expires_at": session.expires_at},
                 {"Set-Cookie": ui.sessions.cookie_header(session)})
            return True

    # -- everything else needs a session ---------------------------------
    if not ui.authed(cookie):
        send(401, {"error": "sign in first"})
        return True

    if path == "/ui/peers" and method == "GET":
        send(200, {"peers": ui.peers(), "self": ui.name,
                   "group": cluster_group(ui.cluster_key_path)})
        return True

    send(404, {"error": "no such route"})
    return True
