"""Local owner invitation actions and TLS challenge endpoints."""
from __future__ import annotations

import base64
import io
import ipaddress
import json
import ssl
from typing import Any

import qrcode
from qrcode.image.svg import SvgPathFillImage

from .discovery import DiscoveryError, adopt
from .invites import Invitations
from .session import parse_cookie


def store(ui: Any) -> Invitations:
    return ui.invitations


def local(address: str) -> bool:
    try:
        parsed = ipaddress.ip_address(address)
        return parsed.is_loopback or bool(getattr(parsed, "ipv4_mapped", None) and parsed.ipv4_mapped.is_loopback)
    except ValueError:
        return False


def person_session(route: Any) -> bool:
    session = route.ui.sessions.get(parse_cookie(route.cookie))
    return session is not None and session.who in {"passphrase", "ticket", "setup"}


def ui_route(route: Any) -> bool:
    if route.path not in ("/ui/fleet/invites", "/ui/fleet/join-invite", "/ui/fleet/android-devices"):
        return False
    if not local(route.client_ip) or not route.ui.host_ok(route.host_header):
        route.send(403, {"error": "invitation actions require this computer's owner interface"})
        return True
    if not route.path.endswith("/join-invite") and not route.ui.authed(route.cookie):
        route.send(401, {"error": "sign in before inviting an owned computer"})
        return True
    try:
        if route.path.endswith("/android-devices") and not person_session(route):
            route.send(403, {"error": "Android enrollment controls require an owner browser session"})
            return True
        if route.path.endswith("/android-devices") and route.method == "GET":
            route.send(200, {"devices": store(route.ui).active_devices()}, {"Cache-Control": "no-store"})
            return True
        length = int(route.header("Content-Length", "0"))
        if not 0 < length <= 8192:
            raise ValueError("invitation request must be between 1 and 8192 bytes")
        body = route.body()
        if not isinstance(body, dict):
            raise ValueError("a JSON object is required")
        if route.path.endswith("/android-devices"):
            if route.method != "DELETE":
                route.send(405, {"error": "use GET or DELETE"})
                return True
            store(route.ui).revoke(str(body.get("device_id", "")))
            result = {"revoked": True}
        elif route.path.endswith("/invites"):
            if route.method == "POST":
                if body.get("kind") == "android" and not person_session(route):
                    route.send(403, {"error": "Android enrollment requires an owner browser session"})
                    return True
                result = store(route.ui).mint(str(body.get("group", "")), str(body.get("kind", "computer")))
                buffer = io.BytesIO()
                qrcode.make(result["invite"], image_factory=SvgPathFillImage, border=4).save(buffer)
                result["qr"] = "data:image/svg+xml;base64," + base64.b64encode(buffer.getvalue()).decode()
            elif route.method == "DELETE":
                store(route.ui).revoke(str(body.get("id", "")))
                result = {"revoked": True}
            else:
                route.send(405, {"error": "use POST or DELETE"})
                return True
        elif route.method == "POST":
            with route.ui.join_guard():
                result = route.ui.join_invitation(str(body.get("invite", "")))
            session = route.ui.sessions.open("setup", "invite")
            route.ui.record("session.open", who="setup", origin="invite", source=route.client_ip)
            route.send(200, result, {"Set-Cookie": route.ui.sessions.cookie_header(session), "Cache-Control": "no-store"})
            return True
        else:
            route.send(405, {"error": "use POST"})
            return True
        route.send(200, result, {"Cache-Control": "no-store"})
    except (DiscoveryError, ValueError, OSError) as exc:
        route.send(400, {"error": str(exc)})
    return True


def public(ui: Any, handler: Any, raw: bytes | None) -> bool:
    path = handler.path.split("?")[0]
    if path not in ("/join/invite/challenge", "/join/invite/redeem"):
        return False
    if ui is None or not isinstance(handler.connection, ssl.SSLSocket):
        handler._send(403, {"error": "invitation exchange requires TLS"})
        return True
    try:
        if raw is None or not 0 < len(raw) <= 4096:
            raise ValueError("invitation exchange is too large or empty")
        source = ipaddress.ip_address(handler.client_address[0])
        if not source.is_private or source.is_multicast or source.is_unspecified:
            raise ValueError("invitation exchange requires a local network source")
        store(ui).permit(str(source))
        body = json.loads(raw)
        if not isinstance(body, dict):
            raise ValueError("a JSON object is required")
        handler._send(200, store(ui).exchange(path.rsplit("/", 1)[1], body),
                      headers={"Cache-Control": "no-store"})
    except (ValueError, TypeError) as exc:
        handler._send(400, {"error": str(exc)})
    return True


def joined(ui: Any, member: Any) -> dict[str, Any]:
    adopt(member, ui.cluster_key_path)
    ui.rejoined()
    return {"ok": True, "group": member.group}
