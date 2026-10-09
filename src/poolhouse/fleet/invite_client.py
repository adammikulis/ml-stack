"""Certificate-pinned redemption of invitations to owned computers."""
from __future__ import annotations

import base64
import hashlib
import hmac
import http.client
import ipaddress
import json
import re
import socket
import ssl
import time
from typing import Any
from urllib.parse import urlsplit

from .discovery import Membership
from .invites import decode, encode, proof
from .onboard.lan import require_local
from .recovery import parse_recovery
from .tls import local

LIMIT = 16384


def _object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate invitation field")
        result[key] = value
    return result


def _json(raw: bytes) -> dict[str, Any]:
    value = json.loads(raw.decode("utf-8"), object_pairs_hook=_object)
    if not isinstance(value, dict):
        raise ValueError("the invitation response is not an object")
    return value


def _addresses(endpoint: str) -> tuple[Any, list[Any]]:
    parts = urlsplit(endpoint)
    if (parts.scheme != "https" or not parts.hostname or parts.username is not None
            or parts.password is not None or parts.path not in ("", "/")
            or parts.query or parts.fragment or not parts.port
            or any(ord(char) <= 32 or ord(char) == 127 for char in endpoint)):
        raise ValueError("the invitation must name one HTTPS device endpoint")
    addresses = socket.getaddrinfo(parts.hostname, parts.port, type=socket.SOCK_STREAM)
    if not addresses:
        raise ValueError("the invitation device does not resolve")
    for _, _, _, _, address in addresses:
        ip = ipaddress.ip_address(address[0].split("%")[0])
        require_local(str(ip), parts.port)
        if ip.is_loopback or ip.is_unspecified or ip.is_multicast:
            raise ValueError("the invitation must name another device on the network")
    return parts, addresses


def parse_invite(text: str, now: float | None = None) -> dict[str, Any]:
    """Return the validated fields of a current computer invitation."""
    prefix = "poolhouse://enroll?data="
    if not isinstance(text, str) or len(text) > 8192 or not text.startswith(prefix):
        raise ValueError("paste a cluster invitation created on your other computer")
    encoded = text[len(prefix):]
    if not re.fullmatch(r"[A-Za-z0-9_-]+", encoded):
        raise ValueError("invalid invitation encoding")
    raw = decode(encoded)
    if len(raw) > 4096 or encode(raw) != encoded:
        raise ValueError("invalid invitation encoding")
    data = _json(raw)
    if set(data) != {"v", "endpoint", "fingerprint", "id", "secret", "expires", "kind"}:
        raise ValueError("invalid invitation fields")
    if type(data["v"]) is not int or data["v"] != 1 or data["kind"] != "computer":
        raise ValueError("this invitation is not for an owned computer")
    for field, pattern in (("fingerprint", r"[a-f0-9]{64}"), ("id", r"[a-f0-9]{32}"),
                           ("secret", r"[A-Za-z0-9_-]{43}")):
        if not isinstance(data[field], str) or not re.fullmatch(pattern, data[field]):
            raise ValueError("invalid invitation identity")
    if len(decode(data["secret"])) != 32 or encode(decode(data["secret"])) != data["secret"]:
        raise ValueError("invalid invitation secret")
    clock = time.time() if now is None else now
    if type(data["expires"]) is not int or not clock < data["expires"] <= clock + 600:
        raise ValueError("invitation expired or has an invalid lifetime")
    if not isinstance(data["endpoint"], str) or len(data["endpoint"]) > 2048:
        raise ValueError("invalid invitation endpoint")
    _addresses(data["endpoint"])
    return data


def _post(data: dict[str, Any], path: str, fields: dict[str, Any]) -> tuple[dict[str, Any], bytes]:
    parts, addresses = _addresses(data["endpoint"])
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    context.minimum_version = ssl.TLSVersion.TLSv1_2
    context.check_hostname = False
    context.verify_mode = ssl.CERT_NONE
    family, kind, protocol, _, address = addresses[0]
    connection = http.client.HTTPConnection(parts.hostname, parts.port, timeout=10)
    plain = socket.socket(family, kind, protocol)
    try:
        plain.settimeout(10)
        plain.connect(address)
        connection.sock = context.wrap_socket(plain, server_hostname=parts.hostname)
        certificate = connection.sock.getpeercert(binary_form=True) or b""
        if not hmac.compare_digest(hashlib.sha256(certificate).hexdigest(), data["fingerprint"]):
            raise ValueError("the device certificate does not match the invitation")
        connection.request("POST", path, body=json.dumps(fields).encode(),
                           headers={"Content-Type": "application/json", "Connection": "close"})
        response = connection.getresponse()
        if response.status != 200:
            raise ValueError("the other device refused this invitation")
        lengths = response.headers.get_all("Content-Length") or []
        if (len(lengths) != 1 or not re.fullmatch(r"[0-9]{1,8}", lengths[0])
                or not 0 < int(lengths[0]) <= LIMIT or response.headers.get("Transfer-Encoding")):
            raise ValueError("the invitation response has invalid framing")
        raw = response.read(LIMIT + 1)
        if len(raw) != int(lengths[0]):
            raise ValueError("the invitation response is incomplete")
        return _json(raw), certificate
    except (OSError, http.client.HTTPException) as exc:
        raise ValueError("could not complete the secure invitation exchange") from exc
    finally:
        connection.close()
        plain.close()


def redeem(invite: str, device_name: str) -> tuple[Membership, str]:
    """Redeem a pinned single-use computer invitation: its membership, and the certificate (base64 DER)
    of the device that issued it, which the invitation's fingerprint pinned."""
    if (not isinstance(device_name, str) or not 1 <= len(device_name) <= 128
            or any(ord(char) < 32 or ord(char) == 127 for char in device_name)):
        raise ValueError("a bounded device name is required")
    data = parse_invite(invite)
    fields = {"id": data["id"], "kind": "computer", "platform": "computer",
              "device_name": device_name, "public_key": local().beacon}
    answer, _ = _post(data, "/join/invite/challenge", fields)
    challenge = answer.get("challenge")
    if not isinstance(challenge, str) or not re.fullmatch(r"[a-f0-9]{64}", challenge):
        raise ValueError("invalid invitation challenge")
    fields["challenge"] = challenge
    secret = decode(data["secret"])
    fields["proof"] = proof(secret, fields, data["fingerprint"])
    answer, certificate = _post(data, "/join/invite/redeem", fields)
    if not isinstance(answer.get("grant_data"), str) or not isinstance(answer.get("proof"), str):
        raise ValueError("invalid invitation grant")
    grant = decode(answer["grant_data"])
    if len(grant) > 4096 or encode(grant) != answer["grant_data"]:
        raise ValueError("invalid invitation grant")
    expected = hmac.new(secret, ("poolhouse-invite-grant/v1\n" + challenge + "\n").encode()
                       + grant, hashlib.sha256).hexdigest()
    if not hmac.compare_digest(expected, answer["proof"]):
        raise ValueError("the invitation grant was not authenticated")
    document = _json(grant)
    if set(document) != {"kind", "group", "key", "mode"} or document.pop("kind") != "computer":
        raise ValueError("the invitation granted unexpected access")
    return parse_recovery(json.dumps(document)), base64.b64encode(certificate).decode()
