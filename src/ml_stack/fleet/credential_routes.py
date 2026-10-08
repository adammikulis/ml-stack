"""The authenticated Fleet API for named provider credentials."""
from __future__ import annotations

from ml_stack import credentials
from ml_stack.fleet.room_routes import LOOPBACK, _origin_ok
from ml_stack.fleet.session import parse_cookie


def route(request) -> bool:
    """Handle credential listing, saving and removal without returning secret values."""
    if request.method == "GET":
        request.send(200, {"credentials": credentials.describe()})
        return True
    if request.method not in ("POST", "DELETE"):
        return False
    session = request.ui.sessions.get(parse_cookie(request.cookie))
    if (request.client_ip != LOOPBACK or not request.ui.host_ok(request.host_header)
            or request.header("Authorization") or request.header("X-ML-Stack-Token")
            or session is None or not session.credentialed):
        request.send(403, {"error": "credential changes require a person in this machine's browser"})
        return True
    if not _origin_ok(request.header("Origin"), request.host_header):
        request.send(403, {"error": "credential changes require the local UI origin"})
        return True
    try:
        size = int(request.header("Content-Length", "0") or 0)
    except ValueError:
        size = -1
    if not 0 <= size <= 65536:
        request.send(400, {"error": "credential request exceeds its size bound"})
        return True
    body = request.body()
    try:
        if not isinstance(body, dict):
            raise credentials.CredentialError("credential request must be an object")
        if request.method == "POST":
            if set(body) != {"name", "value"} or not isinstance(body["name"], str) \
                    or not isinstance(body["value"], str):
                raise credentials.CredentialError("save a credential name and value")
            where = credentials.set(body["name"], body["value"])
            request.send(200, {"saved": body["name"], "source": "credentials file", "where": where})
        else:
            if set(body) != {"name"} or not isinstance(body["name"], str):
                raise credentials.CredentialError("remove a credential by name")
            removed = credentials.unset(body["name"])
            request.send(200, {"removed": removed, "name": body["name"]})
    except credentials.CredentialError as error:
        request.send(400, {"error": str(error)})
    return True
