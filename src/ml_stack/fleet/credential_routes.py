"""The authenticated Fleet API for named provider credentials."""
from __future__ import annotations

from ml_stack import credentials


def route(request) -> bool:
    """Handle credential listing, saving and removal without returning secret values."""
    if request.method == "GET":
        request.send(200, {"credentials": credentials.describe()})
        return True
    if request.method not in ("POST", "DELETE"):
        return False
    body = request.body()
    try:
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
