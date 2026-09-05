"""Stdlib HTTP: JSON, bytes, and open bodies a caller reads itself."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any

USER_AGENT = "ml-stack"

# 5xx from a server means "busy / still loading", not "your request is wrong".
RETRY_STATUS = frozenset({500, 502, 503, 504})


class ServerError(RuntimeError):
    """The server answered, but with an error."""

    def __init__(self, message: str, *, status: int | None = None, body: str = "",
                 headers: Any = None) -> None:
        super().__init__(message)
        self.status = status
        self.body = body
        # An HTTPError's headers, which look a name up whatever its case.
        self.headers: Any = headers if headers is not None else {}


class ServerUnreachable(ServerError):
    """Nothing is listening, or the connection died mid-request."""


@dataclass(frozen=True, slots=True)
class Retry:
    """How many attempts a request gets, and what it retries."""

    tries: int = 1
    backoff: float = 0.5
    on_status: frozenset[int] = field(default=RETRY_STATUS)
    when_unreachable: bool = True


ONCE = Retry()


@dataclass(frozen=True, slots=True)
class Reply:
    """The status, body and headers of one answer."""

    status: int
    body: bytes
    headers: Any


def build_request(url: str, *, data: bytes | None = None, method: str | None = None,
                  headers: dict[str, str] | None = None,
                  token: str = "") -> urllib.request.Request:
    """A request carrying the caller's headers, a user agent and a bearer token."""
    sent = {"User-Agent": USER_AGENT}
    sent.update(headers or {})
    if token:
        sent["Authorization"] = f"Bearer {token}"
    return urllib.request.Request(
        url, data=data,
        method=method or ("POST" if data is not None else "GET"),
        headers=sent)


def open_stream(url: str, *, data: bytes | None = None, method: str | None = None,
                headers: dict[str, str] | None = None, token: str = "",
                timeout: float = 180.0, retry: Retry = ONCE) -> Any:
    """The open response for ``url``, for a caller that reads the body itself."""
    delay = retry.backoff
    tries = max(1, retry.tries)
    last: ServerError | None = None

    for attempt in range(tries):
        request = build_request(url, data=data, method=method, headers=headers, token=token)
        try:
            return urllib.request.urlopen(request, timeout=timeout)  # noqa: S310

        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:500]
            last = ServerError(f"{url} -> HTTP {exc.code}: {detail}", status=exc.code,
                               body=detail, headers=exc.headers)
            if exc.code not in retry.on_status:
                raise last from exc

        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as exc:
            last = ServerUnreachable(f"cannot reach {url} ({exc})")
            if not retry.when_unreachable:
                raise last from exc

        if attempt < tries - 1:
            time.sleep(delay)
            delay = min(delay * 1.5, 2.0)

    assert last is not None
    raise last


def request_bytes(url: str, *, data: bytes | None = None, method: str | None = None,
                  headers: dict[str, str] | None = None, token: str = "",
                  timeout: float = 180.0, retry: Retry = ONCE) -> Reply:
    """Send a request and read the whole answer."""
    with open_stream(url, data=data, method=method, headers=headers, token=token,
                     timeout=timeout, retry=retry) as response:
        return Reply(int(response.status), response.read(), response.headers)


def request_json(url: str, *, payload: dict[str, Any] | None = None,
                 method: str | None = None, timeout: float = 180.0, tries: int = 1,
                 backoff: float = 0.5, headers: dict[str, str] | None = None,
                 token: str = "") -> Any:
    """Send a JSON request and parse the JSON response."""
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    sent = {"Content-Type": "application/json"}
    sent.update(headers or {})

    reply = request_bytes(url, data=data, method=method, headers=sent, token=token,
                          timeout=timeout, retry=Retry(tries=tries, backoff=backoff))
    body = reply.body.decode("utf-8")
    try:
        return json.loads(body) if body else None
    except json.JSONDecodeError as exc:
        raise ServerError(f"{url} returned non-JSON: {exc}") from exc


def request_stream(url: str, *, payload: dict[str, Any], timeout: float = 180.0,
                   headers: dict[str, str] | None = None, token: str = ""):
    """POST a JSON request and yield each SSE ``data:`` payload, parsed, until ``[DONE]``."""
    sent = {"Content-Type": "application/json", "Accept": "text/event-stream"}
    sent.update(headers or {})
    data = json.dumps(payload).encode("utf-8")
    response = open_stream(url, data=data, method="POST", headers=sent, token=token,
                           timeout=timeout)
    try:
        with response:
            for raw in response:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                body = line[5:].strip()
                if body == "[DONE]":
                    return
                try:
                    yield json.loads(body)
                except json.JSONDecodeError:
                    continue
    except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as exc:
        raise ServerUnreachable(f"cannot reach {url} ({exc})") from exc
