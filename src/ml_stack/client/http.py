"""Stdlib HTTP for talking to a local model server."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any


class ServerError(RuntimeError):
    """The model server answered, but with an error."""

    def __init__(self, message: str, *, status: int | None = None, body: str = "") -> None:
        super().__init__(message)
        self.status = status
        self.body = body


class ServerUnreachable(ServerError):
    """Nothing is listening, or the connection died mid-request."""


# 5xx from a local server means "busy / still loading", not "your request is wrong".
_RETRY_STATUS = frozenset({500, 502, 503, 504})


def request_json(
    url: str,
    *,
    payload: dict[str, Any] | None = None,
    method: str | None = None,
    timeout: float = 180.0,
    tries: int = 1,
    backoff: float = 0.5,
    headers: dict[str, str] | None = None,
) -> Any:
    """Send a JSON request and parse the JSON response."""
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    request_headers = {"Content-Type": "application/json"}
    if headers:
        request_headers.update(headers)

    delay = backoff
    last: Exception | None = None

    for attempt in range(max(1, tries)):
        request = urllib.request.Request(
            url,
            data=data,
            method=method or ("POST" if data is not None else "GET"),
            headers=request_headers,
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                body = response.read().decode("utf-8")
            return json.loads(body) if body else None

        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:500]
            last = ServerError(
                f"{url} -> HTTP {exc.code}: {detail}", status=exc.code, body=detail
            )
            if exc.code not in _RETRY_STATUS:
                raise last from exc

        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as exc:
            last = ServerUnreachable(f"cannot reach {url} ({exc})")

        except json.JSONDecodeError as exc:
            raise ServerError(f"{url} returned non-JSON: {exc}") from exc

        if attempt < tries - 1:
            time.sleep(delay)
            delay = min(delay * 1.5, 2.0)

    assert last is not None
    raise last
