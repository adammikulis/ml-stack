"""Stdlib HTTP for talking to a local model server.

``urllib`` rather than httpx, deliberately: this package has to import on targets where
adding a wheel is a build-system change.

The retry policy covers one specific failure -- a local server answering 500 to a request
that arrives while it is still loading, or while another request is mid-flight. Retrying
a 500 from a *remote* API would be wrong; retrying it from a process you launched sixty
seconds ago is not.
"""

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
    """Send a JSON request and parse the JSON response.

    ``tries > 1`` retries only on a connection failure or a retryable 5xx, with
    multiplicative backoff capped at 2s.
    A 4xx is never retried: the request itself is wrong and will stay wrong.
    """
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
            # Not retryable: a server returning non-JSON is misconfigured, and hammering
            # it just delays the report.
            raise ServerError(f"{url} returned non-JSON: {exc}") from exc

        if attempt < tries - 1:
            time.sleep(delay)
            delay = min(delay * 1.5, 2.0)

    assert last is not None
    raise last
