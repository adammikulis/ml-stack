"""Stdlib HTTP: JSON, bytes, open bodies a caller reads itself, and which URLs may be
fetched at all."""

from __future__ import annotations

import json
import re
import socket
import socketserver
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from http.server import ThreadingHTTPServer
from pathlib import Path
from typing import Any

from ml_stack import gate, macauth
from ml_stack.httpguard import Limits, Refused, resolve, split

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


class Server(ThreadingHTTPServer):
    """A threading HTTP server that names itself by the address it bound."""

    def server_bind(self) -> None:
        # http.server's bind asks socket.getfqdn(host): a reverse lookup that took 35s
        # on a macOS runner.
        socketserver.TCPServer.server_bind(self)
        host, port = self.server_address[:2]
        self.server_name, self.server_port = str(host), int(port)


@dataclass(frozen=True, slots=True)
class Retry:
    """How many attempts a request gets, and what it retries."""

    tries: int = 1
    backoff: float = 0.5
    on_status: frozenset[int] = field(default=RETRY_STATUS)
    when_unreachable: bool = True


ONCE = Retry()

_TRUSTED: ssl.SSLContext | None = None


def trust(cafile: str | Path | None) -> None:
    """Verify HTTPS against this certificate authority from now on, or None for the default.

    A development server signs its own certificate, and the store shipped with Python does
    not know it. One call says where that certificate is, and every request this module
    makes afterwards accepts it.
    """
    global _TRUSTED
    _TRUSTED = ssl.create_default_context(cafile=str(cafile)) if cafile else None


_PINNED: dict[str, ssl.SSLContext] = {}


def pin(netloc: str, context: ssl.SSLContext | None) -> None:
    """Talk to ``host:port`` over HTTPS with ``context``, which trusts that machine's one
    certificate; None forgets it. A host that is not pinned is verified against the usual
    authorities, which a self-signed certificate does not satisfy."""
    if context is None:
        _PINNED.pop(netloc.lower(), None)
    else:
        _PINNED[netloc.lower()] = context


def _https_context(url: str) -> ssl.SSLContext | None:
    if not url.lower().startswith("https://"):
        return None
    return _PINNED.get(urllib.parse.urlsplit(url).netloc.lower(), _TRUSTED)


@dataclass(frozen=True, slots=True)
class Reply:
    """The status, body and headers of one answer."""

    status: int
    body: bytes
    headers: Any


def json_body(raw: bytes) -> dict[str, Any]:
    """A request or response body as the object it holds; empty for anything else."""
    try:
        body = json.loads(raw or b"{}")
    except ValueError:
        return {}
    return body if isinstance(body, dict) else {}


SENSITIVE = re.compile(r"token|key|secret|passw|auth|sig|credential|session", re.IGNORECASE)


def shown(url: str) -> str:
    """``url`` as it may appear in a message or a log: no user or password, and no value in a
    query parameter whose name says it is a secret."""
    parts = urllib.parse.urlsplit(url)
    host = parts.hostname or ""
    netloc = f"[{host}]" if ":" in host else host
    netloc += f":{parts.port}" if parts.port else ""
    query = urllib.parse.urlencode(
        [(k, "***" if SENSITIVE.search(k) else v)
         for k, v in urllib.parse.parse_qsl(parts.query, keep_blank_values=True)], safe="*")
    return urllib.parse.urlunsplit((parts.scheme, netloc, parts.path, query, ""))


def build_request(url: str, *, data: bytes | None = None, method: str | None = None,
                  headers: dict[str, str] | None = None,
                  token: str = "") -> urllib.request.Request:
    """A request carrying the caller's headers, a user agent and its credential.

    A ``token`` (or ``Authorization: Bearer`` header) that is a fleet MAC secret signs the
    request instead of being sent; any other token is sent as a bearer token.
    """
    if urllib.parse.urlsplit(url).scheme not in ("http", "https"):
        raise ServerError(f"only http(s) is fetched, not {shown(url)}")
    sent = {"User-Agent": USER_AGENT}
    sent.update(headers or {})
    verb = method or ("POST" if data is not None else "GET")
    secret = token if token.startswith(macauth.PREFIX) else macauth.unwrap(
        sent.get("Authorization", ""))
    if secret:
        sent.pop("Authorization", None)
        sent.update(macauth.sign(secret, verb, url, data))
    elif token:
        sent["Authorization"] = f"Bearer {token}"
    return urllib.request.Request(url, data=data, method=verb, headers=sent)  # noqa: S310 - http(s) only


class _Guarded(urllib.request.HTTPRedirectHandler):
    """Follows a redirect only to a URL ``guard`` accepts; ``guard`` raises to refuse."""

    def __init__(self, guard: Callable[[str], str]) -> None:
        self.guard = guard

    def redirect_request(self, req: Any, *rest: Any) -> Any:
        """``rest`` is urllib's ``fp, code, msg, headers, newurl``."""
        self.guard(urllib.parse.urljoin(req.full_url, rest[-1]))
        return super().redirect_request(req, *rest)


def _open(request: urllib.request.Request, timeout: float, guard: Callable[[str], str] | None
          ) -> Any:
    context = _https_context(request.full_url)
    if guard is None:
        return urllib.request.urlopen(request, timeout=timeout, context=context)  # noqa: S310
    opener = urllib.request.build_opener(_Guarded(guard), urllib.request.HTTPSHandler(
        context=context))
    return opener.open(request, timeout=timeout)


def open_stream(url: str, *, data: bytes | None = None, method: str | None = None,
                headers: dict[str, str] | None = None, token: str = "",
                timeout: float = 180.0, retry: Retry = ONCE,
                guard: Callable[[str], str] | None = None) -> Any:
    """The open response for ``url``, for a caller that reads the body itself; only http(s).

    ``guard`` is called with every redirect target before it is followed (``check`` is the
    usual one); what it raises ends the request.
    """
    delay = retry.backoff
    tries = max(1, retry.tries)
    last: ServerError | None = None

    for attempt in range(tries):
        request = build_request(url, data=data, method=method, headers=headers, token=token)
        try:
            return _open(request, timeout, guard)

        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:500]
            last = ServerError(f"{shown(url)} -> HTTP {exc.code}: {detail}", status=exc.code,
                               body=detail, headers=exc.headers)
            if exc.code not in retry.on_status:
                raise last from exc

        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as exc:
            last = ServerUnreachable(f"cannot reach {shown(url)} ({exc})")
            if not retry.when_unreachable:
                raise last from exc

        if attempt < tries - 1:
            time.sleep(delay)
            delay = min(delay * 1.5, 2.0)

    if last is None:
        raise ServerError(f"{shown(url)}: no attempt was made")
    raise last


@contextmanager
def _queued(url: str) -> Iterator[None]:
    """Wait for this request's turn when it runs a model on a server in the lease registry."""
    if not gate.is_generation(url):
        yield
        return
    try:
        with gate.turn(url):
            yield
    except gate.QueueTimeout as exc:
        raise ServerError(str(exc), status=429) from exc


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args: Any, **kwargs: Any) -> None:
        return None


def head_once(url: str, *, headers: dict[str, str] | None = None, token: str = "",
              timeout: float = 30.0) -> Reply:
    """A HEAD request whose redirect is returned, not followed.

    A caller that sends a bearer token to a host which redirects to another one (a model
    hub handing out a signed CDN address) reads the ``Location`` here and asks the second
    host without the token.
    """
    request = build_request(url, method="HEAD", headers=headers, token=token)
    opener = urllib.request.build_opener(_NoRedirect)
    try:
        with opener.open(request, timeout=timeout) as response:
            return Reply(int(response.status), b"", response.headers)
    except urllib.error.HTTPError as exc:
        if 300 <= exc.code < 400:
            return Reply(exc.code, b"", exc.headers)
        raise ServerError(f"{url} -> HTTP {exc.code}", status=exc.code,
                          headers=exc.headers) from exc
    except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as exc:
        raise ServerUnreachable(f"cannot reach {url} ({exc})") from exc


def request_bytes(url: str, *, data: bytes | None = None, method: str | None = None,
                  headers: dict[str, str] | None = None, token: str = "",
                  timeout: float = 180.0, retry: Retry = ONCE,
                  guard: Callable[[str], str] | None = None) -> Reply:
    """Send a request and read the whole answer."""
    with _queued(url), open_stream(url, data=data, method=method, headers=headers,
                                   token=token, timeout=timeout, retry=retry, guard=guard) as response:
        return Reply(int(response.status), response.read(), response.headers)


def request_json(url: str, *, payload: dict[str, Any] | None = None,
                 method: str | None = None, timeout: float = 180.0, tries: int = 1,
                 backoff: float = 0.5, headers: dict[str, str] | None = None,
                 token: str = "", guard: Callable[[str], str] | None = None) -> Any:
    """Send a JSON request and parse the JSON response."""
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    sent = {"Content-Type": "application/json"}
    sent.update(headers or {})

    reply = request_bytes(url, data=data, method=method, headers=sent, token=token,
                          timeout=timeout, retry=Retry(tries=tries, backoff=backoff), guard=guard)
    body = reply.body.decode("utf-8")
    try:
        return json.loads(body) if body else None
    except json.JSONDecodeError as exc:
        raise ServerError(f"{shown(url)} returned non-JSON: {exc}") from exc


def request_stream(url: str, *, payload: dict[str, Any], timeout: float = 180.0,
                   headers: dict[str, str] | None = None, token: str = ""):
    """POST a JSON request and yield each SSE ``data:`` payload, parsed, until ``[DONE]``."""
    sent = {"Content-Type": "application/json", "Accept": "text/event-stream"}
    sent.update(headers or {})
    data = json.dumps(payload).encode("utf-8")
    try:
        with _queued(url), open_stream(url, data=data, method="POST", headers=sent,
                                       token=token, timeout=timeout) as response:
            for raw in response:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                body = line[5:].strip()
                if body == "[DONE]":
                    return
                try:
                    event = json.loads(body)
                except json.JSONDecodeError:
                    continue
                if isinstance(event, dict) and "error" in event and "choices" not in event:
                    said = event["error"]
                    said = said.get("message", said) if isinstance(said, dict) else said
                    raise ServerError(f"{shown(url)} -> error mid-stream: {said}", body=body)
                yield event
    except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as exc:
        raise ServerUnreachable(f"cannot reach {shown(url)} ({exc})") from exc


# --- what may be fetched -------------------------------------------------------------------


def _addresses(host: str) -> list[str]:
    """Every address a host name resolves to. Separate so a test can answer for DNS."""
    try:
        return sorted({info[4][0] for info in socket.getaddrinfo(host, None)})
    except socket.gaierror as exc:
        raise Refused(f"cannot resolve {host!r}: {exc}") from exc


def check(url: str) -> str:
    """The URL, if it may be fetched; ``Refused`` otherwise.

    http(s) only, and only to a host whose every address is a public one. ``file:``,
    ``localhost``, ``127.0.0.0/8``, ``10.0.0.0/8``, ``192.168.0.0/16``, ``172.16.0.0/12``,
    link-local and the IPv6 equivalents are all refused, by what the name resolves to.
    `ml_stack.httpguard.fetch` is the fetch that holds to this at connection time.
    """
    parts, host, port = split(url)
    resolve(host, port, Limits(resolver=lambda name, _port: _addresses(name)))
    return urllib.parse.urlunsplit(parts)
