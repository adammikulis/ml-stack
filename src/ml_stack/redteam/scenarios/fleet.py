"""The fleet daemon's HTTP surface, attacked directly: requests that should not be served
without the bearer token, bodies that should be refused, paths that should not leave the
files directory, and the proxy in front of the model."""

from __future__ import annotations

import re
import socket
import time
from dataclasses import dataclass
from urllib.parse import urlsplit

from ml_stack.redteam.lab import Lab
from ml_stack.redteam.report import Attempt, Report
from ml_stack.redteam.scenarios import Options, capped

TARGET = "fleet"
LOOK = 3.0
OPEN = {"/health", "/favicon.ico"}
PROTECTED = ("/models", "/availability", "/jobs", "/speech/providers", "/bench", "/bench/export",
             "/fetch", "/files/x", "/infer/v1/models")
BODY_ROUTES = ("/availability", "/fetch", "/serve", "/bench", "/models/get", "/jobs")


@dataclass(frozen=True, slots=True)
class Reply:
    """What a raw request came back with: the status (0 for none), the body, the seconds."""

    status: int
    body: bytes
    seconds: float
    hung: bool = False


def raw(url: str, request: bytes, wait: float = LOOK) -> Reply:
    """Send ``request`` bytes to the server at ``url`` and read what comes back."""
    parts = urlsplit(url)
    started = time.monotonic()
    chunks = b""
    try:
        with socket.create_connection((parts.hostname or "", parts.port or 80),
                                      timeout=wait) as sock:
            sock.sendall(request)
            sock.settimeout(wait)
            while data := sock.recv(65536):
                chunks += data
                if len(chunks) > 1 << 22:
                    break
    except TimeoutError:
        hung = not chunks
        return Reply(_status(chunks), chunks.partition(b"\r\n\r\n")[2], time.monotonic() - started,
                     hung)
    except OSError:
        pass
    return Reply(_status(chunks), chunks.partition(b"\r\n\r\n")[2], time.monotonic() - started)


def _status(chunks: bytes) -> int:
    line = chunks.partition(b"\r\n")[0].split()
    if len(line) > 1 and line[1].isdigit():
        return int(line[1])
    said = re.search(rb"Error code: (\d+)", chunks)
    return int(said.group(1)) if said else 0


def broke(got: Reply) -> bool:
    """Whether the daemon failed on a request: an error of its own, nothing back, or a hang."""
    return got.hung or got.status in (500, 502, 504) or (got.status == 0 and not got.body)


def get(path: str, *headers: str, method: str = "GET") -> bytes:
    lines = [f"{method} {path} HTTP/1.1", "Host: x", "Connection: close", *headers]
    return ("\r\n".join(lines) + "\r\n\r\n").encode()


def post(path: str, body: bytes, *headers: str, length: int | None = None) -> bytes:
    size = len(body) if length is None else length
    lines = [f"POST {path} HTTP/1.1", "Host: x", "Connection: close", f"Content-Length: {size}",
             "Content-Type: application/json", *headers]
    return ("\r\n".join(lines) + "\r\n\r\n").encode() + body


NOTES = {
    "claims-500MB-sends-2-bytes": "the daemon sizes a read by the claimed length and waits",
    "8MB-json-to-availability": "read and parsed an 8 MB body",
    "8MB-json-to-infer": "forwarded an 8 MB body to the model server",
    "proxy-route": "the proxy forwarded a path other than the chat API",
    "POST/infer/slots/0?action=erase": "a peer cleared a slot's cache",
}


def _add(report: Report, attack_class: str, attack_id: str, won: bool, got: Reply) -> None:
    note = NOTES.get(attack_id) or NOTES.get(attack_class) or f"HTTP {got.status}"
    report.add(Attempt(TARGET, attack_class, attack_id, won, seconds=got.seconds,
                       detail=f"HTTP {got.status}: {note}" if won else f"HTTP {got.status}",
                       error=got.hung and not won))


def run_auth(lab: Lab, report: Report, options: Options) -> None:
    """Every protected route asked for with a missing, wrong or oddly placed token."""
    forms = {
        "no-token": (),
        "wrong-token": ("Authorization: Bearer nope",),
        "empty-bearer": ("Authorization: Bearer ",),
        "lowercase-scheme": (f"Authorization: bearer {lab.token}",),
        "basic-scheme": (f"Authorization: Basic {lab.token}",),
        "token-in-header-x": (f"X-Token: {lab.token}",),
        "token-prefix": (f"Authorization: Bearer {lab.token[:-1]}",),
        "token-suffix": (f"Authorization: Bearer {lab.token}x",),
    }
    for path in capped(list(PROTECTED), options.limit):
        for label, headers in forms.items():
            got = raw(lab.daemon_url, get(path, *headers))
            _add(report, "auth-bypass", f"GET{path}:{label}", got.status in range(200, 300), got)
        query = raw(lab.daemon_url, get(f"{path}?token={lab.token}"))
        _add(report, "auth-bypass", f"GET{path}:token-in-query", query.status in range(200, 300),
             query)
    for path in BODY_ROUTES:
        got = raw(lab.daemon_url, post(path, b"{}"))
        _add(report, "auth-bypass", f"POST{path}:no-token", got.status in range(200, 300), got)
    got = raw(lab.daemon_url, post("/infer/v1/chat/completions", b'{"messages":[]}'))
    _add(report, "auth-bypass", "POST/infer/v1/chat/completions:no-token",
         got.status in range(200, 300), got)
    got = raw(lab.daemon_url, get("/files/x", method="PUT"))
    _add(report, "auth-bypass", "PUT/files/x:no-token", got.status in range(200, 300), got)


def run_malformed(lab: Lab, report: Report, options: Options) -> None:
    """Authenticated requests the daemon should refuse with a 4xx and nothing worse."""
    auth = f"Authorization: Bearer {lab.token}"
    bodies = {"not-json": b"{not json", "empty": b"", "array": b"[1,2]", "null": b"null",
              "deep": b"[" * 5000 + b"]" * 5000, "huge-number": b'{"context": 1e999999}',
              "wrong-types": b'{"argv": 5, "model": []}', "nul": b'{"model": "\\u0000"}'}
    for path in capped(list(BODY_ROUTES), options.limit):
        for label, body in bodies.items():
            got = raw(lab.daemon_url, post(path, body, auth))
            _add(report, "malformed-request", f"POST{path}:{label}",
                 broke(got), got)
    for label, request in {
        "length-not-a-number": post("/availability", b"{}", auth, length=0).replace(
            b"Content-Length: 0", b"Content-Length: abc"),
        "negative-length": post("/availability", b"{}", auth, length=-5),
        "put-length-not-a-number": get("/files/x", auth, "Content-Length: abc", method="PUT"),
        "put-range-nonsense": get("/files/x", auth, "Content-Length: 1",
                                  "Content-Range: bytes zzz", method="PUT") + b"x",
        "get-range-nonsense": get("/files/x", auth, "Range: bytes=abc-"),
        "request-line-only": b"GET\r\n\r\n",
        "bad-version": b"GET /health HTTP/9.9\r\n\r\n",
    }.items():
        got = raw(lab.daemon_url, request)
        _add(report, "malformed-request", label, broke(got), got)


def run_oversized(lab: Lab, report: Report, options: Options) -> None:
    """Bodies bigger than anything the daemon needs: does it refuse, or read them all."""
    del options
    auth = f"Authorization: Bearer {lab.token}"
    claimed = raw(lab.daemon_url, post("/availability", b"{}", auth, length=5 * 10**8), wait=2.0)
    _add(report, "oversized-body", "claims-500MB-sends-2-bytes", broke(claimed), claimed)
    big = b'{"x": "' + b"a" * (8 << 20) + b'"}'
    got = raw(lab.daemon_url, post("/availability", big, auth), wait=10.0)
    _add(report, "oversized-body", "8MB-json-to-availability", got.status not in (400, 413, 431),
         got)
    chat = raw(lab.daemon_url, post("/infer/v1/chat/completions", big, auth), wait=10.0)
    _add(report, "oversized-body", "8MB-json-to-infer", chat.status not in (400, 413, 431), chat)


def run_paths(lab: Lab, report: Report, options: Options) -> None:
    """Paths that try to leave the files directory or the proxy's route."""
    outside = lab.files.parent / "outside.txt"
    outside.write_text("OUTSIDE-THE-FILES-ROOT", encoding="utf-8")
    auth = f"Authorization: Bearer {lab.token}"
    tries = ["/files/../outside.txt", "/files/%2e%2e/outside.txt", "/files/..%2foutside.txt",
             "/files/a/../../outside.txt", "/files//etc/passwd", "/files/..\\outside.txt",
             "/files/%00outside.txt", f"/files/{outside}"]
    for path in capped(tries, options.limit):
        got = raw(lab.daemon_url, get(path, auth))
        _add(report, "path-traversal", path, b"OUTSIDE-THE-FILES-ROOT" in got.body, got)
    for path in ("/infer/slots", "/infer/props", "/infer/metrics", "/infer/slots/0?action=erase",
                 "/infer/../health", "/infer/%2e%2e/models", "/inferX/v1/models"):
        got = raw(lab.daemon_url, get(path, auth))
        won = got.status == 200 and not path.endswith("v1/models")
        _add(report, "proxy-route", path, won, got)
    erase = raw(lab.daemon_url, post("/infer/slots/0?action=erase", b"{}", auth))
    _add(report, "proxy-route", "POST/infer/slots/0?action=erase", erase.status == 200, erase)


async def run(lab: Lab, report: Report, options: Options) -> None:
    for step in (run_auth, run_malformed, run_oversized, run_paths):
        step(lab, report, options)
