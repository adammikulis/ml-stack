"""The graph page's server under hostile requests, over real sockets: traversal, symlinks,
malformed and oversized bodies, other methods, and a page on another origin or host name
talking to a server that listens on loopback."""

from __future__ import annotations

import json
import socket
import time
from pathlib import Path

import pytest
from conftest import threaded_server

from ml_stack.graph.page import render
from ml_stack.graph.serve import Handler, bind

SECRET = "TOP-SECRET-OUTSIDE-THE-EXPORT-ROOT"


@pytest.fixture
def site(tmp_path):
    page = tmp_path / "site" / "index.html"
    page.parent.mkdir()
    page.write_text(render({"nodes": [], "edges": [], "messages": {}}, title="t"), encoding="utf-8")
    export = tmp_path / "export"
    (export / "a").mkdir(parents=True)
    (export / "a" / "b.json").write_text("{}")
    (tmp_path / "secret.txt").write_text(SECRET)
    (export / "link.json").symlink_to(tmp_path / "secret.txt")
    (export / "up").symlink_to(tmp_path)
    return page, export


def ask(url: str, raw: bytes, seconds: float = 4.0) -> tuple[int, bytes]:
    """(status, body) of one raw request; status 0 when the connection was dropped."""
    port = int(url.rsplit(":", 1)[1])
    with socket.create_connection(("127.0.0.1", port), timeout=seconds) as sock:
        sock.sendall(raw)
        got = b""
        try:
            while chunk := sock.recv(65536):
                got += chunk
        except OSError:
            pass
    line = got.split(b"\r\n", 1)[0].split()
    return (int(line[1]) if len(line) > 1 and line[1].isdigit() else 0), got


def request(method: str, path: str, *, headers: str = "", body: bytes = b"") -> bytes:
    head = f"{method} {path} HTTP/1.1\r\nHost: 127.0.0.1\r\n{headers}"
    if body or method == "POST":
        head += f"Content-Length: {len(body)}\r\n"
    return head.encode("latin-1") + b"Connection: close\r\n\r\n" + body


HOSTILE = [
    request("GET", "/export/../secret.txt"), request("GET", "/export/%2e%2e/secret.txt"),
    request("GET", "/export/..%2fsecret.txt"), request("GET", "/export/a/..%5c..%5csecret.txt"),
    request("GET", "/export//etc/passwd"), request("GET", "/export/%252e%252e/secret.txt"),
    request("GET", "/export/link.json"), request("GET", "/export/up/secret.txt"),
    request("GET", "/export/a/b.json%00.html"), request("GET", "/" + "a" * 70_000),
    request("GET", "/export/" + "../" * 200 + "etc/passwd"),
    request("POST", "/ask", body=b"{{{{{"), request("POST", "/ask", body=b"[]"),
    request("POST", "/ask", body=b"null"), request("POST", "/ask", body=b"\xff\xfe"),
    request("POST", "/ask/stream", body=b"{"),
    request("POST", "/draft", body=b"[]"), request("POST", "/request", body=b"{}"),
    request("POST", "/review", body=b'{"id": "../../x", "action": "release"}'),
    request("PUT", "/ask", body=b"{}"), request("DELETE", "/export/a/b.json"),
    request("TRACE", "/"), request("OPTIONS", "/ask"),
    request("POST", "/ask", headers="Content-Length: 3\r\n", body=b"{}"),
]


def test_the_page_server_answers_hostile_requests_without_failing_or_leaking(site):
    page, export = site
    with threaded_server(Handler.configured(site=page, export=export)) as url:
        for raw in HOSTILE:
            status, got = ask(url, raw)
            assert SECRET.encode() not in got, raw[:60]
            assert status in (0, 400, 404, 405, 409, 411, 413, 414, 431, 501), (raw[:60], status)
            assert status != 500, raw[:60]
        assert ask(url, request("GET", "/export/a/b.json"))[0] == 200


def test_the_page_server_listens_on_loopback_only(tmp_path):
    server = bind(["serve", "--site", str(tmp_path / "index.html"), "--port", "0"])
    try:
        assert server.server_address[0] == "127.0.0.1"
    finally:
        server.server_close()


@pytest.mark.xfail(strict=True, reason="a host name other than loopback is answered "
                                       "(DNS rebinding, finding F12)")
def test_a_request_addressed_to_another_host_name_is_refused(site):
    page, export = site
    with threaded_server(Handler.configured(site=page, export=export)) as url:
        raw = b"GET /ask/model HTTP/1.1\r\nHost: attacker.example\r\nConnection: close\r\n\r\n"
        assert ask(url, raw)[0] in (400, 403, 421)


@pytest.mark.xfail(strict=True, reason="a form post from another origin is accepted (finding F13)")
def test_a_post_from_another_origin_is_refused(site):
    page, export = site
    with threaded_server(Handler.configured(site=page, export=export)) as url:
        raw = request("POST", "/ask", headers="Origin: https://attacker.example\r\n"
                      "Content-Type: text/plain\r\n", body=json.dumps({"question": "hi"}).encode())
        assert ask(url, raw)[0] in (400, 403, 421)


@pytest.mark.xfail(strict=True, reason="the body is read to the length claimed (finding F14)")
def test_a_body_that_claims_to_be_huge_is_refused_at_once(site):
    page, export = site
    with threaded_server(Handler.configured(site=page, export=export)) as url:
        began = time.monotonic()
        raw = (b"POST /ask HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Length: 99999999999\r\n"
               b"Connection: close\r\n\r\n{")
        status, _ = ask(url, raw, seconds=3.0)
        assert status == 413 and time.monotonic() - began < 2.0


def test_nothing_in_the_tree_was_written_by_the_hostile_requests(site, tmp_path):
    page, export = site
    before = sorted(p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*"))
    with threaded_server(Handler.configured(site=page, export=export)) as url:
        for raw in HOSTILE:
            ask(url, raw)
    after = sorted(p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*"))
    assert after == before
    assert Path(tmp_path / "secret.txt").read_text() == SECRET
