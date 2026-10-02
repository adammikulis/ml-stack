"""The URL policy guard: which addresses `ml_stack.http.check` will fetch.

DNS is faked throughout, so nothing here resolves a real host.
"""

from __future__ import annotations

import pytest

from ml_stack.http import Refused, Server, check


@pytest.fixture
def public_dns(monkeypatch):
    """Every invented host resolves somewhere public, except *.internal, a LAN."""
    def addresses(host):
        return ["10.0.0.5"] if host.endswith(".internal") else ["1.2.3.4"]
    monkeypatch.setattr("ml_stack.http._addresses", addresses)


def test_a_public_url_passes_the_check(public_dns):
    assert check("https://quenlow.example/about?x=1") == "https://quenlow.example/about?x=1"


@pytest.mark.parametrize("url", [
    "file:///etc/hosts",
    "ftp://quenlow.example/",
    "/just/a/path",
    "http://localhost/admin",
    "http://printer.local/",
    "http://127.0.0.1:8080/",
    "http://[::1]/",
    "http://10.1.2.3/",
    "http://192.168.1.1/",
    "http://172.16.0.9/",
    "http://169.254.169.254/latest/meta-data/",
    "http://intranet.internal/",
])
def test_a_url_on_this_side_of_the_router_is_refused(public_dns, url):
    with pytest.raises(Refused):
        check(url)


def test_a_host_dns_cannot_resolve_is_refused(monkeypatch):
    import socket

    def gone(host, port):
        raise socket.gaierror("not found")
    monkeypatch.setattr("ml_stack.http.socket.getaddrinfo", gone)
    with pytest.raises(Refused, match="cannot resolve"):
        check("https://nowhere.example/")


def test_a_server_binds_without_a_reverse_lookup_and_answers(monkeypatch):
    import socket
    import threading
    import urllib.request
    from http.server import BaseHTTPRequestHandler

    def lookup(*_a):
        raise AssertionError("the bind looked its own address up")

    class Hello(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"hello")

        def log_message(self, *_a):
            pass

    monkeypatch.setattr(socket, "getfqdn", lookup)
    server = Server(("127.0.0.1", 0), Hello)
    try:
        assert server.server_name == "127.0.0.1"
        assert server.server_port == server.server_address[1] != 0
        threading.Thread(target=server.serve_forever, daemon=True).start()
        with urllib.request.urlopen(f"http://127.0.0.1:{server.server_port}/", timeout=5) as r:
            assert r.read() == b"hello"
    finally:
        server.shutdown()
        server.server_close()


def test_a_message_about_a_url_carries_no_password_and_no_secret_parameter():
    from ml_stack.http import shown

    got = shown("https://user:pw@example.com:8443/v1/x?token=abc&model=m&api_key=k&x=1#frag")
    assert "abc" not in got and "pw" not in got and "user" not in got and "api_key=k" not in got
    assert "model=m" in got and "x=1" in got and got.startswith("https://example.com:8443/v1/x")


def test_an_unreachable_url_is_reported_without_its_secrets():
    from ml_stack.http import ServerUnreachable, request_json

    with pytest.raises(ServerUnreachable) as caught:
        request_json("http://127.0.0.1:9/x?access_token=SUPERSECRET&a=1", timeout=1)
    assert "SUPERSECRET" not in str(caught.value) and "a=1" in str(caught.value)


@pytest.mark.parametrize("url", ["file:///etc/passwd", "ftp://example.com/x", "data:text/plain,hi",
                                 "jar:file:///x!/y", "/etc/passwd"])
def test_only_http_and_https_are_opened(url):
    from ml_stack.http import ServerError, open_stream, request_bytes

    with pytest.raises(ServerError, match="only http"):
        open_stream(url)
    with pytest.raises(ServerError, match="only http"):
        request_bytes(url)
