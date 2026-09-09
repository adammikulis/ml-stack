"""The URL policy guard: which addresses `ml_stack.http.check` will fetch.

DNS is faked throughout, so nothing here resolves a real host.
"""

from __future__ import annotations

import pytest

from ml_stack.http import Refused, check


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
