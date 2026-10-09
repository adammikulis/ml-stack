"""Fetching a URL somebody else chose: what is refused before a byte is sent, and what is bounded."""

import gzip
import ipaddress
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler
from typing import ClassVar

import pytest

from ml_stack import httpguard
from ml_stack.http import Server
from ml_stack.httpguard import Limits, Refused, TooLarge, fetch

LOCAL = Limits(allow_hosts=frozenset({"127.0.0.1"}), timeout=2.0, deadline_s=6.0)


class Site(BaseHTTPRequestHandler):
    """Routes by path; every route is a behaviour a hostile server might have."""

    log: ClassVar[list[str]] = []

    def log_message(self, *args):
        pass

    def do_GET(self):
        Site.log.append(self.path)
        route = self.path.split("?")[0]
        getattr(self, "route_" + route.strip("/").replace("-", "_"), self.route_missing)()

    def say(self, status, body=b"", **headers):
        self.send_response(status)
        for name, value in headers.items():
            self.send_header(name.replace("_", "-"), value)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def route_missing(self):
        self.say(404, b"none")

    def route_ok(self):
        self.say(200, b"hello", Content_Type="text/plain")

    def route_big(self):
        self.say(200, b"x" * 100_000)

    def route_bomb(self):
        self.say(200, gzip.compress(b"\0" * 50_000_000), Content_Encoding="gzip")

    def route_gzip(self):
        self.say(200, gzip.compress(b"squeezed"), Content_Encoding="gzip")

    def route_lie(self):
        self.send_response(200)
        self.send_header("Content-Length", "99999999999")
        self.end_headers()

    def route_loop(self):
        self.say(302, Location="/loop")

    def route_metadata(self):
        self.say(302, Location="http://169.254.169.254/latest/meta-data/")

    def route_localhost(self):
        self.say(302, Location=f"http://localhost:{self.server.server_port}/ok")

    def route_scheme(self):
        self.say(302, Location="file:///etc/passwd")

    def route_relative(self):
        self.say(301, Location="ok")

    def route_drip(self):
        self.send_response(200)
        self.end_headers()
        for _ in range(100):
            self.wfile.write(b"x")
            self.wfile.flush()
            time.sleep(0.2)

    def route_silent(self):
        time.sleep(10)

    def route_echo(self):
        self.say(200, repr(sorted(self.headers.keys())).encode())

    def route_same_host(self):
        self.say(302, Location="/echo")

    def route_to_other_host(self):
        self.say(302, Location=f"http://localhost:{self.server.server_port}/echo")


@pytest.fixture
def site():
    Site.log = []
    server = Server(("127.0.0.1", 0), Site)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()
    server.server_close()


# -- which addresses are public -------------------------------------------------------------


@pytest.mark.parametrize("address", [
    "127.0.0.1", "127.255.0.9", "::1", str(ipaddress.IPv4Address(0)), "10.1.2.3", "172.16.0.1", "172.31.255.255",
    "192.168.1.1", "169.254.169.254", "169.254.0.1", "fe80::1", "fc00::1", "fd12:3456::1",
    "100.64.0.1", "224.0.0.1", "ff02::1", "240.0.0.1", "::ffff:127.0.0.1", "::ffff:10.0.0.1",
    "64:ff9b::7f00:1", "2002:7f00:1::1", "fe80::1%en0", "192.0.2.1",
])
def test_addresses_that_are_not_the_public_internet_are_refused(address):
    assert httpguard.allowed_address(address) is False


@pytest.mark.parametrize("address", ["8.8.8.8", "1.1.1.1", "93.184.216.34", "2606:4700::1111",
                                     "::ffff:8.8.8.8"])
def test_public_addresses_are_allowed(address):
    assert httpguard.allowed_address(address) is True


def _resolver(**table):
    def answer(host, port):
        return table[host.replace(".", "_").replace("-", "_")]
    return answer


@pytest.mark.parametrize("url", [
    "file:///etc/passwd", "ftp://example.com/x", "gopher://example.com", "javascript:alert(1)",
    "//example.com/x", "example.com/x", "http://", "http://user:pw@example.com/",
    "http://exa mple.com/", "http://example.com/a\r\nHost: evil", "http://example.com:99999/",
])
def test_a_url_that_is_not_plain_http_to_a_host_is_refused_before_resolving(url):
    def never(host, port):
        raise AssertionError("resolved a URL that should have been refused first")

    with pytest.raises(Refused):
        fetch(url, limits=Limits(resolver=never))


@pytest.mark.parametrize("host", ["localhost", "app.localhost", "printer.local", "db.internal",
                                  "127.0.0.1", "[::1]", "2130706433", "0x7f.1", "017700000001",
                                  "169.254.169.254", "[::ffff:127.0.0.1]"])
def test_hosts_that_mean_this_machine_are_refused(host):
    with pytest.raises(Refused, match=r"this machine|public internet"):
        fetch(f"http://{host}/", limits=Limits(resolver=lambda h, p: ["127.0.0.1"]))


def test_a_name_with_any_private_address_is_refused():
    both = _resolver(mixed_example=["93.184.216.34", "10.0.0.5"])
    with pytest.raises(Refused, match=r"10\.0\.0\.5"):
        fetch("http://mixed.example/", limits=Limits(resolver=both))


def test_a_name_that_does_not_resolve_is_refused():
    with pytest.raises(Refused, match="cannot resolve"):
        fetch("http://no-such-host.invalid/")


# -- the connection is to the address that was checked ---------------------------------------


def test_the_connection_goes_to_the_address_that_was_checked(site):
    asked = []

    def once(host, port):
        asked.append(host)
        return ["127.0.0.1"]

    port = site.rsplit(":", 1)[1]
    got = fetch(f"http://rebind.example:{port}/ok",
                limits=Limits(resolver=once, allow_hosts=frozenset({"rebind.example"})))
    assert got.body == b"hello" and asked == ["rebind.example"], \
        "one answer, used for the connection; the system resolver was never asked"


# -- redirects ---------------------------------------------------------------------------


def test_a_redirect_to_the_metadata_address_is_refused(site):
    with pytest.raises(Refused, match=r"169\.254\.169\.254"):
        fetch(f"{site}/metadata", limits=LOCAL)


def test_a_redirect_to_localhost_is_refused_though_the_first_host_was_allowed(site):
    with pytest.raises(Refused, match="localhost"):
        fetch(f"{site}/localhost", limits=LOCAL)


def test_a_redirect_to_another_scheme_is_refused(site):
    with pytest.raises(Refused, match="only http"):
        fetch(f"{site}/scheme", limits=LOCAL)


def test_a_redirect_loop_ends(site):
    with pytest.raises(Refused, match="redirects"):
        fetch(f"{site}/loop", limits=Limits(allow_hosts=LOCAL.allow_hosts, max_redirects=3))
    assert Site.log.count("/loop") == 4


def test_a_relative_redirect_is_followed_and_recorded(site):
    got = fetch(f"{site}/relative", limits=LOCAL)
    assert got.body == b"hello" and got.url.endswith("/ok")
    assert got.redirects == (f"{site}/relative",)


def test_credentials_do_not_follow_a_redirect_to_another_host(site):
    both = Limits(allow_hosts=frozenset({"127.0.0.1", "localhost"}), timeout=2.0)
    got = fetch(f"{site}/to-other-host", headers={"Authorization": "Bearer SECRET",
                                                   "X-Api-Key": "SECRET"}, limits=both)
    assert b"SECRET" not in got.body
    sent = got.body.lower()
    assert got.url.startswith("http://localhost:") and b"user-agent" in sent
    assert b"authorization" not in sent and b"x-api-key" not in sent


# -- sizes and time ----------------------------------------------------------------------


def test_a_body_over_the_cap_is_refused(site):
    with pytest.raises(TooLarge):
        fetch(f"{site}/big", limits=Limits(allow_hosts=LOCAL.allow_hosts, max_bytes=50_000))
    assert fetch(f"{site}/big", limits=LOCAL).body == b"x" * 100_000


def test_a_length_that_lies_is_refused_without_reading_it(site):
    with pytest.raises(TooLarge):
        fetch(f"{site}/lie", limits=LOCAL)


def test_a_compression_bomb_is_stopped_while_it_expands(site):
    started = time.monotonic()
    with pytest.raises(TooLarge):
        fetch(f"{site}/bomb", limits=Limits(allow_hosts=LOCAL.allow_hosts,
                                           max_bytes=1_000_000))
    assert time.monotonic() - started < 3.0


def test_a_compression_bomb_under_the_byte_cap_is_stopped_by_its_ratio(site):
    with pytest.raises(TooLarge, match="times over"):
        fetch(f"{site}/bomb", limits=Limits(allow_hosts=LOCAL.allow_hosts,
                                           max_bytes=200_000_000))


def test_a_gzip_body_is_expanded(site):
    assert fetch(f"{site}/gzip", limits=LOCAL).body == b"squeezed"


def test_a_server_that_drips_bytes_ends_at_the_deadline(site):
    started = time.monotonic()
    with pytest.raises(Refused, match="time"):
        fetch(f"{site}/drip", limits=Limits(allow_hosts=LOCAL.allow_hosts, timeout=2.0,
                                           deadline_s=1.0))
    assert time.monotonic() - started < 3.0


def test_a_server_that_says_nothing_ends_at_the_timeout(site):
    started = time.monotonic()
    with pytest.raises(Refused, match="cannot reach"):
        fetch(f"{site}/silent", limits=Limits(allow_hosts=LOCAL.allow_hosts, timeout=0.5,
                                             deadline_s=3.0))
    assert time.monotonic() - started < 4.0


def test_a_header_with_a_line_break_is_refused(site):
    with pytest.raises(Refused, match="line break"):
        fetch(f"{site}/ok", headers={"X-A": "b\r\nX-Injected: 1"}, limits=LOCAL)


# -- saving ------------------------------------------------------------------------------


def test_download_writes_the_file_whole_and_leaves_nothing_when_it_fails(site, tmp_path):
    where = tmp_path / "out" / "got.bin"
    assert httpguard.download(f"{site}/big", where, limits=LOCAL) == where
    assert where.read_bytes() == b"x" * 100_000
    other = tmp_path / "out" / "refused.bin"
    with pytest.raises(TooLarge):
        httpguard.download(f"{site}/big", other,
                           limits=Limits(allow_hosts=LOCAL.allow_hosts, max_bytes=10))
    assert not other.exists()
    assert sorted(one.name for one in other.parent.iterdir()) == ["got.bin"]


def test_download_refuses_an_error_status(site, tmp_path):
    with pytest.raises(Refused, match="404"):
        httpguard.download(f"{site}/nowhere", tmp_path / "x", limits=LOCAL)


def test_nothing_listens_means_refused_not_a_traceback():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    with pytest.raises(Refused, match="cannot reach"):
        fetch(f"http://127.0.0.1:{port}/", limits=LOCAL)


def test_credentials_stay_on_a_redirect_within_the_same_host(site):
    got = fetch(f"{site}/same-host", headers={"Authorization": "Bearer SECRET"}, limits=LOCAL)
    assert b"authorization" in got.body.lower()
