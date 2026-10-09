"""Nothing crosses a link beyond this machine in the clear or below TLS 1.3, and no setting says otherwise."""

import contextlib
import http.client as httpclient
import socket
import ssl
import threading

import pytest

from ml_stack import http
from ml_stack.fleet import discovery, projects, tls
from ml_stack.fleet.api import Daemon, make_handler
from ml_stack.fleet.daemon import ALL_INTERFACES, load_or_create_token
from ml_stack.fleet.discovery import Beacon, primary_ip
from ml_stack.fleet.framing import LimitedServer
from ml_stack.fleet.jobs import JobRunner
from ml_stack.fleet.projects import lan_host
from ml_stack.fleet.remote import Peer
from ml_stack.workspace.remote import RemoteWorkspace

MARKER = b"PLANTED-MARKER-7f3a91c2e8b44d05"


@pytest.fixture(autouse=True)
def forget_pins():
    yield
    http._PINNED.clear()


def lan_address():
    address = primary_ip()
    if not address or address.startswith("127."):
        pytest.skip("this machine has no address other than loopback")
    return address


@pytest.fixture
def listener(tmp_path):
    """A real daemon handler on every interface, as the daemon's LAN listener builds it."""
    ident = tls.identity(tmp_path / "tls", "wire")
    root = tmp_path / "d"
    (root / "files").mkdir(parents=True)
    runner = JobRunner(root)
    token = load_or_create_token(root)
    handler = make_handler(Daemon(runner, root / "files", token, name="wire"))
    httpd = LimitedServer((ALL_INTERFACES, 0), handler, tls=tls.server_context(ident))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield ident, httpd.server_port
    runner.shutdown()
    httpd.shutdown()
    httpd.server_close()


class Relay:
    """A TCP relay in front of the listener that keeps every byte that crosses it."""

    def __init__(self, target: tuple[str, int]) -> None:
        self.target, self.seen = target, bytearray()
        self.sock = socket.socket()
        self.sock.bind((target[0], 0))
        self.sock.listen(8)
        self.port = self.sock.getsockname()[1]
        threading.Thread(target=self._accept, daemon=True).start()

    def _accept(self) -> None:
        while True:
            try:
                client, _ = self.sock.accept()
            except OSError:
                return
            upstream = socket.create_connection(self.target, timeout=10)
            for source, sink in ((client, upstream), (upstream, client)):
                threading.Thread(target=self._pump, args=(source, sink), daemon=True).start()

    def _pump(self, source: socket.socket, sink: socket.socket) -> None:
        with contextlib.suppress(OSError):
            while data := source.recv(65536):
                self.seen += data
                sink.sendall(data)
        with contextlib.suppress(OSError):
            sink.shutdown(socket.SHUT_WR)

    def close(self) -> None:
        self.sock.close()


def test_a_request_with_a_planted_marker_crosses_the_wire_only_as_tls_13(listener):
    ident, port = listener
    relay = Relay((lan_address(), port))
    try:
        conn = httpclient.HTTPSConnection(lan_address(), relay.port, context=tls.pinned_context(ident.beacon),
                                           timeout=10)
        conn.request("POST", f"/jobs?q={MARKER.decode()}", body=MARKER, headers={
            "X-Marker": MARKER.decode(), "Content-Type": "application/octet-stream"})
        assert conn.sock.version() == "TLSv1.3"
        conn.getresponse().read()
        conn.close()
    finally:
        relay.close()
    wire = bytes(relay.seen)
    assert len(wire) > 200
    assert MARKER not in wire and b"HTTP/1" not in wire and b"POST /jobs" not in wire
    assert wire[0] == 0x16, "the first thing on the wire is a TLS handshake record"


def test_plain_http_from_another_address_gets_no_answer(listener):
    _, port = listener
    with socket.create_connection((lan_address(), port), timeout=5) as raw:
        raw.sendall(b"GET /health HTTP/1.0\r\nHost: x\r\n\r\n")
        raw.settimeout(5)
        try:
            got = raw.recv(4096)
        except OSError:
            got = b""
    assert b"HTTP/" not in got and b"name" not in got


def test_a_client_that_will_not_go_above_tls_12_is_refused(listener):
    ident, port = listener
    context = tls.pinned_context(ident.beacon)
    context.minimum_version = ssl.TLSVersion.MINIMUM_SUPPORTED
    context.maximum_version = ssl.TLSVersion.TLSv1_2
    with pytest.raises(ssl.SSLError), socket.create_connection((lan_address(), port), timeout=5) as raw:
        context.wrap_socket(raw, server_hostname=lan_address()).close()


def test_a_client_context_offers_nothing_below_tls_13(tmp_path):
    ident = tls.identity(tmp_path / "tls", "wire")
    assert tls.pinned_context(ident.beacon).minimum_version == ssl.TLSVersion.TLSv1_3
    assert tls.server_context(ident).minimum_version == ssl.TLSVersion.TLSv1_3


def test_a_listener_beyond_this_machine_cannot_be_built_without_tls():
    with pytest.raises(ValueError, match="TLS or not at all"):
        LimitedServer((ALL_INTERFACES, 0), lambda *a: None)
    with LimitedServer(("127.0.0.1", 0), lambda *a: None) as local:
        assert local.server_port


def test_the_old_switch_changes_nothing(tmp_path, monkeypatch):
    monkeypatch.setenv("ML_STACK_FLEET_TLS", "off")
    monkeypatch.delenv("ML_STACK_WSL_NETWORK", raising=False)
    monkeypatch.setattr(projects, "primary_ip", lambda: "192.168.40.2")
    assert not hasattr(tls, "disabled")
    assert lan_host(8770) == "https://192.168.40.2:8770"
    bare = Beacon(name="old", host="192.168.40.9", port=8770)
    assert discovery._trusted(bare) is False
    assert discovery._trusted(Beacon(name="here", host="127.0.0.1", port=8770)) is True


@pytest.mark.parametrize("host", ["http://192.168.40.9:8770", "http://studio.local:8770"])
def test_a_project_host_or_peer_over_plain_http_on_the_lan_is_refused(host):
    with pytest.raises(ValueError, match="https"):
        RemoteWorkspace(host, "a" * 32)
    with pytest.raises(ValueError, match="https"):
        Peer(host, "token")
