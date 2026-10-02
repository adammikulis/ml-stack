"""Fleet traffic off this machine is encrypted to a certificate the beacon vouched for."""

import contextlib
import json
import socket
import threading
import time

import pytest

from ml_stack import http, macauth
from ml_stack.fleet import tls
from ml_stack.fleet.api import Daemon, make_handler
from ml_stack.fleet.daemon import ALL_INTERFACES, load_or_create_token
from ml_stack.fleet.discovery import Beacon, discover, primary_ip
from ml_stack.fleet.framing import LimitedServer
from ml_stack.fleet.jobs import JobRunner
from ml_stack.fleet.remote import Peer, PeerError


@pytest.fixture(autouse=True)
def forget_pins():
    yield
    http._PINNED.clear()


def lan_address():
    address = primary_ip()
    if not address or address.startswith("127."):
        pytest.skip("this machine has no address other than loopback")
    return address


def serve(tmp_path, ident, *, name="d"):
    root = tmp_path / name
    (root / "files").mkdir(parents=True)
    runner = JobRunner(root)
    token = load_or_create_token(root)
    handler = make_handler(Daemon(runner, root / "files", token, name=name))
    httpd = LimitedServer((ALL_INTERFACES, 0), handler, tls=tls.server_context(ident))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, runner, token


def stop(httpd, runner):
    runner.shutdown()
    httpd.shutdown()
    httpd.server_close()


# -- the certificate -------------------------------------------------------------------


def test_a_daemon_makes_a_certificate_once_and_keeps_it(tmp_path):
    first = tls.identity(tmp_path / "tls", "box")
    again = tls.identity(tmp_path / "tls", "box")
    assert again.der == first.der and first.fingerprint == again.fingerprint
    assert first.keyfile.stat().st_mode & 0o077 == 0
    assert (tmp_path / "tls").stat().st_mode & 0o077 == 0


def test_a_certificate_close_to_its_end_is_made_again(tmp_path):
    now = time.time()
    old = tls.identity(tmp_path / "tls", "box", now=now - 70 * 86400)
    assert tls.identity(tmp_path / "tls", "box", now=now - 70 * 86400).der == old.der
    fresh = tls.identity(tmp_path / "tls", "box", now=now)
    assert fresh.der != old.der and fresh.not_after > now + 80 * 86400


def test_a_certificate_made_by_openssl_serves_just_as_well(tmp_path, monkeypatch):
    import shutil

    if shutil.which("openssl") is None:
        pytest.skip("no openssl on this machine")
    monkeypatch.setattr(tls, "_with_cryptography", lambda *a: (_ for _ in ()).throw(ImportError()))
    ident = tls.identity(tmp_path / "tls", "viaopenssl")
    httpd, runner, token = serve(tmp_path, ident)
    try:
        peer = Peer(f"https://127.0.0.1:{httpd.server_port}", token)
        http.pin(f"127.0.0.1:{httpd.server_port}", tls.pinned_context(ident.beacon))
        assert peer.health()["ok"] is True
    finally:
        stop(httpd, runner)


def test_with_neither_cryptography_nor_openssl_the_error_says_what_to_install(tmp_path,
                                                                              monkeypatch):
    monkeypatch.setattr(tls, "_with_cryptography", lambda *a: (_ for _ in ()).throw(ImportError()))
    monkeypatch.setattr(tls.shutil, "which", lambda *_: None)
    with pytest.raises(tls.TlsUnavailable, match="fleet-tls"):
        tls.identity(tmp_path / "tls", "box")


# -- talking to it ----------------------------------------------------------------------


def test_a_pinned_peer_is_served_over_tls_and_the_request_is_signed_inside_it(tmp_path):
    ident = tls.identity(tmp_path / "tls", "d")
    httpd, runner, token = serve(tmp_path, ident)
    try:
        port = httpd.server_port
        http.pin(f"{lan_address()}:{port}", tls.pinned_context(ident.beacon))
        peer = Peer(f"https://{lan_address()}:{port}", token)
        assert peer.health()["name"] == "d"
        assert peer.jobs() == []
    finally:
        stop(httpd, runner)


def test_a_certificate_other_than_the_pinned_one_is_refused(tmp_path):
    ident = tls.identity(tmp_path / "real", "d")
    impostor = tls.identity(tmp_path / "impostor", "d")
    httpd, runner, token = serve(tmp_path, impostor)
    try:
        port = httpd.server_port
        http.pin(f"{lan_address()}:{port}", tls.pinned_context(ident.beacon))
        with pytest.raises(PeerError, match="unreachable"):
            Peer(f"https://{lan_address()}:{port}", token).health()
    finally:
        stop(httpd, runner)


def test_a_certificate_that_has_been_rotated_since_the_beacon_is_refused(tmp_path):
    old = tls.identity(tmp_path / "tls", "d", now=time.time() - 70 * 86400)
    renewed = tls.identity(tmp_path / "tls", "d")
    assert renewed.der != old.der
    httpd, runner, token = serve(tmp_path, renewed)
    try:
        port = httpd.server_port
        http.pin(f"{lan_address()}:{port}", tls.pinned_context(old.beacon))
        with pytest.raises(PeerError, match="unreachable"):
            Peer(f"https://{lan_address()}:{port}", token).health()
        http.pin(f"{lan_address()}:{port}", tls.pinned_context(renewed.beacon))
        assert Peer(f"https://{lan_address()}:{port}", token).health()["ok"]
    finally:
        stop(httpd, runner)


def test_an_expired_certificate_is_refused_even_when_it_is_the_pinned_one(tmp_path):
    expired = tls.identity(tmp_path / "tls", "d", now=time.time() - 200 * 86400, days=90)
    httpd, runner, token = serve(tmp_path, expired)
    try:
        port = httpd.server_port
        http.pin(f"{lan_address()}:{port}", tls.pinned_context(expired.beacon))
        with pytest.raises(PeerError, match="unreachable"):
            Peer(f"https://{lan_address()}:{port}", token).health()
    finally:
        stop(httpd, runner)


def test_a_peer_that_was_never_pinned_is_not_trusted_on_sight(tmp_path):
    ident = tls.identity(tmp_path / "tls", "d")
    httpd, runner, token = serve(tmp_path, ident)
    try:
        with pytest.raises(PeerError, match="unreachable"):
            Peer(f"https://{lan_address()}:{httpd.server_port}", token).health()
    finally:
        stop(httpd, runner)


def test_plain_http_from_another_machine_is_dropped_but_this_machine_may_use_it(tmp_path):
    ident = tls.identity(tmp_path / "tls", "d")
    httpd, runner, token = serve(tmp_path, ident)
    try:
        port = httpd.server_port
        with socket.create_connection((lan_address(), port), timeout=5) as sock:
            sock.sendall(b"GET /health HTTP/1.0\r\n\r\n")
            with contextlib.suppress(ConnectionResetError):
                assert sock.recv(100) == b""
        with socket.create_connection(("127.0.0.1", port), timeout=5) as sock:
            sock.sendall(b"GET /health HTTP/1.0\r\n\r\n")
            assert sock.recv(12).startswith(b"HTTP/1.")
    finally:
        stop(httpd, runner)


def test_a_request_signed_for_one_cluster_is_refused_by_another_over_tls(tmp_path):
    ident = tls.identity(tmp_path / "tls", "d")
    httpd, runner, token = serve(tmp_path, ident)
    try:
        port = httpd.server_port
        context = tls.pinned_context(ident.beacon)
        other = macauth.derive(b"another-clusters-key-of-32-bytes!")
        http.pin(f"{lan_address()}:{port}", context)
        with pytest.raises(PeerError, match="401"):
            Peer(f"https://{lan_address()}:{port}", other).health()
        # the daemon's own request, replayed byte for byte on a second connection
        url = f"https://{lan_address()}:{port}/jobs"
        sent = macauth.sign(token, "GET", url, None)
        raw = (f"GET /jobs HTTP/1.1\r\nHost: {lan_address()}:{port}\r\nAuthorization: "
               f"{sent['Authorization']}\r\nConnection: close\r\n\r\n").encode()
        replies = []
        for _ in range(2):
            with socket.create_connection((lan_address(), port), timeout=5) as sock, \
                    context.wrap_socket(sock) as secure:
                secure.sendall(raw)
                replies.append(int(secure.recv(20).split()[1]))
        assert replies == [200, 401]
    finally:
        stop(httpd, runner)


# -- the beacon -------------------------------------------------------------------------


def test_a_beacon_carries_the_certificate_and_names_an_https_address(tmp_path):
    ident = tls.identity(tmp_path / "tls", "d")
    beacon = Beacon(name="d", port=8770, host="10.1.2.3", cert=ident.beacon)
    assert beacon.base_url == "https://10.1.2.3:8770" and beacon.public()["cert"] == ident.beacon
    assert Beacon(name="d", port=8770, host="10.1.2.3").base_url == "http://10.1.2.3:8770"


def test_a_machine_that_offers_no_certificate_is_not_talked_to_unless_that_is_named(
        tmp_path, monkeypatch):
    from ml_stack.fleet import discovery

    ident = tls.identity(tmp_path / "tls", "d")
    bare = Beacon(name="bare", port=1, host="192.0.2.9")
    sealed = Beacon(name="sealed", port=2, host="192.0.2.10", cert=ident.beacon)
    here = Beacon(name="here", port=3, host="127.0.0.1")
    assert discovery._trusted(bare) is False
    assert discovery._trusted(sealed) is True and "192.0.2.10:2" in http._PINNED
    assert discovery._trusted(here) is True
    monkeypatch.setenv(tls.ENV, "off")
    assert discovery._trusted(bare) is True


def test_an_advertised_certificate_is_pinned_by_discovery_over_real_sockets(tmp_path):
    from ml_stack.fleet.discovery import Advertiser, create_cluster_key

    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
        probe.bind(("", 0))
        udp = probe.getsockname()[1]
    ident = tls.identity(tmp_path / "tls", "d")
    key = create_cluster_key(tmp_path / "k").encode()
    tell = Advertiser(Beacon(name="d", port=8770, cert=ident.beacon), key, port=udp,
                      interval_s=30).start()
    try:
        found = discover(key, timeout_s=3.0, port=udp)
        assert [b.name for b in found] == ["d"] and found[0].cert == ident.beacon
        assert any(k.endswith(":8770") for k in http._PINNED)
    finally:
        tell.stop()


def test_json_of_a_beacon_never_holds_the_key_the_secret_or_the_private_half(tmp_path):
    ident = tls.identity(tmp_path / "tls", "d")
    text = json.dumps(Beacon(name="d", port=1, cert=ident.beacon).public())
    assert ident.keyfile.read_text().splitlines()[1] not in text
