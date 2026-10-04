"""Joining a cluster with its passphrase: a real daemon, a real advertiser, a real handshake over HTTP
between two machines' roots. Nothing but the loopback address stands in for a LAN."""

from __future__ import annotations

import base64
import json
import re
import socket
import threading

import pytest

from ml_stack import http, macauth, sealing
from ml_stack.fleet.api import Daemon, make_handler
from ml_stack.fleet.availability import Availability
from ml_stack.fleet.daemon import load_or_create_token
from ml_stack.fleet.discovery import (
    Advertiser,
    Beacon,
    DiscoveryError,
    derive_token,
    discover,
    load_cluster_key,
    memberships,
    mint_cluster,
)
from ml_stack.fleet.jobs import JobRunner
from ml_stack.fleet.onboard import joining, pake
from ml_stack.fleet.onboard.joining import Joining, join_by_passphrase
from ml_stack.fleet.remote import Peer, PeerError

WORDS = "quince larch marlow"


def _free_port(kind: int) -> int:
    with socket.socket(socket.AF_INET, kind) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Machine:
    """A daemon in cluster ``lab`` that holds ``WORDS`` and answers joins."""

    def __init__(self, tmp_path, udp: int) -> None:
        self.keyfile = tmp_path / "a" / "cluster.key"
        self.keyfile.parent.mkdir()
        self.member = mint_cluster("lab", self.keyfile)
        root = tmp_path / "a" / "traind"
        (root / "files").mkdir(parents=True)
        self.runner = JobRunner(root)
        self.logged: list[str] = []
        self.captured: list[tuple[str, dict, dict]] = []
        self.joining = Joining(lambda: memberships(self.keyfile), lambda group: WORDS,
                               log=self.logged.append)
        handle = self.joining.handle

        def record(path, body, source):
            status, answer = handle(path, body, source)
            self.captured.append((path, body, answer))
            return status, answer

        self.joining.handle = record
        token = load_or_create_token(root, self.member.key)
        self.httpd = http.Server(("127.0.0.1", 0), make_handler(
            Daemon(self.runner, root / "files", token, name="a", joining=self.joining)))
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        self.port = self.httpd.server_port
        self.advertiser = Advertiser(Beacon(name="a", port=self.port), self.member.key,
                                     port=udp, cluster="lab", interval_s=0.2).start()

    def stop(self) -> None:
        self.advertiser.stop()
        self.runner.shutdown()
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture
def udp() -> int:
    return _free_port(socket.SOCK_DGRAM)


@pytest.fixture
def machine(tmp_path, udp):
    box = Machine(tmp_path, udp)
    try:
        yield box
    finally:
        box.stop()


def _join(tmp_path, udp, words=WORDS, name="b"):
    return join_by_passphrase(words, "lab", tmp_path / name / "cluster.key", timeout_s=1.0, port=udp)


def test_a_new_machine_given_the_passphrase_receives_the_cluster_key(machine, tmp_path, udp):
    got = _join(tmp_path, udp)
    assert got.group == "lab" and got.key == machine.member.key
    assert load_cluster_key(tmp_path / "b" / "cluster.key") == machine.member.key
    assert any("joined from" in line for line in machine.logged)


def test_the_joined_machine_finds_and_drives_the_first_with_what_it_received(machine, tmp_path, udp):
    got = _join(tmp_path, udp)
    found = discover(got.key, timeout_s=2.0, port=udp)
    assert [b.name for b in found] == ["a"]
    peer = Peer(f"http://127.0.0.1:{machine.port}", derive_token(got.key))
    assert peer.health()["name"] == "a"
    assert peer.jobs() == []


def test_a_wrong_passphrase_is_refused_and_nothing_is_stored(machine, tmp_path, udp):
    with pytest.raises(DiscoveryError, match="none accepts that passphrase"):
        _join(tmp_path, udp, "not the words")
    assert memberships(tmp_path / "b" / "cluster.key") == []
    assert [a[3] for a in machine.joining.attempts if a[3].startswith("refused")] == [
        "refused, wrong passphrase"]


def test_a_source_that_keeps_failing_is_locked_out_even_with_the_right_words(machine, tmp_path, udp):
    for _ in range(5):
        with pytest.raises(DiscoveryError):
            _join(tmp_path, udp, "not the words")
    with pytest.raises(DiscoveryError, match="too many attempts"):
        _join(tmp_path, udp)
    assert memberships(tmp_path / "b" / "cluster.key") == []
    assert machine.joining.attempts[-1][3] == "refused, too many attempts"
    assert sum("wrong passphrase" in line for line in machine.logged) == 5


def test_a_passphrase_under_five_characters_is_refused_before_anything_is_sent(machine, tmp_path, udp):
    with pytest.raises(DiscoveryError, match=r"The passphrase needs at least 5 characters\."):
        _join(tmp_path, udp, "abcd")
    assert machine.captured == []


def test_a_machine_that_finds_nobody_makes_the_cluster_and_keeps_it(tmp_path, udp):
    first = _join(tmp_path, udp)
    again = _join(tmp_path, udp)
    assert first.group == "lab" and again.key == first.key
    assert len(base64.urlsafe_b64decode(first.key + b"=")) == 32


def test_a_captured_join_holds_nothing_to_test_a_guess_against(machine, tmp_path, udp):
    got = _join(tmp_path, udp)
    wire = json.dumps(machine.captured).encode()
    assert got.key not in wire and base64.b64decode(base64.b64encode(got.key)) not in wire
    (_, start, started), (_, finish, done) = machine.captured
    for words in (WORDS, "password", "quince"):
        context = joining._context("lab", start["nonce"])
        attacker = pake.start_initiator(words, context=context, mine=joining.CLIENT,
                                        theirs=joining.PLAIN)
        try:
            attacker.receive(started["message"])
        except pake.Bad:
            continue
        assert attacker.confirmation() != finish["confirmation"]
        assert not attacker.check(done["confirmation"])


def test_an_answer_altered_on_the_way_is_refused(machine, tmp_path, udp, monkeypatch):
    real = joining._Call.post

    def altered(self, step, body):
        status, answer = real(self, step, body)
        if step == "finish" and "sealed" in answer:
            raw = bytearray.fromhex(answer["sealed"])
            raw[-1] ^= 1
            answer = {**answer, "sealed": raw.hex()}
        return status, answer

    monkeypatch.setattr(joining._Call, "post", altered)
    with pytest.raises(DiscoveryError):
        _join(tmp_path, udp)
    assert memberships(tmp_path / "b" / "cluster.key") == []


def test_a_finish_for_a_join_nobody_started_is_refused(machine):
    status, answer = machine.joining.handle("/join/v1/finish", {"id": "0" * 32, "confirmation": "x"},
                                            "127.0.0.1")
    assert status == 409 and "no join" in answer["error"]


def test_a_finish_from_another_source_is_refused(machine):
    context = joining._context("lab", "n")
    session = pake.start_initiator(WORDS, context=context, mine=joining.CLIENT, theirs=joining.PLAIN)
    _, started = machine.joining.handle("/join/v1/start", {"group": "lab", "nonce": "n",
                                                           "message": session.message}, "10.0.0.5")
    session.receive(started["message"])
    status, _ = machine.joining.handle("/join/v1/finish", {"id": started["id"],
                                                           "confirmation": session.confirmation()}, "10.0.0.6")
    assert status == 409


def test_a_machine_with_no_passphrase_to_check_takes_nobody_in(tmp_path):
    keyfile = tmp_path / "c.key"
    mint_cluster("lab", keyfile)
    keeper = Joining(lambda: memberships(keyfile), lambda group: None)
    status, answer = keeper.handle("/join/v1/start", {"group": "lab", "message": "00"}, "10.0.0.5")
    assert status == 404 and "cannot take" in answer["error"]


# -- traffic -------------------------------------------------------------------------------------


@pytest.fixture
def served(tmp_path):
    root = tmp_path / "d"
    (root / "files").mkdir(parents=True)
    runner = JobRunner(root)
    key = mint_cluster("lab", tmp_path / "d.key").key
    token = derive_token(key)
    httpd = http.Server(("127.0.0.1", 0), make_handler(Daemon(runner, root / "files", token, name="sealed-box",
                                                                    schedule=Availability())))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{httpd.server_port}", token
    finally:
        runner.shutdown()
        httpd.shutdown()
        httpd.server_close()


def _raw(url, method, headers, body=b""):
    import http.client

    host, target = macauth.parts(url)
    conn = http.client.HTTPConnection(host, timeout=5)
    try:
        conn.request(method, target, body=body or None, headers=headers)
        response = conn.getresponse()
        return response.status, response.read(), response
    finally:
        conn.close()


def _signed(url, token, method, payload, tamper=False):
    stamp = macauth.Stamp.now()
    host, target = macauth.parts(url)
    body = sealing.seal(sealing.box_key(token), json.dumps(payload).encode(),
                        sealing.request_data(method, target, host, f"{stamp.at:.0f}", stamp.nonce))
    if tamper:
        body = body[:-1] + bytes([body[-1] ^ 1])
    return {**macauth.sign(token, method, url, body, stamp), sealing.HEADER: "2"}, body


def _opened(token, headers, status, answer):
    nonce = re.search(r"n=([0-9a-f]+)", headers["Authorization"])[1]
    return json.loads(sealing.open_(sealing.box_key(token), answer, sealing.response_data(nonce, status)))


def test_what_a_peer_sends_is_not_readable_on_the_wire(served):
    import unittest.mock as mock

    base, token = served
    seen = []
    real = http.build_request

    def spy(*args, **kwargs):
        request = real(*args, **kwargs)
        seen.append(request.data)
        return request

    with mock.patch.object(http, "build_request", spy):
        got = Peer(base, token)._json("POST", "/availability", {"action": "reserve", "holder": "hunter2-holder"})
    assert got["reserved"]
    assert seen and b"hunter2-holder" not in seen[0] and b"reserve" not in seen[0]


def test_what_a_peer_is_told_is_not_readable_on_the_wire(served):
    base, token = served
    url = f"{base}/availability"
    status, answer, reply = _raw(url, "GET", macauth.sign(token, "GET", url, None) | {sealing.HEADER: "2"})
    assert status == 200 and reply.getheader(sealing.HEADER) == "1"
    assert b"available" not in answer
    with pytest.raises(ValueError):
        json.loads(answer)


def test_a_health_answer_is_sealed_and_opens_for_the_peer_that_asked(served):
    base, token = served
    url = f"{base}/health"
    stamp = macauth.Stamp.now()
    status, answer, _ = _raw(url, "GET", macauth.sign(token, "GET", url, None, stamp) | {sealing.HEADER: "2"})
    assert status == 200 and b"sealed-box" not in answer
    opened = sealing.open_(sealing.box_key(token), answer, sealing.response_data(stamp.nonce, 200))
    assert json.loads(opened)["name"] == "sealed-box"
    with pytest.raises(sealing.SealError):
        sealing.open_(sealing.box_key(token), answer, sealing.response_data("another-nonce-value", 200))


def test_a_body_altered_after_it_was_signed_over_is_refused(served):
    base, token = served
    url = f"{base}/availability"
    headers, body = _signed(url, token, "POST", {"action": "pause"}, tamper=True)
    status, answer, _ = _raw(url, "POST", headers, body)
    assert status == 400 and "did not authenticate" in _opened(token, headers, status, answer)["error"]


def test_a_body_sealed_for_another_request_is_refused(served):
    base, token = served
    stamp = macauth.Stamp.now()
    other = macauth.parts(f"{base}/jobs")
    body = sealing.seal(sealing.box_key(token), b'{"action": "pause"}',
                        sealing.request_data("POST", other[1], other[0], f"{stamp.at:.0f}", stamp.nonce))
    url = f"{base}/availability"
    headers = {**macauth.sign(token, "POST", url, body, stamp), sealing.HEADER: "2"}
    status, answer, _ = _raw(url, "POST", headers, body)
    assert status == 400 and "did not authenticate" in _opened(token, headers, status, answer)["error"]


def test_a_body_sent_without_sealing_is_refused(served):
    base, token = served
    url = f"{base}/availability"
    body = b'{"action": "pause"}'
    status, answer, _ = _raw(url, "POST", macauth.sign(token, "POST", url, body), body)
    assert status == 400 and b"sent sealed" in answer


def test_a_sealed_request_sent_twice_is_refused_the_second_time(served):
    base, token = served
    url = f"{base}/availability"
    headers, body = _signed(url, token, "POST", {"action": "resume"})
    assert _raw(url, "POST", headers, body)[0] == 200
    status, answer, _ = _raw(url, "POST", headers, body)
    assert status == 401 and b"already seen" in answer


def test_an_answer_that_should_be_sealed_and_is_not_is_refused_by_the_peer(served, monkeypatch):
    base, token = served
    monkeypatch.setattr(sealing, "seal", lambda key, plain, data: plain)
    with pytest.raises(PeerError):
        Peer(base, token).health()
