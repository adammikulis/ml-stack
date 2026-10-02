"""Pairing over real TLS sockets on loopback: the right code, the wrong code, a machine in the
middle, replays, and requests that were never accepted."""

import http.client
import json
import threading
from http.server import BaseHTTPRequestHandler

import pytest
from onboard_support import Clock, Recorder, grant_for, identity, requests

from ml_stack.fleet import tls
from ml_stack.fleet.framing import Limited, LimitedServer
from ml_stack.fleet.onboard.pairing import (
    API,
    Hooks,
    PairError,
    PairingClient,
    PairingServer,
    unverified_context,
)
from ml_stack.fleet.onboard.requests import State


@pytest.fixture
def world(tmp_path):
    rec = Recorder()
    acceptor, joiner = identity(tmp_path, "acceptor"), identity(tmp_path, "joiner")
    rq = requests(tmp_path, rec, Clock())
    told = []
    server = PairingServer(rq, acceptor, Hooks(lambda r: grant_for(acceptor), told.append),
                           address=("127.0.0.1", 0), bus=rec.bus).start()
    yield type("World", (), {"rq": rq, "rec": rec, "server": server, "acceptor": acceptor,
                             "joiner": joiner, "told": told})
    server.stop()


def client(world, port=None):
    return PairingClient("127.0.0.1", port or world.server.port,
                         fingerprint=world.joiner.fingerprint)


def ask(c):
    return c.ask(name="kitchen-pi", hostname="kitchen.local", model="Linux aarch64")


def test_the_right_code_pairs_and_delivers_the_grant(world):
    c = client(world)
    request_id = ask(c)
    assert c.state() == "pending"
    assert world.told and world.told[0].id == request_id        # the owner was told
    code = world.rq.accept(request_id).code
    assert c.state() == "accepted"
    grant = c.finish(code)
    assert (grant.group, grant.key, grant.salt) == ("home", "secret-cluster-key", "c2FsdA")
    assert grant.certificate == world.acceptor.beacon
    assert world.rq.get(request_id).state is State.PAIRED
    device = world.rq.devices.all()[0]
    assert device.fingerprint == world.joiner.fingerprint and device.shared_cluster_key
    assert c.server_fingerprint == world.acceptor.fingerprint


def test_a_code_typed_with_a_space_still_works(world):
    c = client(world)
    code = world.rq.accept(ask(c)).code
    assert c.finish(f"{code[:3]} {code[3:]}").key


def test_a_wrong_code_gets_two_more_tries_then_the_request_is_dead(world):
    c = client(world)
    request_id = ask(c)
    code = world.rq.accept(request_id).code
    wrong = "000000" if code != "000000" else "111111"
    tries = []
    for _ in range(3):
        with pytest.raises(PairError) as err:
            c.finish(wrong)
        tries.append(err.value.tries_left)
    assert tries == [2, 1, 0]
    assert world.rq.get(request_id).state is State.FAILED
    with pytest.raises(PairError):                              # the right code is too late
        c.finish(code)
    assert world.rq.devices.all() == []
    assert world.rec.of("onboard.pair.locked")


def test_the_server_reveals_nothing_checkable_before_the_asker_proves_the_code(world):
    """An asker who guesses can only learn yes or no, once per try: the reply to a bad
    confirmation has no confirmation, grant or tag in it."""
    c = client(world)
    request_id = ask(c)
    world.rq.accept(request_id)
    from ml_stack.fleet.onboard import spake
    guess = spake.start_initiator("000000", context=b"x", mine=world.joiner.fingerprint,
                                  theirs=world.acceptor.fingerprint)
    status, body = c._call("POST", f"{API}/{request_id}/exchange", {"message": guess.message})
    assert status == 200 and set(body) == {"message"}
    status, body = c._call("POST", f"{API}/{request_id}/confirm", {"confirmation": "0" * 64})
    assert status == 403 and set(body) == {"error"}


def test_a_request_nobody_accepted_cannot_start_an_exchange(world):
    c = client(world)
    request_id = ask(c)
    from ml_stack.fleet.onboard import spake
    guess = spake.start_initiator("123456", context=b"x", mine="a", theirs="b")
    status, body = c._call("POST", f"{API}/{request_id}/exchange", {"message": guess.message})
    assert status == 409 and set(body) == {"error"}
    with pytest.raises(PairError):
        c.finish("123456")
    assert world.rq.get(request_id).attempts == 0               # no try was spent either


def test_a_confirmation_cannot_be_replayed(world):
    c = client(world)
    request_id = ask(c)
    code = world.rq.accept(request_id).code
    from ml_stack.fleet.onboard import spake
    from ml_stack.fleet.onboard.pairing import context_for
    session = spake.start_initiator(code, context=context_for(request_id, c.nonce),
                                    mine=world.joiner.fingerprint,
                                    theirs=world.acceptor.fingerprint)
    _, body = c._call("POST", f"{API}/{request_id}/exchange", {"message": session.message})
    session.receive(body["message"])
    mine = {"confirmation": session.confirmation()}
    status, _ = c._call("POST", f"{API}/{request_id}/confirm", mine)
    assert status == 200
    status, _ = c._call("POST", f"{API}/{request_id}/confirm", mine)     # recorded and resent
    assert status == 409


def test_an_identical_request_is_refused_while_the_first_waits(world):
    c, d = client(world), client(world)
    ask(c)
    with pytest.raises(PairError) as err:
        ask(d)
    assert err.value.status == 409
    assert len(world.told) == 1                                 # one notification, not two


def test_a_notifier_that_breaks_does_not_lose_the_request(tmp_path):
    rec = Recorder()
    acceptor, joiner = identity(tmp_path, "a"), identity(tmp_path, "j")
    rq = requests(tmp_path, rec)

    def broken(_):
        raise RuntimeError("no display")

    with PairingServer(rq, acceptor, Hooks(lambda r: grant_for(acceptor), broken),
                       address=("127.0.0.1", 0), bus=rec.bus) as server:
        c = PairingClient("127.0.0.1", server.port, fingerprint=joiner.fingerprint)
        request_id = ask(c)
    assert rq.get(request_id).state is State.PENDING
    assert "onboard.notify.failed" in rec.kinds()


def test_plain_http_is_refused_even_from_this_machine(world):
    conn = http.client.HTTPConnection("127.0.0.1", world.server.port, timeout=5)
    conn.request("POST", API, body=b"{}")
    response = conn.getresponse()
    assert response.status == 403
    assert world.rq.pending() == []


def test_oversize_and_malformed_bodies_are_refused(world):
    conn = http.client.HTTPSConnection("127.0.0.1", world.server.port, timeout=5,
                                       context=unverified_context())
    conn.request("POST", API, body=b"x" * 40_000)
    assert conn.getresponse().status == 413
    conn = http.client.HTTPSConnection("127.0.0.1", world.server.port, timeout=5,
                                       context=unverified_context())
    conn.request("POST", API, body=b"not json")
    assert conn.getresponse().status == 400
    conn = http.client.HTTPSConnection("127.0.0.1", world.server.port, timeout=5,
                                       context=unverified_context())
    conn.request("POST", API, body=b'{"fingerprint": "zz"}', headers={"Content-Length": "20"})
    assert conn.getresponse().status == 400


def test_an_address_that_probes_ids_is_locked_out(world):
    c = client(world)
    statuses = []
    for _ in range(25):
        statuses.append(c._call("GET", f"{API}/{'0' * 32}")[0])
    assert statuses[0] == 404 and statuses[-1] == 429


def test_a_peer_cannot_be_swapped_for_another_mid_conversation(world, tmp_path):
    c = client(world)
    ask(c)
    other = identity(tmp_path, "other")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    with PairingServer(requests(elsewhere, Recorder()), other,
                       Hooks(lambda r: grant_for(other)), address=("127.0.0.1", 0)) as impostor:
        c.port = impostor.port          # same client object, the machine behind the port changed
        with pytest.raises(PairError, match="different certificate"):
            c.state()


# -- a machine in the middle ---------------------------------------------------------------
class Relay:
    """A TLS-terminating relay with its own certificate: what a hostile machine on the path
    would be. It passes every request to the real server unchanged and every answer back; with
    ``rewrite`` it also swaps the real server's fingerprint in the first answer for its own,
    the cleverest thing it can do."""

    def __init__(self, ident, upstream_port, *, rewrite):
        self.upstream_port, self.rewrite, self.ident = upstream_port, rewrite, ident
        relay = self

        class Handler(Limited, BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.0"

            def log_message(self, *_):
                return

            def _pass(self):
                length = int(self.headers.get("Content-Length") or 0)
                body = self.rfile.read(length) if length else None
                up = http.client.HTTPSConnection("127.0.0.1", relay.upstream_port,
                                                 context=unverified_context(), timeout=10)
                up.request(self.command, self.path, body=body,
                           headers={"Content-Type": "application/json"})
                answer = up.getresponse()
                data = answer.read()
                if relay.rewrite and b'"server"' in data:
                    doc = json.loads(data)
                    doc["server"] = relay.ident.fingerprint
                    data = json.dumps(doc).encode()
                self.send_response(answer.status)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            do_GET = do_POST = _pass

        self.httpd = LimitedServer(("127.0.0.1", 0), Handler, tls=tls.server_context(ident))
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        self.port = self.httpd.server_address[1]

    def stop(self):
        self.httpd.shutdown()
        self.httpd.server_close()


def test_a_machine_in_the_middle_cannot_complete_a_pairing_even_with_the_right_code(
        world, tmp_path):
    relay = Relay(identity(tmp_path, "mitm"), world.server.port, rewrite=True)
    try:
        c = client(world, relay.port)
        request_id = ask(c)                  # the relay's lie is consistent: the asker sees nothing
        assert c.server_fingerprint == relay.ident.fingerprint != world.acceptor.fingerprint
        code = world.rq.accept(request_id).code
        with pytest.raises(PairError):
            c.finish(code)                   # the right code, and still no pairing
        assert world.rq.get(request_id).state is not State.PAIRED
        assert world.rq.devices.all() == []
        assert world.rec.of("onboard.pair.wrong_code")
    finally:
        relay.stop()


def test_unrewritten_relay_is_caught_before_any_code_is_asked_for(world, tmp_path):
    relay = Relay(identity(tmp_path, "mitm2"), world.server.port, rewrite=False)
    try:
        with pytest.raises(PairError, match="not the one it names"):
            ask(client(world, relay.port))
    finally:
        relay.stop()



class Bluffer(PairingServer):
    """A listener that hands out a grant without having proved it knew the code."""

    def confirm(self, request_id, confirmation):
        answer = super().confirm(request_id, confirmation)
        return {**answer, "confirmation": "0" * 64}


def test_the_asker_refuses_a_grant_from_a_machine_that_did_not_prove_the_code(tmp_path):
    rec = Recorder()
    acceptor, joiner = identity(tmp_path, "a"), identity(tmp_path, "j")
    rq = requests(tmp_path, rec, Clock())
    with Bluffer(rq, acceptor, Hooks(lambda r: grant_for(acceptor)),
                 address=("127.0.0.1", 0), bus=rec.bus) as server:
        c = PairingClient("127.0.0.1", server.port, fingerprint=joiner.fingerprint)
        code = rq.accept(ask(c)).code
        with pytest.raises(PairError, match="did not prove"):
            c.finish(code)


def _open_exchange(c, world, request_id, code, *, context=None):
    """The asker's side of one exchange, by hand: returns the session after the server's reply."""
    from ml_stack.fleet.onboard import spake
    from ml_stack.fleet.onboard.pairing import context_for
    session = spake.start_initiator(code, context=context or context_for(request_id, c.nonce),
                                    mine=world.joiner.fingerprint,
                                    theirs=world.acceptor.fingerprint)
    status, body = c._call("POST", f"{API}/{request_id}/exchange", {"message": session.message})
    if status == 200:
        session.receive(body["message"])
    return status, session


def test_an_asker_who_opens_exchanges_and_never_confirms_runs_out_of_tries(world):
    c = client(world)
    request_id = ask(c)
    code = world.rq.accept(request_id).code
    statuses = [_open_exchange(c, world, request_id, code)[0] for _ in range(4)]
    assert statuses == [200, 200, 200, 429]
    assert world.rq.get(request_id).state is State.FAILED


def test_a_malformed_message_costs_a_try_and_the_third_closes_the_request(world):
    c = client(world)
    request_id = ask(c)
    world.rq.accept(request_id)
    for _ in range(3):
        status, _ = c._call("POST", f"{API}/{request_id}/exchange", {"message": "04" + "11" * 64})
        assert status == 400
    assert world.rq.get(request_id).state is State.FAILED
    assert len(world.rec.of("onboard.pair.wrong_code")) == 3


def test_a_wrong_confirmation_cannot_be_retried_on_the_same_exchange(world):
    """If it could, one exchange would allow a million guesses at the code without another
    try being spent."""
    c = client(world)
    request_id = ask(c)
    code = world.rq.accept(request_id).code
    _, session = _open_exchange(c, world, request_id, code)
    status, _ = c._call("POST", f"{API}/{request_id}/confirm", {"confirmation": "0" * 64})
    assert status == 403
    status, _ = c._call("POST", f"{API}/{request_id}/confirm",
                        {"confirmation": session.confirmation()})      # the right one, too late
    assert status == 409
    assert world.rq.devices.all() == []


class BadTag(PairingServer):
    def confirm(self, request_id, confirmation):
        return {**super().confirm(request_id, confirmation), "tag": "0" * 64}


def test_the_asker_refuses_a_grant_whose_tag_is_not_the_exchanges(tmp_path):
    rec = Recorder()
    acceptor, joiner = identity(tmp_path, "a"), identity(tmp_path, "j")
    rq = requests(tmp_path, rec, Clock())
    with BadTag(rq, acceptor, Hooks(lambda r: grant_for(acceptor)),
                address=("127.0.0.1", 0), bus=rec.bus) as server:
        c = PairingClient("127.0.0.1", server.port, fingerprint=joiner.fingerprint)
        code = rq.accept(ask(c)).code
        with pytest.raises(PairError, match="tag"):
            c.finish(code)
