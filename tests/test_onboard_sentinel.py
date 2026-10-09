"""Onboarding events reach a sentinel through the adapter docs/onboarding.md shows, and each
step of a pairing leaves its event. The sentinel half runs only where sentinel is merged."""

import pytest
from onboard_support import Clock, Recorder, identity, info, requests

from ml_stack.fleet.onboard.pairing import Grant, Hooks, PairError, PairingClient, PairingServer
from ml_stack.fleet.onboard.requests import Refused


@pytest.fixture(autouse=True)
def needs_spake2():
    pytest.importorskip("spake2")



def watch_onboarding(bus, sentinel, Event, Severity):
    """The adapter from docs/onboarding.md, verbatim."""
    return bus.subscribe(lambda e: sentinel.bus.emit(
        Event(e.kind, Severity.parse(e.severity), "onboard", e.subject, e.evidence, e.ts)))


def test_a_whole_pairing_leaves_the_events_a_watcher_needs(tmp_path):
    rec = Recorder()
    acceptor, joiner = identity(tmp_path, "a"), identity(tmp_path, "j")
    rq = requests(tmp_path, rec, Clock())
    with PairingServer(rq, acceptor, Hooks(lambda r: Grant(group="g", key="k")),
                       bus=rec.bus, address=("127.0.0.1", 0)) as server:
        c = PairingClient("127.0.0.1", server.port, fingerprint=joiner.fingerprint)
        rid = c.ask(name="n", hostname="h", model="m")
        code = rq.accept(rid, mine=True).code
        wrong = "000000" if code != "000000" else "111111"
        with pytest.raises(PairError):
            c.finish(wrong)
        c.finish(code)
    kinds = rec.kinds()
    assert kinds == ["onboard.request.received", "onboard.request.accepted",
                     "onboard.pair.wrong_code", "onboard.pair.succeeded"]
    assert [e.severity for e in rec.events] == ["notice", "notice", "warning", "notice"]
    assert not any(code in str(e.evidence) or code in e.subject for e in rec.events)


def test_every_refusal_and_lock_is_an_event(tmp_path):
    rec = Recorder()
    rq = requests(tmp_path, rec, Clock())
    first = rq.submit(info("1" * 64), "10.0.0.5")
    with pytest.raises(Refused):                      # a second ask from the same device
        rq.submit(info("1" * 64, nonce="cd" * 16), "10.0.0.6")
    rq.decline(first.id)
    assert rec.kinds() == ["onboard.request.received", "onboard.request.refused",
                           "onboard.request.declined"]


def test_the_adapter_hands_events_to_a_real_sentinel(tmp_path):
    sentinel = pytest.importorskip("ml_stack.sentinel")
    rec = Recorder()
    s = sentinel.Sentinel(tmp_path / "sentinel")
    stop = watch_onboarding(rec.bus, s, sentinel.Event, sentinel.Severity)
    rec.bus.emit("onboard.pair.locked", "critical", "request:abc", reason="x")
    stop()
    seen = s.bus.recent(kind="onboard.")
    assert [(e.kind, e.severity.name, e.source) for e in seen] == \
        [("onboard.pair.locked", "CRITICAL", "onboard")]
