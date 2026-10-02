"""Finding pairable machines: announcements in, a bounded and aging list out."""

import json

from onboard_support import Clock, Recorder

from ml_stack.fleet.onboard import nearby as near

FP = "ab" * 32


def say(**more):
    body = {"v": 1, "kind": "pair-open", "name": "kitchen-pi", "hostname": "kitchen.local",
            "model": "Linux aarch64", "port": 8772, "fingerprint": FP, **more}
    return json.dumps(body).encode()


def test_an_announcement_is_heard_by_a_browser_through_the_in_memory_transport():
    hub = near.MemoryHub()
    a = near.Announcer(hub.endpoint("10.0.0.7"),
                       near.Presence("kitchen-pi", "kitchen.local", "Linux", 8772, FP))
    browser = near.Browser(hub.endpoint("10.0.0.2"))
    a.announce()
    found = browser.listen(0.2)
    assert [(n.name, n.address, n.port, n.fingerprint) for n in found] == \
        [("kitchen-pi", "10.0.0.7", 8772, FP)]
    assert found[0].public()["fingerprint_short"] == "abab abab abab abab"


def test_announcement_over_real_udp_sockets_on_loopback():
    listener = near.UdpTransport(bind="127.0.0.1", port=0, destinations=[])
    sender = near.UdpTransport(bind="127.0.0.1", port=0,
                               destinations=[("127.0.0.1", listener.port)])
    try:
        a = near.Announcer(sender, near.Presence("den-mac", "den", "Darwin arm64", 8772, FP),
                           interval_s=0.05).start()
        found = near.Browser(listener).listen(0.5)
        a.stop()
        assert found and found[0].name == "den-mac" and found[0].address == "127.0.0.1"
    finally:
        sender.close()
        listener.close()


def test_the_address_is_the_datagrams_source_never_a_claimed_one():
    hub = near.MemoryHub()
    liar = hub.endpoint("10.0.0.66")
    browser = near.Browser(hub.endpoint("10.0.0.2"))
    liar.send(say(address="10.0.0.1", host="10.0.0.1"))
    assert [n.address for n in browser.listen(0.2)] == ["10.0.0.66"]


def test_anything_that_is_not_a_well_formed_announcement_is_dropped():
    hub = near.MemoryHub()
    liar = hub.endpoint("10.0.0.66")
    browser = near.Browser(hub.endpoint("10.0.0.2"))
    for bad in (b"hello", b"[]", say(v=2), say(kind="other"), say(port=0), say(port=70000),
                say(port=True), say(port="8772"), say(fingerprint="zz"),
                say(fingerprint=FP.upper()), b"x" * 5000, say(name="n" * 5000)):
        liar.send(bad)
    assert browser.listen(0.2) == []


def test_names_are_cleaned_before_they_reach_a_terminal():
    hub = near.MemoryHub()
    browser = near.Browser(hub.endpoint("10.0.0.2"))
    hub.endpoint("10.0.0.66").send(say(name="pi\n\x1b[2J\u202eEVIL", hostname="\x07bell"))
    found = browser.listen(0.2)
    assert found and "\x1b" not in found[0].name and "\n" not in found[0].name
    assert "\u202e" not in found[0].name and "\x07" not in found[0].hostname


def test_the_list_is_capped_per_source_and_overall_and_entries_age_out():
    clock, rec = Clock(), Recorder()
    hub = near.MemoryHub()
    flood = hub.endpoint("10.0.0.66")
    browser = near.Browser(hub.endpoint("10.0.0.2"), near.Bounds(ttl_s=30, most=6, per_source=4),
                           bus=rec.bus, clock=clock)
    for i in range(10):
        flood.send(say(fingerprint=f"{i:064x}"))
    assert len(browser.listen(0.2)) == 4          # one address gets four places
    assert rec.of("onboard.nearby.flood")
    other = hub.endpoint("10.0.0.67")
    for i in range(10, 20):
        other.send(say(fingerprint=f"{i:064x}"))
    assert len(browser.listen(0.2)) == 6          # the overall cap
    clock.advance(31)
    assert browser.current() == []


def test_a_machine_does_not_list_itself():
    hub = near.MemoryHub()
    me = near.Browser(hub.endpoint("10.0.0.2"), ignore=frozenset({FP}))
    hub.endpoint("10.0.0.7").send(say())
    assert me.listen(0.2) == []


def test_udp_transport_survives_a_closed_socket_and_a_silent_network():
    t = near.UdpTransport(bind="127.0.0.1", port=0, destinations=[("127.0.0.1", 9)])
    assert t.receive(0.05) is None
    t.send(b"nobody listens")
    t.close()
    assert t.receive(0.05) is None
