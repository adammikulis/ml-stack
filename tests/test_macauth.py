"""A signed request: what a MAC covers, what is refused, and what is only accepted once."""

import pytest

from poolhouse import macauth
from poolhouse.macauth import Authenticator, Lockout, Stamp

KEY = b"a-cluster-key-of-thirty-two-bytes"
SECRET = macauth.derive(KEY)
OTHER = macauth.derive(b"some-other-clusters-key-32-bytes")
URL = "http://10.1.2.3:8770/jobs?tail=5"
NOW = 1_700_000_000.0


class Clock:
    def __init__(self, at=NOW):
        self.at = at

    def __call__(self):
        return self.at


def _send(auth, **changed):
    """One POST to /jobs, signed with SECRET; ``changed`` alters what is signed or sent."""
    sent = {"secret": SECRET, "method": "POST", "url": URL, "body": b'{"argv": ["true"]}',
            "host": "10.1.2.3:8770", "stamp": Stamp(NOW, "n" * 24), "who": "10.9.9.9",
            "target": "/jobs?tail=5", "sent_method": None, **changed}
    headers = macauth.sign(sent["secret"], sent["method"], sent["url"], sent["body"],
                           sent["stamp"])
    headers["Host"] = sent["host"]
    return auth.check(sent["sent_method"] or sent["method"], sent["target"], headers,
                      sent["body"], sent["who"])


def _auth(**kw):
    return Authenticator(lambda: [SECRET], clock=kw.pop("clock", Clock()), **kw)


def test_the_secret_is_derived_the_same_way_on_both_ends_and_is_not_the_key():
    assert macauth.derive(KEY) == SECRET and SECRET.startswith(macauth.PREFIX)
    assert KEY.decode() not in SECRET and SECRET != OTHER


def test_a_signed_request_is_accepted():
    assert _send(_auth()).ok


def test_the_secret_is_not_in_what_is_sent():
    headers = macauth.sign(SECRET, "GET", URL, None)
    assert SECRET not in str(headers) and SECRET[len(macauth.PREFIX):] not in str(headers)


@pytest.mark.parametrize("change", [
    {"sent_method": "GET"}, {"target": "/jobs?tail=6"}, {"target": "/other?tail=5"},
    {"host": "10.1.2.4:8770"},
], ids=["method", "query", "path", "host"])
def test_changing_what_the_mac_covers_refuses_the_request(change):
    assert not _send(_auth(), **change).ok


def test_a_different_body_is_refused():
    auth = _auth()
    headers = macauth.sign(SECRET, "POST", URL, b"one", Stamp(NOW, "n" * 24))
    headers["Host"] = "10.1.2.3:8770"
    assert not auth.check("POST", "/jobs?tail=5", headers, b"two", "1.1.1.1").ok
    assert not auth.check("POST", "/jobs?tail=5", headers, b"", "1.1.1.1").ok


def test_an_empty_body_and_no_body_sign_alike():
    assert _send(_auth(), body=None).ok


def test_the_wrong_secret_is_refused_and_so_is_one_of_another_cluster():
    assert not _send(_auth(), secret=OTHER).ok
    assert not _send(_auth(), secret="mlsk1.guess").ok


def test_a_machine_in_two_clusters_answers_to_both():
    auth = Authenticator(lambda: [SECRET, OTHER], clock=Clock())
    assert _send(auth, secret=OTHER).ok
    assert _send(auth, stamp=Stamp(NOW, "m" * 24)).ok


def test_a_request_is_accepted_once():
    auth = _auth()
    assert _send(auth).ok
    again = _send(auth)
    assert not again.ok and "already seen" in again.reason


def test_a_second_request_with_a_new_nonce_is_accepted():
    auth = _auth()
    assert _send(auth).ok and _send(auth, stamp=Stamp(NOW, "z" * 24)).ok


def test_a_request_outside_the_window_is_refused_and_says_the_clocks():
    clock = Clock()
    auth = _auth(clock=clock)
    clock.at = NOW + 121
    late = _send(auth)
    assert not late.ok and "clocks" in late.reason
    clock.at = NOW - 121
    assert not _send(auth, stamp=Stamp(NOW, "q" * 24)).ok
    clock.at = NOW + 119
    assert _send(auth, stamp=Stamp(NOW, "r" * 24)).ok


def test_a_nonce_outlasts_the_window_it_was_accepted_in():
    clock = Clock()
    auth = _auth(clock=clock)
    assert _send(auth).ok
    clock.at = NOW + 100
    assert not _send(auth).ok, "still inside the window the first copy was good for"


def test_the_nonce_cache_is_bounded():
    auth = _auth(most=50)
    for index in range(500):
        assert _send(auth, stamp=Stamp(NOW, f"{index:024d}")).ok
    assert len(auth._nonces) <= 50


@pytest.mark.parametrize("header", [
    "", "Bearer mlsk1.abc", "Poolhouse-MAC", "Poolhouse-MAC k=,t=,n=,s=",
    "Poolhouse-MAC k=abc,t=notnumber,n=" + "n" * 24 + ",s=" + "0" * 64,
    "Poolhouse-MAC k=abc,t=1700000000,n=short,s=" + "0" * 64,
    "Poolhouse-MAC k=abc,t=1700000000,n=" + "n" * 24 + ",s=tooshort",
    "Poolhouse-MAC " + "garbage," * 50,
])
def test_a_header_that_is_not_a_signature_is_refused(header):
    got = _auth().check("GET", "/jobs", {"Authorization": header, "Host": "h"}, None, "1.1.1.1")
    assert not got.ok


def test_no_secrets_means_nothing_is_accepted():
    assert not _send(Authenticator(lambda: [], clock=Clock())).ok
    assert not _send(Authenticator(lambda: [""], clock=Clock())).ok


def test_unwrap_finds_a_mac_secret_in_a_bearer_header_and_nothing_else():
    assert macauth.unwrap(f"Bearer {SECRET}") == SECRET
    assert macauth.unwrap("Bearer plain-api-key") == ""
    assert macauth.unwrap("Basic abc") == ""


# -- lockout ---------------------------------------------------------------------------


def test_an_address_that_keeps_failing_is_locked_out_even_for_a_good_request():
    tick = Clock(0.0)
    auth = _auth(lockout=Lockout(failures=3, lock_s=30, clock=tick))
    for _ in range(3):
        assert not _send(auth, secret=OTHER).ok
    refused = _send(auth)
    assert not refused.ok and refused.locked
    assert _send(auth, who="10.8.8.8", stamp=Stamp(NOW, "w" * 24)).ok, "another address is fine"
    tick.at = 31
    assert _send(auth, stamp=Stamp(NOW, "v" * 24)).ok, "the lock ends"


def test_failures_that_are_spread_out_do_not_lock():
    tick = Clock(0.0)
    auth = _auth(lockout=Lockout(failures=3, window_s=10, clock=tick))
    for step in range(10):
        tick.at = step * 20
        assert not _send(auth, secret=OTHER).ok
    assert _send(auth).ok


def test_a_success_clears_the_failures():
    auth = _auth(lockout=Lockout(failures=3, clock=Clock(0.0)))
    for _ in range(2):
        _send(auth, secret=OTHER)
    assert _send(auth).ok
    for _ in range(2):
        _send(auth, secret=OTHER)
    assert _send(auth, stamp=Stamp(NOW, "u" * 24)).ok


def test_the_lockout_table_is_bounded():
    lockout = Lockout(failures=99, most=10)
    for index in range(100):
        lockout.failed(f"10.0.0.{index}")
    assert len(lockout._seen) <= 10
