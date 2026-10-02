"""The pairing exchange (SPAKE2 from the spake2 package): agrees on a right code, fails on a
wrong one or a changed identity, refuses hostile messages."""

import sys

import pytest

from ml_stack.fleet.onboard import pake

CTX = b"ctx"


@pytest.fixture(autouse=True)
def needs_spake2():
    pytest.importorskip("spake2")


def run(code_a="123456", code_b="123456", seen=("bb", "aa"), ctx=(CTX, CTX)):
    """Both ends run; ``seen`` is the certificate each believes the other presents."""
    a = pake.start_initiator(code_a, context=ctx[0], mine="aa", theirs=seen[0])
    b = pake.start_responder(code_b, context=ctx[1], mine="bb", theirs=seen[1])
    a.receive(b.message)
    b.receive(a.message)
    return a, b


def test_same_code_agrees_and_both_confirmations_check():
    a, b = run()
    assert b.check(a.confirmation())
    assert a.check(b.confirmation())


def test_wrong_code_fails_both_ways():
    a, b = run(code_b="123457")
    assert not b.check(a.confirmation())
    assert not a.check(b.confirmation())


def test_a_different_certificate_on_either_leg_fails_even_with_the_right_code():
    a, b = run(seen=("mitm", "aa"))           # the asker saw another certificate than B's own
    assert not b.check(a.confirmation())
    a, b = run(seen=("bb", "mitm"))           # the accepter was told another asker identity
    assert not b.check(a.confirmation())


def test_each_exchange_is_new_so_a_recorded_confirmation_is_useless_in_the_next():
    first_a, _ = run()
    _, second_b = run()
    assert not second_b.check(first_a.confirmation())


def test_a_different_context_fails():
    a, b = run(ctx=(b"one", b"two"))
    assert not b.check(a.confirmation())


@pytest.mark.parametrize("bad", ["", "zz", "41" + "00" * 32, "41" + "01" + "00" * 31,
                                 "42" + "11" * 32, "41" + "ff" * 32, "41" * 80, None, 7])
def test_zero_wrong_side_off_group_and_malformed_messages_are_refused(bad):
    b = pake.start_responder("123456", context=CTX, mine="bb", theirs="aa")
    with pytest.raises(pake.Bad):
        b.receive(bad)


def test_the_payload_tag_binds_the_payload_to_the_exchange():
    a, b = run()
    tag = b.seal(b"grant")
    assert a.open(b"grant", tag)
    assert not a.open(b"grant!", tag)
    assert not a.open(b"grant", "0" * 64)
    other_a, _ = run()
    assert not other_a.open(b"grant", tag)


def test_a_missing_library_is_a_clear_error_naming_the_extra(monkeypatch):
    monkeypatch.setitem(sys.modules, "spake2", None)        # what a failed import looks like
    with pytest.raises(pake.PakeUnavailable, match="fleet-onboard"):
        pake.start_initiator("123456", context=CTX, mine="aa", theirs="bb")
