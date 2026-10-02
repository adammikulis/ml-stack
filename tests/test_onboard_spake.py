"""The pairing exchange: agrees on a right code, fails on a wrong one or a changed identity."""

import pytest

from ml_stack.fleet.onboard import spake

CTX = b"ctx"


def run(code_a="123456", code_b="123456", seen=("bb", "aa")):
    """Both ends run; ``seen`` is the certificate each believes the other presents."""
    a = spake.start_initiator(code_a, context=CTX, mine="aa", theirs=seen[0])
    b = spake.start_responder(code_b, context=CTX, mine="bb", theirs=seen[1])
    a.receive(b.message)
    b.receive(a.message)
    return a, b


def test_curve_arithmetic_matches_the_cryptography_package():
    ec = pytest.importorskip("cryptography.hazmat.primitives.asymmetric.ec")
    k = 0x1234567890ABCDEF1234567890ABCDEF1234567890ABCDEF1234567890ABCDEF
    pub = ec.derive_private_key(k, ec.SECP256R1()).public_key().public_numbers()
    assert spake.multiply(k, spake.G) == (pub.x, pub.y)
    assert spake.multiply(spake.N, spake.G) is None
    assert spake.on_curve(spake.M) and spake.on_curve(spake.NN) and spake.M != spake.NN


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
    a = spake.start_initiator("123456", context=b"one", mine="aa", theirs="bb")
    b = spake.start_responder("123456", context=b"two", mine="bb", theirs="aa")
    a.receive(b.message)
    b.receive(a.message)
    assert not b.check(a.confirmation())


@pytest.mark.parametrize("bad", ["", "04" + "00" * 64, "05" + "11" * 64, "zz", "04" + "11" * 64,
                                 None, 7])
def test_off_curve_or_malformed_points_are_refused(bad):
    b = spake.start_responder("123456", context=CTX, mine="bb", theirs="aa")
    with pytest.raises(spake.Bad):
        b.receive(bad)


def test_the_payload_tag_binds_the_payload_to_the_exchange():
    a, b = run()
    tag = b.seal(b"grant")
    assert a.open(b"grant", tag)
    assert not a.open(b"grant!", tag)
    assert not a.open(b"grant", "0" * 64)
    other_a, _ = run()
    assert not other_a.open(b"grant", tag)


def test_a_message_that_cancels_the_blinding_is_refused_not_crashed_on():
    """Someone who knows the code can send the one message that makes the shared point the
    point at infinity; the other end refuses it instead of failing on it."""
    b = spake.start_responder("123456", context=CTX, mine="bb", theirs="aa")
    w = spake.word_from_code("123456", CTX)
    with pytest.raises(spake.Bad, match="degenerate"):
        b.receive(spake.encode(spake.multiply(w, spake.M)))
