"""A cluster has its own random salt, learned from a machine already in it."""

import json
import socket
import time

import pytest

from ml_stack.fleet import discovery
from ml_stack.fleet.discovery import (
    PROTOCOL,
    Advertiser,
    Beacon,
    DiscoveryError,
    Salting,
    check_passphrase,
    find_salt,
    join_cluster,
    key_from_passphrase,
    memberships,
    new_salt,
)

WORDS = "correct horse battery staple"
SALT = b"0123456789abcdef"


@pytest.fixture
def port(monkeypatch):
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
        probe.bind(("", 0))
        free = probe.getsockname()[1]
    monkeypatch.setenv("ML_STACK_DISCOVERY_PORT", str(free))
    return free


def _member(tmp_path, salt, words=WORDS, group="home", name="a"):
    path = tmp_path / f"{name}.key"
    key = join_cluster(words, group=group, path=path, salting=Salting(salt))
    tell = Advertiser(Beacon(name=name, port=8770), key, port=int(
        __import__("os").environ["ML_STACK_DISCOVERY_PORT"]), interval_s=30)
    tell.salt = memberships(path)[0].salt_bytes()
    return tell.start(), key, path


def test_the_same_words_make_a_different_key_under_each_salt():
    one = key_from_passphrase(WORDS, group="home", salt=new_salt())
    two = key_from_passphrase(WORDS, group="home", salt=new_salt())
    assert one != two
    assert key_from_passphrase(WORDS, group="home", salt=SALT) == key_from_passphrase(
        WORDS, group="home", salt=SALT)
    assert key_from_passphrase(WORDS, group="home", salt=SALT) != key_from_passphrase(
        WORDS, group="work", salt=SALT)


def test_a_salt_too_short_to_be_one_is_refused():
    with pytest.raises(DiscoveryError, match="16 bytes"):
        key_from_passphrase(WORDS, group="home", salt=b"short")


def test_a_new_cluster_gets_a_random_salt_that_is_recorded_and_checks_the_words(tmp_path):
    path = tmp_path / "k"
    join_cluster(WORDS, group="home", path=path)
    again = tmp_path / "k2"
    join_cluster(WORDS, group="home", path=again)
    first, second = memberships(path)[0], memberships(again)[0]
    assert first.salt and first.salt != second.salt and first.key != second.key
    assert check_passphrase(WORDS, group="home", path=path)
    assert not check_passphrase("some other words entirely", group="home", path=path)


def test_a_machine_that_joins_learns_the_salt_and_ends_with_the_same_key(tmp_path, port):
    tell, key, _ = _member(tmp_path, SALT)
    try:
        theirs = tmp_path / "second.key"
        got = join_cluster(WORDS, group="home", path=theirs, salting=Salting(search_s=3.0))
        assert got == key and memberships(theirs)[0].salt_bytes() == SALT
    finally:
        tell.stop()


def test_the_right_cluster_is_picked_among_two_on_one_network(tmp_path, port):
    other_salt = b"fedcba9876543210"
    a, key_a, _ = _member(tmp_path, SALT, words="the first clusters words", name="a")
    b, key_b, _ = _member(tmp_path, other_salt, words="the second clusters words", name="b")
    try:
        assert find_salt("the second clusters words", group="home", timeout_s=3.0) == (
            other_salt, key_b)
        assert find_salt("the first clusters words", group="home", timeout_s=3.0) == (SALT, key_a)
    finally:
        a.stop()
        b.stop()


def test_words_no_cluster_here_accepts_are_an_error_not_a_new_cluster(tmp_path, port):
    tell, _, _ = _member(tmp_path, SALT)
    try:
        with pytest.raises(DiscoveryError, match="none accepts"):
            find_salt("a passphrase nobody here uses", group="home", timeout_s=2.0)
        with pytest.raises(DiscoveryError):
            join_cluster("a passphrase nobody here uses", group="home",
                         path=tmp_path / "x.key", salting=Salting(search_s=2.0))
        assert memberships(tmp_path / "x.key") == []
    finally:
        tell.stop()


def test_a_network_with_no_cluster_says_nothing_answered(port):
    assert find_salt(WORDS, group="home", timeout_s=0.8) is None


def test_a_salt_told_to_the_wrong_group_is_not_accepted(tmp_path, port):
    tell, _, _ = _member(tmp_path, SALT, group="home")
    try:
        with pytest.raises(DiscoveryError, match="none accepts"):
            find_salt(WORDS, group="work", timeout_s=2.0)
    finally:
        tell.stop()


def test_a_random_key_cluster_has_no_salt_to_tell(tmp_path, port):
    path = tmp_path / "k"
    key = discovery.create_cluster_key(path, group="ml-stack").encode()
    tell = Advertiser(Beacon(name="r", port=1), key, port=port, interval_s=30).start()
    try:
        assert find_salt(WORDS, group="ml-stack", timeout_s=0.8) is None
    finally:
        tell.stop()


# -- the protocol version ---------------------------------------------------------------


def test_this_is_a_new_protocol_so_an_old_peer_is_refused():
    assert PROTOCOL == 2
    key = b"k" * 32
    old = discovery._sign(key, {"v": 1, "kind": "beacon", "t": time.time(), "nonce": "n"})
    now = discovery._sign(key, {"v": PROTOCOL, "kind": "beacon", "t": time.time(), "nonce": "n"})
    assert discovery._verify(key, old, kind="beacon") is None
    assert discovery._verify(key, now, kind="beacon") is not None


def test_a_hello_in_the_old_protocol_gets_no_answer(tmp_path, port):
    tell, _, _ = _member(tmp_path, SALT)
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.settimeout(1.0)
            sock.sendto(json.dumps({"v": 1, "kind": "hello", "nonce": "x"}).encode(),
                        ("127.0.0.1", port))
            with pytest.raises(TimeoutError):
                sock.recvfrom(65535)
    finally:
        tell.stop()
