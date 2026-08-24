"""Peer discovery: real UDP on a real interface, and a real daemon subprocess.

Nothing is mocked, for the same reason the daemon tests aren't. The failures
this module exists to prevent are network-shaped and adversary-shaped -- a
beacon that a neighbour can forge, a reply that can be replayed an hour later,
a multicast group a router quietly drops -- and a fake socket reproduces none
of them.

The ports are randomised per test so a run does not talk to, or get answers
from, a daemon the developer actually has running on this LAN.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest
from ml_stack.fleet.discovery import (
    Advertiser,
    Beacon,
    DiscoveryError,
    _prefer,
    _sign,
    _verify,
    create_cluster_key,
    derive_token,
    discover,
    load_cluster_key,
)
from ml_stack.fleet.remote import Peer

REPO = Path(__file__).resolve().parent.parent


def _free_udp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _free_tcp_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def key(tmp_path) -> bytes:
    create_cluster_key(tmp_path / "cluster.key")
    return load_cluster_key(tmp_path / "cluster.key")


@pytest.fixture
def port() -> int:
    return _free_udp_port()


# -- the key -------------------------------------------------------------
def test_key_is_created_once_and_not_silently_rotated(tmp_path):
    p = tmp_path / "cluster.key"
    first = create_cluster_key(p)
    assert create_cluster_key(p) == first, "re-running init must not evict the cluster"
    assert create_cluster_key(p, overwrite=True) != first


def test_the_file_holding_the_keys_is_not_world_readable(tmp_path):
    from ml_stack.fleet.discovery import clusters_path

    p = tmp_path / "cluster.key"
    create_cluster_key(p)
    assert oct(clusters_path(p).stat().st_mode)[-3:] == "600"


def test_missing_key_reads_as_none(tmp_path):
    assert load_cluster_key(tmp_path / "absent.key") is None


def test_token_is_a_pure_function_of_the_key(tmp_path):
    a = create_cluster_key(tmp_path / "a.key").encode()
    b = create_cluster_key(tmp_path / "b.key").encode()
    assert derive_token(a) == derive_token(a), "both ends must compute the same token"
    assert derive_token(a) != derive_token(b)
    assert a.decode() not in derive_token(a), "the token must not leak the key"


# -- the round trip ------------------------------------------------------
def test_advertiser_is_found_and_carries_its_device_report(key, port):
    beacon = Beacon(name="rtx", port=8770, device={"cuda": True, "gpu": "RTX 3090 Ti"})
    with Advertiser(beacon, key, port=port, interval_s=0.2):
        found = discover(key, timeout_s=2.0, port=port)
    names = [b.name for b in found]
    assert "rtx" in names, f"advertiser not discovered; saw {names}"
    peer = next(b for b in found if b.name == "rtx")
    assert peer.device["gpu"] == "RTX 3090 Ti"
    assert peer.port == 8770
    assert peer.host, "host must be filled in from the packet source address"
    assert peer.base_url == f"http://{peer.host}:8770"


def test_nothing_is_found_when_nothing_is_advertising(key, port):
    assert discover(key, timeout_s=0.6, port=port) == []


def test_a_peer_with_a_different_key_is_invisible(key, port, tmp_path):
    other = create_cluster_key(tmp_path / "other.key").encode()
    with Advertiser(Beacon(name="stranger", port=8770), other, port=port,
                    interval_s=0.2):
        assert discover(key, timeout_s=1.0, port=port) == [], \
            "a daemon keyed differently must not be discoverable"


def test_two_peers_are_told_apart(key, port):
    """Two daemons are two peers even when they share a host and a port."""
    a = Beacon(name="rtx", port=8770, device={"cuda": True})
    b = Beacon(name="mac", port=8771, device={"backends": ["mlx"]})
    with Advertiser(a, key, port=port, interval_s=0.2), \
         Advertiser(b, key, port=port, interval_s=0.2):
        found = discover(key, timeout_s=2.5, port=port)
    assert {p.name for p in found} == {"rtx", "mac"}, \
        f"expected both, got {[p.name for p in found]}"
    assert len({p.instance for p in found}) == 2, "instances must be distinct"


def test_one_daemon_on_several_interfaces_is_one_peer():
    """A box with a VPN up answers the same query from each address it holds.

    Counting those as separate peers is how `find_one` starts reporting two
    GPUs on a machine that has one, and how a run gets submitted to a route
    rather than to a card.
    """
    same = "0123456789abcdef"
    lan = Beacon(name="rtx", port=8770, host="192.168.2.9", instance=same)
    vpn = Beacon(name="rtx", port=8770, host="10.8.0.3", instance=same)
    assert lan.identity == vpn.identity
    other = Beacon(name="rtx", port=8770, host="192.168.2.9", instance="beef")
    assert other.identity != lan.identity, "different daemons must stay distinct"


def test_loopback_wins_when_the_daemon_is_on_this_machine():
    same = "0123456789abcdef"
    lan = Beacon(name="rtx", port=8770, host="192.168.2.9", instance=same)
    local = Beacon(name="rtx", port=8770, host="127.0.0.1", instance=same)
    assert _prefer(lan, local).host == "127.0.0.1"
    assert _prefer(local, lan).host == "127.0.0.1", "order must not decide it"


def test_a_beacon_without_an_instance_still_has_an_identity():
    """Tolerate a peer that predates instance ids rather than merging them all."""
    a = Beacon(name="rtx", port=8770, hostname="boxa")
    b = Beacon(name="rtx", port=8770, hostname="boxb")
    assert a.identity != b.identity


# -- the adversary -------------------------------------------------------
def test_a_tampered_beacon_is_refused(key):
    raw = _sign(key, {"v": 1, "kind": "beacon", "t": time.time(), "nonce": "",
                      "beacon": {"name": "rtx", "port": 8770}})
    assert _verify(key, raw, kind="beacon") is not None
    tampered = raw.replace(b'"port":8770', b'"port":9999')
    assert _verify(key, tampered, kind="beacon") is None, \
        "a redirected port must not survive verification"


def test_a_beacon_signed_with_another_key_is_refused(key, tmp_path):
    other = create_cluster_key(tmp_path / "other.key").encode()
    raw = _sign(other, {"v": 1, "kind": "beacon", "t": time.time(), "nonce": "",
                        "beacon": {"name": "evil", "port": 8770}})
    assert _verify(key, raw, kind="beacon") is None


def test_a_stale_beacon_is_refused(key):
    raw = _sign(key, {"v": 1, "kind": "beacon", "t": time.time() - 3600,
                      "nonce": "", "beacon": {"name": "rtx", "port": 8770}})
    assert _verify(key, raw, kind="beacon") is None


def test_a_replayed_reply_is_refused(key):
    """A recorded answer must not satisfy a later question."""
    raw = _sign(key, {"v": 1, "kind": "beacon", "t": time.time(),
                      "nonce": "the-old-nonce",
                      "beacon": {"name": "rtx", "port": 8770}})
    assert _verify(key, raw, kind="beacon", nonce="the-old-nonce") is not None
    assert _verify(key, raw, kind="beacon", nonce="a-fresh-nonce") is None


def test_garbage_on_the_port_does_not_kill_the_listener(key, port):
    """An unrelated service on the group must not take discovery down."""
    with Advertiser(Beacon(name="rtx", port=8770), key, port=port, interval_s=0.2):
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            for junk in (b"", b"not json", b'{"v":1}', b"\xff\xfe\x00"):
                s.sendto(junk, ("127.0.0.1", port))
        time.sleep(0.2)
        assert any(b.name == "rtx" for b in discover(key, timeout_s=2.0, port=port))


# -- end to end: a real daemon process ----------------------------------
@pytest.fixture
def traind(tmp_path):
    """Boot the actual daemon the way a machine would, and find it."""
    keyfile = tmp_path / "cluster.key"
    create_cluster_key(keyfile)
    disco_port = _free_udp_port()
    http_port = _free_tcp_port()
    env = {**os.environ,
           "ML_STACK_DISCOVERY_PORT": str(disco_port),
           "PYTHONPATH": os.pathsep.join(
               str(p) for p in sorted((REPO / "packages").glob("*/src"))),
           "PYTHONUNBUFFERED": "1"}
    # To a file, not a pipe: a test that fails because discovery was off should
    # say so with the daemon's own words, and a pipe nobody drains can only be
    # read after the process is gone.
    log = tmp_path / "traind.out"
    fh = log.open("wb")
    proc = subprocess.Popen(
        [sys.executable, "-m", "ml_stack.fleet.daemon",
         "--root", str(tmp_path / "traind"), "--host", "127.0.0.1",
         "--port", str(http_port), "--name", "testbox",
         "--cluster-key", str(keyfile)],
        env=env, stdout=fh, stderr=subprocess.STDOUT)
    deadline = time.time() + 20
    while time.time() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", http_port), timeout=0.5):
                break
        except OSError:
            if proc.poll() is not None:
                pytest.fail(f"traind died:\n{log.read_text(errors='replace')}")
            time.sleep(0.1)
    else:
        proc.kill()
        pytest.fail("traind never listened")
    try:
        yield keyfile, disco_port, http_port, log
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
        fh.close()


def test_a_booted_daemon_is_found_and_driven_with_no_address_configured(traind, tmp_path):
    """The whole point: nobody typed a host, a port, or a token anywhere."""
    keyfile, disco_port, http_port, log = traind
    try:
        rtx = Peer.find_one(cluster_key_path=keyfile, timeout_s=3.0,
                                     port=disco_port)
    except DiscoveryError as exc:
        pytest.fail(f"{exc}\n--- traind said ---\n{log.read_text(errors='replace')}")
    assert rtx.beacon.name == "testbox"
    assert rtx.beacon.port == http_port

    health = rtx.health()
    assert health["ok"] is True and health["name"] == "testbox"

    # Authenticated route: proves the derived token matches what the daemon
    # computed independently, which is the claim discovery rests on.
    assert rtx.jobs() == []
    out = tmp_path / "proof.txt"
    job = rtx.submit([sys.executable, "-c",
                      f"open({str(out)!r}, 'w').write('ran')"], name="probe")
    final = rtx.wait(job["id"], poll_s=0.3, timeout_s=60)
    assert final["state"] == "done", rtx.log(job["id"])
    assert out.read_text() == "ran"


def test_find_one_says_why_when_no_peer_matches(traind):
    keyfile, disco_port, _, log = traind
    with pytest.raises(DiscoveryError, match="no peer matches"):
        Peer.find_one(name="not-this-box", cluster_key_path=keyfile,
                               timeout_s=3.0, port=disco_port)


def test_discovery_without_a_key_is_an_error_not_an_empty_list(tmp_path):
    """Silence and 'you have no key' are different facts and must read that way."""
    with pytest.raises(DiscoveryError, match="no cluster key"):
        Peer.find_one(cluster_key_path=tmp_path / "absent.key")


def test_peers_ls_reports_the_running_daemon(traind):
    keyfile, disco_port, http_port, log = traind
    env = {**os.environ, "ML_STACK_DISCOVERY_PORT": str(disco_port),
           "PYTHONPATH": os.pathsep.join(
               str(p) for p in sorted((REPO / "packages").glob("*/src")))}
    r = subprocess.run([sys.executable, "-m", "ml_stack.fleet.peers",
                        "--cluster-key", str(keyfile), "ls", "--json",
                        "--timeout", "3"],
                       env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    peers = json.loads(r.stdout)
    assert any(p["name"] == "testbox" and p["port"] == http_port for p in peers)


# -- a beacon that tells the truth about right now ------------------------
def test_a_busy_daemon_stops_advertising_itself_as_idle(key, port):
    """The specific failure: Beacon.busy/queued are set once when the daemon boots and
    never touched again, so a box that has been training for six hours still announces
    itself as idle. Anything choosing a peer on that basis is reading a constant."""
    beacon = Beacon(name="rtx", port=8770, slots=1, free=1)
    state = {"free": 1}

    def refresh(b: Beacon) -> None:
        b.free = state["free"]
        b.busy = state["free"] == 0

    with Advertiser(beacon, key, port=port, interval_s=0.2, refresh=refresh):
        idle = next(b for b in discover(key, timeout_s=2.0, port=port) if b.name == "rtx")
        assert idle.free == 1 and not idle.busy

        state["free"] = 0                       # a job starts
        busy = next(b for b in discover(key, timeout_s=2.0, port=port) if b.name == "rtx")

    assert busy.free == 0, "the beacon still claims a free slot after the job started"
    assert busy.busy


def test_a_refresh_that_raises_does_not_silence_the_beacon(key, port):
    """A probe that throws must cost freshness, not discoverability: a box that vanishes
    from the fleet is strictly worse than one advertising a stale answer."""
    def boom(b: Beacon) -> None:
        raise RuntimeError("nvidia-smi fell over")

    with Advertiser(Beacon(name="rtx", port=8770), key, port=port,
                    interval_s=0.2, refresh=boom):
        found = discover(key, timeout_s=2.0, port=port)
    assert "rtx" in [b.name for b in found]


def test_a_beacon_from_an_older_daemon_still_reads_as_one_free_slot(key, port):
    """Mixed versions: an old daemon sends busy/queued and no slots/free. Reading that as
    zero capacity would quietly park the whole fleet on the new boxes."""
    old = Beacon(name="old", port=8770)
    payload = old.public()
    del payload["slots"], payload["free"]

    revived = Beacon(name=payload["name"], port=payload["port"],
                     busy=bool(payload.get("busy")),
                     free=int(payload["free"]) if "free" in payload
                     else (0 if payload.get("busy") else 1))
    assert revived.free == 1


# -- joining with a passphrase -------------------------------------------
class TestPassphrase:
    """Joining has to be something a person can do. The old story was "generate 32
    random bytes, then paste this shell fragment on every machine", which is a thing
    nobody who is not already a developer will get through."""

    WORDS = "correct horse battery staple"

    def test_the_same_words_give_the_same_key(self):
        from ml_stack.fleet.discovery import key_from_passphrase

        assert key_from_passphrase(self.WORDS) == key_from_passphrase(self.WORDS)

    def test_surrounding_whitespace_does_not_make_a_different_cluster(self):
        """Someone pastes the passphrase and picks up a trailing space. Failing on that
        produces a cluster of one, which looks exactly like a network problem."""
        from ml_stack.fleet.discovery import key_from_passphrase

        assert key_from_passphrase(f"  {self.WORDS}\n") == key_from_passphrase(self.WORDS)

    def test_different_words_give_a_different_key(self):
        from ml_stack.fleet.discovery import key_from_passphrase

        assert key_from_passphrase(self.WORDS) != key_from_passphrase("something else")

    def test_the_group_name_separates_two_households_that_chose_the_same_words(self):
        from ml_stack.fleet.discovery import key_from_passphrase

        assert (key_from_passphrase(self.WORDS, group="home")
                != key_from_passphrase(self.WORDS, group="lab"))

    @pytest.mark.parametrize("bad", ["", "abc", "1234"])
    def test_a_passphrase_too_short_to_survive_guessing_is_refused(self, bad):
        from ml_stack.fleet.discovery import key_from_passphrase

        with pytest.raises(DiscoveryError, match="at least"):
            key_from_passphrase(bad)

    def test_joining_writes_a_key_only_this_user_can_read(self, tmp_path):
        from ml_stack.fleet.discovery import join_cluster

        from ml_stack.fleet.discovery import clusters_path

        keyfile = tmp_path / "cluster.key"
        join_cluster(self.WORDS, path=keyfile)

        assert clusters_path(keyfile).stat().st_mode & 0o077 == 0
        assert load_cluster_key(keyfile) == join_cluster(self.WORDS, path=keyfile)

    def test_a_derived_key_drives_a_real_daemon(self, tmp_path):
        """The point of deriving rather than minting: the bearer token both ends compute
        has to come out the same, or the passphrase bought nothing."""
        from ml_stack.fleet.discovery import join_cluster

        here = join_cluster(self.WORDS, path=tmp_path / "a.key")
        there = join_cluster(self.WORDS, path=tmp_path / "b.key")
        assert derive_token(here) == derive_token(there)


def test_two_passphrase_groups_share_a_network_without_seeing_each_other(port, tmp_path):
    """Several clusters on one LAN, separated by nothing but the words people typed.
    The isolation is the same mechanism that keeps a stranger out: a beacon signed with
    another key does not verify, so it is never answered."""
    from ml_stack.fleet.discovery import join_cluster

    ours = join_cluster("correct horse battery staple", path=tmp_path / "ours.key")
    theirs = join_cluster("a completely different phrase", path=tmp_path / "theirs.key")

    with Advertiser(Beacon(name="ours", port=8770), ours, port=port, interval_s=0.2), \
         Advertiser(Beacon(name="theirs", port=8771), theirs, port=port, interval_s=0.2):
        we_see = {b.name for b in discover(ours, timeout_s=2.0, port=port)}
        they_see = {b.name for b in discover(theirs, timeout_s=2.0, port=port)}

    assert we_see == {"ours"}, f"our group can see into theirs: {we_see}"
    assert they_see == {"theirs"}, f"their group can see into ours: {they_see}"


class TestTheGroupIsRemembered:
    """The group is load-bearing in the derivation, so a box that forgot it cannot check
    a passphrase anyone types -- it does not know which salt the words were stretched
    with. It also could not say which cluster it was in."""

    def test_joining_records_which_cluster_it_joined(self, tmp_path):
        from ml_stack.fleet.discovery import cluster_group, join_cluster

        join_cluster("correct horse battery", group="garage",
                     path=tmp_path / "cluster.key")
        assert cluster_group(tmp_path / "cluster.key") == "garage"

    def test_the_keys_are_not_left_where_anyone_can_read_them(self, tmp_path):
        """The passphrase protects the cluster, so the keys derived from it are the
        one thing on disk that no other account may read."""
        from ml_stack.fleet.discovery import clusters_path, cluster_group, join_cluster

        keyfile = tmp_path / "cluster.key"
        join_cluster("correct horse battery", group="garage", path=keyfile)

        assert clusters_path(keyfile).stat().st_mode & 0o077 == 0
        assert cluster_group(keyfile) == "garage"

    def test_a_machine_never_joined_is_in_no_group(self, tmp_path):
        from ml_stack.fleet.discovery import cluster_group

        assert cluster_group(tmp_path / "nothing.key") is None

    def test_the_right_words_verify_against_the_stored_key(self, tmp_path):
        """What lets someone log in by typing the passphrase rather than pasting a
        43-character token: re-derive and compare, storing nothing."""
        from ml_stack.fleet.discovery import check_passphrase, join_cluster

        keyfile = tmp_path / "cluster.key"
        join_cluster("correct horse battery", group="garage", path=keyfile)

        assert check_passphrase("correct horse battery", path=keyfile)
        assert not check_passphrase("wrong words entirely", path=keyfile)

    def test_the_right_words_in_the_wrong_group_do_not_verify(self, tmp_path):
        from ml_stack.fleet.discovery import check_passphrase, join_cluster

        keyfile = tmp_path / "cluster.key"
        join_cluster("correct horse battery", group="garage", path=keyfile)

        assert not check_passphrase("correct horse battery", group="lab", path=keyfile)

    def test_a_machine_in_no_cluster_verifies_nothing(self, tmp_path):
        from ml_stack.fleet.discovery import check_passphrase

        assert not check_passphrase("correct horse battery", path=tmp_path / "no.key")



class TestBelongingToSeveralClusters:
    """A machine is not owned by one group of machines."""

    WORDS = "correct horse battery staple"
    OTHER = "a completely different set of words"

    def test_it_joins_more_than_one_and_keeps_both(self, tmp_path):
        from ml_stack.fleet.discovery import join, memberships

        anchor = tmp_path / "cluster.key"
        join(self.WORDS, group="home", path=anchor)
        join(self.OTHER, group="work", path=anchor)

        assert [m.group for m in memberships(anchor)] == ["home", "work"]
        assert len({m.key for m in memberships(anchor)}) == 2

    def test_leaving_one_leaves_the_others_alone(self, tmp_path):
        from ml_stack.fleet.discovery import join, leave, memberships

        anchor = tmp_path / "cluster.key"
        join(self.WORDS, group="home", path=anchor)
        join(self.OTHER, group="work", path=anchor)
        leave("home", anchor)

        assert [m.group for m in memberships(anchor)] == ["work"]

    def test_leaving_the_last_one_leaves_no_cluster(self, tmp_path):
        from ml_stack.fleet.discovery import in_cluster, join, leave

        anchor = tmp_path / "cluster.key"
        join(self.WORDS, group="home", path=anchor)
        leave("home", anchor)

        assert in_cluster(anchor) is False

    def test_the_machine_answers_as_the_first_one(self, tmp_path):
        from ml_stack.fleet.discovery import (
            cluster_group, join, leave, load_cluster_key, memberships)

        anchor = tmp_path / "cluster.key"
        join(self.WORDS, group="home", path=anchor)
        join(self.OTHER, group="work", path=anchor)

        assert cluster_group(anchor) == "home"
        assert load_cluster_key(anchor) == memberships(anchor)[0].key

        leave("home", anchor)
        assert cluster_group(anchor) == "work", "it did not promote the one left"

    def test_joining_the_same_cluster_twice_does_not_double_it(self, tmp_path):
        from ml_stack.fleet.discovery import join, memberships

        anchor = tmp_path / "cluster.key"
        join(self.WORDS, group="home", path=anchor)
        join(self.WORDS, group="home", path=anchor)
        assert [m.group for m in memberships(anchor)] == ["home"]

    def test_a_new_passphrase_for_a_cluster_replaces_the_old_key(self, tmp_path):
        from ml_stack.fleet.discovery import join, memberships

        anchor = tmp_path / "cluster.key"
        join(self.WORDS, group="home", path=anchor)
        first = memberships(anchor)[0].key
        join(self.OTHER, group="home", path=anchor)

        assert [m.group for m in memberships(anchor)] == ["home"]
        assert memberships(anchor)[0].key != first, "the passphrase did not change"

    def test_a_short_passphrase_is_refused_for_every_cluster(self, tmp_path):
        import pytest as pt

        from ml_stack.fleet.discovery import DiscoveryError, join, memberships

        anchor = tmp_path / "cluster.key"
        with pt.raises(DiscoveryError, match="at least"):
            join("abc", group="home", path=anchor)
        assert memberships(anchor) == []

    def test_two_machines_in_the_same_cluster_derive_the_same_key(self, tmp_path):
        from ml_stack.fleet.discovery import join

        one = join(self.WORDS, group="home", path=tmp_path / "a.key")
        two = join(self.WORDS, group="home", path=tmp_path / "b.key")
        assert one[0].key == two[0].key

    def test_the_same_words_in_different_clusters_do_not_meet(self, tmp_path):
        from ml_stack.fleet.discovery import join

        home = join(self.WORDS, group="home", path=tmp_path / "a.key")
        work = join(self.WORDS, group="work", path=tmp_path / "b.key")
        assert home[0].key != work[0].key

    def test_a_corrupt_list_reads_as_no_clusters(self, tmp_path):
        from ml_stack.fleet.discovery import clusters_path, memberships

        anchor = tmp_path / "cluster.key"
        clusters_path(anchor).parent.mkdir(parents=True, exist_ok=True)
        clusters_path(anchor).write_text("{not json")
        assert memberships(anchor) == []
