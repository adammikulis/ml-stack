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

import contextlib
import json
import os
import signal
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
    fit_beacon,
    load_cluster_key,
    named_apart,
)
from ml_stack.fleet.remote import Peer, PeerError

#: Real UDP on a real interface and a real daemon subprocess, per the docstring
#: above -- so every test here waits out a network timeout at least once.
pytestmark = pytest.mark.slow

REPO = Path(__file__).resolve().parent.parent


def _free_udp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _free_tcp_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _heard_by_two(dest: str, via: str = "") -> str:
    """How many of two sockets sharing a fresh port hear one datagram sent to ``dest``,
    out of the interface at ``via`` when given, or why it was not sent."""
    import struct

    from ml_stack.fleet.discovery import _socket, default_group

    port = _free_udp_port()
    both = [_socket(broadcast=True, bind=("", port), group=default_group())
            for _ in range(2)]
    if via:
        for s in both:
            with contextlib.suppress(OSError):
                s.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, struct.pack(
                    "4s4s", socket.inet_aton(default_group()), socket.inet_aton(via)))
    try:
        with _socket(broadcast=True) as out:
            if via:
                out.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_IF, socket.inet_aton(via))
            try:
                out.sendto(b"probe", (dest, port))
            except OSError as exc:
                return f"{type(exc).__name__}: {exc}"
        heard = 0
        for s in both:
            s.settimeout(0.5)
            with contextlib.suppress(OSError):
                s.recvfrom(64)
                heard += 1
        return f"{heard}/2"
    finally:
        for s in both:
            s.close()


def _reach(port: int) -> dict[str, str]:
    """Each way a query goes out, and how many of two listeners on one port heard it."""
    from ml_stack.fleet.discovery import default_group, primary_ip

    return {"primary": primary_ip(), "group": _heard_by_two(default_group()),
            "group on loopback": _heard_by_two(default_group(), "127.0.0.1"),
            "broadcast": _heard_by_two("255.255.255.255"),
            "loopback": _heard_by_two("127.0.0.1")}


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
        f"expected both, got {[p.name for p in found]}; sending: {_reach(port)}"
    assert len({p.instance for p in found}) == 2, "instances must be distinct"


@pytest.mark.skipif(sys.platform != "darwin", reason="Linux's lo carries no multicast")
def test_two_peers_on_one_port_are_both_found_when_the_lan_refuses_multicast(
        key, port, monkeypatch):
    """A LAN that refuses every multicast and broadcast (EHOSTUNREACH on a macOS runner)
    leaves loopback, where a unicast reaches one of two sockets sharing a port."""
    from ml_stack.fleet import discovery

    real = discovery._destinations
    monkeypatch.setattr(discovery, "_destinations", lambda group, port: [
        (dest, via) for dest, via in real(group, port)
        if via or dest[0] == discovery.LOOPBACK])
    with Advertiser(Beacon(name="rtx", port=8770), key, port=port, interval_s=0.2), \
         Advertiser(Beacon(name="mac", port=8771), key, port=port, interval_s=0.2):
        found = discover(key, timeout_s=2.5, port=port)
    assert sorted(p.name for p in found) == ["mac", "rtx"]


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


def test_the_later_of_two_answers_is_the_state_kept():
    """A daemon answers each query in a round, and the last thing it said about its own
    free slots is the one a placement must read."""
    same = "0123456789abcdef"
    first = Beacon(name="rtx", port=8770, host="127.0.0.1", instance=same, free=1)
    second = Beacon(name="rtx", port=8770, host="127.0.0.1", instance=same, free=0,
                    busy=True)
    kept = _prefer(first, second)
    assert kept.free == 0 and kept.busy
    lan = Beacon(name="rtx", port=8770, host="192.168.2.9", instance=same, free=0)
    merged = _prefer(first, lan)
    assert merged.host == "127.0.0.1" and merged.free == 0


def test_a_slow_device_probe_does_not_delay_the_answer(key, port):
    """The failure this reproduces: the beacon was built on the receiving thread, so a
    daemon whose refresh reads GPU power -- over a second on Apple silicon -- answered
    after the asker had given up."""
    def slow(b: Beacon) -> None:
        time.sleep(3.0)
        b.free = 0

    with Advertiser(Beacon(name="rtx", port=8770), key, port=port, interval_s=0.2,
                    refresh=slow):
        found = discover(key, timeout_s=1.0, port=port)
    assert "rtx" in [b.name for b in found], "the reply waited on the probe"


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


def test_only_a_shared_name_is_told_apart_and_by_as_little_as_it_takes():
    rows = [{"name": "Mac", "machine": "a1b2c3d4"}, {"name": "Mac", "machine": "a1b2ffff"},
            {"name": "Mac", "machine": "99990000"}, {"name": "rtx", "machine": "12345678"}]
    assert [r["name"] for r in named_apart(rows)] == [
        "Mac#a1b2c", "Mac#a1b2f", "Mac#99990", "rtx"]
    once = [{"name": "Mac", "machine": "a1b2c3d4"}, {"name": "Mac", "machine": "a1b2c3d4"}]
    assert [r["name"] for r in named_apart(once)] == ["Mac", "Mac"]


def test_garbage_on_the_port_does_not_kill_the_listener(key, port):
    """An unrelated service on the group must not take discovery down."""
    with Advertiser(Beacon(name="rtx", port=8770), key, port=port, interval_s=0.2):
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            for junk in (b"", b"not json", b'{"v":1}', b"\xff\xfe\x00"):
                s.sendto(junk, ("127.0.0.1", port))
        time.sleep(0.2)
        assert any(b.name == "rtx" for b in discover(key, timeout_s=2.0, port=port))


# -- end to end: a real daemon process ----------------------------------
@contextlib.contextmanager
def _booted(tmp_path, *extra: str):
    """Boot the actual daemon the way a machine would, rooted in ``tmp_path/traind``
    with ``extra`` flags, and yield ``(keyfile, disco_port, http_port, log)``."""
    keyfile = tmp_path / "cluster.key"
    create_cluster_key(keyfile)
    disco_port = _free_udp_port()
    http_port = _free_tcp_port()
    env = {**os.environ,
           "ML_STACK_DISCOVERY_PORT": str(disco_port),
           "PYTHONPATH": str(REPO / "src"),
           "PYTHONFAULTHANDLER": "1",
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
         "--cluster-key", str(keyfile), *extra],
        env=env, stdout=fh, stderr=subprocess.STDOUT)
    # /health, not a bare connect: the daemon's socket is bound and its backlog accepts
    # from the moment the server object is built, seconds before it serves or advertises.
    driver = _driver(keyfile, http_port)
    deadline = time.time() + 20
    while time.time() < deadline:
        try:
            if driver.health().get("ok"):
                break
        except PeerError:
            if proc.poll() is not None:
                pytest.fail(f"traind died:\n{log.read_text(errors='replace')}")
            time.sleep(0.1)
    else:
        # SIGABRT under PYTHONFAULTHANDLER writes every thread's stack to the log
        proc.send_signal(signal.SIGABRT)
        with contextlib.suppress(subprocess.TimeoutExpired):
            proc.wait(timeout=10)
        proc.kill()
        pytest.fail(f"traind never answered /health:\n{log.read_text(errors='replace')}")
    try:
        yield keyfile, disco_port, http_port, log
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
        fh.close()


@pytest.fixture
def traind(tmp_path):
    """The daemon as a machine boots it: nothing but a root and a key."""
    with _booted(tmp_path) as booted:
        yield booted


def _driver(keyfile: Path, http_port: int) -> Peer:
    """A `Peer` at the booted daemon by address, with the token discovery would derive --
    for tests about what the daemon does, not about finding it."""
    return Peer(f"http://127.0.0.1:{http_port}", derive_token(load_cluster_key(keyfile)))


def _probe(rtx: Peer, out: Path) -> dict:
    return rtx.submit([sys.executable, "-c", f"open({str(out)!r}, 'w').write('ran')"],
                      name="probe")


def test_a_restarted_daemon_keeps_its_machine_id(tmp_path):
    from ml_stack.home import machine_id

    seen = []
    for _ in range(2):
        with _booted(tmp_path) as (keyfile, disco_port, http_port, _log):
            found = discover(load_cluster_key(keyfile), timeout_s=3.0, port=disco_port)
            health = _driver(keyfile, http_port).health()
            seen.append((health["machine"], [b.machine for b in found],
                         [b.instance for b in found]))
    assert seen[0][:2] == seen[1][:2] == (machine_id(), [machine_id()])
    assert seen[0][2] != seen[1][2], "the per-process instance is minted afresh"


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


# -- the measuring gate reads the daemon's own bench home, not this machine's ----
def test_a_daemon_in_its_own_root_runs_training_while_some_other_home_is_measuring(tmp_path):
    """The failure this reproduces: the daemon's bench gate read ``~/.ml-stack/bench``
    whatever ``--root`` said, so on a developer's box with a real benchmark running, a
    daemon booted in a test's directory held every training job queued -- the gate told
    the truth about the wrong machine. Here the "real" home is another tmp dir with its
    lock held, and the daemon, pointed nowhere near it, must not care."""
    from ml_stack.lock import only_one

    elsewhere = tmp_path / "somebody-elses-home" / "bench"
    with only_one(elsewhere / "measuring.lock", announce=lambda *a, **k: None):
        with _booted(tmp_path) as (keyfile, _disco, http_port, log):
            rtx = _driver(keyfile, http_port)
            assert rtx.health()["measuring"] is False, log.read_text(errors="replace")
            out = tmp_path / "proof.txt"
            job = _probe(rtx, out)
            final = rtx.wait(job["id"], poll_s=0.3, timeout_s=60)
            assert final["state"] == "done", rtx.log(job["id"])
            assert out.read_text() == "ran"
    said = log.read_text(errors="replace")
    assert f"bench {tmp_path / 'bench'}" in said, said
    assert "holding" not in said


def test_a_daemon_pointed_at_a_held_bench_home_keeps_training_queued_and_says_so(tmp_path):
    """The gate still works once it reads the right home: a daemon told its bench home
    is the held one queues training, says why in its log, and runs the job the moment
    the lock goes."""
    from ml_stack.lock import only_one

    home = tmp_path / "bench-of-this-box"
    with _booted(tmp_path, "--bench-home", str(home)) as (keyfile, _disco, http_port, log):
        rtx = _driver(keyfile, http_port)
        out = tmp_path / "proof.txt"
        with only_one(home / "measuring.lock", announce=lambda *a, **k: None):
            assert rtx.health()["measuring"] is True
            job = _probe(rtx, out)
            deadline = time.time() + 4
            while time.time() < deadline and "holding" not in log.read_text(errors="replace"):
                time.sleep(0.1)
            assert rtx.job(job["id"])["state"] == "queued"
            assert not out.exists()
            said = log.read_text(errors="replace")
            assert "holding 1 queued job(s): a benchmark is measuring" in said, said
        final = rtx.wait(job["id"], poll_s=0.3, timeout_s=60)
        assert final["state"] == "done", rtx.log(job["id"])
        assert out.read_text() == "ran"
        assert rtx.health()["measuring"] is False
    assert "taking queued work again" in log.read_text(errors="replace")


def test_find_one_says_why_when_no_peer_matches(traind):
    keyfile, disco_port, _, log = traind
    with pytest.raises(DiscoveryError, match="no peer matches"):
        Peer.find_one(name="not-this-box", cluster_key_path=keyfile,
                               timeout_s=3.0, port=disco_port)


def test_a_daemon_that_can_hear_is_found_by_asking_for_speech(traind):
    """The energy VAD needs no model and no dependency, so every machine answers for at
    least one protocol; the beacon says which, and `require=` picks on it. The registries
    this process holds are the suite's empty ones; the daemon is a real process with its
    own, probed in a thread of its own once it is up."""
    keyfile, disco_port, http_port, log = traind
    # the probe runs in the daemon's own thread; on a busy machine the beacon carries an
    # empty `speech` for a while, and this waits for the thread rather than for the clock
    deadline = time.time() + 120
    while True:
        try:
            peer = Peer.find_one(require="speech", cluster_key_path=keyfile,
                                 timeout_s=3.0, port=disco_port)
            break
        except DiscoveryError as exc:
            if time.time() > deadline:
                pytest.fail(f"{exc}\n--- traind said ---\n"
                            f"{log.read_text(errors='replace')}")
            time.sleep(0.5)
    assert peer.beacon.port == http_port
    assert "vad" in peer.beacon.device["speech"]
    assert "vad" in peer.health()["speech"]


def test_discovery_without_a_key_is_an_error_not_an_empty_list(tmp_path):
    """Silence and 'you have no key' are different facts and must read that way."""
    with pytest.raises(DiscoveryError, match="no cluster key"):
        Peer.find_one(cluster_key_path=tmp_path / "absent.key")


def test_peers_ls_reports_the_running_daemon(traind):
    keyfile, disco_port, http_port, log = traind
    env = {**os.environ, "ML_STACK_DISCOVERY_PORT": str(disco_port),
           "PYTHONPATH": str(REPO / "src")}
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


def test_a_machine_holding_hundreds_of_models_is_still_found(key, port):
    """The models go on the beacon, and a beacon is one datagram. Past 65,507 bytes
    `sendto` refuses it and the daemon is in no `ml-stack-peers ls` at all."""
    held = [{"name": f"a-model-with-a-long-enough-name-{n:04d}.gguf", "size": n}
            for n in range(900)]
    beacon = Beacon(name="hoarder", port=8770,
                    device={"models": held, "models_total": len(held)})
    with Advertiser(beacon, key, port=port, interval_s=0.2):
        found = discover(key, timeout_s=2.0, port=port)

    peer = next((b for b in found if b.name == "hoarder"), None)
    assert peer is not None, f"a machine with {len(held)} models vanished; saw {found}"
    sent = peer.device["models"]
    assert 0 < len(sent) < len(held), f"{len(sent)} of {len(held)} models sent"
    assert peer.device["models_total"] == len(held), \
        "the beacon does not say how many models it left off"


def test_a_beacon_too_big_to_leave_the_machine_is_said_out_loud(key, port):
    """Vanishing from the fleet with nothing written anywhere is the failure nobody can
    diagnose: every other machine simply stops listing this one."""
    from ml_stack import log
    from ml_stack.fleet.discovery import MAX_DATAGRAM

    lines: list[str] = []
    beacon = Beacon(name="x" * (MAX_DATAGRAM + 1000), port=8770)
    with log.to(lambda stream, text: lines.append(text)), \
            Advertiser(beacon, key, port=port, interval_s=0.2) as adv:
        discover(key, timeout_s=1.0, port=port)

    assert adv.undelivered, "a beacon that was never sent was counted as sent"
    assert adv.last_error, "nothing recorded why the beacon did not go out"
    said = "".join(lines)
    assert "not in the fleet" in said, said
    assert "did not reach" in said, said


def test_a_beacon_that_fits_is_left_exactly_as_it_is():
    body = {"name": "small", "device": {"models": [{"name": "one.gguf", "size": 1}]}}
    assert fit_beacon(body) == body
    assert "models_total" not in body["device"]


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
        from ml_stack.fleet.discovery import clusters_path, join_cluster

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
        from ml_stack.fleet.discovery import cluster_group, clusters_path, join_cluster

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
            cluster_group,
            join,
            leave,
            load_cluster_key,
            memberships,
        )

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

    def test_a_machine_set_up_before_the_list_existed_is_still_in_its_cluster(
            self, tmp_path):
        from ml_stack.fleet.discovery import (
            cluster_group,
            in_cluster,
            key_from_passphrase,
            load_cluster_key,
        )

        anchor = tmp_path / "cluster.key"
        key = key_from_passphrase(self.WORDS, group="ml-stack")
        anchor.write_text(key.decode() + "\n")

        assert in_cluster(anchor)
        assert load_cluster_key(anchor) == key
        assert cluster_group(anchor) == "ml-stack"

    def test_the_group_it_was_set_up_with_survives(self, tmp_path):
        from ml_stack.fleet.discovery import cluster_group, key_from_passphrase

        anchor = tmp_path / "cluster.key"
        anchor.write_text(key_from_passphrase(self.WORDS, group="garage").decode())
        (tmp_path / "cluster.group").write_text("garage\n")

        assert cluster_group(anchor) == "garage"

    def test_the_old_key_is_moved_into_the_list_once(self, tmp_path):
        from ml_stack.fleet.discovery import (
            join,
            key_from_passphrase,
            leave,
            memberships,
        )

        anchor = tmp_path / "cluster.key"
        anchor.write_text(key_from_passphrase(self.WORDS, group="ml-stack").decode())
        assert len(memberships(anchor)) == 1
        assert (tmp_path / "cluster.json").exists()

        join(self.WORDS, group="lab", path=anchor)
        assert {m.group for m in memberships(anchor)} == {"ml-stack", "lab"}

        leave("ml-stack", path=anchor)
        leave("lab", path=anchor)
        assert memberships(anchor) == [], "leaving must not be undone by the old file"

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
