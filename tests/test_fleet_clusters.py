"""Choosing a cluster to join: the clusters a machine holds, the ones daemons on the network offer
(real advertisers answering real datagrams on loopback), and the question that picks one."""

from __future__ import annotations

import getpass
import io
import json
import socket
import sys
from pathlib import Path

import pytest

from ml_stack.fleet import join as fleet_join
from ml_stack.fleet.discovery import Advertiser, Beacon, DiscoveryError, check_name, mint_cluster
from ml_stack.fleet.onboard import joining
from ml_stack.fleet.onboard.clusters import (
    Choice,
    Cluster,
    format_clusters,
    known_clusters,
    pick_cluster,
)
from ml_stack.fleet.onboard.joining import find_clusters

WORDS = "quince larch marlow"


def _free_udp() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def udp(monkeypatch) -> int:
    port = _free_udp()
    monkeypatch.setenv("ML_STACK_DISCOVERY_PORT", str(port))
    return port


@pytest.fixture
def advertise(tmp_path, udp):
    """Start a daemon's advertiser for ``cluster`` called ``name``; all are stopped afterwards."""
    started: list[Advertiser] = []

    def start(cluster: str, name: str) -> bytes:
        key = mint_cluster(cluster, tmp_path / f"{name}-{cluster}.key").key
        started.append(Advertiser(Beacon(name=name, port=9 + len(started)), key, port=udp, cluster=cluster,
                                  interval_s=5.0).start())
        return key

    yield start
    for one in started:
        one.stop()


def _names(found: list[Cluster]) -> dict[str, tuple[str, ...]]:
    return {c.name: c.machines for c in found}


# -- what daemons offer ------------------------------------------------------------
def test_a_daemon_advertising_a_cluster_is_found_by_name_and_machine(advertise, udp):
    advertise("lab", "studio")
    offers = find_clusters(timeout_s=1.0, port=udp)
    assert [(o.group, o.machine) for o in offers] == [("lab", "studio")]


def test_the_answer_names_the_cluster_and_carries_no_key(advertise, udp):
    key = advertise("lab", "studio")
    with joining.disc._socket(broadcast=True, bind=("", 0)) as sock:
        joining.disc._say(sock, json.dumps({"v": joining.disc.PROTOCOL, "kind": "join?", "group": "",
                                            "nonce": "n"}).encode(), joining.disc.default_group(), udp)
        sock.settimeout(1.0)
        heard = [sock.recvfrom(65535)[0]]
    said = json.loads(heard[0])
    assert said["group"] == "lab" and said["name"] == "studio"
    assert key not in heard[0]


def test_a_cluster_asked_for_by_name_is_still_answered_only_by_its_holders(advertise, udp):
    advertise("lab", "studio")
    advertise("home", "harrowgate")
    assert {j.port for j in joining.find_joiners("home", timeout_s=1.0, port=udp)} == {10}
    assert joining.find_joiners("attic", timeout_s=0.5, port=udp) == []


def test_two_clusters_on_one_machine_list_with_their_holders(advertise, tmp_path, udp):
    advertise("lab", "studio")
    advertise("lab", "larch")
    advertise("home", "harrowgate")
    mine = tmp_path / "me" / "cluster.key"
    mine.parent.mkdir()
    mint_cluster("lab", mine)
    mint_cluster("attic", mine)
    found = known_clusters(mine, timeout_s=1.0, port=udp)
    assert _names(found) == {"lab": ("this machine", "larch", "studio"),
                             "attic": ("this machine",),
                             "home": ("harrowgate",)}
    assert [c.mine for c in found] == [True, True, False]


def test_this_machines_own_daemon_is_not_listed_as_another_holder(advertise, tmp_path, udp):
    advertise("lab", "studio")
    mine = tmp_path / "me" / "cluster.key"
    mine.parent.mkdir()
    mint_cluster("lab", mine)
    found = known_clusters(mine, timeout_s=1.0, port=udp, self_names=("studio",))
    assert _names(found) == {"lab": ("this machine",)}


def test_the_listing_is_numbered_with_holders():
    lines = format_clusters([Cluster("lab", ("studio", "larch")), Cluster("home", ("harrowgate",))])
    assert lines == ["1) lab - studio, larch", "2) home - harrowgate"]


# -- the question -----------------------------------------------------------------
def _answers(*typed: str):
    queue = list(typed)
    seen: list[str] = []

    def ask(prompt: str) -> str:
        seen.append(prompt)
        return queue.pop(0)

    return ask, seen


FOUND = [Cluster("lab", ("studio", "larch")), Cluster("home", ("harrowgate",))]


def test_a_number_picks_the_listed_cluster():
    ask, _ = _answers("2")
    assert pick_cluster(FOUND, ask, lambda _: None) == Choice("home", True)


def test_a_typed_name_that_is_listed_picks_it_and_another_makes_a_new_one():
    ask, _ = _answers("lab")
    assert pick_cluster(FOUND, ask, lambda _: None) == Choice("lab", True)
    ask, _ = _answers("  attic ")
    assert pick_cluster(FOUND, ask, lambda _: None) == Choice("attic", False)


def test_n_asks_for_the_name_of_a_new_cluster():
    ask, seen = _answers("n", "attic")
    assert pick_cluster(FOUND, ask, lambda _: None) == Choice("attic", False)
    assert "new cluster" in seen[1]


def test_enter_takes_the_default_when_nothing_is_listed():
    ask, seen = _answers("")
    assert pick_cluster([], ask, lambda _: None) == Choice("ml-stack", False)
    assert "[ml-stack]" in seen[0]


def test_enter_alone_does_not_pick_when_clusters_are_listed():
    ask, _ = _answers("", "1")
    told: list[str] = []
    assert pick_cluster(FOUND, ask, told.append) == Choice("lab", True)
    assert any("cannot be empty" in line for line in told)


def test_a_number_outside_the_list_is_asked_again():
    ask, _ = _answers("7", "1")
    told: list[str] = []
    assert pick_cluster(FOUND, ask, told.append).name == "lab"
    assert any("no cluster 7" in line for line in told)


@pytest.mark.parametrize("bad, said", [("a/b", "'/'"), ("a\\b", "'\\\\'"), ("a\tb", "'\\t'"),
                                       ("x" * 65, "at most 64")])
def test_a_name_the_wire_cannot_carry_is_refused_and_says_which(bad, said):
    with pytest.raises(DiscoveryError) as why:
        check_name(bad)
    assert said in str(why.value)
    ask, _ = _answers(bad, "good")
    told: list[str] = []
    assert pick_cluster([], ask, told.append).name == "good"
    assert any(said in line for line in told)


def test_a_name_is_trimmed_and_cannot_be_empty():
    assert check_name("  lab ") == "lab"
    with pytest.raises(DiscoveryError, match="empty"):
        check_name("   ")


def test_joining_by_passphrase_refuses_a_bad_name(tmp_path):
    with pytest.raises(DiscoveryError, match="cannot contain"):
        joining.join_by_passphrase(WORDS, "a/b", tmp_path / "k.key")


# -- the command ------------------------------------------------------------------
class _Terminal(io.StringIO):
    def isatty(self) -> bool:
        return True


@pytest.fixture
def captured(monkeypatch, tmp_path):
    """Replaces the join itself; returns what it was asked to do."""
    asked: dict = {}

    def fake(**kw):
        asked.update(kw)
        return fleet_join.Joined(name="x", port=1, root=Path(kw["root"]), group=kw["group"])

    monkeypatch.setattr(fleet_join, "join_machine", fake)
    monkeypatch.delenv("ML_STACK_PASSPHRASE", raising=False)
    monkeypatch.delenv("ML_STACK_CLUSTER", raising=False)
    monkeypatch.delenv("ML_STACK_NONINTERACTIVE", raising=False)
    return asked


def _join(tmp_path, *more: str) -> int:
    return fleet_join.main(["--cluster-key", str(tmp_path / "cluster.key"), "--root", str(tmp_path),
                            "--port", "1", "join", "--timeout", "0.5", *more])


def _terminal(monkeypatch, typed: list[str], words: list[str]):
    prompts: list[str] = []

    monkeypatch.setattr(sys, "stdin", _Terminal(""))
    monkeypatch.setattr("builtins.input", lambda prompt="": (prompts.append(prompt), typed.pop(0))[1])
    monkeypatch.setattr(getpass, "getpass", lambda prompt="": (prompts.append(prompt), words.pop(0))[1])
    return prompts


def test_a_terminal_picks_a_cluster_by_number_then_types_its_passphrase_once(
        captured, tmp_path, monkeypatch, advertise, udp):
    advertise("lab", "studio")
    advertise("home", "harrowgate")
    prompts = _terminal(monkeypatch, ["1"], [WORDS])
    assert _join(tmp_path, "--mode", "prod") == 0
    assert captured["group"] == "home" and captured["passphrase"] == WORDS
    assert sum("Passphrase" in p for p in prompts) == 1
    assert not any("Again" in p for p in prompts)


def test_a_terminal_that_types_a_new_name_confirms_the_passphrase(captured, tmp_path, monkeypatch, udp):
    prompts = _terminal(monkeypatch, ["attic"], [WORDS, WORDS])
    assert _join(tmp_path, "--mode", "prod") == 0
    assert captured["group"] == "attic"
    assert any("Again" in p for p in prompts)


def test_a_terminal_takes_the_default_name_on_enter_when_none_is_found(captured, tmp_path, monkeypatch, udp):
    _terminal(monkeypatch, [""], [WORDS, WORDS])
    assert _join(tmp_path, "--mode", "prod") == 0
    assert captured["group"] == "ml-stack"


def test_group_skips_the_question(captured, tmp_path, monkeypatch, udp):
    prompts = _terminal(monkeypatch, [], [WORDS, WORDS])
    assert _join(tmp_path, "--group", "attic") == 0
    assert captured["group"] == "attic"
    assert not any("luster" in p for p in prompts)


def test_a_script_is_never_asked(captured, tmp_path, monkeypatch):
    monkeypatch.setattr("builtins.input", lambda *_: pytest.fail("asked a script"))
    monkeypatch.setattr(sys, "stdin", io.StringIO(f"{WORDS}\n"))
    assert _join(tmp_path) == 0
    assert captured["group"] == "" and captured["passphrase"] == ""
    assert sys.stdin.read() == f"{WORDS}\n"


def test_a_terminal_marked_non_interactive_is_never_asked(captured, tmp_path, monkeypatch):
    monkeypatch.setenv("ML_STACK_NONINTERACTIVE", "1")
    monkeypatch.setattr("builtins.input", lambda *_: pytest.fail("asked an agent"))
    monkeypatch.setattr(sys, "stdin", _Terminal(f"{WORDS}\n"))
    assert _join(tmp_path) == 0
    assert captured["group"] == "" and captured["passphrase"] == ""


def test_a_passphrase_from_the_environment_is_never_asked_for_a_name(captured, tmp_path, monkeypatch):
    monkeypatch.setenv("ML_STACK_PASSPHRASE", WORDS)
    monkeypatch.setattr(sys, "stdin", _Terminal(""))
    monkeypatch.setattr("builtins.input", lambda *_: pytest.fail("asked an install script"))
    assert _join(tmp_path) == 0
    assert captured["group"] == "ml-stack"


def test_a_bad_group_name_is_refused_with_a_plain_message(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("ML_STACK_PASSPHRASE", WORDS)
    assert _join(tmp_path, "--group", "a/b") == 2
    assert "A cluster name cannot contain '/'" in capsys.readouterr().err


def test_the_clusters_command_lists_what_the_network_offers_without_joining(
        advertise, tmp_path, udp, capsys):
    advertise("lab", "studio")
    code = fleet_join.main(["--cluster-key", str(tmp_path / "none.key"), "--port", "1",
                            "clusters", "--timeout", "1"])
    assert code == 0
    assert "1) lab - studio" in capsys.readouterr().out
    assert not (tmp_path / "none.key").exists()


def test_the_clusters_command_says_when_there_is_none(tmp_path, udp, capsys):
    assert fleet_join.main(["--cluster-key", str(tmp_path / "none.key"), "--port", "1",
                            "clusters", "--timeout", "0.5"]) == 1
    assert "no cluster" in capsys.readouterr().out
