"""The cluster passphrase kept in the keystore and the recovery file: join stores it, `passphrase` reads
it back to a person only, a recovery file joins another machine, `leave` clears it. The keystore is a
fake keyring backend; the real one is never touched."""

from __future__ import annotations

import socket
import stat

import pytest

from ml_stack import keystore
from ml_stack.fleet import recovery
from ml_stack.fleet.discovery import (
    Advertiser,
    Beacon,
    Salting,
    discover,
    join_cluster,
    load_cluster_key,
    memberships,
)
from ml_stack.fleet.join import leave_machine, main
from ml_stack.keystore import Keystore, Wires
from tests.keystore_support import counting  # noqa: F401

WORDS = "quince larch marlow"


@pytest.fixture
def ks(tmp_path, counting, monkeypatch):  # noqa: F811
    store = Keystore(directory=tmp_path / "ks", wires=Wires(interactive=lambda: True, sleep=lambda _s: None))
    monkeypatch.setattr(recovery, "_store", lambda: store)
    return store


def _udp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _join(path, group="lab"):
    join_cluster(WORDS, group=group, path=path, salting=Salting(salt=b"s" * 16))
    recovery.remember(WORDS, group, path)


def test_the_stored_passphrase_reads_back(tmp_path, ks):
    key = tmp_path / "cluster.key"
    _join(key)
    assert recovery.recall("lab", key) == WORDS
    assert recovery.recall("", key) == WORDS
    assert WORDS not in recovery.passphrases_path(key).read_text()


def test_a_keystore_that_refuses_is_said_once_and_the_join_stands(tmp_path, monkeypatch):
    def refuse():
        raise keystore.KeystoreUnavailable("no keystore here")

    monkeypatch.setattr(recovery, "_store", refuse)
    key = tmp_path / "cluster.key"
    join_cluster(WORDS, group="lab", path=key, salting=Salting(salt=b"s" * 16))
    said: list[str] = []
    recovery.remember(WORDS, "lab", key, say=said.append)
    assert len(said) == 1 and "not saved" in said[0]
    assert load_cluster_key(key) is not None
    assert recovery.recall("lab", key) is None


def test_the_passphrase_command_prints_for_a_person(tmp_path, ks, monkeypatch, capsys):
    key = tmp_path / "cluster.key"
    _join(key)
    monkeypatch.delenv("CLAUDECODE", raising=False)
    monkeypatch.delenv("ML_STACK_NONINTERACTIVE", raising=False)
    monkeypatch.setattr("sys.stdin.isatty", lambda: True, raising=False)
    monkeypatch.setattr("sys.stdout.isatty", lambda: True, raising=False)
    assert main(["--cluster-key", str(key), "passphrase"]) == 0
    assert capsys.readouterr().out.strip() == WORDS


@pytest.mark.parametrize("marker", ["CLAUDECODE", "ML_STACK_NONINTERACTIVE"])
def test_the_passphrase_command_refuses_an_agent(tmp_path, ks, monkeypatch, capsys, marker):
    key = tmp_path / "cluster.key"
    _join(key)
    monkeypatch.setenv(marker, "1")
    monkeypatch.setattr("sys.stdin.isatty", lambda: True, raising=False)
    monkeypatch.setattr("sys.stdout.isatty", lambda: True, raising=False)
    assert main(["--cluster-key", str(key), "passphrase"]) == 2
    assert WORDS not in capsys.readouterr().out


def test_export_then_import_joins_a_fresh_machine_with_the_same_key(tmp_path, ks, monkeypatch):
    first, second = tmp_path / "a" / "cluster.key", tmp_path / "b" / "cluster.key"
    first.parent.mkdir(), second.parent.mkdir()
    _join(first)
    monkeypatch.delenv("CLAUDECODE", raising=False)
    monkeypatch.delenv("ML_STACK_NONINTERACTIVE", raising=False)
    monkeypatch.setattr("sys.stdin.isatty", lambda: True, raising=False)
    monkeypatch.setattr("sys.stdout.isatty", lambda: True, raising=False)
    file = tmp_path / "lab.recovery"
    assert main(["--cluster-key", str(first), "recovery", "export", str(file)]) == 0
    assert stat.S_IMODE(file.stat().st_mode) == 0o600
    assert "run commands on every machine" in file.read_text().splitlines()[1]
    assert main(["--cluster-key", str(second), "recovery", "import", str(file)]) == 0
    assert memberships(second) == memberships(first)
    assert recovery.recall("lab", second) is None
    port = _udp_port()
    with Advertiser(Beacon(name="fresh", port=8770), load_cluster_key(second), port=port, interval_s=0.2):
        found = discover(load_cluster_key(first), timeout_s=2.0, port=port)
    assert "fresh" in [b.name for b in found]


def test_a_file_that_is_not_a_recovery_file_is_refused(tmp_path, capsys):
    bad = tmp_path / "bad"
    bad.write_text("# nothing\n{}\n")
    assert main(["--cluster-key", str(tmp_path / "c.key"), "recovery", "import", str(bad)]) == 2
    assert memberships(tmp_path / "c.key") == []


def test_leave_clears_the_stored_passphrase(tmp_path, ks):
    key = tmp_path / "cluster.key"
    _join(key)
    _join(key, group="home")
    leave_machine(group="lab", root=tmp_path, cluster_key_path=key, say=lambda s: None, unpersist=lambda: [])
    assert recovery.recall("lab", key) is None
    assert recovery.recall("home", key) == WORDS
    leave_machine(root=tmp_path, cluster_key_path=key, say=lambda s: None, unpersist=lambda: [])
    assert not recovery.passphrases_path(key).exists()


def test_join_machine_stores_the_passphrase(tmp_path, ks, monkeypatch):
    from ml_stack.fleet import join as joining

    key = tmp_path / "cluster.key"
    monkeypatch.setattr(joining, "already_running", lambda port: {"name": "quince", "machine": "m1"})
    monkeypatch.setattr(joining, "checks", lambda *a, **k: [])
    monkeypatch.setattr(joining, "peers", lambda **k: [])
    joining.join_machine(passphrase=WORDS, group="lab", root=tmp_path, cluster_key_path=key,
                         enrol=lambda words, group: join_cluster(words, group=group, path=key,
                                                                 salting=Salting(salt=b"s" * 16)),
                         say=lambda s: None)
    assert recovery.recall("lab", key) == WORDS
