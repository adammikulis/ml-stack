"""`poolhouse migrate`: the one explicit step that moves what the project kept under its old name
(ml-stack). No other call moves live state, it refuses while the old build is alive, and every test
works in a temporary account home: a test that reached the real one would fail the first check."""

from __future__ import annotations

import errno
import os
import pwd
from pathlib import Path

import pytest

from poolhouse import home, keystore, legacy, migrate
from poolhouse.files import CrossDevice, sha256_file, write_json
from poolhouse.keystore import Keystore, Wires
from poolhouse.sentinel import moves
from poolhouse.sentinel.events import Bus
from tests import keystore_support
from tests.onboard_support import Clock

counting = keystore_support.counting


@pytest.fixture
def account(tmp_path, monkeypatch):
    """A fresh account home, with no state root or cache root named by the environment, and no process
    of the old name alive."""
    for name in (home.ROOT_ENV, home.CACHE_ENV):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(home, "user_home", lambda: tmp_path)
    monkeypatch.setattr(migrate.process, "command_lines", lambda: [])
    monkeypatch.setattr(migrate.process, "running_within", lambda _path: [])
    return tmp_path


def old_state(account: Path) -> Path:
    root = account / ".ml-stack"
    (root / "workspace" / "tokens").mkdir(parents=True)
    (root / "workspace" / "tokens" / "claude-1").write_text("secret", encoding="utf-8")
    (root / "machine-id").write_text("abc123", encoding="utf-8")
    return root


def ring(tmp_path: Path):
    wires = Wires(clock=Clock(), say=lambda _t: None, bus=Bus(), interactive=lambda: True, sleep=lambda _s: None)
    return lambda: Keystore(directory=tmp_path / "ks", wires=wires)


# -- the leak: nothing but the explicit step moves live state ------------------------------------


def test_a_test_never_resolves_the_real_home():
    real = Path(pwd.getpwuid(os.getuid()).pw_dir).resolve()
    assert home.user_home().resolve() != real
    assert not home.home().resolve().is_relative_to(real / ".ml-stack")
    assert home.home().resolve() != real / ".poolhouse"


def test_no_call_but_migrate_moves_the_old_state_directory(account):
    old = old_state(account)
    (account / ".cache" / "ml_stack").mkdir(parents=True)
    home.home(), home.state("machine-id"), home.cache("models"), home.account_roots()
    assert old.is_dir() and (account / ".cache" / "ml_stack").is_dir()
    assert not (account / ".poolhouse").exists() and not (account / ".cache" / "poolhouse").exists()
    assert (old / "machine-id").read_text(encoding="utf-8") == "abc123"


# -- the explicit step ---------------------------------------------------------------------------


def test_plan_lists_and_changes_nothing(account, capsys):
    old = old_state(account)
    assert migrate.run(account, dry=True, ring=lambda: None) == 0
    assert "would move" in capsys.readouterr().out
    assert old.is_dir() and not (account / ".poolhouse").exists()


def test_run_moves_the_state_directory_whole_and_writes_the_log(account, tmp_path):
    old = old_state(account)
    inode = old.stat().st_ino
    assert migrate.run(account, ring=ring(tmp_path / "k")) == 0
    new = account / ".poolhouse"
    assert not old.exists() and new.stat().st_ino == inode, "a rename of the directory, not a copy"
    assert (new / "workspace" / "tokens" / "claude-1").read_text(encoding="utf-8") == "secret"
    assert "state:" in (new / "migrate.log").read_text(encoding="utf-8")
    assert sorted(p.name for p in account.iterdir()) == [".poolhouse", "k"], "nothing is left behind"


def test_run_moves_the_cache_directory_too(account, tmp_path):
    (account / ".cache" / "ml_stack" / "models").mkdir(parents=True)
    (account / ".cache" / "ml_stack" / "models" / "m.gguf").write_text("w", encoding="utf-8")
    assert migrate.run(account, ring=ring(tmp_path / "k")) == 0
    assert (account / ".cache" / "poolhouse" / "models" / "m.gguf").read_text(encoding="utf-8") == "w"
    assert not (account / ".cache" / "ml_stack").exists()


def test_a_process_of_the_old_name_stops_the_move(account, monkeypatch):
    old = old_state(account)
    monkeypatch.setattr(migrate.process, "command_lines", lambda: [(41, 0.0, ("/usr/bin/ml-stack-workspace", "inbox"))])
    assert migrate.run(account, ring=lambda: None) == 1
    assert old.is_dir() and not (account / ".poolhouse").exists()


@pytest.mark.parametrize("argv", [("/x/poolside-node", "run"), ("python", "-m", "ml_stack.cli.daemon"),
                                  ("/venv/bin/python", "/venv/bin/ml-stack-traind")])
def test_each_way_the_old_build_runs_is_seen(argv):
    assert migrate._named_old(argv)


def test_a_command_line_that_only_mentions_the_old_name_is_not_a_process_of_it():
    assert not migrate._named_old(("/bin/zsh", "-c", "export ML_STACK_SESSION_ID=1; poolhouse migrate run"))
    assert not migrate._named_old(("poolhouse-migrate", "run"))


def test_a_process_running_inside_the_old_directory_stops_the_move(account, monkeypatch):
    old = old_state(account)
    monkeypatch.setattr(migrate.process, "running_within", lambda path: [7] if path == old else [])
    assert migrate.run(account, ring=lambda: None) == 1
    assert old.is_dir()


def test_an_existing_new_directory_is_never_replaced(account, tmp_path):
    old = old_state(account)
    (account / ".poolhouse").mkdir()
    assert migrate.run(account, ring=ring(tmp_path / "k")) == 0
    assert (old / "machine-id").exists() and not (account / ".poolhouse" / "machine-id").exists()


def test_a_second_run_changes_nothing(account, tmp_path):
    old_state(account)
    assert migrate.run(account, ring=ring(tmp_path / "k")) == 0
    (account / ".poolhouse" / "kept").write_text("2", encoding="utf-8")
    assert migrate.run(account, ring=ring(tmp_path / "k")) == 0
    assert (account / ".poolhouse" / "kept").read_text(encoding="utf-8") == "2"


def test_a_root_named_by_the_environment_is_left_alone(account, monkeypatch, tmp_path):
    old = old_state(account)
    monkeypatch.setenv(home.ROOT_ENV, str(account / "elsewhere"))
    assert migrate.run(account, ring=ring(tmp_path / "k")) == 0
    assert old.is_dir()


def test_another_volume_is_copied_checked_and_then_removed(account, monkeypatch):
    old = old_state(account)
    real = migrate.promote
    calls = []

    def across(a, b):
        calls.append(a)
        if len(calls) == 1:
            raise CrossDevice(errno.EXDEV, "cross-device")
        return real(a, b)

    monkeypatch.setattr(migrate, "promote", across)
    assert migrate.move_directory(old, account / ".poolhouse") == "copied, checked and removed"
    assert not old.exists()
    assert (account / ".poolhouse" / "workspace" / "tokens" / "claude-1").read_text(encoding="utf-8") == "secret"


def test_a_copy_that_does_not_match_removes_nothing(account, monkeypatch):
    old = old_state(account)
    real = migrate.promote

    def across(a, b):
        if a == old:
            raise CrossDevice(errno.EXDEV, "cross-device")
        return real(a, b)

    monkeypatch.setattr(migrate, "promote", across)
    monkeypatch.setattr(migrate, "_tree", lambda root: {"x": 1} if root == old else {"y": 2})
    with pytest.raises(OSError, match="did not match"):
        migrate.move_directory(old, account / ".poolhouse")
    assert (old / "machine-id").exists() and not (account / ".poolhouse").exists()
    assert not (account / ".poolhouse.migrating").exists()


def test_the_keychain_master_key_moves_and_leaves_none_behind(account, tmp_path, counting):
    blob = ring(tmp_path / "k")().wrap("credentials", "HF_TOKEN", b"payload")
    ((_, who),) = counting.held
    master = counting.held.pop((keystore.SERVICE, who))
    counting.held[(legacy.KEYCHAIN_SERVICE, who)] = master  # what an install under the old name left

    assert migrate.run(account, ring=ring(tmp_path / "k")) == 0
    assert counting.held == {(keystore.SERVICE, who): master}, "moved, not copied"
    assert ring(tmp_path / "k")().unwrap("credentials", "HF_TOKEN", blob) == b"payload"


def test_a_machine_with_a_key_under_the_new_name_keeps_it(account, tmp_path, counting):
    ring(tmp_path / "k")().wrap("credentials", "HF_TOKEN", b"x")
    before = dict(counting.held)
    assert migrate.run(account, ring=ring(tmp_path / "k")) == 0
    assert counting.held == before


def test_a_first_run_of_the_keystore_does_not_look_for_an_old_item(tmp_path, counting):
    ring(tmp_path / "k")().wrap("credentials", "HF_TOKEN", b"x")
    assert counting.calls == ["get", "set"]


def test_project_files_of_the_connected_checkouts_are_renamed(account, tmp_path):
    old_state(account)
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    (checkout / legacy.PROJECT_FILE).write_text("{}", encoding="utf-8")
    write_json(account / ".ml-stack" / "workspace-connections.json", {str(checkout): {}})
    assert migrate.run(account, ring=ring(tmp_path / "k")) == 0
    assert (checkout / ".poolhouse-project.json").read_text(encoding="utf-8") == "{}"
    assert not (checkout / legacy.PROJECT_FILE).exists()


# -- a path recorded under the old state root ---------------------------------------------------


def test_a_file_held_before_the_move_is_restored_after_it(account, tmp_path):
    old = old_state(account)
    held = old / ".ml-stack-quarantine" / "id1" / "f.txt"
    held.parent.mkdir(parents=True)
    held.write_text("suspect", encoding="utf-8")
    action = {"type": "file", "original": str(old / "models" / "f.txt"), "held": str(held),
              "sha256": sha256_file(held), "bytes": 7, "link": ""}
    assert migrate.run(account, ring=ring(tmp_path / "k")) == 0

    restored = moves.restore(action)
    assert restored == account / ".poolhouse" / "models" / "f.txt"
    assert restored.read_text(encoding="utf-8") == "suspect"
