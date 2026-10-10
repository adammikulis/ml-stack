"""`poolhouse migrate` where it can go wrong: both names present, two keys, a rerun after a kill, a copy
that looks right and is not, and what counts as a process of the old build."""

from __future__ import annotations

from pathlib import Path

import pytest

from poolhouse import home, keystore, legacy, migrate
from poolhouse.files import CrossDevice, write_json
from tests import keystore_support, migrate_support

counting = keystore_support.counting
account = migrate_support.account
old_state = migrate_support.old_state
ring = migrate_support.ring


# -- (a) both names present ---------------------------------------------------------------------


def test_both_directories_present_is_a_failure_that_says_the_fix(account, tmp_path, capsys):
    old = old_state(account)
    (account / ".poolhouse").mkdir()
    (account / ".poolhouse" / "theirs").write_text("n", encoding="utf-8")
    assert migrate.run(account, ring=ring(tmp_path / "k")) != 0
    said = capsys.readouterr().err
    assert str(old) in said and "merge" in said.lower() and "remove" in said
    assert (old / "machine-id").exists() and (account / ".poolhouse" / "theirs").exists()


# -- (b) the keychain: three different "no" answers ----------------------------------------------


def held_pair(tmp_path, counting, *, new: str | None, old: str | None):
    make = ring(tmp_path / "k")
    make().wrap("credentials", "X", b"x")
    ((_, who),) = counting.held
    counting.held.clear()
    if new is not None:
        counting.held[(keystore.SERVICE, who)] = new
    if old is not None:
        counting.held[(legacy.KEYCHAIN_SERVICE, who)] = old
    return make, who


KEY_A, KEY_B = "v1:" + "QUFB" * 10 + "QUE=", "v1:" + "QkJC" * 10 + "QkI="


def test_no_key_under_the_old_name_is_said_so_and_is_not_a_failure(account, tmp_path, counting, capsys):
    make, _who = held_pair(tmp_path, counting, new=None, old=None)
    assert migrate.run(account, ring=make) == 0
    assert "no master key under the old name" in capsys.readouterr().out


def test_two_different_keys_stay_where_they_are_and_fail(account, tmp_path, counting, capsys):
    make, who = held_pair(tmp_path, counting, new=KEY_A, old=KEY_B)
    assert migrate.run(account, ring=make) != 0
    assert "stays locked" in capsys.readouterr().err
    assert counting.held == {(keystore.SERVICE, who): KEY_A, (legacy.KEYCHAIN_SERVICE, who): KEY_B}


def test_the_same_key_under_both_names_drops_the_old_one(account, tmp_path, counting):
    make, who = held_pair(tmp_path, counting, new=KEY_A, old=KEY_A)
    assert migrate.run(account, ring=make) == 0
    assert counting.held == {(keystore.SERVICE, who): KEY_A}


def test_a_copy_that_reads_back_wrong_is_deleted_and_the_old_key_stays(account, tmp_path, counting, capsys):
    make, who = held_pair(tmp_path, counting, new=None, old=KEY_A)
    real = counting.set_password

    def garbled(service, user, password):
        real(service, user, KEY_B if service == keystore.SERVICE else password)

    counting.set_password = garbled
    assert migrate.run(account, ring=make) != 0
    assert "read back" in capsys.readouterr().err
    assert counting.held == {(legacy.KEYCHAIN_SERVICE, who): KEY_A}


# -- (c) a rerun with another account, and a project file that is already there --------------------


def test_a_rerun_for_another_account_still_renames_the_project_files(account, tmp_path, monkeypatch):
    old_state(account)
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    (checkout / legacy.PROJECT_FILE).write_text("{}", encoding="utf-8")
    write_json(account / ".ml-stack" / "workspace-connections.json", {str(checkout): {}})
    assert migrate.run(account, ring=ring(tmp_path / "k")) == 0
    (checkout / ".poolhouse-project.json").rename(checkout / legacy.PROJECT_FILE)  # a kill left it undone
    monkeypatch.setattr(home, "user_home", lambda: tmp_path / "somebody-else")
    assert migrate.run(account, ring=ring(tmp_path / "k")) == 0
    assert (checkout / ".poolhouse-project.json").exists() and not (checkout / legacy.PROJECT_FILE).exists()


def test_an_existing_project_file_of_the_new_name_is_never_overwritten(account, tmp_path):
    old_state(account)
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    (checkout / legacy.PROJECT_FILE).write_text("old", encoding="utf-8")
    (checkout / ".poolhouse-project.json").write_text("new", encoding="utf-8")
    write_json(account / ".ml-stack" / "workspace-connections.json", {str(checkout): {}})
    assert migrate.run(account, ring=ring(tmp_path / "k")) != 0
    assert (checkout / ".poolhouse-project.json").read_text(encoding="utf-8") == "new"
    assert (checkout / legacy.PROJECT_FILE).read_text(encoding="utf-8") == "old"


# -- (d) the copy across volumes ----------------------------------------------------------------


def across(monkeypatch, old: Path):
    real = migrate.promote

    def promote(a, b):
        if Path(a) == old:
            raise CrossDevice(18, "cross-device")
        return real(a, b)

    monkeypatch.setattr(migrate, "promote", promote)


def corrupt_copies(monkeypatch, change):
    real = migrate.shutil.copytree

    def copy(src, dst, *args, **kwargs):
        out = real(src, dst, *args, **kwargs)
        if str(dst).endswith(".migrating"):
            change(Path(dst))
        return out

    monkeypatch.setattr(migrate.shutil, "copytree", copy)


def test_a_copy_with_the_same_sizes_and_other_bytes_is_refused(account, monkeypatch):
    old = old_state(account)
    across(monkeypatch, old)
    corrupt_copies(monkeypatch, lambda d: (d / "machine-id").write_text("abc124", encoding="utf-8"))
    with pytest.raises(OSError, match="did not match"):
        migrate.move_directory(old, account / ".poolhouse")
    assert (old / "machine-id").read_text(encoding="utf-8") == "abc123"


def test_a_copy_with_another_symlink_target_is_refused(account, monkeypatch):
    old = old_state(account)
    (old / "link").symlink_to("machine-id")
    across(monkeypatch, old)

    def retarget(d: Path) -> None:
        (d / "link").unlink()
        (d / "link").symlink_to("workspace")

    corrupt_copies(monkeypatch, retarget)
    with pytest.raises(OSError, match="did not match"):
        migrate.move_directory(old, account / ".poolhouse")


def test_a_copy_with_other_permissions_is_refused(account, monkeypatch):
    old = old_state(account)
    (old / "machine-id").chmod(0o600)
    across(monkeypatch, old)
    corrupt_copies(monkeypatch, lambda d: (d / "machine-id").chmod(0o644))
    with pytest.raises(OSError, match="did not match"):
        migrate.move_directory(old, account / ".poolhouse")


def test_a_kill_between_the_copy_and_the_removal_is_finished_by_the_next_run(account, tmp_path, monkeypatch):
    old = old_state(account)
    across(monkeypatch, old)
    real = migrate.shutil.rmtree

    def killed(path, *args, **kwargs):
        if Path(path) == old:
            raise KeyboardInterrupt
        return real(path, *args, **kwargs)

    monkeypatch.setattr(migrate.shutil, "rmtree", killed)
    with pytest.raises(KeyboardInterrupt):
        migrate.run(account, ring=ring(tmp_path / "k"))
    assert old.exists() and (account / ".poolhouse" / "machine-id").exists()
    monkeypatch.setattr(migrate.shutil, "rmtree", real)
    assert migrate.run(account, ring=ring(tmp_path / "k")) == 0
    assert not old.exists() and (account / ".poolhouse" / "machine-id").read_text(encoding="utf-8") == "abc123"


def test_the_log_is_written_even_when_the_first_step_raises(account, monkeypatch):
    old = old_state(account)

    def boom(a, b):
        raise PermissionError("denied")

    monkeypatch.setattr(migrate, "move_directory", boom)
    assert migrate.run(account, ring=lambda: None) == 2
    assert "stopped" in (old / "migrate.log").read_text(encoding="utf-8")


# -- (e) what is a process of the old build -------------------------------------------------------


@pytest.mark.parametrize("argv", [
    ("python", "-u", "-m", "ml_stack.cli.daemon"),
    ("/usr/bin/python3", "-X", "dev", "-u", "-m", "ml_stack.node_launch", "supervise"),
    ("uv", "run", "ml-stack-serve", "up"),
    ("uv", "run", "--with", "x", "python", "-m", "ml_stack.bench.run"),
    ("/Users/x/repos/ml-stack/.venv/bin/ml-stack-serve", "up"),
    ("python3.13", "/opt/ml-stack/bin/ml-stack-traind"),
])
def test_these_are_processes_of_the_old_build(argv):
    assert migrate._named_old(argv)


@pytest.mark.parametrize("argv", [
    ("vim", "ml-stack-notes.md"),
    ("/usr/bin/less", "/var/log/ml-stack.log"),
    ("bash", "ml-stack-setup.sh"),
    ("python", "-c", "print('ml-stack')"),
    ("python", "-m", "poolhouse.node_launch", "supervise"),
    ("uv", "run", "poolhouse-serve"),
    ("tail", "-f", "ml_stack.log"),
    ("/x/ml-stack/.build-venv/bin/python", "/x/run-jedi-language-server.py"),
    ("/Users/x/repos/ml-stack/.venv/bin/python", "-c", "pass"),
])
def test_these_only_mention_the_old_name(argv):
    assert not migrate._named_old(argv)


def test_a_process_inside_the_old_cache_or_state_directory_counts(account, monkeypatch):
    old_cache = account / ".cache" / "ml_stack"
    monkeypatch.setattr(migrate.process, "running_within", lambda path: [9] if path == old_cache else [])
    assert [line for line in migrate.running(account / ".ml-stack", old_cache) if line.startswith("9:")]
    assert migrate.running(account / ".ml-stack") == []


def test_a_language_server_in_a_checkout_named_for_the_old_build_does_not_count(account, monkeypatch):
    argv = ("/x/ml-stack/.build-venv/bin/python", "/x/run-jedi-language-server.py")
    monkeypatch.setattr(migrate.process, "command_lines", lambda: [(9876, 0.0, argv)])
    assert migrate.running(account / ".ml-stack") == []
