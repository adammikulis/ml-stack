"""The keystore directory on the hosts that have no POSIX modes or locks: Windows (icacls at creation)
and WSL with the state root on a Windows drive (refused). Subprocess and /proc are replaced; nothing
touches a real keystore or ACL."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from ml_stack import keystore, platform, private_path
from tests import keystore_support, test_keystore

counting = keystore_support.counting
make = test_keystore.make


@pytest.fixture
def as_windows(monkeypatch):
    """Windows as far as `platform` and the keystore can tell, with `icacls` recorded not run."""
    ran: list[list[str]] = []
    monkeypatch.setattr(platform, "is_windows", lambda: True)
    monkeypatch.setenv("USERNAME", "fixture-user")
    monkeypatch.setattr(platform.subprocess, "run",
                        lambda argv, **_k: ran.append(list(argv)) or subprocess.CompletedProcess(argv, 0))
    monkeypatch.setattr(keystore.human, "protect", lambda _d: None)
    return ran


def test_windows_cuts_the_keystore_directory_to_the_owner_at_creation(tmp_path, counting, as_windows):
    make(tmp_path)._ensure_dir()
    assert as_windows == [["icacls", str(tmp_path / "ks"), "/inheritance:r", "/grant:r",
                           "fixture-user:(OI)(CI)F"]]


def test_a_refused_icacls_is_reported_not_swallowed(tmp_path, counting, as_windows, monkeypatch, caplog):
    monkeypatch.setattr(platform.subprocess, "run",
                        lambda argv, **_k: subprocess.CompletedProcess(argv, 5))
    with caplog.at_level("WARNING", logger="ml_stack.keystore"):
        make(tmp_path)._ensure_dir()
    assert "could not restrict" in caplog.text


def test_posix_uses_chmod_and_never_icacls(tmp_path, counting, monkeypatch):
    ran: list[list[str]] = []
    monkeypatch.setattr(platform.subprocess, "run", lambda argv, **_k: ran.append(list(argv)))
    make(tmp_path)._ensure_dir()
    assert ran == [] and ((tmp_path / "ks").stat().st_mode & 0o077) == 0


def test_private_dir_without_a_username_says_it_did_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(platform, "is_windows", lambda: True)
    monkeypatch.delenv("USERNAME", raising=False)
    assert platform.private_dir(tmp_path) is False


def test_a_state_root_on_a_windows_drive_is_refused_before_anything_is_made(tmp_path, counting, monkeypatch):
    monkeypatch.setattr(private_path, "windows_mount", lambda path: True)
    store = make(tmp_path)
    with pytest.raises(keystore.KeystoreUnavailable, match="ML_STACK_HOME"):
        store.subkey("memory", "a")
    assert not (tmp_path / "ks").exists()


@pytest.mark.skipif(sys.platform == "win32", reason="/proc/mounts is read on POSIX")
def test_windows_mount_reads_the_mount_table(tmp_path, monkeypatch):
    drive = tmp_path / "mnt" / "c"
    table = f"C:\\134 {drive} 9p rw 0 0\next4 / ext4 rw 0 0\n"
    monkeypatch.setattr(Path, "read_text", lambda self, **_k: (
        "5.15.90.1-microsoft-standard-WSL2" if self.name == "osrelease" else table))
    assert private_path.windows_mount(drive / "Users" / "me" / ".ml-stack") is True
    assert private_path.windows_mount(tmp_path / "home") is False


def test_backend_names_the_ring_in_use_and_why_there_is_none(tmp_path, counting, monkeypatch):
    assert make(tmp_path).backend() == "tests.keystore_support.CountingRing"
    monkeypatch.setenv(keystore.ENV_NO_REAL, "1")
    monkeypatch.setattr(keystore, "is_real", lambda ring: True)
    assert make(tmp_path).backend().startswith("none: the machine's own keystore is switched off")
