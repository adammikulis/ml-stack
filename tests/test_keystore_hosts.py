"""The keystore directory on the hosts that have no POSIX modes or locks: Windows (icacls at creation)
and WSL with the state root on a Windows drive (refused). Subprocess and /proc are replaced; nothing
touches a real keystore or ACL."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from poolhouse import keystore, platform, private_path
from tests import keystore_support, test_keystore

counting = keystore_support.counting
make = test_keystore.make


SID = "S-1-5-21-111-222-333-1001"


@pytest.fixture
def as_windows(monkeypatch):
    """Windows as far as `platform` and the keystore can tell: `whoami` answers a SID and `icacls` is
    recorded, not run."""
    ran: list[list[str]] = []

    def run(argv, **_k):
        ran.append(list(argv))
        out = f'"HOST\\fixture","{SID}"\n' if argv[0] == "whoami" else ""
        return subprocess.CompletedProcess(argv, 0, stdout=out)

    monkeypatch.setattr(platform, "is_windows", lambda: True)
    monkeypatch.setattr(platform.subprocess, "run", run)
    monkeypatch.setattr(keystore.human, "protect", lambda _d: None)
    return ran


def test_windows_grants_the_sid_then_drops_the_inherited_entries_at_creation(tmp_path, counting, as_windows):
    make(tmp_path)._ensure_dir()
    where = str(tmp_path / "ks")
    assert [a for a in as_windows if a[0] == "icacls"] == [
        ["icacls", where, "/grant:r", f"*{SID}:(OI)(CI)F"], ["icacls", where, "/inheritance:r"]]


def test_a_failed_grant_never_reaches_the_step_that_would_lock_everyone_out(tmp_path, counting, as_windows,
                                                                            monkeypatch, caplog):
    def run(argv, **_k):
        as_windows.append(list(argv))
        return subprocess.CompletedProcess(argv, 5 if argv[0] == "icacls" else 0,
                                           stdout=f'"HOST\\fixture","{SID}"\n')

    monkeypatch.setattr(platform.subprocess, "run", run)
    with caplog.at_level("WARNING", logger="poolhouse.keystore"):
        make(tmp_path)._ensure_dir()
    assert "could not restrict" in caplog.text
    assert not any(a[-1] == "/inheritance:r" for a in as_windows)


def test_posix_uses_chmod_and_never_icacls(tmp_path, counting, monkeypatch):
    ran: list[list[str]] = []
    monkeypatch.setattr(platform.subprocess, "run", lambda argv, **_k: ran.append(list(argv)))
    make(tmp_path)._ensure_dir()
    assert ran == [] and ((tmp_path / "ks").stat().st_mode & 0o077) == 0


def test_private_dir_without_a_sid_says_it_did_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(platform, "is_windows", lambda: True)
    monkeypatch.setattr(platform.subprocess, "run", lambda argv, **_k: subprocess.CompletedProcess(argv, 1, stdout=""))
    assert platform.private_dir(tmp_path) is False


def test_a_state_root_on_a_windows_drive_is_refused_before_anything_is_made(tmp_path, counting, monkeypatch):
    monkeypatch.setattr(private_path, "windows_mount", lambda path: True)
    store = make(tmp_path)
    with pytest.raises(keystore.KeystoreUnavailable, match="POOLHOUSE_HOME"):
        store.subkey("memory", "a")
    assert not (tmp_path / "ks").exists()


@pytest.mark.skipif(sys.platform == "win32", reason="/proc/mounts is read on POSIX")
def test_windows_mount_reads_the_mount_table(tmp_path, monkeypatch):
    drive = tmp_path / "mnt" / "c"
    table = f"C:\\134 {drive} 9p rw 0 0\next4 / ext4 rw 0 0\n"
    monkeypatch.setattr(Path, "read_text", lambda self, **_k: (
        "5.15.90.1-microsoft-standard-WSL2" if self.name == "osrelease" else table))
    assert private_path.windows_mount(drive / "Users" / "me" / ".poolhouse") is True
    assert private_path.windows_mount(tmp_path / "home") is False


def test_backend_names_the_ring_in_use_and_why_there_is_none(tmp_path, counting, monkeypatch):
    assert make(tmp_path).backend() == "tests.keystore_support.CountingRing"
    monkeypatch.setenv(keystore.ENV_NO_REAL, "1")
    monkeypatch.setattr(keystore, "is_real", lambda ring: True)
    assert make(tmp_path).backend().startswith("none: the machine's own keystore is switched off")


@pytest.mark.parametrize("name", ["with space", "semi;colon", "dollar$(calc)", "back`tick`", "-leading", "pct%PATH%"])
def test_a_hostile_directory_name_reaches_icacls_as_one_argument(tmp_path, as_windows, name):
    target = tmp_path / name
    platform.private_dir(target)
    icacls = [a for a in as_windows if a[0] == "icacls"]
    assert icacls and all(a[1] == str(target) and len(a) in (3, 4) for a in icacls)
    assert all(a[0] in ("whoami", "icacls") for a in as_windows)


@pytest.mark.parametrize("answer", ['"HOST\\u","not-a-sid"\n', '"HOST\\u","S-1-5-21;calc"\n', "", '"x"\n'])
def test_whoami_output_that_is_not_a_sid_is_never_handed_to_icacls(tmp_path, monkeypatch, answer):
    ran: list[list[str]] = []

    def run(argv, **_k):
        ran.append(list(argv))
        return subprocess.CompletedProcess(argv, 0, stdout=answer)

    monkeypatch.setattr(platform, "is_windows", lambda: True)
    monkeypatch.setattr(platform.subprocess, "run", run)
    assert platform.private_dir(tmp_path) is False
    assert not any(a[0] == "icacls" for a in ran)
