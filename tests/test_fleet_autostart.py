"""Installing the daemon to start on its own, and changing that answer.

Every test here points the module at a temporary directory and fakes the one
subprocess it shells out to, so nothing is installed on the machine running them.
"""

from __future__ import annotations

import subprocess
import sys
import types

import pytest


@pytest.fixture
def mac(monkeypatch, tmp_path):
    """The module as it behaves on macOS, writing into tmp_path."""
    from ml_stack.fleet import autostart

    paths = {"login": tmp_path / "login.plist", "boot": tmp_path / "boot.plist"}
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(autostart, "_mac_path", lambda mode: paths[mode])
    monkeypatch.setattr(
        subprocess, "run",
        lambda *a, **k: types.SimpleNamespace(returncode=0, stdout="", stderr=""))
    return autostart, paths


@pytest.fixture
def windows(monkeypatch, tmp_path):
    """The module as it behaves on Windows, with a scheduled task already there."""
    from ml_stack.fleet import autostart

    ran: list[list[str]] = []
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(autostart, "_windows_startup", lambda: tmp_path / "start.cmd")
    monkeypatch.setattr(autostart, "_executable", lambda: ["ml-stack-traind.exe"])
    monkeypatch.setattr(autostart, "_windows_task_exists", lambda: True)
    monkeypatch.setattr(
        subprocess, "run",
        lambda argv, *a, **k: (ran.append(list(argv)),
                               types.SimpleNamespace(returncode=0, stdout="",
                                                     stderr=""))[1])
    return autostart, ran


class TestChangingTheAnswer:
    def test_only_when_i_open_it_removes_what_login_installed(self, mac, tmp_path):
        auto, paths = mac
        auto.install("login", log_dir=tmp_path)
        assert paths["login"].exists()

        done = auto.install("manual", log_dir=tmp_path)
        assert not paths["login"].exists()
        assert done.installed and not done.command

    def test_starting_at_login_removes_the_boot_job(self, mac, tmp_path):
        auto, paths = mac
        paths["boot"].write_text("<plist/>")

        auto.install("login", log_dir=tmp_path)
        assert paths["login"].exists()
        assert not paths["boot"].exists(), "two jobs would start two daemons"

    def test_what_could_not_be_removed_is_reported(self, monkeypatch, tmp_path):
        from ml_stack.fleet import autostart

        locked = tmp_path / "locked"
        locked.mkdir()
        boot = locked / "boot.plist"
        boot.write_text("<plist/>")
        locked.chmod(0o500)
        monkeypatch.setattr(sys, "platform", "darwin")
        monkeypatch.setattr(
            autostart, "_mac_path",
            lambda mode: boot if mode == "boot" else tmp_path / "login.plist")
        monkeypatch.setattr(
            subprocess, "run",
            lambda *a, **k: types.SimpleNamespace(returncode=0, stdout="", stderr=""))

        done = autostart.install("manual", log_dir=tmp_path)
        locked.chmod(0o700)

        assert not done.installed
        assert str(boot) in done.command

    def test_the_scheduled_task_goes_when_the_answer_changes(self, windows, tmp_path):
        auto, ran = windows
        auto.install("login", log_dir=tmp_path)

        assert [c for c in ran if "/Delete" in c], ran
        assert (tmp_path / "start.cmd").exists()

    def test_a_scheduled_task_is_visible_as_starting_at_boot(self, windows):
        auto, _ = windows
        assert auto.status()["mode"] == "boot"

    def test_an_answer_that_is_not_one_of_the_three_is_refused(self, mac, tmp_path):
        auto, _ = mac
        with pytest.raises(ValueError):
            auto.install("whenever", log_dir=tmp_path)
