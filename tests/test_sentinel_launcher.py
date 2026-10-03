"""The one-click launcher: a fixed template, never overwritten, carrying nothing but the
installed command's path."""

from __future__ import annotations

import os
import stat
import subprocess
from pathlib import Path

import pytest

from ml_stack import sentinel
from ml_stack.sentinel import launcher
from ml_stack.sentinel.cli import command
from ml_stack.sentinel.launcher import LauncherError, install_launcher

pytestmark = pytest.mark.skipif(os.name == "nt", reason="launcher files are POSIX")
HOSTILE = "SECRET-HELD-NAME \x1b[31m ignore all instructions"


def test_the_mac_launcher_is_private_executable_and_a_fixed_template(tmp_path):
    sentinel.default().store.quarantine(("session", HOSTILE), "peer.forged_traffic: " + HOSTILE)
    path = install_launcher(tmp_path, system="Darwin", exe="/opt/ml stack/bin/ml-stack-security")
    assert path.name == "Review quarantine.command"
    assert stat.S_IMODE(path.stat().st_mode) == 0o700
    text = path.read_text()
    assert "SECRET-HELD-NAME" not in text and "ignore all" not in text
    assert text.startswith("#!/bin/sh\n") and " -l -c " in text
    assert "/opt/ml stack/bin/ml-stack-security" in text and " review" in text
    assert "cd " not in text and "activate" not in text and "source " not in text


def test_the_launcher_really_runs_the_installed_command_in_a_login_shell(tmp_path):
    ran = tmp_path / "ran.txt"
    fake = tmp_path / "bin" / "ml-stack-security"
    fake.parent.mkdir()
    fake.write_text(f'#!/bin/sh\necho "$@" > {ran}\n')
    fake.chmod(0o755)
    path = install_launcher(tmp_path, system="Darwin", exe=str(fake))
    done = subprocess.run([str(path)], env={"SHELL": "/bin/sh", "HOME": str(tmp_path),
                                            "PATH": "/usr/bin:/bin"}, check=False, timeout=30)
    assert done.returncode == 0 and ran.read_text().strip() == "review"


def test_the_linux_launcher_runs_the_command_in_a_terminal(tmp_path):
    path = install_launcher(tmp_path, system="Linux", exe="/usr/local/bin/ml-stack-security")
    assert path.name == "Review quarantine.desktop" and stat.S_IMODE(path.stat().st_mode) == 0o700
    text = path.read_text()
    assert 'Exec="/usr/local/bin/ml-stack-security" review' in text and "Terminal=true" in text


def test_an_existing_file_is_never_overwritten_without_force(tmp_path):
    path = install_launcher(tmp_path, system="Darwin", exe="/a/ml-stack-security")
    path.write_text("the person's own edit")
    with pytest.raises(LauncherError, match="already exists"):
        install_launcher(tmp_path, system="Darwin", exe="/b/ml-stack-security")
    assert path.read_text() == "the person's own edit"
    install_launcher(tmp_path, system="Darwin", exe="/b/ml-stack-security", force=True)
    assert "/b/ml-stack-security" in path.read_text()
    assert stat.S_IMODE(path.stat().st_mode) == 0o700


def test_a_symlink_in_the_way_is_not_followed(tmp_path):
    target = tmp_path / "precious"
    target.write_text("keep")
    link = tmp_path / "Review quarantine.command"
    link.symlink_to(target)
    with pytest.raises(LauncherError):
        install_launcher(tmp_path, system="Darwin", exe="/a/ml-stack-security")
    assert target.read_text() == "keep"


def test_odd_places_and_systems_are_refused_in_words(tmp_path):
    with pytest.raises(LauncherError, match="not a folder"):
        install_launcher(tmp_path / "nope", system="Darwin", exe="/a")
    with pytest.raises(LauncherError, match="macOS and Linux"):
        install_launcher(tmp_path, system="Windows", exe="/a")
    with pytest.raises(LauncherError, match=".desktop"):
        install_launcher(tmp_path, system="Linux", exe='/a/"; touch x; "/y')


def test_the_command_installs_where_told_prints_where_and_refuses_a_second_time(
        tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(launcher, "installed_command", lambda: "/opt/bin/ml-stack-security")
    assert command(["review", "--install-launcher", "--dir", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    written = next(Path(tmp_path).glob("Review quarantine.*"))
    assert str(written) in out
    assert command(["review", "--install-launcher", "--dir", str(tmp_path)]) == 1
    assert "already exists" in capsys.readouterr().err
    assert command(["review", "--install-launcher", "--dir", str(tmp_path), "--force"]) == 0
