"""Account-root isolation, private snapshot cleanup and fixed Git launch inputs."""

import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from poolhouse import home
from poolhouse.activity import source_snapshot as source

pytestmark = pytest.mark.redteam


@pytest.fixture(autouse=True)
def account_lookup(tmp_path, monkeypatch):
    if os.name == "posix":
        monkeypatch.setattr(home.pwd, "getpwuid", lambda uid: SimpleNamespace(pw_dir=str(tmp_path / "account")))


def test_spoofed_home_preserves_actual_account_roots_and_explicit_overrides(tmp_path, monkeypatch):
    if os.name != "posix":
        pytest.skip("POSIX account identity")
    account = tmp_path / "account"
    actual = {account / home.DEFAULT_NAME, account / ".cache" / "poolhouse"}
    monkeypatch.setenv("HOME", str(tmp_path / "spoofed"))
    monkeypatch.setattr(home, "user_home", lambda: tmp_path / "spoofed")
    override = {"POOLHOUSE_HOME": str(tmp_path / "state"), "POOLHOUSE_CACHE": str(tmp_path / "cache")}
    assert set(source.protected_roots(override)) == actual | {Path(value) for value in override.values()}


@pytest.mark.parametrize("failure", [RuntimeError, KeyboardInterrupt])
def test_partial_snapshot_construction_removes_its_private_metadata(tmp_path, monkeypatch, failure):
    storage = tmp_path / "storage"
    storage.mkdir(mode=0o700)
    root = tmp_path / "source"
    root.mkdir()

    def prepare(snapshot):
        (snapshot.directory / "partial").write_text("incomplete")
        raise failure("fixture preparation failed")

    monkeypatch.setattr(source.SourceSnapshot, "prepare", prepare)
    with pytest.raises(failure, match="fixture preparation failed"):
        source.SourceSnapshot(root, storage)
    assert list(storage.iterdir()) == []


def test_git_inventory_launch_excludes_inherited_configuration_and_input(tmp_path, monkeypatch):
    monkeypatch.setenv("GIT_CONFIG_COUNT", "1")
    monkeypatch.setenv("GIT_CONFIG_VALUE_0", "/foreign/helper")
    monkeypatch.setenv("GIT_TRACE2_EVENT", str(tmp_path / "trace"))
    seen = []

    def launch(command, **options):
        seen.append((command, options))
        raise RuntimeError("fixture launch stopped")

    monkeypatch.setattr(source, "actual_git", lambda: tmp_path / "verified-git")
    monkeypatch.setattr(source, "start_process", launch)
    with pytest.raises(RuntimeError, match="fixture launch stopped"):
        source.git_output(["ls-files", "-z"], cwd=tmp_path)
    command, options = seen[0]
    assert command[0] == str(tmp_path / "verified-git")
    assert command[-2:] == ["ls-files", "-z"]
    assert "core.fsmonitor=false" in command
    assert "core.hooksPath=" + os.devnull in command
    assert options["cwd"] == tmp_path and options["close_fds"] is True
    assert options["stdin"] == source.subprocess.DEVNULL
    assert not ({"GIT_CONFIG_COUNT", "GIT_CONFIG_VALUE_0", "GIT_TRACE2_EVENT"} & options["env"].keys())
    assert options["env"]["GIT_CONFIG_GLOBAL"] == os.devnull
    assert options["env"]["GIT_CONFIG_SYSTEM"] == os.devnull


@pytest.fixture
def git_asset(tmp_path, monkeypatch):
    if os.name != "posix":
        pytest.skip("POSIX owned executable contract")
    root = tmp_path / "homebrew"
    image = root / "Cellar/git/1/bin/git"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"\xcf\xfa\xed\xfefixture")
    image.chmod(0o500)
    (root / "opt").mkdir()
    (root / "opt/git").symlink_to(image.parent.parent, target_is_directory=True)
    monkeypatch.setattr(source, "GIT_ROOTS", (root,))
    monkeypatch.setattr(source.sys, "platform", "darwin")
    monkeypatch.setattr(source.grp, "getgrnam", lambda name: SimpleNamespace(gr_gid=os.getgid()))
    return root, image


def test_actual_git_uses_owned_fixed_asset_without_inherited_path(git_asset, tmp_path, monkeypatch):
    root, image = git_asset
    monkeypatch.setenv("PATH", str(tmp_path / "hostile"))
    assert source.actual_git() == image
    (root / "opt/git").unlink()
    with pytest.raises(RuntimeError, match="Apple shim refused"):
        source.actual_git()


@pytest.mark.parametrize("attack", ["redirect", "writable", "hardlink", "not-image", "fifo", "changed", "intermediate"])
def test_actual_git_refuses_hostile_asset(git_asset, tmp_path, monkeypatch, attack):
    root, image = git_asset
    if attack == "redirect":
        (root / "opt/git").unlink()
        (root / "opt/git").symlink_to(tmp_path, target_is_directory=True)
        (tmp_path / "bin").mkdir()
        (tmp_path / "bin/git").write_bytes(b"foreign")
    elif attack == "intermediate":
        redirect = tmp_path / "redirect"
        redirect.symlink_to(image.parent.parent, target_is_directory=True)
        (root / "opt/git").unlink()
        (root / "opt/git").symlink_to(redirect, target_is_directory=True)
    elif attack == "writable":
        image.chmod(0o777)
    elif attack == "hardlink":
        os.link(image, tmp_path / "linked-image")
    elif attack == "not-image":
        image.chmod(0o700)
        image.write_bytes(b"#!/bin/sh")
    elif attack == "fifo":
        image.unlink()
        os.mkfifo(image)
    else:
        original = source.os.fstat
        calls = []

        def changed(descriptor):
            calls.append(descriptor)
            if len(calls) == 2:
                image.chmod(0o644)
            return original(descriptor)

        monkeypatch.setattr(source.os, "fstat", changed)
    with pytest.raises(RuntimeError, match="Git"):
        source.actual_git()
