"""A `--system` install with no unlocked keystore warns and still installs: the sealed stores fail
closed at boot, so the machine is safe, only less useful. Nothing here needs root or a keystore."""

from __future__ import annotations

import json

import pytest

from ml_stack.fleet import autostart, autostart_keystore


def provision(home_dir, *, at=1.0):
    state = home_dir / ".ml-stack" / "keystore"
    state.mkdir(parents=True)
    (state / "provisioned.json").write_text(json.dumps({"at": at}))


def test_a_user_without_a_master_is_told_to_unlock_and_what_stays_locked(tmp_path):
    said = autostart_keystore.notice("wren", tmp_path, "linux")
    assert "ml-stack-security unlock" in said and "wren" in said
    assert "fails closed" in said and "signing key" in said


def test_a_user_with_a_master_on_linux_hears_nothing(tmp_path):
    provision(tmp_path)
    assert autostart_keystore.notice("wren", tmp_path, "linux") == ""


def test_on_a_mac_even_a_provisioned_user_is_told_the_login_keychain_is_locked_at_boot(tmp_path):
    provision(tmp_path)
    said = autostart_keystore.notice("wren", tmp_path, "darwin")
    assert "login keychain is locked" in said and "first login" in said


@pytest.mark.parametrize("platform", ["linux", "darwin"])
def test_the_install_warns_and_still_installs(tmp_path, monkeypatch, platform):
    target = tmp_path / "unit"
    made = autostart.SystemService(platform, "x", str(target), "[Unit]\n", "install", "wren", {})
    warned: list[str] = []
    monkeypatch.setattr(autostart, "system_service", lambda *_a, **_k: made)
    monkeypatch.setattr(autostart, "_run", lambda _argv: 0)
    monkeypatch.setattr(autostart, "warn", warned.append)
    monkeypatch.setattr(autostart, "say", lambda _m: None)
    assert autostart._install_system("wren", str(tmp_path / "nobody-home")) == 0
    assert target.read_text() == "[Unit]\n"
    assert any("ml-stack-security unlock" in line for line in warned)
