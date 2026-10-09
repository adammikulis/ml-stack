"""Persisted interface preferences and validation."""

import pytest

from ml_stack.fleet.settings import Settings
from ml_stack.fleet.ui import UI


def test_advanced_options_preference_survives_restart(tmp_path):
    path = tmp_path / "settings.json"
    settings = Settings()
    ui = UI(name="demo-device")
    ui.settings = settings
    ui.settings_path = path
    assert settings.always_show_advanced is False
    assert "error" not in ui.apply_prefs({"always_show_advanced": True})
    assert Settings.load(path).always_show_advanced is True
    assert "error" not in ui.apply_prefs({"always_show_advanced": False})
    assert Settings.load(path).always_show_advanced is False


@pytest.mark.parametrize("value", ["false", 1, 0, None, [], {}])
def test_invalid_interface_preference_does_not_change_other_settings(tmp_path, value):
    path = tmp_path / "settings.json"
    settings = Settings()
    ui = UI(name="demo-device")
    ui.settings = settings
    ui.settings_path = path
    result = ui.apply_prefs({"always_show_advanced": value, "context": 16384})
    assert "boolean" in result["error"]
    assert settings.context == 8192
    assert settings.always_show_advanced is False
    assert not path.exists()


@pytest.mark.parametrize("limit", [256, None])
def test_default_output_limit_is_persisted_independently(tmp_path, limit):
    path = tmp_path / "settings.json"
    ui = UI(name="demo-device")
    ui.settings = Settings(context=32768)
    ui.settings_path = path
    assert "error" not in ui.apply_prefs({"chat_max_output_tokens": limit})
    saved = Settings.load(path)
    assert saved.chat_max_output_tokens == limit
    assert saved.context == 32768


@pytest.mark.parametrize("limit", [True, False, 0, -1, 1.5, "2048", [], {}])
def test_invalid_output_default_does_not_apply_other_preferences(tmp_path, limit):
    ui = UI(name="demo-device")
    ui.settings = Settings()
    ui.settings_path = tmp_path / "settings.json"
    result = ui.apply_prefs({"chat_max_output_tokens": limit, "always_show_advanced": True})
    assert "positive integer" in result["error"]
    assert ui.settings.always_show_advanced is False
    assert ui.settings.chat_max_output_tokens == 8192
