"""Appearance persistence, complete presets, and atomic validation."""

from copy import deepcopy

import pytest

from ml_stack.fleet.settings import Settings
from ml_stack.fleet.themes import BRAND, default_appearance, registry, resolve, validate_appearance
from ml_stack.fleet.ui import UI


def custom():
    theme = registry()[0]
    theme.update(id="custom-1", name="Quiet water", font_size=16, density="compact")
    theme["colors"]["pink"] = "#AA3377"
    return {"active_theme": "custom-1", "saved_themes": [theme]}


def test_both_presets_include_owner_palette_and_distinct_neutrals():
    themes = registry()
    assert {row["base"] for row in themes} == {"light", "dark"}
    for row in themes:
        assert {key: row["colors"][key] for key in BRAND} == BRAND
    assert themes[0]["colors"]["panel"] != themes[1]["colors"]["panel"]
    themes[0]["colors"]["pink"] = "#000000"
    assert registry()[0]["colors"]["pink"] == "#ff5fa2"


def test_custom_theme_persists_without_changing_advanced_preference(tmp_path):
    ui = UI(name="theme-device")
    ui.settings = Settings(always_show_advanced=True)
    ui.settings_path = tmp_path / "settings.json"
    assert "error" not in ui.apply_prefs({"appearance": custom()})
    loaded = Settings.load(ui.settings_path)
    assert loaded.always_show_advanced is True
    assert resolve(loaded.appearance)["colors"]["pink"] == "#aa3377"
    assert resolve(loaded.appearance)["font_size"] == 16
    assert resolve(loaded.appearance)["density"] == "compact"
    assert "error" not in ui.apply_prefs({"appearance": default_appearance()})
    assert resolve(Settings.load(ui.settings_path).appearance)["id"] == "dark"


@pytest.mark.parametrize("change", [
    lambda a: a.update(active_theme="missing"),
    lambda a: a.update(saved_themes="bad"),
    lambda a: a.update(extra=True),
    lambda a: a["saved_themes"][0].update(id="dark"),
    lambda a: a["saved_themes"][0].update(id="<script>"),
    lambda a: a["saved_themes"][0].update(name="\n"),
    lambda a: a["saved_themes"][0].update(name="a" * 49),
    lambda a: a["saved_themes"][0].update(base=[]),
    lambda a: a["saved_themes"][0].update(font_size=True),
    lambda a: a["saved_themes"][0].update(font_size=21),
    lambda a: a["saved_themes"][0].update(density="dense"),
    lambda a: a["saved_themes"][0]["colors"].update(pink="url(https://example.invalid)"),
    lambda a: a["saved_themes"][0]["colors"].update(panel="red;display:none"),
    lambda a: a["saved_themes"][0]["colors"].update(unknown="#ffffff"),
    lambda a: a["saved_themes"][0]["colors"].pop("text"),
    lambda a: a["saved_themes"].append(deepcopy(a["saved_themes"][0])),
])
def test_invalid_theme_rejects_entire_preference_request(tmp_path, change):
    ui = UI(name="theme-device")
    ui.settings = Settings()
    ui.settings_path = tmp_path / "settings.json"
    before = ui.settings.public()
    appearance = custom()
    change(appearance)
    result = ui.apply_prefs({"appearance": appearance, "always_show_advanced": True, "context": 16384})
    assert result["error"]
    assert ui.settings.public() == before
    assert not ui.settings_path.exists()


def test_custom_theme_count_is_bounded():
    appearance = custom()
    appearance["saved_themes"] *= 17
    with pytest.raises(ValueError, match="16"):
        validate_appearance(appearance)
