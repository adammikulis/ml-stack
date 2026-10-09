"""Validated interface themes and preset registry."""

from __future__ import annotations

import re
from copy import deepcopy
from typing import Any

BRAND = {"pink": "#ff5fa2", "orange": "#ff8a3d", "yellow": "#ffd166",
         "cyan": "#2de2e6", "navy": "#1b1f3a"}
_NEUTRALS = {
    "dark": {"background": "#141518", "panel": "#202126", "sunken": "#101114",
             "text": "#f4f4f5", "muted": "#b2b3bb", "border": "#41424b"},
    "light": {"background": "#f3f4f6", "panel": "#ffffff", "sunken": "#e8e9ed",
              "text": "#202126", "muted": "#62636d", "border": "#c9cbd2"},
}
PRESETS = {base: {"id": base, "name": f"Poolhouse {base.title()}", "base": base,
                  "colors": {**BRAND, **colors}, "font_size": 14, "density": "comfortable"}
           for base, colors in _NEUTRALS.items()}
_HEX = re.compile(r"#[0-9a-fA-F]{6}\Z")
_ID = re.compile(r"[a-z][a-z0-9-]{0,31}\Z")


def default_appearance() -> dict[str, Any]:
    return {"active_theme": "dark", "saved_themes": []}


def registry() -> list[dict[str, Any]]:
    return deepcopy(list(PRESETS.values()))


def validate_theme(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict) or set(raw) != {"id", "name", "base", "colors", "font_size", "density"}:
        raise ValueError("A theme needs an ID, name, base, colors, font size, and density.")
    ident, name, base = raw["id"], raw["name"], raw["base"]
    if not isinstance(ident, str) or not _ID.fullmatch(ident) or ident in PRESETS:
        raise ValueError("Custom theme IDs use 1-32 lowercase letters, digits, or hyphens.")
    if not isinstance(name, str) or not name.strip() or len(name) > 48 or any(ord(c) < 32 for c in name):
        raise ValueError("Theme names need 1-48 visible characters.")
    if not isinstance(base, str) or base not in PRESETS:
        raise ValueError("Choose a light or dark base.")
    colors = raw["colors"]
    if not isinstance(colors, dict) or set(colors) != set(PRESETS[base]["colors"]):
        raise ValueError("Provide every brand and interface color.")
    if any(not isinstance(v, str) or not _HEX.fullmatch(v) for v in colors.values()):
        raise ValueError("Colors must be six-digit hexadecimal values, such as #ff5fa2.")
    size = raw["font_size"]
    if type(size) is not int or not 12 <= size <= 20:
        raise ValueError("Font size must be an integer from 12 to 20.")
    if raw["density"] not in ("compact", "comfortable"):
        raise ValueError("Choose compact or comfortable density.")
    return {**raw, "name": name.strip(), "colors": {k: v.lower() for k, v in colors.items()}}


def validate_appearance(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict) or set(raw) != {"active_theme", "saved_themes"}:
        raise ValueError("Appearance needs an active theme and saved themes.")
    rows = raw["saved_themes"]
    if not isinstance(rows, list) or len(rows) > 16:
        raise ValueError("Save up to 16 custom themes.")
    themes = [validate_theme(row) for row in rows]
    ids = [row["id"] for row in themes]
    if len(set(ids)) != len(ids):
        raise ValueError("Each saved theme needs a unique ID.")
    active = raw["active_theme"]
    if not isinstance(active, str) or active not in {*PRESETS, *ids}:
        raise ValueError("Choose an existing theme.")
    return {"active_theme": active, "saved_themes": themes}


def resolve(appearance: Any) -> dict[str, Any]:
    valid = validate_appearance(appearance)
    active = valid["active_theme"]
    return deepcopy(PRESETS[active] if active in PRESETS else
                    next(row for row in valid["saved_themes"] if row["id"] == active))
