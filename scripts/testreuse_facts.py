"""Facts the reuse key compares against the machine: which pinned distributions are installed now."""

from __future__ import annotations

from importlib import metadata


def installed_pins(pins: list[str]) -> list[str]:
    """The ``name==version`` each pinned name resolves to now; a name that is not installed reads ``name==missing``.

    A name that is installed twice (two site-packages on the path) resolves to the copy an import
    would find first, which is the one ``metadata.version`` returns and the one the pin recorded.
    """
    found = []
    for name in sorted({pin.split("==")[0] for pin in pins}):
        try:
            found.append(f"{name}=={metadata.version(name)}")
        except metadata.PackageNotFoundError:
            found.append(f"{name}==missing")
    return found


def changed_pins(pins: list[str]) -> str:
    """Empty when every pin still matches the installed version, else the first that does not, in words."""
    now = dict(pin.split("==", 1) for pin in installed_pins(pins))
    for pin in sorted(pins):
        name, _, version = pin.partition("==")
        if now.get(name) != version:
            return f"installed distribution changed: {name} {version} -> {now.get(name)}"
    return ""
