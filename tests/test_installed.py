"""What a full install has, checked against what pyproject.toml offers.

`ml_stack.installed.STANDARD` is what `install.sh` asks pip for and what `ml-stack-setup`
reports one line per. An extra renamed in pyproject.toml and not here would install
nothing and report nothing missing.
"""

from __future__ import annotations

import tomllib
from importlib.metadata import version
from pathlib import Path

import pytest

from ml_stack import installed
from ml_stack.installed import STANDARD, Capability, extras, missing

PYPROJECT = Path(__file__).resolve().parent.parent / "pyproject.toml"


def offered() -> dict[str, list[str]]:
    raw = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    return raw["project"]["optional-dependencies"]


def test_every_extra_a_full_install_asks_for_is_one_pip_can_install():
    have = offered()
    for one in STANDARD:
        assert one.extra in have, f"no [{one.extra}] extra in pyproject.toml"


def test_the_module_each_extra_is_read_by_is_one_that_extra_installs():
    """The import stands in for the extra. A module no requirement provides reads as
    missing on a machine that has everything."""
    have = offered()
    for one in STANDARD:
        named = {req.split(">")[0].split("<")[0].split("[")[0].strip().replace("-", "_")
                 for req in have[one.extra]}
        assert one.module in named, f"[{one.extra}] does not install {one.module}"


def test_a_module_that_is_here_reads_as_present():
    assert Capability("core", "the standard library", "reads JSON", "json").present()


def test_a_module_that_is_not_here_reads_as_absent():
    assert not Capability("none", "nothing", "does nothing", "no_such_module").present()


def test_the_extras_are_named_the_way_pip_takes_them():
    assert extras() == ",".join(one.extra for one in STANDARD)
    assert " " not in extras()


def test_what_is_missing_is_a_subset_of_what_a_full_install_has():
    assert set(missing()) <= set(STANDARD)


def test_every_installed_extra_holds_the_version_its_pin_asks_for():
    """One red for a stale environment, rather than a module failing to import somewhere.

    mlx-lm sat four minor versions under its pin for a fortnight: the two spec modules died
    at collection and their ten tests stopped running with nothing said.
    """
    behind = installed.unmet(offered())
    assert not behind, "installed below the pin:\n  " + "\n  ".join(
        f"{name} {have}, {spec} asked for (pip install -e '.[{extra}]')"
        for extra, name, have, spec in behind)


def test_a_package_nobody_installed_is_not_a_finding():
    assert installed.unmet({"absent": ["ml-stack-no-such-package>=9"]}) == []


def test_a_package_at_its_floor_is_not_a_finding():
    have = version("pytest")
    assert installed.unmet({"test": [f"pytest>={have}"]}) == []


def test_a_package_under_its_floor_is_a_finding():
    assert [(e, n) for e, n, _, _ in installed.unmet({"test": ["pytest>=999.0"]})] == [
        ("test", "pytest")]


def test_a_requirement_naming_another_extra_is_not_looked_up():
    assert installed.unmet({"one": ["ml-stack[two]"], "two": ["nope>=9"]}) == []


def test_the_pins_are_read_from_the_installed_distributions_own_metadata():
    """A wheel carries no pyproject.toml, so a real machine reads them from here."""
    assert set(installed.declared()) >= {c.extra for c in installed.STANDARD}


def test_the_store_says_which_macos_it_needs_when_this_one_is_older(monkeypatch):
    monkeypatch.setattr(installed.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(installed.platform, "mac_ver", lambda: ("14.7.1", ("", "", ""), "arm64"))
    said = installed.store_wants_newer_macos()
    assert "macOS 15 or newer" in said and "14.7.1" in said


def test_a_new_enough_macos_is_told_nothing():
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(installed.platform, "mac_ver", lambda: ("15.0", ("", "", ""), "arm64"))
    monkeypatch.setattr(installed.platform, "system", lambda: "Darwin")
    assert installed.store_wants_newer_macos() == ""
    monkeypatch.undo()


def test_a_machine_that_is_not_a_mac_is_told_nothing(monkeypatch):
    monkeypatch.setattr(installed.platform, "system", lambda: "Linux")
    assert installed.store_wants_newer_macos() == ""
