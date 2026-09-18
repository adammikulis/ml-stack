"""What a full install has, checked against what pyproject.toml offers.

`ml_stack.installed.STANDARD` is what `install.sh` asks pip for and what `ml-stack-setup`
reports one line per. An extra renamed in pyproject.toml and not here would install
nothing and report nothing missing.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

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
