"""The wheel has to contain the files the code reads at runtime."""

from __future__ import annotations

import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent


def build(package: str, out: Path) -> Path:
    done = subprocess.run(
        [sys.executable, "-m", "build", "--wheel", "--outdir", str(out),
         str(REPO / "packages" / package)],
        capture_output=True, text=True)
    if done.returncode != 0:
        pytest.skip(f"cannot build wheels here: {done.stderr[-300:]}")
    return sorted(out.glob("*.whl"))[-1]


@pytest.mark.slow
def test_the_web_interface_ships_in_the_fleet_wheel(tmp_path):
    """Served from disk at runtime, so a wheel without them is a daemon whose UI 404s."""
    wheel = build("ml-stack-fleet", tmp_path)
    names = zipfile.ZipFile(wheel).namelist()

    for asset in ("index.html", "style.css", "app.js"):
        assert f"ml_stack/fleet/web/{asset}" in names, f"{asset} missing from {wheel.name}"


@pytest.mark.slow
def test_the_contract_data_ships_in_the_contracts_wheel(tmp_path):
    """contracts/ lives at the repo root and is force-included at build time."""
    wheel = build("ml-stack-contracts", tmp_path)
    names = zipfile.ZipFile(wheel).namelist()

    assert any(n.endswith("model_tiers.json") for n in names)
    assert any(n.endswith("json.gbnf") for n in names)
    assert any("recipes/" in n and n.endswith(".json") for n in names)


def test_every_package_declares_a_console_script_that_exists():
    import tomllib

    for pyproject in sorted((REPO / "packages").glob("*/pyproject.toml")):
        data = tomllib.loads(pyproject.read_text())
        for name, target in data.get("project", {}).get("scripts", {}).items():
            module, _, attr = target.partition(":")
            path = pyproject.parent / "src" / Path(*module.split(".")).with_suffix(".py")
            assert path.exists(), f"{name} points at {module}, which is not a file"
            assert f"def {attr}(" in path.read_text(), f"{module} has no {attr}()"
