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


# -- the release page ----------------------------------------------------
WORKFLOW = REPO / ".github" / "workflows" / "release.yml"


def downloads_step() -> str:
    """The shell of the step that writes the download links into the release body."""
    text = WORKFLOW.read_text()
    start = text.index("- name: what to download")
    body = text[text.index("run: |", start):]
    lines = body.splitlines()[1:]
    indent = len(lines[0]) - len(lines[0].lstrip())
    out = []
    for line in lines:
        if line.strip() and len(line) - len(line.lstrip()) < indent:
            break
        out.append(line[indent:])
    return "\n".join(out)


def test_every_bundle_the_release_builds_is_linked_from_its_notes():
    """An asset renamed in the matrix and not in the notes is a dead link."""
    import re

    text = WORKFLOW.read_text()
    built = set(re.findall(r"^\s+asset: (\S+)$", text, re.M))
    assert built, "the release builds no bundles"
    step = downloads_step()
    for asset in built:
        assert f"{asset}.zip" in step, f"{asset}.zip is built but not offered"


@pytest.mark.skipif(sys.platform == "win32", reason="the step is written for bash")
def test_the_notes_offer_only_what_actually_built(tmp_path):
    """A platform whose runner was unavailable must not leave a link to nothing."""
    import os

    (tmp_path / "artifacts" / "wheels").mkdir(parents=True)
    (tmp_path / "artifacts" / "mac").mkdir()
    (tmp_path / "artifacts" / "mac" / "ml-stack-macos-arm64.zip").write_text("x")
    (tmp_path / "artifacts" / "install.sh").write_text("x")
    (tmp_path / "artifacts" / "install.ps1").write_text("x")
    (tmp_path / "artifacts" / "wheels" / "ml_stack_fleet-0-py3-none-any.whl").write_text("x")
    out = tmp_path / "out"
    out.touch()

    done = subprocess.run(
        ["bash", "-e", "-c", downloads_step()], cwd=tmp_path, capture_output=True,
        text=True,
        env={**os.environ, "GITHUB_REF_NAME": "v9.9.9",
             "GITHUB_REPOSITORY": "owner/repo", "GITHUB_OUTPUT": str(out)})
    assert done.returncode == 0, done.stderr

    body = out.read_text()
    assert "releases/download/v9.9.9/ml-stack-macos-arm64.zip" in body
    assert "ml-stack-windows-x86_64" not in body, "linked a bundle that did not build"
    assert "ml-stack-linux-x86_64" not in body, "linked a bundle that did not build"
    assert "the 1 `ml_stack_*.whl`" in body
