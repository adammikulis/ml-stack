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


def workflow_step(name: str) -> str:
    """The shell of one step of the release workflow."""
    text = WORKFLOW.read_text()
    start = text.index(f"- name: {name}")
    body = text[text.index("run: |", start):]
    lines = body.splitlines()[1:]
    indent = len(lines[0]) - len(lines[0].lstrip())
    out = []
    for line in lines:
        if line.strip() and len(line) - len(line.lstrip()) < indent:
            break
        out.append(line[indent:])
    return "\n".join(out)


def downloads_step() -> str:
    return workflow_step("what to download")


def test_every_bundle_the_release_builds_is_linked_from_its_notes():
    """An asset renamed in the matrix and not in the notes is a dead link."""
    import re

    text = WORKFLOW.read_text()
    built = set(re.findall(r"^\s+asset: (\S+)$", text, re.M))
    assert built, "the release builds no bundles"
    step = downloads_step()
    for asset in built:
        assert asset in step, f"{asset} is built but not offered"


@pytest.mark.skipif(sys.platform == "win32", reason="the step is written for bash")
def test_the_notes_offer_only_what_actually_built(tmp_path):
    """A platform whose runner was unavailable must not leave a link to nothing."""
    import os

    (tmp_path / "artifacts" / "wheels").mkdir(parents=True)
    (tmp_path / "artifacts" / "mac").mkdir()
    (tmp_path / "artifacts" / "mac" / "ml-stack-macos-arm64-v9.9.9.zip").write_text("x")
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
    assert "releases/download/v9.9.9/ml-stack-macos-arm64-v9.9.9.zip" in body
    assert "ml-stack-windows-x86_64" not in body, "linked a bundle that did not build"
    assert "ml-stack-linux-x86_64" not in body, "linked a bundle that did not build"
    assert "the 1 `ml_stack_*.whl`" in body


def git(*args, cwd):
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)


@pytest.mark.skipif(sys.platform == "win32", reason="the step is written for bash")
def test_the_notes_say_what_changed_and_leave_out_what_did_not(tmp_path):
    """A version bump and a paragraph of prose are not news to someone downloading it."""
    import os

    repo = tmp_path / "repo"
    repo.mkdir()
    git("init", "-q", "-b", "main", cwd=repo)
    git("config", "user.email", "t@example.com", cwd=repo)
    git("config", "user.name", "T", cwd=repo)

    def commit(subject, path, body="x"):
        target = repo / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(body)
        git("add", "-A", cwd=repo)
        git("commit", "-qm", subject, cwd=repo)

    commit("Before the tag", "packages/ml-stack-fleet/src/a.py")
    git("tag", "v0.1.0", cwd=repo)
    commit("Chat with a model on any machine", "packages/ml-stack-fleet/src/b.py")
    commit("Write down what is pending", "HANDOFF.md")
    commit("Version 0.2.0", "packages/ml-stack-fleet/pyproject.toml")
    commit("Explain the recipes", "docs/FEATURES.md")
    git("tag", "v0.2.0", cwd=repo)

    out = tmp_path / "out"
    out.touch()
    done = subprocess.run(
        ["bash", "-e", "-c", workflow_step("what changed")], cwd=repo,
        capture_output=True, text=True,
        env={**os.environ, "GITHUB_REF_NAME": "v0.2.0",
             "GITHUB_REPOSITORY": "owner/repo", "GITHUB_OUTPUT": str(out)})
    assert done.returncode == 0, done.stderr

    body = out.read_text()
    assert "- Chat with a model on any machine" in body
    assert "Write down what is pending" not in body
    assert "Version 0.2.0" not in body
    assert "Explain the recipes" not in body
    assert "Before the tag" not in body, "listed a commit from the previous release"
    assert "compare/v0.1.0...v0.2.0" in body


def test_the_name_of_a_download_says_which_release_it_is():
    """Two downloads in one folder are two files, and the updater and both installers
    still find theirs: they look for ml-stack-<os>-<arch> inside the name."""
    import re

    text = WORKFLOW.read_text()
    packed = re.search(r'pack\.py dist/bundle "([^"]+)"', text)
    assert packed, "nothing packs the bundle"
    assert packed.group(1) == "dist/${{ matrix.asset }}-$stamp.zip", packed.group(1)

    for asset in re.findall(r"^\s+asset: (\S+)$", text, re.M):
        assert packed.group(1).startswith("dist/${{ matrix.asset }}-"), (
            f"{asset} would not keep its platform in the name")
