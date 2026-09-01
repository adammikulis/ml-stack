"""The wheel has to contain the files the code reads at runtime."""

from __future__ import annotations

import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent


def build(out: Path) -> Path:
    done = subprocess.run(
        [sys.executable, "-m", "build", "--wheel", "--outdir", str(out), str(REPO)],
        capture_output=True, text=True)
    if done.returncode != 0:
        pytest.skip(f"cannot build wheels here: {done.stderr[-300:]}")
    return sorted(out.glob("*.whl"))[-1]


@pytest.mark.slow
def test_the_web_interface_ships_in_the_fleet_wheel(tmp_path):
    """Served from disk at runtime, so a wheel without them is a daemon whose UI 404s."""
    wheel = build(tmp_path)
    names = zipfile.ZipFile(wheel).namelist()

    for asset in ("index.html", "style.css", "app.js"):
        assert f"ml_stack/fleet/web/{asset}" in names, f"{asset} missing from {wheel.name}"


@pytest.mark.slow
def test_the_contract_data_ships_in_the_contracts_wheel(tmp_path):
    """contracts/ lives at the repo root and is force-included at build time."""
    wheel = build(tmp_path)
    names = zipfile.ZipFile(wheel).namelist()

    assert any(n.endswith("model_tiers.json") for n in names)
    assert any(n.endswith("json.gbnf") for n in names)
    assert any("recipes/" in n and n.endswith(".json") for n in names)


def console_scripts() -> dict:
    import tomllib

    return tomllib.load((REPO / "pyproject.toml").open("rb"))["project"]["scripts"]


def test_serving_a_model_has_a_command_of_its_own():
    """Without it, answering 'what is serving, on which port' is lsof and curl."""
    assert console_scripts().get("ml-stack-serve") == "ml_stack.serve.cli:main"


def test_every_console_script_points_at_something_that_exists():
    scripts = console_scripts()
    assert scripts, "the package installs no commands"
    for name, target in scripts.items():
        module, _, attr = target.partition(":")
        path = REPO / "src" / Path(*module.split(".")).with_suffix(".py")
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
    (tmp_path / "artifacts" / "wheels" / "ml_stack-0-py3-none-any.whl").write_text("x")
    out = tmp_path / "out"
    out.touch()

    done = subprocess.run(
        ["bash", "-e", "-c", downloads_step()], cwd=tmp_path, capture_output=True,
        text=True,
        env={**os.environ, "TAG": "v9.9.9",
             "GITHUB_REPOSITORY": "owner/repo", "GITHUB_OUTPUT": str(out)})
    assert done.returncode == 0, done.stderr

    body = out.read_text()
    assert "releases/download/v9.9.9/ml-stack-macos-arm64-v9.9.9.zip" in body
    assert "ml-stack-windows-x86_64" not in body, "linked a bundle that did not build"
    assert "ml-stack-linux-x86_64" not in body, "linked a bundle that did not build"
    assert "pip install ml-stack" in body


def git(*args, cwd):
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)


@pytest.mark.skipif(sys.platform == "win32", reason="the step is written for bash")
def test_a_tag_with_a_changelog_entry_uses_it(tmp_path):
    """The release page says what CHANGELOG.md says, so editing the file is enough and
    a second run of the job leaves the body the same rather than twice as long."""
    import os

    (tmp_path / "CHANGELOG.md").write_text(
        "# Changelog\n\n## [0.2.0](link) (2026-01-01)\n\n* the new thing\n\n"
        "## [0.1.0](link) (2025-01-01)\n\n* the old thing\n")
    out = tmp_path / "out"
    out.touch()
    done = subprocess.run(
        ["bash", "-e", "-c", workflow_step("what changed")], cwd=tmp_path,
        capture_output=True, text=True,
        env={**os.environ, "TAG": "v0.2.0", "GITHUB_REPOSITORY": "owner/repo",
             "GITHUB_OUTPUT": str(out)})
    assert done.returncode == 0, done.stderr

    body = out.read_text()
    assert "* the new thing" in body
    assert "the old thing" not in body, "carried the previous release's notes too"


@pytest.mark.skipif(sys.platform == "win32", reason="the step is written for bash")
def test_a_tag_with_no_entry_falls_back_to_the_subjects(tmp_path):
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

    commit("Before the tag", "src/ml_stack/fleet/a.py")
    git("tag", "v0.1.0", cwd=repo)
    commit("Chat with a model on any machine", "src/ml_stack/fleet/b.py")
    commit("Write down what is pending", "HANDOFF.md")
    commit("Version 0.2.0", "pyproject.toml")
    commit("Explain the recipes", "docs/FEATURES.md")
    git("tag", "v0.2.0", cwd=repo)

    out = tmp_path / "out"
    out.touch()
    done = subprocess.run(
        ["bash", "-e", "-c", workflow_step("what changed")], cwd=repo,
        capture_output=True, text=True,
        env={**os.environ, "TAG": "v0.2.0",
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


# -- the version -------------------------------------------------------
def test_release_please_is_pointed_at_every_file_that_holds_the_version():
    import json
    import re

    config = json.loads((REPO / "release-please-config.json").read_text())
    listed = {e["path"] for e in config["packages"]["."]["extra-files"]}
    carry = {str(p.relative_to(REPO)) for p in (REPO.glob("pyproject.toml"))}
    carry |= {"packaging/ml-stack-app.spec"}

    assert listed == carry, f"registered: {listed}; carrying: {carry}"
    for path in sorted(carry):
        text = (REPO / path).read_text()
        assert "x-release-please-version" in text, f"{path} has no line to bump"
    del re


def test_the_package_is_on_the_version_in_the_manifest():
    import json
    import re

    want = json.loads((REPO / ".release-please-manifest.json").read_text())["."]
    assert (REPO / "version.txt").read_text().strip() == want
    found = re.search(r'^version = "([^"]+)"', (REPO / "pyproject.toml").read_text(), re.M)
    assert found and found.group(1) == want


def test_the_version_in_a_checkout_is_read_without_the_marker():
    """The marker is a comment on the same line; the reader has to stop at it."""
    import json

    from ml_stack.fleet.updates import _version_in_source

    want = json.loads((REPO / ".release-please-manifest.json").read_text())["."]
    assert _version_in_source() == want


# -- what PyPI shows -----------------------------------------------------
def test_the_package_says_what_it_is_and_where_it_came_from():
    """Published without these it is a blank page on PyPI, and the version it went up
    at cannot be uploaded again to fix it."""
    import tomllib

    meta = tomllib.load((REPO / "pyproject.toml").open("rb"))["project"]
    assert meta["name"] == "ml-stack"
    assert meta["readme"] == "README.md" and (REPO / "README.md").is_file()
    assert meta["urls"]["Homepage"].startswith("https://github.com/")
    assert meta["classifiers"], "no classifiers, so it is filed under nothing"
    assert meta["description"].strip() and meta["license"]


# -- one workflow calling another ----------------------------------------
def workflows() -> dict:
    """Every workflow file, read as text. Parsed by hand: PyYAML is not a test
    dependency and CI installs only what the tests import."""
    return {p.name: p.read_text()
            for p in sorted((REPO / ".github" / "workflows").glob("*.yml"))}


def test_a_called_workflow_declares_the_secrets_it_reads():
    """A reusable workflow that reads a secret it never declared does not start: the
    run fails before any job, with nothing in the log to say which secret."""
    import re

    text = workflows()["release.yml"]
    used = {m for m in re.findall(r"secrets\.([A-Z_][A-Z0-9_]*)", text)
            if m != "GITHUB_TOKEN"}
    called = text[text.index("workflow_call:"):]
    for name in used:
        assert name in called, f"{name} is read but not declared under workflow_call"
    if used:
        assert "secrets: inherit" in workflows()["release-please.yml"], (
            "the called workflow would see no secrets")


def test_the_release_is_built_by_the_workflow_that_cuts_it():
    caller = workflows()["release-please.yml"]
    assert "uses: ./.github/workflows/release.yml" in caller
    assert "id-token" not in caller, "nothing here asks PyPI for anything any more"


def test_nothing_uploads_to_pypi():
    """The release builds the wheel and attaches it. Publishing it is a decision, not
    something a tag does on its own."""
    for name, text in workflows().items():
        assert "pypi" not in text.lower(), f"{name} still reaches for PyPI"
