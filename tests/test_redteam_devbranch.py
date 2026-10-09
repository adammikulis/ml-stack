"""Hostile input to ``development_branch``: the environment value and the checkout path."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from ml_stack.devbranch import DEFAULT, development_branch


def _git(cwd: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)


@pytest.fixture(autouse=True)
def _no_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ML_STACK_DEV_BRANCH", raising=False)


def _primary(tmp_path: Path, name: str, branch: str) -> Path:
    root = tmp_path / name
    root.mkdir()
    _git(root, "init", "-q", "-b", branch)
    _git(root, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "--allow-empty", "-m", "x")
    return root


@pytest.mark.parametrize("hostile", ["main", "master", " main ", "\nmain\n"])
def test_the_environment_cannot_name_a_protected_branch(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, hostile: str) -> None:
    monkeypatch.setenv("ML_STACK_DEV_BRANCH", hostile)
    assert development_branch(tmp_path) == DEFAULT


@pytest.mark.parametrize("hostile", ["--upload-pack=touch pwned", "$(touch pwned)", "a;touch pwned", "`touch pwned`"])
def test_an_environment_value_is_returned_as_text_and_never_run(
        monkeypatch: pytest.MonkeyPatch, tmp_path: Path, hostile: str) -> None:
    monkeypatch.setenv("ML_STACK_DEV_BRANCH", hostile)
    assert development_branch(tmp_path) == hostile
    assert not (tmp_path / "pwned").exists()


def test_a_primary_on_main_falls_back_to_the_default(tmp_path: Path) -> None:
    root = _primary(tmp_path, "primary", "main")
    assert development_branch(root) == DEFAULT


def test_a_checkout_path_with_shell_characters_is_one_argument(tmp_path: Path) -> None:
    root = _primary(tmp_path, "a b;touch pwned $(id)", "feature-x")
    assert development_branch(root) == "feature-x"
    assert not (tmp_path / "pwned").exists() and not (root / "pwned").exists()


def test_a_missing_directory_falls_back_to_the_default(tmp_path: Path) -> None:
    assert development_branch(tmp_path / "does not exist") == DEFAULT
