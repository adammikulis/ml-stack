"""The wheel the packaging tests use is rebuilt when its inputs change, whatever their modification times say."""

from __future__ import annotations

import os
from pathlib import Path

from wheel_support import STAMP, fresh_wheel


class Builds:
    """A stand-in for packaging/build.py that writes a wheel and counts the times it was asked to."""

    def __init__(self) -> None:
        self.count = 0

    def __call__(self, repo: Path) -> None:
        self.count += 1
        (repo / "dist").mkdir(exist_ok=True)
        (repo / "dist" / "ml_stack-0.0.1-py3-none-any.whl").write_bytes(b"wheel %d" % self.count)


def tree(tmp_path: Path) -> Path:
    (tmp_path / "src" / "ml_stack").mkdir(parents=True)
    (tmp_path / "src" / "ml_stack" / "a.py").write_text("x = 1\n")
    (tmp_path / "packaging").mkdir()
    (tmp_path / "packaging" / "install.sh").write_text("echo a\n")
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'x'\n")
    return tmp_path


def test_an_unchanged_tree_reuses_its_wheel(tmp_path: Path) -> None:
    repo, build = tree(tmp_path), Builds()
    first = fresh_wheel(repo, build)
    assert fresh_wheel(repo, build) == first and build.count == 1
    assert (repo / "dist" / STAMP).is_file()


def test_a_changed_source_with_an_older_modification_time_still_rebuilds(tmp_path: Path) -> None:
    repo, build = tree(tmp_path), Builds()
    wheel = fresh_wheel(repo, build)
    source = repo / "src" / "ml_stack" / "a.py"
    source.write_text("x = 2\n")
    os.utime(source, (1, 1))
    assert fresh_wheel(repo, build) == wheel and build.count == 2
    assert wheel.read_bytes() == b"wheel 2"


def test_a_changed_installer_or_pyproject_rebuilds(tmp_path: Path) -> None:
    repo, build = tree(tmp_path), Builds()
    fresh_wheel(repo, build)
    (repo / "packaging" / "install.sh").write_text("echo b\n")
    fresh_wheel(repo, build)
    (repo / "pyproject.toml").write_text("[project]\nname = 'y'\n")
    fresh_wheel(repo, build)
    assert build.count == 3


def test_a_wheel_with_no_stamp_is_not_trusted(tmp_path: Path) -> None:
    repo, build = tree(tmp_path), Builds()
    (repo / "dist").mkdir()
    (repo / "dist" / "ml_stack-0.0.1-py3-none-any.whl").write_bytes(b"left over from some other tree")
    assert fresh_wheel(repo, build).read_bytes() == b"wheel 1"
