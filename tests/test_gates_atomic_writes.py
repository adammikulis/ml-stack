"""What atomic-writes counts: the syscall, whatever it is spelled as."""

from __future__ import annotations

from pathlib import Path

import pytest
from gates import atomic_writes


def _count(tmp_path: Path, source: str) -> list[str]:
    where = tmp_path / "src" / "ml_stack" / "one.py"
    where.parent.mkdir(parents=True)
    where.write_text(source, encoding="utf-8")
    return [f.detail for f in atomic_writes.find(tmp_path)]


@pytest.mark.parametrize("line", ["os.replace(a, b)", "os.rename(a, b)", "shutil.move(a, b)"])
def test_the_module_level_spellings_are_found(tmp_path, line: str) -> None:
    assert _count(tmp_path, f"def f():\n    {line}\n")


@pytest.mark.parametrize("line", ["staged.replace(target)", "staged.rename(target)"])
def test_a_path_method_is_the_same_syscall_and_is_found(tmp_path, line: str) -> None:
    """`Path.replace` and `Path.rename` do what `os.replace` does; a gate that reads only
    the module spelling is one a rename walks straight through."""
    assert _count(tmp_path, f"def f():\n    {line}\n")


def test_a_strings_replace_is_left_alone(tmp_path) -> None:
    """`str.replace` takes two arguments and a path's takes one, which tells them apart
    without knowing what the receiver is."""
    assert not _count(tmp_path, 'def f():\n    return name.replace("-", "_")\n')


def test_a_callback_on_self_is_left_alone(tmp_path) -> None:
    """`self.rename(called)` is a handler this repository passes around, not a path."""
    assert not _count(tmp_path, "class A:\n    def f(self):\n        self.rename(called)\n")


def test_the_file_that_owns_the_job_is_exempt(tmp_path) -> None:
    where = tmp_path / "src" / "ml_stack" / "files.py"
    where.parent.mkdir(parents=True)
    where.write_text("def promote(a, b):\n    a.replace(b)\n", encoding="utf-8")
    assert not atomic_writes.find(tmp_path)
