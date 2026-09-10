"""What local-imports counts: a deferred dependency, not a line."""

from __future__ import annotations

from pathlib import Path

from gates import local_imports


def _count(tmp_path: Path, source: str) -> int:
    where = tmp_path / "src" / "ml_stack" / "one.py"
    where.parent.mkdir(parents=True)
    where.write_text(source, encoding="utf-8")
    return len(local_imports.find(tmp_path))


def test_one_module_named_twice_in_a_function_is_one_deferred_dependency(tmp_path) -> None:
    """Splitting an import in two is a formatting change; the function still defers the
    same module, so the number a budget watches must not move."""
    together = _count(tmp_path / "a", "def f():\n    from ml_stack.graph.x import A, b as c\n")
    apart = _count(tmp_path / "b",
                   "def f():\n    from ml_stack.graph.x import A\n"
                   "    from ml_stack.graph.x import b as c\n")
    assert together == apart == 1


def test_two_modules_in_one_function_are_two(tmp_path) -> None:
    assert _count(tmp_path, "def f():\n    from ml_stack.a import A\n    from ml_stack.b import B\n") == 2


def test_the_same_module_deferred_by_two_functions_is_two(tmp_path) -> None:
    assert _count(tmp_path, "def f():\n    from ml_stack.a import A\n\n"
                            "def g():\n    from ml_stack.a import A\n") == 2
