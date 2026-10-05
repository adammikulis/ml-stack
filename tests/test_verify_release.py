"""Every name `docs/verify_release.py` imports from ml_stack still exists; a check that
cannot import its subject reports FAIL for a feature that works."""

from __future__ import annotations

import ast
import importlib
from pathlib import Path
from types import CodeType, FunctionType

import pytest

VERIFIER = Path(__file__).resolve().parents[1] / "docs" / "verify_release.py"


def imported() -> list[tuple[int, str, str]]:
    """``(line, module, name)`` for every ml_stack name the verifier imports."""
    tree = ast.parse(VERIFIER.read_text(encoding="utf-8"), filename=str(VERIFIER))
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("ml_stack"):
            out.extend((node.lineno, node.module or "", alias.name) for alias in node.names)
        elif isinstance(node, ast.Import):
            out.extend((node.lineno, alias.name, "") for alias in node.names
                       if alias.name.startswith("ml_stack"))
    return out


def test_the_verifier_imports_something_from_the_library():
    assert len(imported()) > 20, "the verifier stopped reaching into ml_stack"


@pytest.mark.parametrize("line,module,name",
                         imported(), ids=lambda v: str(v) if isinstance(v, str) else None)
def test_a_name_the_verifier_imports_still_exists(line, module, name):
    try:
        held = importlib.import_module(module)
    except ModuleNotFoundError as exc:
        if (exc.name or "").startswith("ml_stack"):
            pytest.fail(f"verify_release.py:{line} imports {module}, which is gone")
        pytest.skip(f"{module} needs {exc.name}, which is not installed here")
    if name:
        assert hasattr(held, name), (
            f"verify_release.py:{line} imports {name} from {module}, which no longer has it")


def test_offline_setup_checks_use_random_keys_and_group_scoped_join_authentication(tmp_path):
    tree = ast.parse(VERIFIER.read_text(encoding='utf-8'))
    ran = []
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef) or not node.decorator_list:
            continue
        decorator = node.decorator_list[0]
        if not isinstance(decorator, ast.Call) or not decorator.args \
                or not isinstance(decorator.args[0], ast.Constant) or decorator.args[0].value != 'Setup':
            continue
        claim = decorator.args[1].value
        if 'two groups' in claim:
            continue  # This check requires live UDP; the offline checks do not.
        node.decorator_list = []
        scope = {'TMP': tmp_path, 'WORDS': 'correct horse battery staple'}
        module = compile(ast.Module(body=[node], type_ignores=[]), str(VERIFIER), 'exec')
        code = next(value for value in module.co_consts if isinstance(value, CodeType))
        assert isinstance(FunctionType(code, scope)(), str)
        ran.append(claim)
    assert len(ran) == 3
