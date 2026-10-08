"""Source-bound zero-resource admission refuses missing authority."""
import sys

import pytest
from test_kernel_isolation import zero_resource_plan


def source(tmp_path, body):
    tests = tmp_path / "tests"
    tests.mkdir()
    path = tests / "test_example.py"
    path.write_text(body)
    return path


def test_source_plan_admits_builtin_fixture_and_detects_changed_source(tmp_path):
    path = source(tmp_path, "def test_example(tmp_path):\n    pass\n")
    plan = zero_resource_plan([sys.executable, "-m", "pytest", "tests/test_example.py"], tmp_path, [path])
    assert not plan.resources
    assert plan.proof()["sources"]
    path.write_text("def test_example(tmp_path):\n    raise AssertionError\n")
    with pytest.raises(RuntimeError, match="source identity changed"):
        plan.recheck()


@pytest.mark.parametrize("body,match", [
    ("def test_example(unknown):\n    pass\n", "unresolved fixture"),
    ("from test_fixture_plan import resources\n@resources('browser')\ndef test_example():\n    pass\n", "owned fixture allocation"),
])
def test_source_plan_refuses_unresolved_or_unallocated_fixtures(tmp_path, body, match):
    path = source(tmp_path, body)
    with pytest.raises(RuntimeError, match=match):
        zero_resource_plan([sys.executable, "-m", "pytest", "tests/test_example.py"], tmp_path, [path])
