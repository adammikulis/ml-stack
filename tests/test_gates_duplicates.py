"""The gate that finds functions written twice."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent


def _load():
    spec = importlib.util.spec_from_file_location(
        "_gates_duplicates_under_test", REPO / "scripts" / "gates" / "duplicates.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


duplicates = _load()


def tree(tmp_path: Path, **modules: str) -> Path:
    where = tmp_path / "src" / "ml_stack"
    where.mkdir(parents=True)
    for name, text in modules.items():
        (where / f"{name}.py").write_text(text, encoding="utf-8")
    return tmp_path


COUNTER = '''
def tally(rows):
    """How often each kind appears."""
    counts = {}
    for row in rows:
        kind = str(row.get("kind") or "")
        if not kind:
            continue
        counts[kind] = counts.get(kind, 0) + 1
    return sorted(counts.items())
'''

RENAMED = '''
def count_kinds(entries):
    """The same routine under another vocabulary."""
    held = {}
    for one in entries:
        label = str(one.get("kind") or "")
        if not label:
            continue
        held[label] = held.get(label, 0) + 1
    return sorted(held.items())
'''


def test_the_same_routine_under_another_vocabulary_is_found(tmp_path):
    root = tree(tmp_path, one=COUNTER, two=RENAMED)
    found = duplicates.find(root)
    assert len(found) == 2
    assert {one.path for one in found} == {"src/ml_stack/one.py", "src/ml_stack/two.py"}
    assert "count_kinds" in found[0].detail and "src/ml_stack/two.py" in found[0].detail


def test_a_shared_shape_over_different_apis_is_not_a_duplicate(tmp_path):
    root = tree(tmp_path, one='''
import json

def read_one(path):
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    parsed = json.loads(text)
    if not isinstance(parsed, dict):
        return None
    return parsed
''', two='''
import tomllib

def read_two(path):
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    parsed = tomllib.loads(text)
    if not isinstance(parsed, dict):
        return None
    return parsed
''')
    assert duplicates.find(root) == []


SMALL = '''
def {name}({arg}):
    {arg} = str({arg}).strip()
    if not {arg}:
        return ""
    return {arg}.casefold()
'''


def test_a_function_under_the_floor_is_left_alone(tmp_path):
    root = tree(tmp_path, one=SMALL.format(name="tidy", arg="word"),
                two=SMALL.format(name="clean", arg="text"))
    assert duplicates.find(root, floor=9) == []
    assert len(duplicates.find(root, floor=3)) == 2


def test_dataclass_bodies_and_dunder_init_are_left_alone(tmp_path):
    root = tree(tmp_path, one='''
class One:
    def __init__(self, a, b, c, d):
        self.a = a
        self.b = b
        self.c = c
        self.d = d
''', two='''
class Two:
    def __init__(self, a, b, c, d):
        self.a = a
        self.b = b
        self.c = c
        self.d = d
''')
    assert duplicates.find(root, floor=1) == []


def test_test_modules_are_not_read(tmp_path):
    root = tree(tmp_path, one=COUNTER)
    (root / "src" / "ml_stack" / "test_two.py").write_text(RENAMED, encoding="utf-8")
    assert duplicates.find(root) == []


def test_the_checker_carries_the_gate_interface():
    assert duplicates.NAME == "duplicate-bodies"
    assert isinstance(duplicates.OWNER, str)
    assert duplicates.describe().strip()


@pytest.mark.parametrize("name, one, two", [
    ("plurals", "src/ml_stack/graph/tidy.py", "src/ml_stack/ingest/fold.py"),
    ("_read", "src/ml_stack/serve/fit.py", "src/ml_stack/serve/profile.py"),
    ("writable_file", "src/ml_stack/serve/fit.py", "src/ml_stack/serve/profile.py"),
    ("local_file", "src/ml_stack/serve/fit.py", "src/ml_stack/serve/profile.py"),
    ("_human", "src/ml_stack/hub.py", "src/ml_stack/serve/layout.py"),
    ("_human", "src/ml_stack/serve/fit.py", "src/ml_stack/serve/preflight.py"),
])
def test_the_copies_this_gate_was_written_for_are_reported(name, one, two):
    """The known pairs, so a normalisation that quietly tightens is caught."""
    for where in (one, two):
        if f"def {name}(" not in (REPO / where).read_text(encoding="utf-8"):
            pytest.skip(f"{where} no longer defines {name}")
    together = [members for members in duplicates.index(REPO).values()
                if {(entry.path, entry.name) for entry in members} >= {(one, name), (two, name)}]
    assert together, f"{name} in {one} and {two} should hash the same"
