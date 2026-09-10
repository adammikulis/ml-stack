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


SPAN_ESTIMATE = '''
def span(seconds):
    """``45 s``, ``26 min``, ``2 h 10 min`` -- the shapes `history.parse_duration` reads."""
    whole = int(round(seconds))
    if whole < 60:
        return f"{whole} s"
    minutes = int(round(whole / 60))
    if minutes < 60:
        return f"{minutes} min"
    return f"{minutes // 60} h {minutes % 60:02d} min"
'''

SPAN_LORA = '''
def span(seconds):
    """``45 s``, ``26 min``, ``2 h 10 min`` -- the shapes the bench's estimates print."""
    whole = int(round(seconds))
    if whole < 60:
        return f"{whole} s"
    minutes = int(round(whole / 60))
    if minutes < 60:
        return f"{minutes} min"
    return f"{minutes // 60} h {minutes % 60:02d} min"
'''

EMIT_PLAIN = '''
def emit(events, kind, **fields):
    if events is None:
        return
    row = {"kind": kind}
    row.update(fields)
    try:
        events(row)
    except Exception:
        pass
'''

EMIT_COMMENTED = '''
def emitting(callback, kind, **fields):
    # A listener that raises must not take the run down with it.
    if callback is None:
        return

    row = {"kind": kind}
    row.update(fields)

    try:
        callback(row)
    except Exception:
        pass
'''

PLURALS_ONE = '''
def plurals(word):
    """Every spelling of ``word`` a label might use."""
    out = {word}
    if word.endswith("y") and len(word) > 2:
        out.add(word[:-1] + "ies")
    elif word.endswith(("s", "x", "z", "ch", "sh")):
        out.add(word + "es")
    else:
        out.add(word + "s")
    return sorted(out)
'''

PLURALS_TWO = '''
def spellings(term):
    seen = {term}
    if term.endswith("y") and len(term) > 2:
        seen.add(term[:-1] + "ies")
    elif term.endswith(("s", "x", "z", "ch", "sh")):
        seen.add(term + "es")
    else:
        seen.add(term + "s")
    return sorted(seen)
'''

# The shapes this gate was written for. Each pair was a real copy in the tree once, and
# each has been collapsed since; the fixture keeps the shape after the copy is gone.
KNOWN_COPIES: tuple[tuple[str, str, str], ...] = (
    ("two docstrings over one body", SPAN_ESTIMATE, SPAN_LORA),
    ("a rename, comments and blank lines", EMIT_PLAIN, EMIT_COMMENTED),
    ("a rename with the docstring dropped", PLURALS_ONE, PLURALS_TWO),
)


def test_the_gate_still_knows_the_shapes_it_was_written_for():
    assert KNOWN_COPIES, "a gate with nothing pinned is a gate nobody is checking"


@pytest.mark.parametrize("what, one, two", KNOWN_COPIES,
                         ids=[case[0] for case in KNOWN_COPIES])
def test_the_copies_this_gate_was_written_for_are_reported(what, one, two, tmp_path):
    """The known shapes, so a normalisation that quietly tightens is caught."""
    root = tree(tmp_path, one=one, two=two)
    found = duplicates.find(root)
    assert len(found) == 2, f"{what} should hash the same"
