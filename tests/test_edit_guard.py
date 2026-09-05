"""The Claude Code hook that refuses an edit writing what src/ml_stack already has."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
GUARD = REPO / "scripts" / "hooks" / "claude-edit-guard"

BLOCKED, ALLOWED = 2, 0

TALLY = '''"""A module."""


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

RENAMED = '''"""Another module."""


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


@pytest.fixture
def tree(tmp_path):
    where = tmp_path / "src" / "ml_stack"
    where.mkdir(parents=True)
    (where / "one.py").write_text(TALLY, encoding="utf-8")
    (where / "two.py").write_text('"""Two."""\n\nfrom __future__ import annotations\n',
                                  encoding="utf-8")
    return tmp_path


def run(given: dict, *, tool: str = "Write", cache: Path, guard: Path = GUARD, **env):
    done = subprocess.run(
        [str(guard)], text=True, capture_output=True,
        env={**os.environ, "MLSTACK_GUARD_CACHE": str(cache), **env},
        input=json.dumps({"tool_name": tool, "tool_input": given}))
    assert done.returncode in (BLOCKED, ALLOWED), done.stderr
    said = done.stderr if done.returncode == BLOCKED else done.stdout
    if done.returncode == BLOCKED:
        assert said.startswith("blocked: "), "a refusal has to say what to do instead"
    return done.returncode, said


def write(tree: Path, name: str, body: str) -> dict:
    return {"file_path": str(tree / "src" / "ml_stack" / name), "content": body}


def edit(tree: Path, name: str, added: str) -> dict:
    return {"file_path": str(tree / "src" / "ml_stack" / name),
            "old_string": "from __future__ import annotations",
            "new_string": "from __future__ import annotations\n\n" + added}


def test_a_body_the_tree_already_holds_is_refused(tree, tmp_path):
    code, said = run(write(tree, "three.py", RENAMED), cache=tmp_path / "cache")
    assert code == BLOCKED
    assert "`count_kinds` has the same body as `tally`" in said
    assert "src/ml_stack/one.py:4" in said


def test_the_same_body_arriving_through_multiedit_is_refused(tree, tmp_path):
    given = {"file_path": str(tree / "src" / "ml_stack" / "two.py"),
             "edits": [{"old_string": "from __future__ import annotations",
                        "new_string": "from __future__ import annotations\n\n" + RENAMED}]}
    code, said = run(given, tool="MultiEdit", cache=tmp_path / "cache")
    assert code == BLOCKED
    assert "src/ml_stack/one.py" in said


def test_editing_a_function_that_is_already_there_is_not_a_duplicate(tree, tmp_path):
    """Rewriting one line of `tally` must not report `tally` against itself."""
    given = {"file_path": str(tree / "src" / "ml_stack" / "one.py"),
             "old_string": '"""How often each kind appears."""',
             "new_string": '"""How often each kind is named."""'}
    assert run(given, tool="Edit", cache=tmp_path / "cache")[0] == ALLOWED


@pytest.mark.parametrize("added, expected", [
    ("def fetch(url):\n    with urllib.request.urlopen(url) as r:\n        return r.read()",
     "ml_stack.http.request_json"),
    ("def live():\n    return list(psutil.process_iter(['pid']))",
     "ml_stack.serve.process"),
    ("def save(path, obj):\n    tmp = path.with_suffix('.json.tmp')\n"
     "    tmp.write_text(json.dumps(obj))\n    os.replace(tmp, path)",
     "ml_stack.files.write_json"),
    ("HOME = Path('~/.ml-stack/two').expanduser()", "~/.ml-stack"),
    ('def notes(a):\n    """One.\n' + "\n".join(f"    line {i}" for i in range(14))
     + '\n    """\n    return a', "docstring"),
    ("def wide(a, b, c, d, e, f, g, h, i):\n    return a", "9 parameters"),
])
def test_the_routines_that_have_an_owner_are_refused(tree, tmp_path, added, expected):
    code, said = run(edit(tree, "two.py", added), tool="Edit", cache=tmp_path / "cache")
    assert code == BLOCKED
    assert expected in said


@pytest.mark.parametrize("added, expected", [
    ("def tally(rows, seen):\n    out = []\n    for row in rows:\n"
     "        out.append(str(row))\n    return out", "already exists at"),
    ("def command():\n    parser = argparse.ArgumentParser()\n"
     "    parser.add_argument('x')\n    return parser.parse_args()", "ArgumentParser"),
])
def test_the_advisory_rules_speak_but_let_the_edit_through(tree, tmp_path, added, expected):
    code, said = run(edit(tree, "two.py", added), tool="Edit", cache=tmp_path / "cache")
    assert code == ALLOWED
    assert expected in said


def test_ordinary_new_code_is_written_without_a_word(tree, tmp_path):
    code, said = run(write(tree, "three.py", '"""Three."""\n\n\n'
                           'def shout(word: str) -> str:\n    return str(word).upper()\n'),
                     cache=tmp_path / "cache")
    assert (code, said) == (ALLOWED, "")


@pytest.mark.parametrize("name", ["notes.md", "test_one.py"])
def test_only_library_modules_are_guarded(tree, tmp_path, name):
    assert run(write(tree, name, RENAMED), cache=tmp_path / "cache")[0] == ALLOWED


def test_a_file_outside_src_ml_stack_is_left_alone(tree, tmp_path):
    given = {"file_path": str(tree / "elsewhere.py"), "content": RENAMED}
    assert run(given, cache=tmp_path / "cache")[0] == ALLOWED


def test_the_guard_can_be_switched_off_for_a_session(tree, tmp_path):
    assert run(write(tree, "three.py", RENAMED), cache=tmp_path / "cache",
               MLSTACK_GUARD="off")[0] == ALLOWED


def test_a_corrupt_index_is_rebuilt_rather_than_believed(tree, tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    for stale in ("index.json",):
        (cache / stale).write_text("{not json", encoding="utf-8")
    assert run(write(tree, "three.py", RENAMED), cache=cache)[0] == BLOCKED
    for held in cache.glob("*.json"):
        held.write_text("{not json", encoding="utf-8")
    assert run(write(tree, "three.py", RENAMED), cache=cache)[0] == BLOCKED


def test_a_guard_that_cannot_run_does_not_stop_the_work(tree, tmp_path):
    """The detector is missing beside this copy; a broken guard must not block every edit."""
    alone = tmp_path / "elsewhere" / "hooks"
    alone.mkdir(parents=True)
    shutil.copy2(GUARD, alone / GUARD.name)
    code, said = run(write(tree, "three.py", RENAMED), cache=tmp_path / "cache",
                     guard=alone / GUARD.name)
    assert code == ALLOWED
    assert "claude-edit-guard did not run" in said


def test_only_the_writing_tools_are_guarded(tree, tmp_path):
    done = subprocess.run(
        [str(GUARD)], text=True, capture_output=True,
        env={**os.environ, "MLSTACK_GUARD_CACHE": str(tmp_path / "cache")},
        input=json.dumps({"tool_name": "Read",
                          "tool_input": write(tree, "three.py", RENAMED)}))
    assert done.returncode == ALLOWED


def test_a_malformed_event_is_not_an_edit_to_refuse(tmp_path):
    done = subprocess.run([str(GUARD)], text=True, capture_output=True,
                          env={**os.environ, "MLSTACK_GUARD_CACHE": str(tmp_path / "cache")},
                          input="not json at all")
    assert done.returncode == ALLOWED
