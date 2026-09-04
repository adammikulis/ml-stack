"""`ml-stack-audit`: every tracked file read for a person's details, reported, never blocked.

The recogniser here is a fake that finds what it is told to, so nothing depends on presidio.
Everything named is invented.
"""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

from ml_stack.redact import audit, hook

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "audit-pii"


@dataclass
class Hit:
    entity_type: str
    score: float
    start: int
    end: int


class FakeEngine:
    """Reports each configured phrase, wherever it occurs, as the kind and score given."""

    def __init__(self, **phrases: tuple[str, float]) -> None:
        self.phrases = {k.replace("_", " "): v for k, v in phrases.items()}
        self.calls = 0

    def analyze(self, text: str, language: str) -> list[Hit]:
        self.calls += 1
        hits = []
        for phrase, (kind, score) in self.phrases.items():
            at = text.find(phrase)
            while at != -1:
                hits.append(Hit(kind, score, at, at + len(phrase)))
                at = text.find(phrase, at + 1)
        return hits


def repo(tmp_path: Path, fixtures: str = "", graph: dict | None = None, **files: str) -> Path:
    """A git repository with those files committed."""
    where = tmp_path / "repo"
    where.mkdir()

    def git(*a: str) -> None:
        subprocess.run(["git", "-C", str(where), *a], check=True, capture_output=True)

    git("init", "-q")
    git("config", "user.email", "nobody@example.invalid")
    git("config", "user.name", "Nobody")
    (where / "known-fixtures.txt").write_text(fixtures, encoding="utf-8")
    (tmp_path / "graph.json").write_text(json.dumps(graph or {"nodes": [], "messages": {}}))
    for name, body in files.items():
        (where / name).parent.mkdir(parents=True, exist_ok=True)
        (where / name).write_text(body, encoding="utf-8")
    git("add", "known-fixtures.txt", *files)
    git("commit", "-q", "-m", "files")
    return where


def wiring(tmp_path: Path) -> dict[str, str]:
    return {"NAMES_GRAPH": str(tmp_path / "graph.json"), "NAMES_SCRAPE": "",
            "NAMES_FIXTURES": "known-fixtures.txt", "HOME": str(tmp_path)}


def run(where: Path, tmp_path: Path, *argv: str, engine=None) -> tuple[int, str]:
    said = io.StringIO()
    code = audit.main(["--root", str(where), *argv], env=wiring(tmp_path), stdout=said,
                      engine=engine)
    return code, said.getvalue()


def test_a_card_number_in_a_tracked_file_is_reported(tmp_path):
    where = repo(tmp_path, **{"notes/pay.md": "the card was 0000 1111 2222 3333\n"})
    engine = FakeEngine(**{"0000 1111 2222 3333": ("CREDIT_CARD", 1.0)})
    code, said = run(where, tmp_path, engine=engine)
    assert code == 1, said
    assert "notes/pay.md:1" in said and "credit card" in said and "0000 1111 2222 3333" in said
    assert "a guess, not a verdict" in said


def test_nothing_found_says_so_and_exits_zero(tmp_path):
    where = repo(tmp_path, **{"a.md": "the kiln is warm\n"})
    code, said = run(where, tmp_path, engine=FakeEngine())
    assert code == 0
    assert "nothing to look at" in said


def test_a_hit_under_the_floor_is_not_reported_until_the_floor_is_lowered(tmp_path):
    where = repo(tmp_path, **{"a.md": "Marla Quinn fired the kiln\n"})
    engine = FakeEngine(Marla_Quinn=("PERSON", 0.5))
    assert run(where, tmp_path, engine=engine)[0] == 0
    code, said = run(where, tmp_path, "--floor", "0.4", engine=engine)
    assert code == 1, said
    assert "Marla Quinn" in said


def test_the_noisy_kinds_are_reported_only_with_all(tmp_path):
    where = repo(tmp_path, **{"a.md": "seen in Loomcast Bay on the day\n"})
    engine = FakeEngine(Loomcast_Bay=("LOCATION", 0.9))
    assert run(where, tmp_path, engine=engine)[0] == 0
    code, said = run(where, tmp_path, "--all", engine=engine)
    assert code == 1, said
    assert "Loomcast Bay" in said and "location" in said


def test_an_invented_name_on_the_allow_list_is_not_reported(tmp_path):
    where = repo(tmp_path, fixtures="Marla Quinn\n", **{"a.py": 'who = "Marla Quinn"\n'})
    engine = FakeEngine(Marla_Quinn=("PERSON", 0.9))
    code, said = run(where, tmp_path, engine=engine)
    assert code == 0, said


def test_an_identifier_is_not_a_card_number(tmp_path):
    where = repo(tmp_path, **{"a.py": 'out["opportunities"] = test_a_thing\n'})
    engine = FakeEngine(**{'out["opportunities"]': ("CREDIT_CARD", 1.0),
                           "test_a_thing": ("IBAN_CODE", 1.0)})
    assert run(where, tmp_path, engine=engine)[0] == 0


def test_a_name_from_the_database_is_reported_without_presidio(tmp_path, monkeypatch):
    where = repo(tmp_path, graph={"nodes": [{"kind": "person", "label": "Wren Halloway"}],
                                  "messages": {}},
                 **{"a.md": "ask Wren Halloway\n"})
    monkeypatch.setattr(audit, "recogniser", lambda: None)
    code, said = run(where, tmp_path)
    assert code == 1, said
    assert "Wren Halloway" in said and "someone in the data" in said
    assert "presidio is not installed" in said


def test_an_untracked_file_is_not_read(tmp_path):
    where = repo(tmp_path, **{"a.md": "warm\n"})
    (where / "scratch.md").write_text("Marla Quinn\n", encoding="utf-8")
    engine = FakeEngine(Marla_Quinn=("PERSON", 0.9))
    assert run(where, tmp_path, engine=engine)[0] == 0


def test_the_hooks_skip_suffixes_and_data_files_are_not_read(tmp_path):
    where = repo(tmp_path, **{"a.png": "Marla Quinn", "b.json": '{"who": "Marla Quinn"}',
                              "c.min.js": "Marla Quinn"})
    engine = FakeEngine(Marla_Quinn=("PERSON", 0.9))
    assert run(where, tmp_path, engine=engine)[0] == 0
    assert engine.calls == 0


def test_json_prints_one_object_per_finding(tmp_path):
    where = repo(tmp_path, **{"a.md": "Marla Quinn\n", "b.md": "Marla Quinn again\n"})
    engine = FakeEngine(Marla_Quinn=("PERSON", 0.9))
    code, said = run(where, tmp_path, "--json", engine=engine)
    assert code == 1
    rows = json.loads(said)
    assert [(r["path"], r["line"]) for r in rows] == [("a.md", 1), ("b.md", 1)]
    assert all("Marla Quinn" in r["what"] for r in rows)


def test_staged_is_the_hook_over_the_index_and_tracked_is_the_working_tree(tmp_path):
    where = repo(tmp_path, graph={"nodes": [{"kind": "person", "label": "Wren Halloway"}],
                                  "messages": {}},
                 **{"a.md": "warm\n"})
    (where / "a.md").write_text("ask Wren Halloway\n", encoding="utf-8")
    engine = FakeEngine()
    assert run(where, tmp_path, "--staged", engine=engine)[0] == 0
    assert run(where, tmp_path, "--tracked", engine=engine)[0] == 1
    subprocess.run(["git", "-C", str(where), "add", "a.md"], check=True)
    code, said = run(where, tmp_path, "--staged", engine=engine)
    assert code == 1, said
    assert "Wren Halloway" in said


def test_the_default_root_is_the_repository_around_the_working_directory(tmp_path, monkeypatch):
    where = repo(tmp_path, **{"a.md": "Marla Quinn\n"})
    monkeypatch.chdir(where)
    engine = FakeEngine(Marla_Quinn=("PERSON", 0.9))
    said = io.StringIO()
    assert audit.main([], env=wiring(tmp_path), stdout=said, engine=engine) == 1


def test_fixtures_on_the_command_line_win_over_the_environment(tmp_path):
    where = repo(tmp_path, **{"a.md": "Marla Quinn\n"})
    (where / "allowed.txt").write_text("Marla Quinn\n", encoding="utf-8")
    engine = FakeEngine(Marla_Quinn=("PERSON", 0.9))
    assert run(where, tmp_path, engine=engine)[0] == 1
    assert run(where, tmp_path, "--fixtures", "allowed.txt", engine=engine)[0] == 0


def test_findings_take_the_kinds_and_the_floor(tmp_path):
    """The hook's own `_findings` reads any kind it is given, so the audit and the hook share
    one reader of one file."""
    engine = FakeEngine(Marla_Quinn=("PERSON", 0.9), **{"GB33 0000": ("IBAN_CODE", 0.7)})
    rules = hook.shapes()
    found = list(hook._findings("a.md", "Marla Quinn GB33 0000\n", set(), set(), engine, rules,
                                kinds=frozenset({"IBAN_CODE"}), floor=0.6))
    assert [w for _, _, w in found] == ["'GB33 0000' reads as iban code"]
    assert list(hook._findings("a.md", "GB33 0000\n", set(), set(), engine, rules,
                               kinds=frozenset({"IBAN_CODE"}), floor=0.8)) == []


@pytest.mark.slow
def test_the_script_runs_the_module_over_a_checkout(tmp_path):
    where = repo(tmp_path, graph={"nodes": [{"kind": "person", "label": "Wren Halloway"}],
                                  "messages": {}},
                 **{"a.md": "ask Wren Halloway\n"})
    done = subprocess.run(["sh", str(SCRIPT), "--root", str(where)], capture_output=True,
                          text=True, env={**os.environ, **wiring(tmp_path),
                                          "PYTHON": sys.executable})
    assert done.returncode == 1, done.stdout + done.stderr
    assert "Wren Halloway" in done.stdout


def test_a_wrong_flag_is_refused_with_the_usage(capsys):
    with pytest.raises(SystemExit) as left:
        audit.main(["--what"])
    assert left.value.code == 2
    assert "usage:" in capsys.readouterr().err
