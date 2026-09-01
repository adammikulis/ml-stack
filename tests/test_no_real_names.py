"""The hook that refuses a person's name in a commit.

It is the last thing standing between a real community and a public repository, and it had
no tests. Everything named here is invented.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

HOOK = Path(__file__).resolve().parent.parent / "scripts" / "hooks" / "no-real-names"


def repo(tmp_path: Path, graph: dict | None = None, fixtures: str = "") -> Path:
    """A git repository with a graph to check against, and nothing staged yet."""
    where = tmp_path / "repo"
    where.mkdir()
    def run(*a):
        return subprocess.run(a, cwd=where, check=True, capture_output=True)

    run("git", "init", "-q")
    run("git", "config", "user.email", "nobody@example.invalid")
    run("git", "config", "user.name", "Nobody")
    (where / "known-fixtures.txt").write_text(fixtures)
    (tmp_path / "graph.json").write_text(json.dumps(graph or {"nodes": [], "messages": {}}))
    return where


def check(where: Path, tmp_path: Path, **files: str) -> tuple[int, str]:
    """Stage those files and run the hook. Returns (exit code, what it said)."""
    for name, body in files.items():
        (where / name).write_text(body)
        subprocess.run(["git", "add", name], cwd=where, check=True, capture_output=True)
    done = subprocess.run(["sh", str(HOOK)], cwd=where, capture_output=True, text=True,
                          env={**os.environ,
                               "NAMES_GRAPH": str(tmp_path / "graph.json"),
                               "NAMES_SCRAPE": "",
                               "NAMES_FIXTURES": "known-fixtures.txt",
                               "SKIP_NAME_CHECK": ""})
    return done.returncode, done.stdout + done.stderr


PEOPLE = {"nodes": [{"id": "person:1", "kind": "person", "label": "Wren Halloway"},
                    {"id": "person:2", "kind": "person", "label": "Bo Ng"},
                    {"id": "person:3", "kind": "person", "label": "Li"},
                    {"id": "org:1", "kind": "org", "label": "Tinsley Works"}],
          "messages": {}}


def test_a_name_from_the_graph_is_refused(tmp_path):
    where = repo(tmp_path, PEOPLE)
    code, said = check(where, tmp_path, notes="Ask Wren Halloway about the kiln.\n")
    assert code == 1
    assert "Wren Halloway" in said


def test_a_short_real_name_keeps_its_protection(tmp_path):
    """Li, Bo, Ng, Wu are names. A length floor meant for guessed-at handles dropped anything
    under four characters out of the list entirely, so the shortest real names -- which a
    heuristic is least likely to catch either -- had no protection at all."""
    where = repo(tmp_path, PEOPLE)
    code, said = check(where, tmp_path, notes="Li signed it off.\n")
    assert code == 1, said
    assert "'Li'" in said

    # and an org name that short stays out: those are guessed at, and two letters of
    # lowercase is an ordinary word, not a company
    other = tmp_path / "b"
    other.mkdir()
    where2 = repo(other, {"nodes": [{"id": "org:2", "kind": "org", "label": "Co"}],
                          "messages": {}})
    assert check(where2, other, notes="Co-ordinate the release.\n")[0] == 0


def test_a_one_letter_surname_is_shaped_like_a_name(tmp_path):
    """The heuristic demanded two letters of surname, so a person called "Jane O" who was
    not already in the graph passed straight through it."""
    where = repo(tmp_path, {"nodes": [], "messages": {}})
    code, said = check(where, tmp_path, notes='greeted = "Jane O"\n')
    assert code == 1
    assert "Jane O" in said


def test_an_invented_name_on_the_allow_list_passes(tmp_path):
    where = repo(tmp_path, PEOPLE, fixtures="Jane O\nWren Halloway\n")
    code, _ = check(where, tmp_path, notes='greeted = "Jane O"\nAsk Wren Halloway.\n')
    assert code == 0


def test_ordinary_prose_is_not_refused(tmp_path):
    where = repo(tmp_path, PEOPLE)
    code, said = check(where, tmp_path, notes="The kiln needs firing before the studio opens.\n")
    assert code == 0, said


def test_a_word_that_merely_contains_a_name_is_not_a_name(tmp_path):
    """Matching is word-bounded, which is what makes protecting a two-letter name affordable."""
    where = repo(tmp_path, PEOPLE)
    code, said = check(where, tmp_path, notes="The bongo drums and the Ngultrum exchange.\n")
    assert code == 0, said


def test_an_email_address_is_refused(tmp_path):
    where = repo(tmp_path, PEOPLE)
    code, said = check(where, tmp_path, notes="write to someone@elsewhere.example\n")
    assert code == 1 and "email" in said.lower()


@pytest.mark.parametrize("body", ["nothing to see", "a = 1"])
def test_a_clean_file_commits(tmp_path, body):
    where = repo(tmp_path, PEOPLE)
    assert check(where, tmp_path, notes=body + "\n")[0] == 0


def test_a_name_from_the_graph_is_refused_inside_a_json_file(tmp_path):
    """`docs/bench-runs.json` is committed to a public repository, and the shape rule is
    deliberately off for data files -- so the exact list has to carry json on its own."""
    where = repo(tmp_path, graph={"nodes": [{"kind": "person", "label": "Marta Quillon"}]})
    code, said = check(where, tmp_path,
                       **{"runs.json": '[{"label": "asked by Marta Quillon", "f1": 0.7}]'})
    assert code == 1
    assert "Marta Quillon" in said and "runs.json" in said


def test_an_org_from_the_graph_is_refused_inside_a_json_file(tmp_path):
    """A bench label names whatever was measured, and a real community's name is as much a
    leak as a person's."""
    where = repo(tmp_path, graph={"nodes": [{"kind": "org", "label": "Brayfield Survey Co"}]})
    code, said = check(where, tmp_path,
                       **{"runs.json": '[{"label": "Brayfield Survey Co nightly", "f1": 0.7}]'})
    assert code == 1
    assert "Brayfield Survey Co" in said


def test_a_data_file_full_of_proper_nouns_still_commits(tmp_path):
    """The shape rule is off for json/csv on purpose: a gazetteer's towns and a map's
    countries are quoted proper nouns and none of them are people. Turning it on there would
    refuse every data file, which is the same as turning the hook off."""
    where = repo(tmp_path, graph={"nodes": []})
    code, said = check(where, tmp_path,
                       **{"places.json": '["Dunmore", "Calderwick", "Ashby Weald"]'})
    assert code == 0, said


def test_the_bench_export_shape_commits(tmp_path):
    """What `ml-stack-bench show --export` actually writes: totals and server settings, no
    question, no entry, no answer."""
    where = repo(tmp_path, graph={"nodes": [{"kind": "person", "label": "Marta Quillon"}]})
    code, said = check(where, tmp_path, **{"bench-runs.json": json.dumps([{
        "label": "gptoss-plain", "model": "gpt-oss-120b-mxfp4-00001-of-00003.gguf",
        "f1": 0.6, "recall": 0.7, "precision": 0.6, "questions": 34, "seconds": 487,
        "context": 32768, "slots": 2, "sampling": {"temperature": 0.0}}])})
    assert code == 0, said
