"""The hook that refuses a person's name in a commit.

It is the last thing standing between a real community and a public repository, and it had
no tests. Everything named here is invented.
"""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from ml_stack.redact import hook

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


def stage(where: Path, files: dict[str, str]) -> None:
    for name, body in files.items():
        (where / name).write_text(body)
        subprocess.run(["git", "add", name], cwd=where, check=True, capture_output=True)


def wiring(tmp_path: Path) -> dict[str, str]:
    return {**os.environ,
            "NAMES_GRAPH": str(tmp_path / "graph.json"),
            "NAMES_SCRAPE": "",
            "NAMES_FIXTURES": "known-fixtures.txt",
            "SKIP_NAME_CHECK": ""}


def check(where: Path, tmp_path: Path, **files: str) -> tuple[int, str]:
    """Stage those files and run the hook in this process. Returns (exit code, what it said)."""
    stage(where, files)
    said = io.StringIO()
    code = hook.main(env=wiring(tmp_path), root=where, stdout=said)
    return code, said.getvalue()


def check_wrapper(where: Path, tmp_path: Path, script: str = str(HOOK),
                  python: str = sys.executable, **files: str) -> tuple[int, str]:
    """Stage those files and run the shell wrapper the way git does."""
    stage(where, files)
    done = subprocess.run(["sh", script], cwd=where, capture_output=True, text=True,
                          env={**wiring(tmp_path), "PYTHON": python})
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
    # assembled so the commit hook does not read this file as holding an address
    address = "someone@" + "elsewhere.co"
    code, said = check(where, tmp_path, notes=f"write to {address}\n")
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


def test_geography_is_not_shaped_like_a_person(tmp_path):
    """"North Carolina", "Colorado River", "San Francisco Bay Area": a gazetteer and a
    geocoder's tests are full of quoted pairs that look like names and are places. The shape
    rule stands down when the first word is a direction or place prefix, or the last a place
    kind. Mutation: drop the `is_place` clause."""
    where = repo(tmp_path, graph={"nodes": []})
    code, said = check(where, tmp_path, **{"geo.py": (
        'SHORTHAND = {"nc": "North Carolina", "sf": "San Francisco"}\n'
        'rows = ["Colorado River", "Raleigh County", "United Kingdom", "The Bay"]\n')})
    assert code == 0, said


def test_a_person_quoted_beside_a_place_is_still_refused(tmp_path):
    """The place rule must not widen into a hole: an ordinary name-shaped pair on the same
    line as a place is still flagged."""
    where = repo(tmp_path, graph={"nodes": []})
    code, said = check(where, tmp_path, **{"geo.py": 'x = ["North Carolina", "Bea Marlow"]\n'})
    assert code == 1
    assert "Bea Marlow" in said


def test_a_reserved_documentation_domain_is_not_a_contact(tmp_path):
    """`ada.lovelace@pellard.example`, `one@example.com`: RFC 2606 reserves these so that
    nobody can be reached at them, which is exactly why tests use them. Mutation: drop the
    RESERVED_DOMAIN check."""
    where = repo(tmp_path, graph={"nodes": []})
    code, said = check(where, tmp_path, **{"t.py": (
        'A = "ada.lovelace@pellard.example"\nB = "one@example.com"\nC = "x@site.test"\n')})
    assert code == 0, said


def test_a_real_looking_address_is_still_refused(tmp_path):
    where = repo(tmp_path, graph={"nodes": []})
    address = "ada.lovelace@" + "pellard.co"
    code, said = check(where, tmp_path, **{"t.py": f'A = "{address}"\n'})
    assert code == 1
    assert "pellard.co" in said


def test_the_middle_of_a_uuid_is_not_a_phone_number(tmp_path):
    """`6f1b2a3c-4d5e-4f60-8172-839405a6b7c8` holds `60-8172-839405`, which is digits and
    dashes and the right length. Mutation: drop the UUIDISH check."""
    where = repo(tmp_path, graph={"nodes": []})
    code, said = check(where, tmp_path, **{"t.py": (
        'NS = "6f1b2a3c-4d5e-4f60-8172-839405a6b7c8"\n')})
    assert code == 0, said


def test_a_job_title_is_not_shaped_like_a_person(tmp_path):
    """A role catalogue is a page of "Software Engineer", "Account Manager", "Site Reliability
    Engineer". Mutation: drop the `is_role` clause."""
    where = repo(tmp_path, graph={"nodes": []})
    code, said = check(where, tmp_path, **{"roles.py": (
        'TITLES = ["Software Engineer", "Account Manager", "Site Reliability Engineer",\n'
        '          "Payroll Specialist", "People Partner", "Technical Writer"]\n')})
    assert code == 0, said


def test_a_file_may_declare_itself_a_catalogue_of_invented_labels(tmp_path):
    """"All Hands", "Hack Week", "Winter Party": a catalogue of event names is name-shaped
    and no heuristic will ever know every shape. A file saying `no-real-names: shapes off`
    in its first lines turns off the shape rule for itself and nothing else -- a name from
    the graph in that file is still refused. Mutation: drop the `shapes_off` clause."""
    where = repo(tmp_path, PEOPLE)
    catalogue = ('"""Event names.  no-real-names: shapes off"""\n'
                 'EVENTS = ["All Hands", "Hack Week", "Winter Party"]\n')
    code, said = check(where, tmp_path, **{"events.py": catalogue})
    assert code == 0, said
    code, said = check(where, tmp_path, **{"events.py": catalogue + 'X = "Ada Lovelace"\n'})
    assert code == 1 and "Ada Lovelace" in said


def test_the_recogniser_is_built_once_per_process():
    """One Presidio load serves every check in the process."""
    assert hook.recogniser() is hook.recogniser()


def test_the_shell_wrapper_runs_the_hook_end_to_end(tmp_path):
    """`sh scripts/hooks/no-real-names`, the way git runs it: a clean file commits, a name
    from the graph is refused with the same exit codes and message as in-process."""
    where = repo(tmp_path, PEOPLE)
    code, said = check_wrapper(where, tmp_path, notes="The kiln needs firing.\n")
    assert code == 0, said
    code, said = check_wrapper(where, tmp_path, notes="Ask Wren Halloway about the kiln.\n")
    assert code == 1
    assert "Wren Halloway" in said and "refusing to commit" in said


def test_the_wrapper_finds_the_source_tree_when_ml_stack_is_not_installed(tmp_path):
    """Run through a `.git/hooks/pre-commit` symlink with a Python that has no site-packages
    (`-I -S`): the wrapper resolves the symlink to find `../../src`, and the exact list still
    refuses the name. Presidio is absent from that Python, so the hook says so."""
    where = repo(tmp_path, PEOPLE)
    bare = tmp_path / "bare-python"
    bare.write_text(f'#!/bin/sh\nexec "{sys.executable}" -I -S "$@"\n')
    bare.chmod(0o755)
    link = where / ".git" / "hooks" / "pre-commit"
    link.parent.mkdir(exist_ok=True)
    link.symlink_to(HOOK)
    code, said = check_wrapper(where, tmp_path, script=".git/hooks/pre-commit", python=str(bare),
                               notes="Ask Wren Halloway about the kiln.\n")
    assert code == 1, said
    assert "Wren Halloway" in said
    assert "presidio is not installed" in said


def test_the_shape_rules_are_data_and_every_section_the_code_reads_exists():
    """`contracts/name-shapes.json` is well-formed, carries every section `hook.SECTIONS`
    names, both patterns, and a `why` for each section saying what it is for -- so the next
    exception is a data change with a known section, not a code change."""
    from ml_stack.contracts import contracts_dir
    data = json.loads((contracts_dir() / hook.CONTRACT).read_text(encoding="utf-8"))
    for section in hook.SECTIONS:
        assert section in data, f"{hook.CONTRACT} lacks {section}"
        assert data["why"].get(section), f"{hook.CONTRACT} has no why for {section}"
    assert set(data["patterns"]) >= {"uuid", "nameish"}
    rules = hook.shapes()
    assert rules.stood_down("North Carolina") == "place_first: north"
    assert rules.stood_down("Colorado River") == "place_last: river"
    assert rules.stood_down("Software Engineer") == "role_last: engineer"
    assert rules.stood_down("Bea Marlow") is None
    assert rules.reserved("pellard.example") == "reserved_domains: example"
    assert rules.reserved("sub.example.com") is None, "a whole-domain entry matches the whole domain only"


def test_a_rules_file_missing_a_section_is_refused_not_guessed_at(tmp_path):
    from ml_stack.contracts import ContractError
    partial = tmp_path / "partial.json"
    partial.write_text(json.dumps({"place_first": []}))
    with pytest.raises(ContractError, match="place_last"):
        hook.shapes(str(partial))


def test_a_word_added_to_the_data_changes_the_verdict_without_a_code_change(tmp_path):
    """A name-shaped pair no rule stands down -- a hero's title, assembled here so this file
    does not trip the hook itself. Copy the contract, add `knight` to `role_last`, point
    `NAMES_SHAPES` at the copy: the same file commits. The shipped rules are untouched, so
    the same file is still refused without the variable."""
    from ml_stack.contracts import contracts_dir
    data = json.loads((contracts_dir() / hook.CONTRACT).read_text(encoding="utf-8"))
    data["role_last"].append("knight")
    copy = tmp_path / "shapes.json"
    copy.write_text(json.dumps(data))
    hero = "Hollow " + "Knight"
    where = repo(tmp_path, graph={"nodes": []})
    stage(where, {"t.py": f'HERO = "{hero}"\n'})

    said = io.StringIO()
    assert hook.main(env=wiring(tmp_path), root=where, stdout=said) == 1
    assert hero in said.getvalue() and "nothing stood it down" in said.getvalue()

    said = io.StringIO()
    env = {**wiring(tmp_path), "NAMES_SHAPES": str(copy)}
    assert hook.main(env=env, root=where, stdout=said) == 0, said.getvalue()


def test_why_names_the_rule_that_cleared_each_pair(tmp_path):
    """`--why` (or `NAMES_WHY=1`) prints, for every name-shaped pair and contact-shaped run
    a rule stood down, which section and which word did it -- the line to edit next time."""
    where = repo(tmp_path, graph={"nodes": []}, fixtures="Jane O\n")
    body = ('SHORTHAND = {"nc": "North Carolina", "sf": "Colorado River"}\n'
            'TITLE = "Payroll Specialist"\n'
            'WHO = "Jane O"\n'
            'MAIL = "one@example.com"\n'
            'NS = "6f1b2a3c-4d5e-4f60-8172-839405a6b7c8"\n')
    stage(where, {"t.py": body})
    quiet = io.StringIO()
    assert hook.main(env=wiring(tmp_path), root=where, stdout=quiet) == 0
    assert "cleared by" not in quiet.getvalue()

    said = io.StringIO()
    assert hook.main(["--why"], env=wiring(tmp_path), root=where, stdout=said) == 0
    told = said.getvalue()
    assert "t.py:1  'North Carolina' cleared by place_first: north" in told
    assert "t.py:1  'Colorado River' cleared by place_last: river" in told
    assert "t.py:2  'Payroll Specialist' cleared by role_last: specialist" in told
    assert "t.py:3  'Jane O' cleared by fixtures" in told
    assert "'one@example.com' cleared by reserved_domains: example.com" in told
    assert "cleared by patterns: uuid" in told

    said = io.StringIO()
    hook.main(env={**wiring(tmp_path), "NAMES_WHY": "1"}, root=where, stdout=said)
    assert "cleared by place_first: north" in said.getvalue()

    stage(where, {"events.py": '"""no-real-names: shapes off"""\nX = ["Hack Week"]\n'})
    said = io.StringIO()
    hook.main(["--why"], env=wiring(tmp_path), root=where, stdout=said)
    assert "events.py  shape rule off: shapes_off: marker" in said.getvalue()
