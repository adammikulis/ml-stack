"""The hook that refuses a person's name in a commit.

It is the last thing standing between a real community and a public repository, and it had
no tests. Everything named here is invented.
"""

from __future__ import annotations

import io
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

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


def commit(where: Path, files: dict[str, str], message: str) -> str:
    """Stage and commit those files, and return the commit's sha."""
    stage(where, files)
    subprocess.run(["git", "commit", "-q", "-m", message], cwd=where, check=True,
                   capture_output=True)
    done = subprocess.run(["git", "rev-parse", "HEAD"], cwd=where, check=True,
                          capture_output=True, text=True)
    return done.stdout.strip()


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


def shell_path() -> str:
    shell = shutil.which("sh")
    if not shell and os.name == "nt":
        git = shutil.which("git")
        candidate = Path(git).parent.parent / "usr" / "bin" / "sh.exe" if git else None
        shell = str(candidate) if candidate and candidate.is_file() else None
    assert shell, "a POSIX shell is required"
    return shell

def check_wrapper(where: Path, tmp_path: Path, script: str = str(HOOK),
                  python: str = sys.executable, **files: str) -> tuple[int, str]:
    """Stage those files and run the shell wrapper the way git does."""
    stage(where, files)
    source = where / "src" / "ml_stack"
    if not source.exists():
        shutil.copytree(HOOK.parents[2] / "src" / "ml_stack", source)
        shutil.copytree(HOOK.parents[2] / "contracts", where / "contracts")
    done = subprocess.run([shell_path(), script], cwd=where, capture_output=True, text=True,
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


def test_a_merge_is_checked_for_what_it_adds_not_for_what_the_other_parent_brought(tmp_path):
    where = repo(tmp_path, PEOPLE)
    git = lambda *a: subprocess.run(["git", *a], cwd=where, check=True, capture_output=True)  # noqa: E731
    commit(where, {"base.txt": "clean\n"}, "base")
    git("checkout", "-q", "-b", "other")
    commit(where, {"theirs.txt": "Ask Wren Halloway about the kiln.\n"}, "theirs")
    git("checkout", "-q", "-")
    commit(where, {"ours.txt": "clean too\n"}, "ours")
    git("merge", "--no-commit", "--no-ff", "other")
    said = io.StringIO()
    assert hook.main(env=wiring(tmp_path), root=where, stdout=said) == 0, said.getvalue()
    code, said = check(where, tmp_path, added="Ask Wren Halloway about the kiln.\n")
    assert code == 1 and "Wren Halloway" in said


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


def test_technical_phrases_that_look_like_people_are_stood_down(tmp_path, monkeypatch):
    """Named technical phrases are not refused just because Presidio reads them as people."""
    class Engine:
        def analyze(self, text, language):
            for phrase in ("Git metadata", "Cloud Files", "LM Studio", "Bea Marlow",
                           "Cloud Files " + "Bea Marlow"):
                start = text.find(phrase)
                if start >= 0:
                    yield SimpleNamespace(entity_type="PERSON", score=0.99, start=start,
                                          end=start + len(phrase))

    monkeypatch.setattr(hook, "recogniser", Engine)
    where = repo(tmp_path, graph={"nodes": []})
    code, said = check(where, tmp_path, **{"recovery.md": (
        "LM Studio manages local models.\n"
        "Git metadata still points at the original checkout.\n"
        "Cloud Files placeholders remain in the old worktree.\n"
        "Cloud Files " + "Bea Marlow" + " is in the log.\n"
        "Bea Marlow owns the sample checkout.\n")})
    assert code == 1
    assert "LM Studio" not in said
    assert "Git metadata" not in said
    assert not any("Cloud Files" in line and "Bea Marlow" not in line
                   for line in said.splitlines())
    assert "Bea Marlow" in said
    assert "Cloud Files " + "Bea Marlow" in said


def test_the_technical_phrase_rule_does_not_clear_similar_person_names():
    rules = hook.shapes()
    assert rules.in_context("Git metadata") == "context_product: Git metadata"
    assert rules.in_context("Cloud Files") == "context_product: Cloud Files"
    assert rules.in_context("LM Studio") == "context_product: LM Studio"
    assert rules.in_context("LM Studio " + "Bea Marlow") is None
    assert rules.in_context("Bea Marlow " + "LM Studio") is None
    assert rules.in_context("Git " + "Marlow") is None
    assert rules.in_context("Cloud " + "Marlow") is None
    assert rules.in_context("Cloud Files " + "Bea Marlow") is None
    assert rules.in_context("Bea Marlow " + "Cloud Files") is None


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


def test_a_compact_date_time_stamp_is_not_a_phone_number(tmp_path):
    """A log named `NAME-PORT-YYYYMMDD-HHMMSS-PID.log` holds a run of digits and dashes at a
    phone number's length."""
    where = repo(tmp_path, graph={"nodes": []})
    code, said = check(where, tmp_path, **{"t.py": (
        'LOG = f"llama-server-{port}-20260923-170000-1.log"\n')})
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
    # a name from the graph, so the exact list catches it with or without presidio
    code, said = check(where, tmp_path, **{"events.py": catalogue + 'X = "Wren Halloway"\n'})
    assert code == 1 and "Wren Halloway" in said


def test_the_recogniser_is_built_once_per_process():
    """One Presidio load serves every check in the process."""
    assert hook.recogniser() is hook.recogniser()


def test_against_a_revision_checks_a_committed_range_not_the_index(tmp_path):
    """`--against REV`: the diff between REV and HEAD, the shape CI checks a pushed range in,
    rather than the staged index a commit uses."""
    where = repo(tmp_path, PEOPLE)
    base = commit(where, {"notes.txt": "The kiln needs firing.\n"}, "base")
    commit(where, {"notes.txt": "The kiln needs firing.\nAsk Wren Halloway about it.\n"},
          "adds a note")
    said = io.StringIO()
    code = hook.main(["--against", base], env=wiring(tmp_path), root=where, stdout=said)
    assert code == 1
    assert "Wren Halloway" in said.getvalue()

    clean = io.StringIO()
    assert hook.main(["--against", "HEAD"], env=wiring(tmp_path), root=where,
                     stdout=clean) == 0


@pytest.mark.slow


def test_the_shell_wrapper_runs_the_hook_end_to_end(tmp_path):
    """`sh scripts/hooks/no-real-names`, the way git runs it: a clean file commits, a name
    from the graph is refused with the same exit codes and message as in-process."""
    where = repo(tmp_path, PEOPLE)
    code, said = check_wrapper(where, tmp_path, notes="The kiln needs firing.\n")
    assert code == 0, said
    code, said = check_wrapper(where, tmp_path, notes="Ask Wren Halloway about the kiln.\n")
    assert code == 1
    assert "Wren Halloway" in said and "refusing to commit" in said


@pytest.mark.slow


def test_the_wrapper_finds_the_source_tree_when_ml_stack_is_not_installed(tmp_path):
    """The copied hook loads checkout source with no installed site packages."""
    where = repo(tmp_path, PEOPLE)
    bare = tmp_path / "bare-python"
    bare.write_text(f'#!/bin/sh\nexec "{Path(sys.executable).as_posix()}" -I -S "$@"\n')
    bare.chmod(0o755)
    link = where / ".git" / "hooks" / "pre-commit"
    link.parent.mkdir(exist_ok=True)
    shutil.copyfile(HOOK, link)
    code, said = check_wrapper(where, tmp_path, script=".git/hooks/pre-commit", python=str(bare),
                               notes="Ask Wren Halloway about the kiln.\n")
    assert code == 1, said
    assert "Wren Halloway" in said
    assert "presidio is not installed" in said


def test_the_wrapper_prefers_the_current_checkout_to_an_installed_package(tmp_path):
    where = repo(tmp_path, graph={"nodes": []})
    source = where / "src" / "ml_stack" / "redact"
    source.mkdir(parents=True)
    (where / "src" / "ml_stack" / "__init__.py").write_text("")
    (source / "__init__.py").write_text("")
    (source / "hook.py").write_text(
        "from pathlib import Path\n"
        "def main(argv=None):\n"
        "    print(Path(__file__).resolve())\n"
        "    return 0\n"
    )
    stale = tmp_path / "installed" / "ml_stack" / "redact"
    stale.mkdir(parents=True)
    (tmp_path / "installed" / "ml_stack" / "__init__.py").write_text("")
    (stale / "__init__.py").write_text("")
    (stale / "hook.py").write_text("def main(argv=None):\n    return 1\n")

    shell = shell_path()
    done = subprocess.run(
        [shell, str(HOOK)],
        cwd=where,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHON": sys.executable,
             "PYTHONPATH": str(tmp_path / "installed")},
    )
    assert done.returncode == 0, done.stderr
    assert str(source / "hook.py") in done.stdout


@pytest.mark.slow


def test_a_clean_commit_is_still_refused_without_presidio(tmp_path):
    """A file with nothing the exact list or the shape rule would catch still fails the hook
    when presidio cannot be imported: without it, a name the hook has never seen passes
    unnoticed, so the hook refuses rather than passing silently."""
    where = repo(tmp_path, {"nodes": [], "messages": {}})
    bare = tmp_path / "bare-python"
    bare.write_text(f'#!/bin/sh\nexec "{Path(sys.executable).as_posix()}" -I -S "$@"\n')
    bare.chmod(0o755)
    code, said = check_wrapper(where, tmp_path, python=str(bare),
                               notes="The kiln needs firing before the studio opens.\n")
    assert code == 1, said
    assert "presidio is not installed" in said
    assert "Activate the project environment" in said
    assert "python -m pip install '.[privacy]'" in said
    assert "python -m spacy download en_core_web_sm" in said


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


def test_only_the_lines_a_commit_adds_are_judged_not_what_the_file_already_said(tmp_path):
    where = repo(tmp_path, PEOPLE)
    commit(where, {"notes.txt": "Ask Wren Halloway about the kiln.\n"}, "old text, accepted then")
    code, said = check(where, tmp_path, **{"notes.txt": "Ask Wren Halloway about the kiln.\nA clean new line.\n"})
    assert code == 0, said
    code, said = check(where, tmp_path, **{"notes.txt": "Ask Wren Halloway about the kiln.\nA clean new line.\n"
                                                         "And Wren Halloway again.\n"})
    assert code == 1 and "notes.txt:3" in said and "notes.txt:1" not in said


def test_a_revision_that_is_an_option_is_refused_and_a_hostile_path_reaches_git_as_one_argument(tmp_path):
    import subprocess

    from ml_stack.redact.added import added_lines

    def git(*args):
        subprocess.run(["git", "-C", str(tmp_path), *args], check=True, capture_output=True)

    git("init", "-q")
    git("config", "user.email", "t@example.invalid")
    git("config", "user.name", "t")
    hostile = tmp_path / "a b;$(touch pwned)`x`.txt"
    hostile.write_text("one\n")
    git("add", "--all")
    git("commit", "-q", "-m", "base")
    hostile.write_text("one\ntwo\n")
    git("add", "--all")
    assert added_lines(str(tmp_path), hostile.name) == {2}
    assert not (tmp_path / "pwned").exists()
    with pytest.raises(ValueError):
        added_lines(str(tmp_path), hostile.name, against="--output=" + str(tmp_path / "out"))
    assert not (tmp_path / "out").exists()


FRICTION_LINES = (
    "App Attest keys live in the Secure Enclave.\n"
    "Seal with NaCl Box before sending.\n"
    "RunPod offers Community Cloud and Secure Cloud pods.\n"
    "See 4.3 Disputes for the process.\n"
    "The Earned tier unlocks after review.\n"
    "Use the worktree path shown above, and the console metadata.\n"
    "Serve qwen3-30b-a3b-2507-q4-1234567890-instruct and claude-sonnet-5-5-20260301.\n"
    '<path d="M12.5 3-4.2 8.1 9.3-7.7 2 1.1-5.5 6.6z"/>\n'
    '<path d="M1234-5678 9012-3456 L7890-1234"/>\n')


def test_technical_phrases_model_ids_and_path_data_are_not_people(tmp_path):
    """Phrases and numbers an agent writes in ordinary docs and markup. Mutation: drop the
    `benign` calls in `hook` and each of these is refused again."""
    where = repo(tmp_path, graph={"nodes": []})
    code, said = check(where, tmp_path, **{"notes.md": FRICTION_LINES})
    assert code == 0, said


def test_real_contacts_are_still_refused_beside_the_friction_lines(tmp_path):
    where = repo(tmp_path, graph={"nodes": []})
    code, said = check(where, tmp_path, **{"notes.md": FRICTION_LINES
                                           + "Write to bea.marlow@" + "gmail.com or +1 (415) " + "555-0134.\n"})
    assert code == 1
    assert "bea.marlow@" + "gmail.com" in said and "555-0134" in said


def test_a_phrase_the_documents_already_carry_is_the_repositorys_vocabulary(tmp_path):
    from ml_stack.redact.benign import documented
    where = repo(tmp_path, graph={"nodes": []})
    commit(where, {"glossary.md": "Quartz" + " Lantern is a term.\n"}, "chore: glossary")
    assert documented(str(where), "Quartz" + " Lantern")
    assert not documented(str(where), "Bea Marlow")
    assert not documented(str(where), "quartz lantern")


def test_a_phrase_shaped_like_a_git_option_or_a_pathspec_is_only_searched_for(tmp_path):
    from ml_stack.redact.benign import documented
    where = repo(tmp_path, graph={"nodes": []})
    commit(where, {"glossary.md": "Quartz" + " Lantern is a term.\n"}, "chore: glossary")
    for hostile in ("--output=leak.txt Of", "Quartz --open-files-in-pager=sh", "Quartz; touch leak"):
        assert not documented(str(where), hostile)
    assert not (where / "leak.txt").exists() and not (where / "leak").exists()
