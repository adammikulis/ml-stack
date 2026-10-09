"""scripts/testwritedeny.py: the pytest tree cannot write under the real state root, and the tests that
run `sandbox-exec` themselves are split into a pass of their own. Every kernel check writes only under
a temporary directory standing in for the real root."""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
import testwritedeny as deny

REPO = Path(__file__).resolve().parent.parent
PYTEST = [sys.executable, "-m", "pytest", "-q", "-n", "2", "tests/test_a.py"]


def expression(command):
    """The -m expression after pytest (not the interpreter's own -m)."""
    rest = command[command.index("pytest") + 1:]
    return rest[rest.index("-m") + 1]


def test_the_profile_denies_writes_under_each_root_and_allows_the_rest():
    text = deny.profile([Path("/real/state"), Path("/other")])
    assert text.startswith("(version 1)(allow default)")
    assert '(deny file-write* (subpath "/real/state") (subpath "/other"))' in text


def test_a_path_cannot_close_the_string_and_add_a_rule():
    assert deny.quote('/x") (allow file-write* (subpath "/') == '"/x\\") (allow file-write* (subpath \\"/"'
    with pytest.raises(ValueError, match="control character"):
        deny.quote("/x\n(allow default)")


def test_select_adds_or_extends_the_marker_expression_and_leaves_python_dash_m_alone():
    assert expression(deny.select(PYTEST, False)) == "not seatbelt"
    assert expression(deny.select(PYTEST, True)) == "seatbelt"
    with_m = [sys.executable, "-m", "pytest", "-q", "-m", "not slow and not heavy", "tests"]
    assert expression(deny.select(with_m, False)) == "(not slow and not heavy) and not seatbelt"
    assert expression(deny.select(with_m, True)) == "(not slow and not heavy) and seatbelt"
    assert deny.select(with_m, True)[:3] == with_m[:3] and deny.select(with_m, True)[-1] == "tests"


def test_combined_status_is_the_first_failure_else_pass_else_nothing_ran():
    assert deny.combined([0, 5]) == 0
    assert deny.combined([5, 5]) == 5
    assert deny.combined([1, 0]) == 1
    assert deny.combined([0, 2]) == 2


def test_junit_files_merge_testcases_and_counts(tmp_path):
    head = ('<?xml version="1.0"?><testsuites><testsuite name="pytest" errors="0" failures="1" skipped="0" '
            'tests="2" time="1.500"><testcase name="a"/><testcase name="b"><failure/></testcase></testsuite></testsuites>')
    tail = ('<?xml version="1.0"?><testsuites><testsuite name="pytest" errors="0" failures="0" skipped="1" '
            'tests="1" time="0.250"><testcase name="c"><skipped/></testcase></testsuite></testsuites>')
    one, two = tmp_path / "one.xml", tmp_path / "two.xml"
    one.write_text(head)
    two.write_text(tail)
    deny.merge_junit(one, two)
    from xml.etree import ElementTree
    suite = ElementTree.parse(one).getroot().find("testsuite")  # noqa: S314 - written above
    assert [c.get("name") for c in suite.iter("testcase")] == ["a", "b", "c"]
    assert (suite.get("tests"), suite.get("failures"), suite.get("skipped"), suite.get("time")) == ("3", "1", "1", "1.750")


@pytest.fixture
def usable(monkeypatch, tmp_path):
    monkeypatch.setattr(deny, "works", lambda: True)
    monkeypatch.setattr(deny, "denied_here", lambda _root: False)
    monkeypatch.setattr(deny, "protected", lambda _repo: [tmp_path / "state"])
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "seatbelt-modules.txt").write_text("# comment\ntest_sb.py  # why\n")
    return tmp_path


def test_a_run_of_both_kinds_is_two_passes_the_marked_one_without_the_denial(usable):
    plan = deny.passes([sys.executable, "-m", "pytest", "-q", "tests"], {}, usable)
    assert [p.denied for p in plan] == [True, False]
    assert [expression(p.command) for p in plan] == ["not seatbelt", "seatbelt"]


def test_naming_only_ordinary_files_is_one_denied_pass_and_only_marked_ones_one_free_pass(usable):
    ordinary = deny.passes([sys.executable, "-m", "pytest", "tests/test_a.py::test_x"], {}, usable)
    assert [p.denied for p in ordinary] == [True]
    marked = deny.passes([sys.executable, "-m", "pytest", "tests/test_sb.py"], {}, usable)
    assert [p.denied for p in marked] == [False]


def test_no_marked_modules_means_the_command_runs_once_unchanged(usable):
    (usable / "tests" / "seatbelt-modules.txt").write_text("# none\n")
    command = [sys.executable, "-m", "pytest", "-q", "tests"]
    plan = deny.passes(command, {}, usable)
    assert plan == [deny.Pass(command, True)]


def test_only_zero_opts_out_and_a_user_set_one_is_not_taken_as_proof(usable, capsys):
    command = [sys.executable, "-m", "pytest", "tests"]
    assert deny.passes(command, {deny.ENV: "0"}, usable) is None
    assert "write denial is off" in capsys.readouterr().err
    assert deny.passes(command, {deny.ENV: "1"}, usable) is not None


def test_a_process_already_denied_is_found_by_effect_and_not_wrapped_again(usable, monkeypatch):
    monkeypatch.setattr(deny, "denied_here", lambda _root: True)
    assert deny.passes([sys.executable, "-m", "pytest", "tests"], {}, usable) is None


def test_where_the_denial_cannot_be_applied_the_run_is_left_alone_with_a_notice(usable, monkeypatch, capsys):
    monkeypatch.setattr(deny, "works", lambda: False)
    assert deny.passes([sys.executable, "-m", "pytest", "tests"], {}, usable) is None
    assert "cannot be applied" in capsys.readouterr().err


@pytest.mark.skipif(sys.platform != "darwin", reason="Seatbelt is macOS; Linux uses bwrap and is not run here")
def test_the_kernel_refuses_a_write_under_the_root_and_allows_one_outside_it(tmp_path):
    root, outside = tmp_path / "state", tmp_path / "elsewhere"
    root.mkdir()
    outside.mkdir()
    code = ("import sys, pathlib\n"
            "for target in sys.argv[1:]:\n"
            " try:\n  pathlib.Path(target).write_text('x'); print('wrote', target)\n"
            " except OSError as exc:\n  print('refused', target, exc.errno)\n")
    wrapped = deny.wrapper([root.resolve()], [sys.executable, "-c", code, str(root / "a"), str(outside / "b")])
    done = subprocess.run(wrapped, capture_output=True, text=True, check=False, timeout=60)
    assert f"refused {root / 'a'} 1" in done.stdout, done
    assert f"wrote {outside / 'b'}" in done.stdout
    assert not (root / "a").exists() and (outside / "b").exists()


@pytest.mark.skipif(sys.platform != "darwin", reason="Seatbelt is macOS")
def test_a_second_sandbox_cannot_be_applied_inside_the_denial_which_is_why_marked_modules_run_apart(tmp_path):
    wrapped = deny.wrapper([tmp_path.resolve()], ["/usr/bin/sandbox-exec", "-p", "(version 1)(allow default)", "/usr/bin/true"])
    done = subprocess.run(wrapped, capture_output=True, text=True, check=False, timeout=60)
    assert done.returncode != 0 and "sandbox_apply" in done.stderr


def test_every_test_module_that_names_the_seatbelt_backend_is_listed():
    """A module that reaches the Seatbelt backend and is not in seatbelt-modules.txt would fail with
    sandbox_apply EPERM under the denial. The list is checked against the source, not remembered."""
    reaching = re.compile(r"sandbox_kit|sandbox\.seatbelt|import Seatbelt|sandbox-exec")
    reaches = {path.name for path in (REPO / "tests").glob("test_*.py") if reaching.search(path.read_text(encoding="utf-8"))}
    assert reaches <= deny.listed(REPO), sorted(reaches - deny.listed(REPO))


def test_every_listed_module_exists():
    assert all((REPO / "tests" / name).is_file() for name in deny.listed(REPO))


@pytest.mark.parametrize(("given", "joined"), [(["-m", "slow"], ["-m", "(slow) and not seatbelt"]),
                                               (["-mslow"], ["-m=(slow) and not seatbelt"]),
                                               (["-m=slow"], ["-m=(slow) and not seatbelt"])])
def test_every_spelling_of_the_marker_option_is_extended_not_duplicated(given, joined):
    out = deny.select([sys.executable, "-m", "pytest", "-q", *given, "tests"], False)
    assert out[3:] == ["-q", *joined, "tests"]


def test_the_junit_path_is_found_and_replaced_in_both_spellings(tmp_path):
    for form in (["--junitxml=a.xml"], ["--junitxml", "a.xml"]):
        command = [sys.executable, "-m", "pytest", *form, "tests"]
        assert deny.junit_path(command) == Path("a.xml")
        assert deny.junit_path(deny.with_junit(command, tmp_path / "b.xml")) == tmp_path / "b.xml"
    assert deny.junit_path([sys.executable, "-m", "pytest"]) is None


def test_a_pass_that_died_before_writing_junit_does_not_lose_the_other(tmp_path):
    good, missing = tmp_path / "good.xml", tmp_path / "missing.xml"
    good.write_text('<testsuites><testsuite name="p" errors="0" failures="0" skipped="0" tests="1" time="1">'
                    '<testcase name="a"/></testsuite></testsuites>')
    deny.merge_junit(good, missing)
    assert "testcase" in good.read_text()
    deny.merge_junit(missing, good)  # the first died: the second becomes the file
    assert "testcase" in missing.read_text()
    good.write_text("")
    deny.merge_junit(missing, good)  # an empty file is not a crash
    assert deny.combined([139, 0]) == 139


def test_bubblewrap_skips_a_root_that_does_not_exist_and_dies_with_its_parent(tmp_path):
    here = tmp_path / "here"
    here.mkdir()
    argv = deny.bubblewrap([here, tmp_path / "absent"], ["true"])
    assert argv.count("--ro-bind") == 1 and str(here) in argv and str(tmp_path / "absent") not in argv
    assert "--die-with-parent" in argv and "--new-session" in argv


def test_the_protected_roots_cover_the_cache_and_each_override_and_a_dropped_root_is_said(tmp_path, monkeypatch, capsys):
    state, cache, job = tmp_path / "s", tmp_path / "c", tmp_path / "jobs"
    monkeypatch.setattr(deny.home, "account_roots", lambda: (state, cache))
    monkeypatch.setenv("ML_STACK_HOME", str(state))
    monkeypatch.setenv("ML_STACK_CACHE", str(cache))
    monkeypatch.setenv("MLSTACK_JOBS_HOME", str(job))
    monkeypatch.setattr(deny.tempfile, "gettempdir", lambda: str(tmp_path / "t"))
    monkeypatch.setattr(deny.Path, "home", classmethod(lambda cls: tmp_path / "h"))
    got = deny.protected(tmp_path / "checkout")
    assert {state.resolve(), cache.resolve(), job.resolve()} <= set(got)
    monkeypatch.setattr(deny.tempfile, "gettempdir", lambda: str(state / "tmp"))
    assert state.resolve() not in deny.protected(tmp_path / "checkout")
    assert "holds the temporary directory" in capsys.readouterr().err


@pytest.mark.skipif(sys.platform != "darwin", reason="Seatbelt is macOS")
def test_a_hard_link_to_a_file_in_the_root_is_written_through_or_blocked_and_this_is_which(tmp_path):
    """Seatbelt judges the path a process opens. If a link outside the root to a file inside it lets a
    write through, a test could change real state by making that link first; that is documented, and
    this pins what the kernel does."""
    root, outside = tmp_path / "state", tmp_path / "elsewhere"
    root.mkdir()
    outside.mkdir()
    (root / "f").write_text("before")
    os.link(root / "f", outside / "alias")
    code = ("import sys, pathlib\ntry:\n pathlib.Path(sys.argv[1]).write_text('after'); print('through')\n"
            "except OSError:\n print('blocked')")
    done = subprocess.run(deny.wrapper([root.resolve()], [sys.executable, "-c", code, str(outside / "alias")]),
                          capture_output=True, text=True, check=False, timeout=60)
    assert done.stdout.strip() == "through", done
    assert (root / "f").read_text() == "after"
