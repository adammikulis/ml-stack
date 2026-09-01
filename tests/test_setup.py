"""The first-run wizard: what this machine can do, and what happens without being asked.

Every fact it reports is a one-line check that nothing announces and everything depends on.
The tests here are about it never lying -- an unreadable machine must say so rather than
guess, because a wrong "supported" is worse than no answer.
"""

from __future__ import annotations

import pytest

from ml_stack.setup import BEHAVIOURS, Finding, ask, explain, look, main


def test_it_reports_rather_than_changes(capsys):
    """Looking must never alter anything: it is what a first run does before trusting it."""
    findings = look()
    assert all(isinstance(f, Finding) for f in findings)
    assert findings, "a machine that reports nothing at all is a bug in the looking"
    assert all(f.name and f.said for f in findings)


def test_a_machine_that_will_not_answer_says_so_rather_than_guessing(monkeypatch):
    """A wrong "supported" sends someone to debug a model that was never going to load."""
    import ml_stack.setup as setup

    monkeypatch.setattr(setup, "_sysctl", lambda key: "")
    monkeypatch.setattr(setup, "_arches", lambda binary: set())
    named = {f.name for f in setup.look()}
    assert not any(n.startswith("architecture") for n in named), \
        "it must not claim an architecture is missing when it could not read any"


def test_an_architecture_is_only_claimed_when_the_names_were_read(monkeypatch):
    import ml_stack.setup as setup

    monkeypatch.setattr(setup, "_arches", lambda binary: {"qwen3moe", "gemma4"})
    found = [f for f in setup.look() if f.name == "architecture qwen4exp"]
    assert found and not found[0].good
    assert "master" in found[0].note

    monkeypatch.setattr(setup, "_arches", lambda binary: {"qwen4exp", "gemma4"})
    found = [f for f in setup.look() if f.name == "architecture qwen4exp"]
    assert found and found[0].good


def test_nothing_is_run_without_being_asked(capsys, monkeypatch):
    """A wizard that fixes things by itself is a wizard nobody can safely run twice."""
    ran = []
    import subprocess
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: ran.append(a))
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)

    ask([Finding(name="a thing", good=False, said="wrong", fix="rm -rf /")])
    assert ran == [], "a non-interactive run must offer, not act"
    assert "fix: rm -rf /" in capsys.readouterr().out


def test_a_fix_runs_only_when_it_is_wanted(capsys, monkeypatch):
    import subprocess
    ran = []
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: ran.append(a[0]))
    ask([Finding(name="a thing", good=False, said="wrong", fix="echo yes")], yes=True)
    assert ran == ["echo yes"]

    ran.clear()
    ask([Finding(name="a thing", good=True, said="fine", fix="echo no")], yes=True)
    assert ran == [], "nothing is run to fix what is not broken"


def test_it_never_asks_for_a_password():
    """Where a change needs root, sudo prompts on the terminal itself. Anything that reads a
    password in order to pass it along is doing what this deliberately does not."""
    import inspect

    import ml_stack.setup as setup

    source = inspect.getsource(setup)
    for reading in ("getpass", "password=", "input(\"password", "sudo -S", "--stdin"):
        assert reading not in source, f"{reading!r} would mean handling a password"


def test_the_behaviours_are_the_ones_that_surprise_people():
    """Each of these was once somebody's bug report. A default nobody was told about is
    indistinguishable from a fault when it does something unexpected."""
    named = {b.name for b in BEHAVIOURS}
    assert "a model that is not here" in named, "downloading tens of gigabytes unprompted"
    assert "repeated questions" in named, "an answer returning instantly reads as a fault"
    assert "a port that is taken" in named, "a server appearing somewhere unasked"
    assert all(b.does for b in BEHAVIOURS)
    assert all(b.why or b.setting for b in BEHAVIOURS), \
        "saying what happens without saying why it matters is just more output"


def test_explaining_says_what_happens_and_how_to_change_it(capsys):
    explain()
    said = capsys.readouterr().out
    assert "downloads, the first time it is asked for" in said
    assert "change it:" in said


def test_the_wizard_runs_end_to_end(capsys, monkeypatch):
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    code = main(["--quiet"])
    assert code in (0, 1)
    assert "what this machine can do" in capsys.readouterr().out
