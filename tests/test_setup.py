"""The first-run wizard: what this machine can do, and what happens without being asked.

Every fact it reports is a one-line check that nothing announces and everything depends on.
The tests here are about it never lying -- an unreadable machine must say so rather than
guess, because a wrong "supported" is worse than no answer.
"""

from __future__ import annotations

import json

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


def test_an_architecture_is_only_claimed_when_the_names_were_read(monkeypatch, tmp_path):
    import ml_stack.setup as setup
    fake = tmp_path / "llama-server"
    fake.write_text("#!/bin/sh\necho usage\n")
    fake.chmod(0o755)
    monkeypatch.setattr("ml_stack.serve.binary.find_binary", lambda *a, **k: fake)

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


def _server_answering(tmp_path, help_text: str):
    binary = tmp_path / "llama-server"
    binary.write_text("#!/bin/sh\nif [ \"$1\" = --help ]; then cat <<'HELP'\n"
                      + help_text + "HELP\nfi\nexit 0\n")
    binary.chmod(0o755)
    return binary


def test_a_flag_the_build_lacks_is_reported_with_the_nearest_it_has(tmp_path, monkeypatch):
    """`--draft-max` became `--spec-draft-n-max`; a build listing only the new name must
    say so here, before a load finds out."""
    import ml_stack.serve.binary as binary_module
    import ml_stack.setup as setup
    from ml_stack.serve import backend

    monkeypatch.setattr(backend, "_FLAGS", {})
    monkeypatch.setattr(setup, "_arches", lambda binary: {"gemma4"})
    stand_in = _server_answering(tmp_path, "-m, --model FNAME   model path\n"
                                           "--kv-unified   one cache for every slot\n")
    monkeypatch.setattr(binary_module, "find_binary", lambda *a, **k: stand_in)
    found = [f for f in setup.look() if f.name == "flags this build lacks"]
    assert found and not found[0].good
    assert "--kv-unified-per-slot" in found[0].said
    assert "no --kv-unified-per-slot, it has --kv-unified" in found[0].note
    assert "--spec-draft-n-max" in found[0].said


def test_a_build_that_answers_every_flag_is_not_mentioned(tmp_path, monkeypatch):
    import ml_stack.serve.binary as binary_module
    import ml_stack.setup as setup
    from ml_stack.serve import backend
    from ml_stack.serve.backend import LlamaServerBackend, emitted_flags

    monkeypatch.setattr(backend, "_FLAGS", {})
    monkeypatch.setattr(setup, "_arches", lambda binary: {"gemma4"})
    quiet = _server_answering(tmp_path, "-m, --model FNAME   model path\n")
    everything = "\n".join(f"{flag} X   described" for flag in
                           emitted_flags(LlamaServerBackend(binary=quiet))) + "\n"
    stand_in = _server_answering(tmp_path, everything)
    monkeypatch.setattr(backend, "_FLAGS", {})
    monkeypatch.setattr(binary_module, "find_binary", lambda *a, **k: stand_in)
    assert not [f for f in setup.look() if f.name == "flags this build lacks"]


def test_a_missing_architecture_offers_the_build_fix(monkeypatch, tmp_path):
    """`ml-stack-serve build` is the fix for a release lagging master by an architecture --
    `ml-stack-setup --yes` runs whatever a finding's `fix` names, so this is what makes
    `--yes` actually get a missing architecture rather than just naming the gap."""
    import ml_stack.setup as setup
    fake = tmp_path / "llama-server"
    fake.write_text("#!/bin/sh\necho usage\n")
    fake.chmod(0o755)
    monkeypatch.setattr("ml_stack.serve.binary.find_binary", lambda *a, **k: fake)

    monkeypatch.setattr(setup, "_arches", lambda binary: {"gemma4"})
    found = [f for f in setup.look() if f.name == "architecture qwen4exp"]
    assert found and found[0].fix == "ml-stack-serve build"

    monkeypatch.setattr(setup, "_arches", lambda binary: {"gemma4", "qwen4exp"})
    found = [f for f in setup.look() if f.name == "architecture qwen4exp"]
    assert found and found[0].fix == ""


def test_lacking_flags_offer_the_build_fix(tmp_path, monkeypatch):
    import ml_stack.serve.binary as binary_module
    import ml_stack.setup as setup
    from ml_stack.serve import backend

    monkeypatch.setattr(backend, "_FLAGS", {})
    monkeypatch.setattr(setup, "_arches", lambda binary: {"gemma4"})
    stand_in = _server_answering(tmp_path, "-m, --model FNAME   model path\n")
    monkeypatch.setattr(binary_module, "find_binary", lambda *a, **k: stand_in)
    found = [f for f in setup.look() if f.name == "flags this build lacks"]
    assert found and found[0].fix == "ml-stack-serve build"


def test_a_managed_build_is_labelled_by_commit_and_age_not_by_version(tmp_path):
    """BUILD.json is what tells a managed build apart from a brew one -- read it rather than
    guessing from the path, since both sit at some arbitrary directory."""
    import ml_stack.setup as setup

    binary = tmp_path / "llama-server"
    binary.write_text("#!/bin/sh\nexit 0\n")
    binary.chmod(0o755)
    (tmp_path / "BUILD.json").write_text(json.dumps({
        "commit": "abc1234", "built_at": "2020-01-01T00:00:00+00:00", "version": "x"}))

    label = setup._build_label(str(binary))
    assert "abc1234" in label
    assert "d old" in label, "2020 is certainly more than a day before today"


def test_an_unmanaged_build_is_labelled_by_its_own_version_output(tmp_path):
    import ml_stack.setup as setup

    binary = tmp_path / "llama-server"
    binary.write_text("#!/bin/sh\nif [ \"$1\" = --version ]; then echo 'version: 0.3.0'; fi\n")
    binary.chmod(0o755)

    assert setup._build_label(str(binary)) == "version: 0.3.0"


def test_a_build_that_prints_no_help_is_not_accused_of_lacking_anything(tmp_path, monkeypatch):
    """Unknown is not none: a stand-in that says nothing must not be reported as lacking
    every flag there is."""
    import ml_stack.serve.binary as binary_module
    import ml_stack.setup as setup
    from ml_stack.serve import backend

    monkeypatch.setattr(backend, "_FLAGS", {})
    monkeypatch.setattr(setup, "_arches", lambda binary: set())
    silent = tmp_path / "llama-server"
    silent.write_text("#!/bin/sh\nexit 0\n")
    silent.chmod(0o755)
    monkeypatch.setattr(binary_module, "find_binary", lambda *a, **k: silent)
    assert not [f for f in setup.look() if f.name == "flags this build lacks"]


def test_every_command_the_package_installs_is_looked_for_on_path(monkeypatch, tmp_path):
    """A new entry point is invisible until `pip install -e . && pyenv rehash`; a queue
    died on `command not found` for it. Setup names each missing command and the line."""
    import shutil

    import ml_stack.setup as setup

    monkeypatch.setattr(setup, "_checkout", lambda: tmp_path)
    monkeypatch.setattr(shutil, "which",
                        lambda name, *a, **k: None if name == "ml-stack-ingest"
                        else f"/opt/bin/{name}")
    found = [f for f in setup.look() if f.name == "commands on PATH"]
    assert found and not found[0].good
    assert "ml-stack-ingest" in found[0].said
    assert "ml-stack-serve" not in found[0].said, "only what is missing is named"
    assert found[0].fix == f"pip install -e {tmp_path} && pyenv rehash"

    monkeypatch.setattr(shutil, "which", lambda name, *a, **k: f"/opt/bin/{name}")
    found = [f for f in setup.look() if f.name == "commands on PATH"]
    assert found and found[0].good
    assert "ml-stack-serve" in found[0].note


def test_the_commands_are_read_from_the_install_and_the_checkouts_pyproject(monkeypatch,
                                                                           tmp_path):
    """The installed metadata says what was installed; the checkout's pyproject says what
    should be -- and the difference is the command nobody can run yet."""
    import ml_stack.setup as setup

    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "ml-stack"\n[project.scripts]\n'
        'ml-stack-serve = "ml_stack.serve.cli:main"\n'
        'ml-stack-newthing = "ml_stack.newthing:main"\n')
    monkeypatch.setattr(setup, "_checkout", lambda: tmp_path)
    names = setup._scripts()
    assert {"ml-stack-serve", "ml-stack-setup", "ml-stack-ingest"} <= set(names)
    assert "ml-stack-newthing" in names, "named in the checkout, not yet installed"
    assert names == sorted(set(names))


def test_the_printed_report_names_the_missing_command_and_the_line(monkeypatch, capsys,
                                                                   tmp_path):
    import shutil

    import ml_stack.setup as setup

    monkeypatch.setattr(setup, "_checkout", lambda: tmp_path)
    monkeypatch.setattr(shutil, "which",
                        lambda name, *a, **k: None if name == "ml-stack-jobs"
                        else f"/opt/bin/{name}")
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    assert setup.main(["--quiet"]) == 1
    out = capsys.readouterr().out
    assert "ml-stack-jobs" in out
    assert f"fix: pip install -e {tmp_path} && pyenv rehash" in out
