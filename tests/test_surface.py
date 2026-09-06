"""``ml-stack-surface``: the walk, what it is safe to run, the normalising and the diff."""

from __future__ import annotations

from pathlib import Path

import pytest

from ml_stack.surface import ops
from ml_stack.surface.cli import main

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "src"


def test_every_safe_invocation_names_a_command_this_checkout_installs():
    installed = set(ops.installed(SRC))
    named = {(spoken.split() or [""])[0] for spoken in ops.SAFE}
    assert named <= installed, f"SAFE names commands that do not exist: {named - installed}"


def test_the_umbrella_is_one_of_the_commands_walked():
    assert "" in ops.installed(SRC)


@pytest.mark.slow
def test_a_command_is_walked_with_every_subcommand_it_declares():
    found = [node.spoken for node in ops.walk(SRC, ["world"])]
    assert found == ["world", "world make", "world questions", "world simulate",
                     "world emit", "world check"]


@pytest.mark.slow
def test_a_command_with_no_subcommands_is_a_single_node():
    found = ops.walk(SRC, ["claude"])
    assert [node.spoken for node in found] == ["claude"]
    assert "usage: ml-stack-claude" in found[0].help


@pytest.mark.slow
def test_a_parser_that_requires_a_subcommand_refuses_a_bare_argv():
    by_words = {node.spoken: node for node in ops.walk(SRC, ["world"])}
    assert by_words["world"].refuses
    assert by_words["world make"].refuses


@pytest.mark.slow
def test_a_command_that_cannot_be_asked_is_recorded_rather_than_skipped():
    found = ops.walk(SRC, ["no-such-command"])
    assert len(found) == 1
    assert found[0].help.startswith("[")


def _node(spoken: str, refuses: bool) -> ops.Node:
    return ops.Node(tuple(spoken.split()), f"usage: {spoken}\n", refuses)


def test_a_refused_parser_is_run_bare_and_an_accepting_one_is_not():
    nodes = [_node("world", True), _node("world make", True), _node("claude", False)]
    made = {run.slug: run.verdict for run in ops.invocations(nodes)}
    assert made == {"world": "refused", "world.make": "refused"}


def test_a_declared_invocation_is_run_where_no_parser_refuses():
    made = {run.slug: run.words for run in ops.invocations([_node("serve", True)])}
    assert made["serve.status"] == ("serve", "status")
    assert made["serve.fit"] == ("serve", "fit", "--room", "24G")


def test_a_declared_invocation_for_a_command_outside_the_walk_is_left_out():
    assert ops.declared([_node("claude", False)]) == []


def test_nothing_is_run_at_all_when_only_help_is_asked_for():
    assert ops.invocations([_node("world", True)], help_only=True) == []


def test_a_node_with_no_invocation_is_named_as_help_only():
    node = _node("claude", False)
    taken = ops.Taken((node,), ())
    assert taken.help_only == (node,)


def test_what_a_rerun_changes_is_settled_out_of_an_invocation():
    was = ("started 2026-09-05T11:22:33, pid 4821 on 192.168.1.9:8080\n"
           "read 12.5 GiB in 4.2 s, 61% of 24G")
    now = ("started 2026-09-06T09:01:02, pid 91 on 10.0.0.2:8081\n"
           "read 12.5 GiB in 9.9 s, 61% of 24G")
    assert ops.settle(was, SRC, output=True) == ops.settle(now, SRC, output=True)


def test_a_moved_flag_survives_the_settling():
    was = ops.settle("  --room SIZE   how much\n", SRC, output=True)
    now = ops.settle("  --space SIZE  how much\n", SRC, output=True)
    assert was != now


def test_a_default_in_the_help_is_left_as_it_is():
    text = "  --timeout T  seconds to wait (default: 60 s)\n"
    assert ops.settle(text, SRC, output=False) == text


def test_this_machine_is_settled_out_of_the_help():
    text = f"  --home DIR  where the records are (default: {Path.home()}/.ml-stack/jobs)\n"
    assert ops.settle(text, SRC, output=False) == (
        "  --home DIR  where the records are (default: ~/.ml-stack/jobs)\n")


@pytest.mark.slow
def test_two_captures_of_the_same_commands_do_not_differ(tmp_path):
    before, after = tmp_path / "before", tmp_path / "after"
    ops.capture(before, src=SRC, only=["world"])
    ops.capture(after, src=SRC, only=["world"])
    assert ops.compare(before, after) == []


@pytest.mark.slow
def test_a_capture_writes_the_help_of_every_node_and_says_what_it_ran(tmp_path):
    taken = ops.capture(tmp_path, src=SRC, only=["world"])
    written = {path.name for path in (tmp_path / "help").iterdir()}
    assert written == {f"{node.slug}.txt" for node in taken.nodes}
    assert (tmp_path / "manifest.json").exists()
    assert {run.verdict for run in taken.runs} == {"refused"}


@pytest.mark.slow
def test_a_changed_flag_shows_up_in_the_diff(tmp_path):
    before, after = tmp_path / "before", tmp_path / "after"
    ops.capture(before, src=SRC, only=["world"])
    ops.capture(after, src=SRC, only=["world"])
    moved = after / "help" / "world.make.txt"
    moved.write_text(moved.read_text().replace("--seed", "--sown"))
    changed = ops.compare(before, after)
    assert [one.name for one in changed] == ["world make --help"]
    assert "-  --seed" in changed[0].diff and "+  --sown" in changed[0].diff


@pytest.mark.slow
def test_the_command_captures_and_then_says_nothing_differs(tmp_path, capsys):
    for name in ("before", "after"):
        assert main(["capture", "--out", str(tmp_path / name), "--only", "world",
                     "--help-only"]) == 0
    capsys.readouterr()
    assert main(["diff", str(tmp_path / "before"), str(tmp_path / "after")]) == 0
    assert "nothing differs" in capsys.readouterr().out


@pytest.mark.slow
def test_the_command_exits_1_when_a_capture_differs(tmp_path, capsys):
    for name in ("before", "after"):
        main(["capture", "--out", str(tmp_path / name), "--only", "world", "--help-only"])
    gone = tmp_path / "after" / "help" / "world.emit.txt"
    gone.unlink()
    capsys.readouterr()
    assert main(["diff", str(tmp_path / "before"), str(tmp_path / "after")]) == 1
    assert "world emit --help" in capsys.readouterr().out


@pytest.mark.slow
def test_listing_says_what_would_be_run_and_what_would_not(tmp_path, capsys):
    assert main(["list", "--only", "claude,world"]) == 0
    printed = capsys.readouterr().out
    assert "claude" in printed and "not declared safe" in printed
    assert "world make" in printed and "bare" in printed
