"""A project board that cannot be reached warns and carries on; a board that said no still says no."""

import argparse

from ml_stack.http import ServerError
from ml_stack.workspace import cli
from ml_stack.workspace.identity import BoardUnavailable, Denied


def _run(cmd, error):
    def run(_args):
        raise error
    return cli._guarded(run)(argparse.Namespace(cmd=cmd, json=False))


def test_advisory_commands_succeed_with_a_warning_when_the_board_is_unreachable(capsys):
    for cmd in ("announce", "claim", "heartbeat", "release"):
        assert _run(cmd, BoardUnavailable("project board unavailable: connection refused")) == 0
    assert "was not recorded" in capsys.readouterr().err


def test_other_commands_and_real_refusals_still_fail(capsys):
    assert _run("send", BoardUnavailable("project board unavailable: connection refused")) == 3
    assert _run("announce", Denied("this token may not announce")) == 3
    assert _run("announce", ServerError("HTTP 400", status=400)) == 3
