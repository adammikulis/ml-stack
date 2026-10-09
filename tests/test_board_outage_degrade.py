"""A board that said no still says no, and a project board that cannot be reached is an error, not a silent success."""

import argparse

from ml_stack.http import ServerError
from ml_stack.workspace import cli
from ml_stack.workspace.identity import BoardUnavailable, Denied


def _run(cmd, error):
    def run(_args):
        raise error
    return cli._guarded(run)(argparse.Namespace(cmd=cmd, json=False))


def test_an_unreachable_project_board_and_real_refusals_fail_for_every_command(capsys):
    for cmd in ("send", "join", "release"):
        assert _run(cmd, BoardUnavailable("project board unavailable: connection refused")) == 3
    assert "was not recorded" not in capsys.readouterr().err
    assert _run("announce", Denied("this token may not announce")) == 3
    assert _run("announce", ServerError("HTTP 400", status=400)) == 3
