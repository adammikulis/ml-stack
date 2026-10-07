"""Redacted native hook failure diagnostics."""

import io

import pytest

from ml_stack import harnesshook
from ml_stack.workspace.identity import Denied


@pytest.mark.parametrize("reason, expected", [
    ("the token expired", "Denied: the token expired"),
    ("native roots must belong to the selected canonical project",
     "Denied: native roots must belong to the selected canonical project"),
    ("token file /private/person/session: secret-token-value",
     "Denied: workspace credential file could not be loaded"),
    ("project board unavailable: https://private-host/?token=secret-token-value",
     "Denied: project board is unavailable"),
    ("unknown refusal with secret-token-value", "Denied: workspace authorization refused"),
])
def test_run_reports_authorization_category_without_private_values(monkeypatch, capsys, reason, expected):
    def refused(*_args):
        raise Denied(reason)

    monkeypatch.setattr(harnesshook, "post", refused)
    output = io.StringIO()
    assert harnesshook.run(["post"], io.StringIO("{}"), output) == 2
    diagnostic = capsys.readouterr().err
    assert expected in diagnostic
    assert "secret-token-value" not in diagnostic
    assert "/private/person" not in diagnostic
    assert output.getvalue() == ""


def test_unexpected_errors_do_not_expose_exception_contents(monkeypatch, capsys):
    def crashed(*_args):
        raise ValueError("secret-token-value")

    monkeypatch.setattr(harnesshook, "post", crashed)
    assert harnesshook.run(["post"], io.StringIO("{}"), io.StringIO()) == 2
    assert capsys.readouterr().err == "ml-stack hook failed, call blocked: ValueError\n"


def test_unhandled_denial_reports_redacted_reason_and_blocks(monkeypatch, capsys):
    exits = []
    monkeypatch.setattr(harnesshook.os, "_exit", exits.append)
    harnesshook._block(Denied, Denied("the token expired"), None)
    assert exits == [2]
    assert capsys.readouterr().err == "ml-stack hook failed, call blocked: Denied: the token expired\n"
