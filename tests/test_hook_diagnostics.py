"""Redacted native hook failure diagnostics."""

import io

import pytest

from poolhouse import harnesshook
from poolhouse.workspace.identity import Denied


@pytest.mark.parametrize("reason, expected", [
    ("the token expired", "Denied: the token expired"),
    ("native roots must belong to the selected project",
     "Denied: native roots must belong to the selected project"),
    ("token file /fixture/session: secret-token-value",
     "Denied: token file /fixture/session:"),
    ("project board unavailable: https://private-host/?token=secret-token-value",
     "Denied: project board unavailable:"),
    ("unknown refusal with secret-token-value", "Denied: unknown refusal with"),
])
def test_run_reports_authorization_category_without_private_values(monkeypatch, capsys, reason, expected):
    monkeypatch.setenv("POOLHOUSE_TEST_TOKEN", "secret-token-value")

    def refused(*_args):
        raise Denied(reason)

    monkeypatch.setattr(harnesshook, "post", refused)
    output = io.StringIO()
    assert harnesshook.run(["post"], io.StringIO("{}"), output) == 0
    diagnostic = capsys.readouterr().err
    assert expected in diagnostic
    assert "secret-token-value" not in diagnostic
    assert "diagnostic=" in output.getvalue()
    assert "secret-token-value" not in output.getvalue()


def test_unexpected_errors_do_not_expose_exception_contents(monkeypatch, capsys):
    monkeypatch.setenv("POOLHOUSE_TEST_TOKEN", "secret-token-value")

    def crashed(*_args):
        raise ValueError("secret-token-value")

    monkeypatch.setattr(harnesshook, "post", crashed)
    assert harnesshook.run(["post"], io.StringIO("{}"), io.StringIO()) == 0
    diagnostic = capsys.readouterr().err
    assert "poolhouse hook failed, notification unavailable:" in diagnostic
    assert "ValueError:" in diagnostic
    assert "secret-token-value" not in diagnostic


def test_unhandled_denial_reports_redacted_reason_and_blocks(monkeypatch, capsys):
    exits = []
    monkeypatch.setattr(harnesshook.os, "_exit", exits.append)
    secret = "mlws1.worker." + "a" * 43
    harnesshook._block(Denied, Denied("the token expired " + secret), None)
    assert exits == [2]
    diagnostic = capsys.readouterr().err
    assert "Denied: the token expired" in diagnostic
    assert secret not in diagnostic
