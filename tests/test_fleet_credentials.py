"""Credential management through Fleet never returns or repeats a saved value."""

from pathlib import Path

import pytest
from test_fleet_ui import Serving

from ml_stack import credentials


@pytest.fixture
def server(tmp_path, monkeypatch):
    monkeypatch.setenv("ML_STACK_HOME", str(tmp_path / "state"))
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    made = Serving(tmp_path, secure=False)
    try:
        yield made
    finally:
        made.close()


def test_ui_saves_multiple_named_credentials_without_returning_values(server):
    values = {"HF_TOKEN": "hf-test-one-value", "ANTHROPIC_API_KEY": "sk-test-two-value"}
    for name, value in values.items():
        status, result, _ = server.call("/ui/credentials", method="POST",
                                        body={"name": name, "value": value})
        assert status == 200 and result["saved"] == name
        assert value not in str(result)

    status, result, _ = server.call("/ui/credentials")
    assert status == 200 and {row["name"] for row in result["credentials"]} >= set(values)
    assert not any(value in str(result) for value in values.values())
    assert credentials.get("HF_TOKEN") == values["HF_TOKEN"]
    path = credentials.file_path()
    assert Path(path).stat().st_mode & 0o077 == 0

    status, result, _ = server.call("/ui/credentials", method="DELETE", body={"name": "HF_TOKEN"})
    assert status == 200 and result == {"removed": True, "name": "HF_TOKEN"}
    assert credentials.get("HF_TOKEN") is None
    assert credentials.get("ANTHROPIC_API_KEY") == values["ANTHROPIC_API_KEY"]


def test_joined_credential_management_requires_a_signed_in_session(server):
    from test_fleet_ui import WORDS

    server.call("/ui/setup/done", method="POST")
    server.call("/ui/setup/join", method="POST",
                body={"mode": "create", "passphrase": WORDS, "group": "home"})
    status, result, _ = server.call("/ui/credentials", method="POST",
                                    body={"name": "HF_TOKEN", "value": "hf-private"})
    assert status == 401 and "sign in" in result["error"]
    status, signed, headers = server.call("/ui/session", method="POST",
                                         body={"passphrase": WORDS, "group": "home"})
    assert status == 200 and signed["signed_in"]
    cookie = headers["Set-Cookie"].split(";", 1)[0]
    status, result, _ = server.call("/ui/credentials", method="POST", cookie=cookie,
                                    body={"name": "HF_TOKEN", "value": "hf-private"})
    assert status == 200 and "hf-private" not in str(result)
