"""Credential management through Fleet never returns or repeats a saved value."""

from pathlib import Path

import pytest
from launch_support import sign_in, ticket
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
    cookie = sign_in(server)
    for name, value in values.items():
        status, result, _ = server.call("/ui/credentials", method="POST", cookie=cookie,
                                        body={"name": name, "value": value})
        assert status == 200 and result["saved"] == name
        assert value not in str(result)

    status, result, _ = server.call("/ui/credentials", cookie=cookie)
    assert status == 200 and {row["name"] for row in result["credentials"]} >= set(values)
    assert not any(value in str(result) for value in values.values())
    assert credentials.get("HF_TOKEN") == values["HF_TOKEN"]
    path = credentials.file_path()
    assert Path(path).stat().st_mode & 0o077 == 0

    status, result, _ = server.call("/ui/credentials", method="DELETE", cookie=cookie, body={"name": "HF_TOKEN"})
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


@pytest.mark.redteam
@pytest.mark.parametrize("body", [None, [], {"name": "HF_TOKEN", "value": []}, {"name": "HF_TOKEN"}])
def test_credential_schema_and_origin_fail_before_persistence(server, body):
    cookie = sign_in(server)
    status, result, _ = server.call("/ui/credentials", method="POST", body=body, cookie=cookie)
    assert status == 400 and "error" in result
    assert not credentials.file_path().exists()
    status, _, _ = server.call("/ui/credentials", method="POST", cookie=cookie,
        body={"name": "HF_TOKEN", "value": "isolated"}, headers={"Origin": "https://foreign.example"})
    assert status == 403 and not credentials.file_path().exists()


@pytest.mark.slow
def test_person_saves_and_removes_a_credential_in_settings(server, playwright):
    from playwright.sync_api import expect

    server.ui.settings.setup_done = True
    with playwright.chromium.launch(headless=True) as browser:
        page = browser.new_page()
        page.goto(f"http://127.0.0.1:{server.port}/ui/?launch_ticket={ticket(server)[1]['ticket']}#settings")
        page.get_by_role("tab", name="Credentials", exact=True).click()
        panel = page.locator("#settings-credentials")
        panel.get_by_label("Credential name", exact=True).fill("OPENAI_API_KEY")
        panel.get_by_label("Token or API key", exact=True).fill("isolated-browser-secret")
        panel.get_by_role("button", name="Save credential", exact=True).click()
        expect(panel.get_by_text("OPENAI_API_KEY · credentials file", exact=False)).to_be_visible()
        expect(panel.get_by_label("Token or API key", exact=True)).to_have_value("")
        assert "isolated-browser-secret" not in page.content()
        panel.get_by_label("Credential name", exact=True).fill("OPENAI_API_KEY")
        panel.get_by_role("button", name="Remove credential", exact=True).click()
        expect(panel.get_by_text("OPENAI_API_KEY · credentials file", exact=False)).to_have_count(0)
        assert credentials.get("OPENAI_API_KEY") is None


@pytest.mark.redteam
@pytest.mark.parametrize("method", ["POST", "DELETE"])
@pytest.mark.parametrize("access", ["token-session", "authorization", "token-header"])
def test_agent_access_cannot_change_person_credentials(server, method, access):
    credentials.set("HF_TOKEN", "isolated-person-secret")
    before = credentials.file_path().read_bytes()
    cookie = ""
    headers = {}
    if access == "token-session":
        session = server.ui.sessions.open("token", "token")
        cookie = server.ui.sessions.cookie_header(session).split(";", 1)[0]
    elif access == "authorization":
        headers["Authorization"] = "Bearer isolated-agent-token"
    else:
        headers["X-ML-Stack-Token"] = "isolated-agent-token"
    body = {"name": "HF_TOKEN"}
    if method == "POST":
        body["value"] = "isolated-agent-secret"
    status, result, _ = server.call("/ui/credentials", method=method,
                                   body=body, cookie=cookie, headers=headers)
    assert status == 403 and "person" in result["error"]
    assert credentials.file_path().read_bytes() == before
