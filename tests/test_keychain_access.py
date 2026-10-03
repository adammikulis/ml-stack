"""Keychain enrollment, cancellation, and read-only status behavior."""

from types import SimpleNamespace

import keyring
import pytest
from onboard_support import FileKeyring

from ml_stack import credentials
from ml_stack.credentials import keychain
from ml_stack.fleet.onboard import signing, signing_cli


@pytest.fixture(autouse=True)
def isolated_keychain(monkeypatch, tmp_path):
    monkeypatch.setattr(keychain, "_BLOCKED", False)
    monkeypatch.setenv("ML_STACK_HOME", str(tmp_path / "state"))
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    monkeypatch.setenv("ML_STACK_TEST_KEYRING", str(tmp_path / "keyring.json"))
    monkeypatch.setattr(keyring, "get_keyring", lambda: FileKeyring())


def test_background_lookup_does_not_discover_or_query_native_keychain(monkeypatch):
    calls = []
    monkeypatch.setattr(credentials, "_keyring", lambda: calls.append("discovery"))
    assert credentials.get("HF_TOKEN") is None
    credentials.status("HF_TOKEN")
    credentials.describe()
    assert calls == []


def test_enrolled_keychain_is_used_and_status_never_reads_secret(monkeypatch):
    values = {}
    calls = []
    backend = SimpleNamespace(errors=keyring.errors,
                              set_password=lambda service, name, value: values.update({name: value}),
                              get_password=lambda service, name: calls.append(name) or values.get(name))
    monkeypatch.setattr(credentials, "_keyring", lambda: backend)
    credentials.set("HF_TOKEN", "invented-token", keychain=True)
    assert credentials.status("HF_TOKEN") == {"present": True, "source": "keychain", "checked": False}
    credentials.describe()
    assert calls == []
    assert credentials.get("HF_TOKEN") == "invented-token"
    assert calls == ["HF_TOKEN"]


def test_cancelled_reads_are_latched_until_explicit_retry(monkeypatch):
    calls = []

    def cancelled(*_args):
        calls.append("read")
        raise keyring.errors.KeyringLocked("cancelled")

    backend = SimpleNamespace(errors=keyring.errors, get_password=cancelled,
                              set_password=lambda *_: None)
    monkeypatch.setattr(credentials, "_keyring", lambda: backend)
    credentials.set("HF_TOKEN", "invented-token", keychain=True)
    for _ in range(3):
        with pytest.raises(credentials.CredentialError, match="explicitly retry"):
            credentials.get("HF_TOKEN")
    assert calls == ["read"]
    with pytest.raises(credentials.CredentialError):
        credentials.get("HF_TOKEN", keychain=True)
    assert calls == ["read", "read"]


def test_write_failure_is_actionable_and_does_not_expose_native_message(monkeypatch):
    def fail(*_):
        raise keyring.errors.PasswordSetError("invented-private-token")

    monkeypatch.setattr(credentials, "_keyring", lambda: SimpleNamespace(
        errors=keyring.errors, set_password=fail))
    with pytest.raises(credentials.CredentialError, match="default keychain") as error:
        credentials.set("HF_TOKEN", "invented-private-token", keychain=True)
    assert "invented-private-token" not in str(error.value)
    assert not credentials.file_path().exists()


def test_signing_show_does_not_create_a_key_or_discover_keychain(monkeypatch, tmp_path):
    monkeypatch.setattr(signing, "keyring_usable", lambda: pytest.fail("read-only keychain discovery"))
    keys = signing.SigningKeys(tmp_path / "signing")
    args = SimpleNamespace(action="show", json=True)
    assert signing_cli._run(args, keys, tmp_path / "signing") == 0
    assert keys.peek() == {}
    assert not keys.meta_path.exists()


def test_signing_write_failure_never_falls_back_or_repeats(monkeypatch, tmp_path):
    calls = []

    def fail(*_):
        calls.append("write")
        raise keyring.errors.PasswordSetError("native failure")

    monkeypatch.setattr(signing, "keyring_usable", lambda: True)
    monkeypatch.setattr(keyring, "set_password", fail)
    keys = signing.SigningKeys(tmp_path / "signing")
    for _ in range(2):
        with pytest.raises(signing.KeyStoreError, match="default keychain"):
            keys.meta()
    assert calls == ["write"]
    assert not keys.meta_path.exists()
    assert not keys.file_path.exists()
