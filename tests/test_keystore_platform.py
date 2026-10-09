"""The keystore on every platform: who may cause a prompt, what never hangs, which backends may hold the
key, and what happens when the key is gone. No test touches the machine's own keystore: each installs the
in-memory fake from `tests/keystore_support.py` (see AGENTS.md, "ML_STACK_NO_REAL_KEYSTORE")."""

from __future__ import annotations

import sys

import keyring
import pytest
from keyring.backend import KeyringBackend

from ml_stack import keystore, keystore_guard
from tests import keystore_support

counting = keystore_support.counting
REAL_INTERACTIVE = keystore.interactive


def test_only_windows_has_a_service_session(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(keystore_guard, "_session_id", lambda: 0)
    assert keystore_guard.service_session() is False


@pytest.mark.skipif(sys.platform != "darwin", reason="the Security framework is macOS's")
def test_macos_can_be_told_to_fail_rather_than_show_a_dialog():
    import ctypes
    framework = keystore_guard._security_framework()
    allowed = ctypes.c_ubyte(1)
    try:
        assert keystore_guard.forbid_prompts() is True
        assert framework.SecKeychainGetUserInteractionAllowed(ctypes.byref(allowed)) == 0
        assert allowed.value == 0
    finally:
        framework.SecKeychainSetUserInteractionAllowed(ctypes.c_ubyte(1))


@pytest.mark.skipif(sys.platform == "darwin", reason="macOS is where it acts")
def test_elsewhere_forbidding_prompts_does_nothing():
    assert keystore_guard.forbid_prompts() is False


def test_bounded_returns_the_value_and_carries_the_exception():
    assert keystore_guard.bounded(lambda: 7, 5) == 7
    with pytest.raises(ZeroDivisionError):
        keystore_guard.bounded(lambda: 1 / 0, 5)


class PlaintextRing(KeyringBackend):
    """Stands in for `keyrings.alt.file.PlaintextKeyring`: a usable backend that writes the key to a file."""

    __module__ = "keyrings.alt.file"
    priority = 1  # type: ignore[assignment]

    def __init__(self) -> None:
        super().__init__()
        self.calls: list[str] = []

    def get_password(self, service, username):
        self.calls.append("get")

    def set_password(self, service, username, password):
        self.calls.append("set")

    def delete_password(self, service, username):
        self.calls.append("delete")


@pytest.fixture
def plaintext():
    before, ring = keyring.get_keyring(), PlaintextRing()
    keyring.set_keyring(ring)
    yield ring
    keyring.set_keyring(before)


def test_a_chainer_holding_a_plaintext_backend_is_refused_too(tmp_path, plaintext):
    from types import SimpleNamespace
    assert keystore_guard.unsafe_backend(SimpleNamespace(backends=[PlaintextRing()]))
