"""The keystore on every platform: who may cause a prompt, what never hangs, which backends may hold the
key, and what happens when the key is gone. No test touches the machine's own keystore: each installs the
in-memory fake from `tests/keystore_support.py` (see AGENTS.md, "ML_STACK_NO_REAL_KEYSTORE")."""

from __future__ import annotations

import io
import os
import stat
import sys
import threading
import time
from pathlib import Path

import keyring
import pytest
from keyring.backend import KeyringBackend

from ml_stack import keystore, keystore_guard, person
from ml_stack.keystore import Keystore, Wires
from tests import keystore_support
from tests.test_keystore import Said, make

counting = keystore_support.counting
REAL_INTERACTIVE = keystore.interactive


class Tty(io.StringIO):
    def isatty(self) -> bool:
        return True


@pytest.fixture
def at_a_desktop(monkeypatch):
    """A terminal on stdin and a desktop session, and no agent marker."""
    for marker in person.AGENT_MARKERS:
        monkeypatch.delenv(marker, raising=False)
    monkeypatch.setattr(sys, "stdin", Tty())
    monkeypatch.setattr(keystore, "_desktop", lambda: True)


# -- who may cause a prompt ---------------------------------------------------------------------------


def test_a_person_at_a_desktop_is_interactive(at_a_desktop):
    assert REAL_INTERACTIVE() is True


@pytest.mark.parametrize("marker", person.AGENT_MARKERS)
def test_every_agent_marker_makes_a_process_background_whatever_its_terminal_and_desktop(at_a_desktop, monkeypatch,
                                                                                         marker):
    monkeypatch.setenv(marker, "1")
    assert REAL_INTERACTIVE() is False


def test_an_agent_on_a_desktop_cannot_make_the_master(tmp_path, counting, at_a_desktop, monkeypatch):
    monkeypatch.setenv("CLAUDECODE", "1")
    monkeypatch.setattr(keystore, "interactive", REAL_INTERACTIVE)
    ks = Keystore(directory=tmp_path / "ks", wires=Wires(say=Said()))
    with pytest.raises(keystore.KeystoreLocked, match="ml-stack-security unlock"):
        ks.subkey("memory", "a")
    assert counting.calls == [] and counting.held == {}


@pytest.mark.parametrize(("session", "expected"), [(0, False), (1, True), (None, True)])
def test_a_windows_service_session_has_no_desktop(monkeypatch, session, expected):
    def session_id() -> int:
        if session is None:
            raise OSError("no answer")
        return session

    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(keystore_guard, "_session_id", session_id)
    assert keystore._desktop() is expected


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


def test_a_background_call_forbids_prompts_and_a_person_s_does_not(tmp_path, counting, monkeypatch):
    asked: list[str] = []
    monkeypatch.setattr(keystore_guard, "forbid_prompts", lambda: asked.append("forbid") or True)
    make(tmp_path).subkey("memory", "a")
    assert asked == []
    make(tmp_path, person=False).subkey("memory", "a")
    assert asked == ["forbid"]


# -- nothing hangs ---------------------------------------------------------------------------------------


class StuckRing(KeyringBackend):
    """A backend whose read waits on a dialog nobody can answer."""

    priority = 1  # type: ignore[assignment]

    def __init__(self) -> None:
        super().__init__()
        self.release = threading.Event()
        self.reads = 0

    def get_password(self, service, username):
        self.reads += 1
        self.release.wait(30)

    def set_password(self, service, username, password):
        raise AssertionError("nothing may be stored behind a stuck read")

    def delete_password(self, service, username):
        raise AssertionError("nothing may be deleted behind a stuck read")


@pytest.fixture
def stuck():
    before, ring = keyring.get_keyring(), StuckRing()
    keyring.set_keyring(ring)
    yield ring
    ring.release.set()
    keyring.set_keyring(before)


@pytest.mark.parametrize("person_here", [True, False])
def test_a_backend_that_never_answers_is_refused_for_good_not_waited_on(tmp_path, stuck, monkeypatch, person_here):
    monkeypatch.setattr(keystore_guard, "PERSON_WAIT_S", 0.3)
    monkeypatch.setattr(keystore_guard, "BACKGROUND_WAIT_S", 0.3)
    ks = make(tmp_path, person=person_here)
    if not person_here:
        ks._put_doc("provisioned.json", {"at": 1.0})
    began = time.monotonic()
    with pytest.raises(keystore.KeystoreDenied):
        ks.subkey("memory", "a")
    assert time.monotonic() - began < 5
    with pytest.raises(keystore.KeystoreDenied):
        ks.subkey("memory", "a")
    assert stuck.reads == 1, "a refusal is remembered; the stuck call is not repeated"
    assert (tmp_path / "ks" / "denied.json").exists()


def test_bounded_returns_the_value_and_carries_the_exception():
    assert keystore_guard.bounded(lambda: 7, 5) == 7
    with pytest.raises(ZeroDivisionError):
        keystore_guard.bounded(lambda: 1 / 0, 5)


# -- which backend may hold the key -----------------------------------------------------------------------


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


def test_a_plaintext_file_keyring_never_holds_the_master(tmp_path, plaintext):
    ks = make(tmp_path)
    assert ks.available() is False
    with pytest.raises(keystore.KeystoreUnavailable, match="plain file"):
        ks.subkey("memory", "a")
    assert plaintext.calls == []
    assert not (tmp_path / "ks" / "provisioned.json").exists()


def test_a_chainer_holding_a_plaintext_backend_is_refused_too(tmp_path, plaintext):
    from types import SimpleNamespace
    assert keystore_guard.unsafe_backend(SimpleNamespace(backends=[PlaintextRing()]))


def test_the_libsecret_backend_counts_as_the_machines_own_keystore():
    class Libsecret(KeyringBackend):
        __module__ = "keyring.backends.libsecret"
        priority = 1  # type: ignore[assignment]

        def get_password(self, service, username):
            return None

        def set_password(self, service, username, password):
            return None

    assert keystore.is_real(Libsecret())


# -- the key is gone ----------------------------------------------------------------------------------------


def test_a_lost_master_is_not_quietly_replaced_by_a_second_one(tmp_path, counting):
    first = make(tmp_path).subkey("memory", "a")
    counting.held.clear()
    counting.calls.clear()
    with pytest.raises(keystore.KeystoreMissing, match="no longer holds the key"):
        make(tmp_path).subkey("memory", "a")
    assert counting.count("set") == 0 and counting.held == {}
    again = make(tmp_path)
    assert again.provision() is True, "a person who accepts starting over may"
    assert again.subkey("memory", "a") != first


# -- modes ---------------------------------------------------------------------------------------------------


@pytest.mark.skipif(os.name != "posix", reason="POSIX modes")
def test_the_state_directory_is_private_even_when_something_else_made_it_first(tmp_path, counting):
    directory = tmp_path / "ks"
    directory.mkdir(mode=0o755)
    directory.chmod(0o755)
    make(tmp_path).subkey("memory", "a")
    assert stat.S_IMODE(directory.stat().st_mode) == 0o700
    for held in (f for f in directory.iterdir() if f.suffix != ".lock"):
        assert stat.S_IMODE(held.stat().st_mode) & 0o077 == 0, held.name
    assert Path(directory / "flight.lock").exists()
