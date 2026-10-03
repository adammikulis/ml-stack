"""Fake keyring backends for the keystore tests: one in memory that counts and can refuse, one
over a file that separate processes share and that logs every call."""

from __future__ import annotations

import json
import os
from pathlib import Path

import keyring
import pytest
from keyring.backend import KeyringBackend
from keyring.compat import properties
from keyring.errors import KeyringError, PasswordDeleteError

from ml_stack import keystore


class CountingRing(KeyringBackend):
    """The keyring interface over a dict; ``calls`` lists every call and ``refuse`` makes them fail."""

    priority = 1  # type: ignore[assignment]

    def __init__(self) -> None:
        super().__init__()
        self.held: dict[tuple[str, str], str] = {}
        self.calls: list[str] = []
        self.refuse: type[Exception] | None = None
        self.refuse_ops: set[str] | None = None

    def _enter(self, name: str) -> None:
        self.calls.append(name)
        if self.refuse is not None and (self.refuse_ops is None or name in self.refuse_ops):
            raise self.refuse("declined")

    def get_password(self, service, username):
        self._enter("get")
        return self.held.get((service, username))

    def set_password(self, service, username, password):
        self._enter("set")
        self.held[(service, username)] = password

    def delete_password(self, service, username):
        self._enter("delete")
        if (service, username) not in self.held:
            raise PasswordDeleteError(username)
        del self.held[(service, username)]

    def count(self, name: str) -> int:
        return self.calls.count(name)


class CountingFileRing(KeyringBackend):
    """Entries in the JSON file named by ``ML_STACK_TEST_KEYRING``; each call appends a line to
    ``<file>.calls``. Inert while the variable is unset."""

    @properties.classproperty
    def priority(cls) -> float:
        return 5 if "ML_STACK_TEST_KEYRING" in os.environ else 0

    @staticmethod
    def _file() -> Path:
        return Path(os.environ["ML_STACK_TEST_KEYRING"])

    def _note(self, name: str) -> None:
        with open(f"{self._file()}.calls", "a") as out:
            out.write(name + "\n")

    def _all(self) -> dict:
        try:
            return json.loads(self._file().read_text())
        except (OSError, ValueError):
            return {}

    def get_password(self, service, username):
        self._note("get")
        return self._all().get(f"{service}/{username}")

    def set_password(self, service, username, password):
        self._note("set")
        rows = self._all()
        rows[f"{service}/{username}"] = password
        self._file().write_text(json.dumps(rows))

    def delete_password(self, service, username):
        self._note("delete")
        rows = self._all()
        if f"{service}/{username}" not in rows:
            raise PasswordDeleteError(username)
        del rows[f"{service}/{username}"]
        self._file().write_text(json.dumps(rows))


@pytest.fixture
def counting():
    before = keyring.get_keyring()
    fake = CountingRing()
    keyring.set_keyring(fake)
    yield fake
    keyring.set_keyring(before)


__all__ = ["CountingFileRing", "CountingRing", "KeyringError", "counting", "keystore"]
