"""A keystore held in memory for the memory tests, installed through keyring's own interface so
the real key code runs against it."""

from __future__ import annotations

import keyring
import pytest
from keyring.backend import KeyringBackend


class MemoryRing(KeyringBackend):
    """The keyring interface over a dict."""

    priority = 1  # type: ignore[assignment]

    def __init__(self) -> None:
        super().__init__()
        self.held: dict[tuple[str, str], str] = {}

    def get_password(self, service, username):
        return self.held.get((service, username))

    def set_password(self, service, username, password):
        self.held[(service, username)] = password

    def delete_password(self, service, username):
        self.held.pop((service, username), None)


@pytest.fixture(autouse=True)
def ring():
    before = keyring.get_keyring()
    fake = MemoryRing()
    keyring.set_keyring(fake)
    yield fake
    keyring.set_keyring(before)
