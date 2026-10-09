"""Shared pieces of the onboarding tests: real certificates, a movable clock, a recorder for
the event bus. Nothing here replaces a network or a file system."""

from __future__ import annotations

import json
import os
from pathlib import Path

from keyring.backend import KeyringBackend
from keyring.compat import properties
from keyring.errors import PasswordDeleteError

from poolhouse.fleet import tls
from poolhouse.fleet.onboard.events import Bus, Event
from poolhouse.fleet.onboard.pairing import Grant
from poolhouse.fleet.onboard.requests import Limits, Requests


class Clock:
    def __init__(self, start: float = 1_000_000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class Recorder:
    """A bus that remembers what it was told, kind by kind."""

    def __init__(self) -> None:
        self.bus = Bus()
        self.events: list[Event] = []
        self.bus.subscribe(self.events.append)

    def kinds(self) -> list[str]:
        return [e.kind for e in self.events]

    def of(self, kind: str) -> list[Event]:
        return [e for e in self.events if e.kind == kind]


def identity(root: Path, name: str) -> tls.Identity:
    return tls.identity(root / name, name)


def requests(root: Path, rec: Recorder, clock: Clock | None = None,
             limits: Limits | None = None) -> Requests:
    return Requests(root / "requests.json", limits=limits, bus=rec.bus,
                    clock=clock or Clock())


def info(fingerprint: str, nonce: str = "ab" * 16, **more: str) -> dict[str, str]:
    return {"name": "kitchen-pi", "hostname": "kitchen.local", "model": "Linux aarch64",
            "fingerprint": fingerprint, "nonce": nonce, **more}


def grant_for(ident: tls.Identity) -> Grant:
    return Grant(group="home", key="secret-cluster-key", certificate=ident.beacon)


class FileKeyring(KeyringBackend):
    """A real keyring backend whose entries live in the JSON file named by
    ``POOLHOUSE_TEST_KEYRING``: stands in for the Keychain so tests never touch it, and lets
    separate processes share one keystore."""

    @properties.classproperty
    def priority(cls) -> float:
        """Not a candidate while no test names a file: keyring's chainer lists every backend
        class it has seen, and an unset variable must not make this one answer for the others."""
        if "POOLHOUSE_TEST_KEYRING" not in os.environ:
            return 0  # the chainer keeps only backends above zero
        return 5

    @staticmethod
    def _file() -> Path:
        return Path(os.environ["POOLHOUSE_TEST_KEYRING"])

    def _all(self) -> dict:
        try:
            return json.loads(self._file().read_text())
        except (OSError, ValueError):
            return {}

    def get_password(self, service, username):
        return self._all().get(f"{service}/{username}")

    def set_password(self, service, username, password):
        rows = self._all()
        rows[f"{service}/{username}"] = password
        self._file().write_text(json.dumps(rows))

    def delete_password(self, service, username):
        rows = self._all()
        if f"{service}/{username}" not in rows:
            raise PasswordDeleteError(username)
        del rows[f"{service}/{username}"]
        self._file().write_text(json.dumps(rows))
