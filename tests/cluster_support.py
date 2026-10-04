"""A cluster on this machine without the network: the same words and group make the same key."""

from __future__ import annotations

import base64
import hashlib
from collections.abc import Sequence
from pathlib import Path

import pytest

from ml_stack.fleet.discovery import DEFAULT_CLUSTER, Membership, _write_memberships, memberships
from tests.keystore_support import counting  # noqa: F401

KEEPING = False
"""Set by `a_keystore`: while it is, a test that joins also stores the passphrase."""


def any_command(argv: Sequence[str]) -> list[str]:
    """What a test daemon runs for ``POST /jobs``: whatever it is sent."""
    return list(argv)


def key_for(words: str, group: str = DEFAULT_CLUSTER) -> bytes:
    """The key a test cluster named ``group`` with passphrase ``words`` has."""
    raw = hashlib.sha256(f"test-cluster/{group}/{words.strip()}".encode()).digest()
    return base64.urlsafe_b64encode(raw).rstrip(b"=")


def join(words: str, *, group: str = "", path: Path | str | None = None) -> list[Membership]:
    """Add the cluster ``group`` that ``words`` make; with `a_keystore` in force, keep the passphrase too."""
    from ml_stack.fleet import recovery

    group = group or DEFAULT_CLUSTER
    _write_memberships([*[m for m in memberships(path) if m.group != group],
                        Membership(group=group, key=key_for(words, group))], path)
    if KEEPING:
        recovery.remember(words, group, path, say=lambda _line: None)
    return memberships(path)


def join_cluster(words: str, *, group: str = DEFAULT_CLUSTER, path: Path | str | None = None) -> bytes:
    """Join ``group`` and answer as it; the key."""
    join(words, group=group, path=path)
    return key_for(words, group)


@pytest.fixture
def a_keystore(tmp_path, counting, monkeypatch):  # noqa: F811
    """The passphrase store over a fake keyring backend; the machine's own is never touched."""
    from ml_stack.fleet import recovery
    from ml_stack.keystore import Keystore, Wires

    store = Keystore(directory=tmp_path / "ks", wires=Wires(interactive=lambda: True, sleep=lambda _s: None))
    monkeypatch.setattr(recovery, "_store", lambda: store)
    monkeypatch.setitem(globals(), "KEEPING", True)
    return store
