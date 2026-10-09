"""A cluster on this machine without the network: the same words and group make the same key."""

from __future__ import annotations

import base64
import hashlib
from collections.abc import Sequence
from pathlib import Path

import pytest

from poolhouse.fleet.discovery import DEFAULT_CLUSTER, Membership, _write_memberships, memberships
from poolhouse.fleet.onboard.joining import join_secret
from tests.keystore_support import counting  # noqa: F401


def any_command(argv: Sequence[str]) -> list[str]:
    """What a test daemon runs for ``POST /jobs``: whatever it is sent."""
    return list(argv)


def key_for(words: str, group: str = DEFAULT_CLUSTER) -> bytes:
    """The key a test cluster named ``group`` with passphrase ``words`` has."""
    # a passphrase goes through a password-hashing function, as in the real join; the cost
    # is the smallest scrypt takes, because a test makes many of these
    raw = hashlib.scrypt(words.strip().encode(), salt=f"test-cluster/{group}".encode(),
                         n=16, r=1, p=1, dklen=32)
    return base64.urlsafe_b64encode(raw).rstrip(b"=")


def join(words: str, *, group: str = "", path: Path | str | None = None) -> list[Membership]:
    """Add the cluster ``group`` that ``words`` make, knowing the passphrase as a machine that joined with it does."""
    group = group or DEFAULT_CLUSTER
    _write_memberships([*[m for m in memberships(path) if m.group != group],
                        Membership(group=group, key=key_for(words, group), join=join_secret(words, group))], path)
    return memberships(path)


def join_cluster(words: str, *, group: str = DEFAULT_CLUSTER, path: Path | str | None = None) -> bytes:
    """Join ``group`` and answer as it; the key."""
    join(words, group=group, path=path)
    return key_for(words, group)


@pytest.fixture
def a_keystore(tmp_path, counting, monkeypatch):  # noqa: F811
    """The passphrase store over a fake keyring backend; the machine's own is never touched."""
    from poolhouse.fleet import recovery
    from poolhouse.keystore import Keystore, Wires

    store = Keystore(directory=tmp_path / "ks", wires=Wires(interactive=lambda: True, sleep=lambda _s: None))
    monkeypatch.setattr(recovery, "_store", lambda: store)
    return store
