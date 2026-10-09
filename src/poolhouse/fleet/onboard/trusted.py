"""The signed file lists this machine accepted, kept for the load-time check (docs/sentinel.md).

`TrustedLists.ingest` keeps a list only if a key a person pinned signed it, it has not expired and its serial is not
below the highest kept for that key (replay). `lookup` verifies each kept list again under the keys pinned now, so an
un-paired or revoked key stops counting. Expiry is not applied at lookup: it bounds fetching, not checking held bytes.
"""

from __future__ import annotations

import base64
import binascii
import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from poolhouse import home
from poolhouse.files import read_json
from poolhouse.sentinel.sealed import SealedFile
from poolhouse.sentinel.store import sentinel_dir

from .manifest import Entry, Manifest, ManifestError, key_fingerprint, verify

__all__ = ["Listed", "TrustedLists", "pinned_keys", "remember"]

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Listed:
    """A file named by an accepted list: its ``entry`` and the list's ``serial`` and key."""

    entry: Entry
    serial: int
    key_id: str


def _key(text: Any) -> bytes | None:
    try:
        raw = base64.b64decode(str(text), validate=True)
    except (binascii.Error, ValueError):
        return None
    return raw if len(raw) == 32 else None


def pinned_keys(directory: Path | None = None) -> tuple[list[bytes], list[str]]:
    """``(keys, revoked key ids)`` a person pinned on this machine: the owner's key of each
    paired peer, the key in ``trust.json`` and this machine's own signing key."""
    where = Path(directory) if directory else home.state("onboard")
    keys: list[bytes] = []
    revoked: list[str] = []
    peers = read_json(where / "peers.json", {})
    for row in peers.get("peers", []) if isinstance(peers, dict) else []:
        if isinstance(row, dict) and (k := _key(row.get("signing_key"))) is not None:
            keys.append(k)
    for name in ("trust.json", "signing.json"):
        doc = read_json(where / name, {})
        if not isinstance(doc, dict):
            continue
        for field in ("signing_key", "public"):
            if (k := _key(doc.get(field))) is not None:
                keys.append(k)
        revoked += [str(r) for r in doc.get("revoked", []) if isinstance(r, str)]
    return list(dict.fromkeys(keys)), revoked


class TrustedLists:
    """The accepted lists, sealed at ``path`` (default: sentinel's state directory)."""

    def __init__(self, path: Path | None = None,
                 keys: Callable[[], tuple[list[bytes], list[str]]] = pinned_keys) -> None:
        self._file = SealedFile(path if path is not None else sentinel_dir() / "signed-lists.json")
        self._keys = keys

    def _lists(self) -> dict[str, dict[str, Any]]:
        loaded = self._file.load()
        if loaded.status == "tampered":
            logger.warning("the signed-list store failed its seal; no list is used")
        lists = loaded.payload.get("lists", {})
        return lists if isinstance(lists, dict) else {}

    def high_water(self, key_id: str) -> int:
        """The highest serial accepted so far for ``key_id`` (-1 when none)."""
        return int(self._lists().get(key_id, {}).get("serial", -1))

    def ingest(self, raw: bytes, pinned: bytes, *, now: float | None = None,
               revoked_keys: tuple[str, ...] = ()) -> Manifest:
        """Accept ``raw`` if ``pinned`` signed it, it has not expired and its serial is not
        below the highest accepted for that key. `ManifestError` otherwise (an unsigned list
        and an old one alike), and nothing is stored."""
        held = self.high_water(key_fingerprint(pinned))
        manifest = verify(raw, pinned, now=now, min_serial=max(held, 0),
                          revoked_keys=revoked_keys)
        lists = self._lists()
        lists[key_fingerprint(pinned)] = {"serial": manifest.serial, "raw": raw.decode("utf-8")}
        self._file.save({"lists": lists})
        return manifest

    def lookup(self, name: str) -> list[Listed]:
        """Every entry named ``name`` in a kept list that still verifies under a key pinned
        now, newest serial first. A kept list whose key is gone or revoked, or whose
        signature no longer holds, is skipped."""
        lists = self._lists()
        if not lists:
            return []
        keys, revoked = self._keys()
        by_id = {key_fingerprint(k): k for k in keys}
        out: list[Listed] = []
        for key_id, row in lists.items():
            pinned = by_id.get(key_id)
            if pinned is None:
                continue
            try:
                manifest = verify(str(row["raw"]).encode(), pinned, now=0.0,
                                  min_serial=int(row.get("serial", 0)), revoked_keys=revoked)
            except (ManifestError, KeyError, TypeError, ValueError):
                logger.warning("a kept signed list for key %s no longer verifies; ignored",
                               key_id[:16])
                continue
            out += [Listed(e, manifest.serial, key_id) for e in manifest.entries if e.name == name]
        return sorted(out, key=lambda x: -x.serial)


def remember(raw: bytes, pinned: bytes) -> None:
    """Keep a list that was just verified for a fetch, for the load-time check. Never raises:
    a list that is old or unsigned is logged and dropped."""
    try:
        TrustedLists().ingest(raw, pinned)
    except (ManifestError, OSError, ImportError) as exc:
        logger.warning("signed list not kept: %s", exc)
