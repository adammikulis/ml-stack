"""Model files from the owner's paired devices before the internet (docs/model-discovery.md).

``hub.pull`` asks `session()`. For each file the session asks every device in the `PeerBook`
(TLS pinned to its certificate, `lan.require_local` addresses only) for its signed manifest and
uses those that list that very file. The digest the bytes must have is the **pin**, the Hub
listing's sha256 in `Wanted`; a digest a peer states never replaces it (only without any
listing digest does a manifest entry signed by the owner's pinned key supply one). Chunks are
checked against the manifest, the whole file against the pin, then it passes the net scan and
quarantine (`net.download.accept`). A peer that sends a bad byte is dropped for the pull, the
partial file is deleted and a critical event goes to the bus and to sentinel. Sharing levels
and quarantine are enforced by the serving device (`sharing.py`, `quarantine_veto`).
Anything that goes wrong answers False and the Hub is used.
"""

from __future__ import annotations

import base64
import binascii
import logging
import shutil
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ml_stack import home, macauth, sentinel
from ml_stack.files import read_json, write_json
from ml_stack.hub import peers as hub_peers
from ml_stack.net import Blocked, Want, sniff
from ml_stack.net.download import accept
from ml_stack.net.hold import staging_dir
from ml_stack.platform import private_file
from ml_stack.sentinel.events import Event, Severity

from .. import tls
from .events import BUS, Bus
from .manifest import Entry, Manifest, ManifestError, verify
from .sharing import NEVER
from .transfer import (
    Cancelled,
    Downloader,
    PeerSource,
    Settings,
    TransferError,
    fetch_manifest,
)

__all__ = ["PeerBook", "PeerSession", "quarantine_veto", "session"]

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1
ASK_TIMEOUT = 4.0
"""Seconds a peer has to answer for its manifest, and for each chunk."""
MIN_RATE = 2 << 20
"""Bytes a second a peer must average (plus ``GRACE`` seconds); slower, and the Hub is used."""
GRACE = 30.0


class PeerBook:
    """The devices to ask, in a private file beside the pairing records.

    Each row: ``name``, ``url`` (https://host:port of its ``fleet share``), ``certificate``
    (its beacon, which the TLS connection is pinned to), ``signing_key`` (the owner's manifest
    key, base64 raw Ed25519 public key), ``device_secret`` (this device's request key from
    pairing, urlsafe base64), ``min_serial`` (the lowest manifest serial still accepted: an
    older list cannot be served again).
    """

    def __init__(self, path: Path | str | None = None) -> None:
        self.path = Path(path) if path else home.state("onboard", "peers.json")

    def _doc(self) -> dict[str, Any]:
        doc = read_json(self.path, {})
        return doc if isinstance(doc, dict) else {}

    def _save(self, doc: dict[str, Any]) -> None:
        write_json(self.path, {"schema_version": SCHEMA_VERSION, **doc})
        private_file(self.path)

    @property
    def enabled(self) -> bool:
        return self._doc().get("enabled", True) is not False

    def set_enabled(self, on: bool) -> None:
        self._save({**self._doc(), "enabled": on})

    def rows(self) -> list[dict[str, Any]]:
        rows = self._doc().get("peers", [])
        return [r for r in rows if isinstance(r, dict) and r.get("name") and r.get("url")]

    def add(self, row: dict[str, Any]) -> None:
        doc = self._doc()
        kept = [r for r in self.rows() if r["name"] != row["name"]]
        self._save({**doc, "peers": [*kept, row]})

    def remove(self, name: str) -> bool:
        doc, rows = self._doc(), self.rows()
        left = [r for r in rows if r["name"] != name]
        self._save({**doc, "peers": left})
        return len(left) != len(rows)

    def seen_serial(self, name: str, serial: int) -> None:
        """Remember the newest manifest serial ``name`` has shown."""
        rows = [{**r, "min_serial": max(int(r.get("min_serial", 0)), serial)}
                if r["name"] == name else r for r in self.rows()]
        self._save({**self._doc(), "peers": rows})


def quarantine_veto(entry: Entry) -> str:
    """The serving side's veto: why a file sentinel holds in quarantine is not handed out.
    A download is quarantined under its sha256 (`ml_stack.net.hold`)."""
    try:
        if sentinel.default().store.blocked("artifact", f"download:{entry.sha256}"):
            return "this copy is in quarantine on the serving device"
    except (OSError, ImportError, ValueError) as exc:
        logger.warning("could not ask sentinel about %s, withholding it: %s", entry.name, exc)
        return "the serving device could not check its quarantine"
    return ""


class _Relay(Bus):
    """A bus whose events also go to the process-wide onboarding bus and to sentinel."""

    def emit(self, kind: str, severity: str, subject: str = "", **evidence: Any):  # type: ignore[no-untyped-def]
        event = super().emit(kind, severity, subject, **evidence)
        BUS.emit(kind, severity, subject, **evidence)
        try:
            sentinel.default().bus.emit(Event(kind, Severity.parse(severity), "onboard", subject,
                                              dict(evidence), event.ts))
        except (OSError, ImportError, ValueError) as exc:
            logger.debug("no sentinel event for %s: %s", kind, exc)
        return event


@dataclass(slots=True)
class _Offer:
    peer: PeerSource
    entry: Entry
    seconds: float


@dataclass(slots=True)
class PeerSession:
    """The peers of one pull."""

    book: PeerBook
    bus: Bus = field(default_factory=_Relay)
    staging: Path | None = None
    bad: set[str] = field(default_factory=set)
    """Peers that sent bad bytes during this pull."""
    notes: list[str] = field(default_factory=list)
    """Why a peer was not used, for the log and for tests."""

    # -- who has it --
    def _source(self, row: dict[str, Any]) -> PeerSource:
        secret = ""
        if row.get("device_secret"):
            secret = macauth.derive(base64.urlsafe_b64decode(
                str(row["device_secret"]) + "=" * (-len(str(row["device_secret"])) % 4)))
        elif row.get("cluster_key"):
            secret = macauth.derive(str(row["cluster_key"]).encode())
        context = tls.pinned_context(str(row["certificate"])) if row.get("certificate") else None
        return PeerSource(str(row["url"]), secret, context, name=str(row["name"]))

    def _ask(self, row: dict[str, Any]) -> tuple[PeerSource, Manifest, float] | None:
        """The peer and its verified manifest, or None with the reason in ``notes``."""
        name = str(row["name"])
        try:
            peer = self._source(row)
            began = time.monotonic()
            raw = fetch_manifest(peer, ASK_TIMEOUT)
            took = time.monotonic() - began
            key = base64.b64decode(str(row["signing_key"]), validate=True)
            manifest = verify(raw, key, min_serial=int(row.get("min_serial", 0)))
        except ManifestError as exc:
            self.notes.append(f"{name}: manifest refused: {exc}")
            self.bus.emit("onboard.peer.bad_manifest", "warning", f"peer:{name}", reason=str(exc))
            return None
        except (OSError, TransferError, KeyError, ValueError, binascii.Error) as exc:
            self.notes.append(f"{name}: not asked: {type(exc).__name__}: {exc}")
            return None
        return peer, manifest, took

    def _offers(self, wanted: hub_peers.Wanted) -> tuple[str, list[_Offer]]:
        """``(pin, offers)``: the digest the file must have and the peers whose signed manifest
        lists that file with that digest."""
        rows = [r for r in self.book.rows() if r["name"] not in self.bad]
        if not rows:
            return "", []
        with ThreadPoolExecutor(max_workers=min(8, len(rows))) as pool:
            asked = [a for a in pool.map(self._ask, rows) if a is not None]
        for peer, manifest, _took in asked:
            self.book.seen_serial(peer.name, manifest.serial)
        pin = wanted.sha256.lower()
        found: list[_Offer] = []
        for peer, manifest, took in asked:
            for entry in manifest.entries:
                if entry.name != wanted.name or (wanted.size and entry.size != wanted.size):
                    continue
                if entry.sharing == NEVER:
                    self.notes.append(f"{peer.name}: {entry.name} may not be shared")
                    continue
                if pin and entry.sha256 != pin:
                    self.notes.append(f"{peer.name}: lists {entry.name} with a different digest")
                    continue
                found.append(_Offer(peer, entry, took))
        if not pin and found:
            pin = found[0].entry.sha256
            found = [o for o in found if o.entry.sha256 == pin]
        found.sort(key=lambda o: o.seconds)
        return pin, found

    # -- fetching --
    def fetch(self, wanted: hub_peers.Wanted, final: Path, *, cancelled: Callable[[], bool],
              progress: Callable[[int], None], phase: Callable[[str], None]) -> bool:
        pin, offers = self._offers(wanted)
        if not offers:
            return False
        same = [o for o in offers if o.entry == offers[0].entry]
        entry, peers = same[0].entry, [o.peer for o in same]
        stage = (self.staging or staging_dir() / "peers") / pin[:32]
        manifest = Manifest(0, 0.0, 0.0, "", (entry,))
        settings = Settings(bus=self.bus, strikes=1, timeout=ASK_TIMEOUT, cancel=cancelled, progress=progress,
                            deadline_s=GRACE + entry.size / MIN_RATE)
        try:
            staged = Downloader(manifest, peers, stage, settings=settings).download(entry.name)
        except Cancelled as exc:
            raise hub_peers.Stopped(wanted.path) from exc
        except TransferError as exc:
            return self._gave_up(peers, stage, wanted, exc)
        phase("verifying")
        try:
            accept(staged, final, Want(
                kind=sniff.expected_kind(wanted.name), sha256=pin, size=entry.size,
                purpose="peer download"), f"peer:{peers[0].name}")
        except Blocked as exc:
            self._distrust(peers, wanted, f"the finished file was refused: {exc}")
            shutil.rmtree(stage, ignore_errors=True)
            return False
        shutil.rmtree(stage, ignore_errors=True)
        return True

    def _gave_up(self, peers: list[PeerSource], stage: Path, wanted: hub_peers.Wanted,
                 why: Exception) -> bool:
        liars = [p for p in peers if p.lies]
        if liars:
            self._distrust(liars, wanted, str(why))
            shutil.rmtree(stage, ignore_errors=True)
        self.notes.append(f"{wanted.name}: peers could not supply it: {why}")
        return False

    def _distrust(self, peers: list[PeerSource], wanted: hub_peers.Wanted, why: str) -> None:
        for peer in peers:
            self.bad.add(peer.name)
            self.bus.emit("onboard.peer.bad_copy", "critical", f"peer:{peer.name}",
                          file=wanted.name, repo=wanted.repo, reason=why[:200])


def session() -> PeerSession | None:
    """The peer session for a pull, or None when peers are off or none is stored."""
    book = PeerBook()
    if not book.enabled or not book.rows():
        return None
    return PeerSession(book)
