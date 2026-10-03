"""Model files from the owner's paired devices before the internet (docs/model-discovery.md).

``hub.pull`` asks `session()`. For each file the session asks every device in the `PeerBook`
(`hub.peerbook`) for its signed manifest over TLS pinned to its certificate, reaching it by `routes`
(this network, then its tailnet address). The **pin** is the Hub listing's sha256 (`Wanted`); a peer's
digest never replaces it. Chunks are checked against the manifest, the whole file against the pin,
then the net scan and quarantine. A peer that sends a bad byte is dropped and sentinel hears of it.
Speed: peers are ranked by the rate measured before, a peer gets at most its ``streams`` requests and
``limit_bps`` bytes a second, a ``metered`` peer is not asked, and a transfer under `FLOOR_BPS` for
`WINDOW_S` seconds is given up for the Hub. Anything that goes wrong answers False.
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
from urllib.parse import urlsplit

from ml_stack import macauth, sentinel
from ml_stack.hub import origins, peers as hub_peers
from ml_stack.hub.peerbook import PeerBook
from ml_stack.net import Blocked, Want, sniff
from ml_stack.net.download import accept
from ml_stack.net.hold import staging_dir
from ml_stack.sentinel.events import Event, Severity

from .. import tls
from ..tailnet import Tailnet, detect
from .events import BUS, Bus
from .manifest import Entry, Manifest, ManifestError, verify
from .peerlearn import share_url
from .requests import Device, Devices
from .routes import Probe, Route, pinned_probe, reach
from .sharing import NEVER, OWNER
from .transfer import (
    Cancelled,
    Downloader,
    PeerSource,
    Settings,
    TooSlow,
    TransferError,
    fetch_manifest,
)

__all__ = ["PeerBook", "PeerSession", "quarantine_veto", "session"]

logger = logging.getLogger(__name__)

ASK_TIMEOUT = 4.0
"""Seconds a peer has to answer for its manifest, and for each chunk."""
FLOOR_BPS = 2 << 20
"""Bytes a second the transfer must keep up; under it for `WINDOW_S`, the Hub is used."""
WINDOW_S = 15.0
UNMEASURED = float("inf")
"""A peer never measured is tried first, so that it gets measured."""


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
    rate: float


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
    probe: Probe = pinned_probe
    tailnet: Callable[[], Tailnet] = lambda: detect()   # looked up when called, so a test can replace detect
    devices: Callable[[], list[Device]] | None = None
    """The paired devices (default: the ones beside the peer book)."""
    origins_path: Path | None = None

    # -- who, and where --
    def _paired(self) -> list[Device]:
        if self.devices is not None:
            return self.devices()
        return Devices(self.book.path.with_name("devices.json")).all()

    def _address(self, row: dict[str, Any]) -> str | None:
        """The URL to reach ``row`` at. A row that came from pairing names a device, whose route
        is chosen now (this network, then the tailnet) and must answer with its pinned
        certificate; a row added by hand is used as it stands. None, with a note, when the
        device is revoked or reachable by neither."""
        url, name = str(row["url"]), str(row["name"])
        fingerprint = str(row.get("fingerprint") or "")
        if not fingerprint:
            return url
        device = next((d for d in self._paired() if d.fingerprint == fingerprint), None)
        if device is None or device.status != "active":
            self.notes.append(f"{name}: no longer a paired device")
            return None
        port = urlsplit(url).port or 443
        found = reach(device, self.tailnet, port, self.probe)
        if found.route is Route.UNREACHABLE:
            self.notes.append(f"{name}: not reachable on the network or the tailnet")
            return None
        return share_url(found.address, port)

    def _source(self, row: dict[str, Any], url: str) -> PeerSource:
        secret = ""
        if row.get("device_secret"):
            secret = macauth.derive(base64.urlsafe_b64decode(
                str(row["device_secret"]) + "=" * (-len(str(row["device_secret"])) % 4)))
        elif row.get("cluster_key"):
            secret = macauth.derive(str(row["cluster_key"]).encode())
        context = tls.pinned_context(str(row["certificate"])) if row.get("certificate") else None
        return PeerSource(url, secret, context, name=str(row["name"]),
                          limit_bps=float(row.get("limit_bps") or 0),
                          streams=int(row.get("streams") or 0))

    def _ask(self, row: dict[str, Any]) -> tuple[PeerSource, Manifest, float, float] | None:
        """The peer, its verified manifest, how long it took to answer and its measured rate; or
        None with the reason in ``notes``."""
        name = str(row["name"])
        if row.get("metered"):
            self.notes.append(f"{name}: on a metered connection, not asked")
            return None
        try:
            url = self._address(row)
            if url is None:
                return None
            peer = self._source(row, url)
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
        return peer, manifest, took, float(row.get("rate") or 0) or UNMEASURED

    def _offers(self, wanted: hub_peers.Wanted) -> tuple[str, list[_Offer]]:
        """``(pin, offers)``: the digest the file must have and the peers whose signed manifest
        lists that file with that digest, the fastest measured first."""
        rows = [r for r in self.book.rows() if r["name"] not in self.bad]
        if not rows:
            return "", []
        with ThreadPoolExecutor(max_workers=min(8, len(rows))) as pool:
            asked = [a for a in pool.map(self._ask, rows) if a is not None]
        for peer, manifest, _took, _rate in asked:
            self.book.seen_serial(peer.name, manifest.serial)
        pin = wanted.sha256.lower()
        found: list[_Offer] = []
        for peer, manifest, took, rate in asked:
            for entry in manifest.entries:
                if entry.name != wanted.name or (wanted.size and entry.size != wanted.size):
                    continue
                if entry.sharing == NEVER:
                    self.notes.append(f"{peer.name}: {entry.name} may not be shared")
                    continue
                if pin and entry.sha256 != pin:
                    self.notes.append(f"{peer.name}: lists {entry.name} with a different digest")
                    continue
                found.append(_Offer(peer, entry, took, rate))
        if not pin and found:
            pin = found[0].entry.sha256
            found = [o for o in found if o.entry.sha256 == pin]
        found.sort(key=lambda o: (-o.rate, o.seconds))
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
        settings = Settings(bus=self.bus, strikes=1, timeout=ASK_TIMEOUT, cancel=cancelled,
                            progress=progress, floor_bps=FLOOR_BPS, window_s=WINDOW_S)
        try:
            staged = Downloader(manifest, peers, stage, settings=settings).download(entry.name)
        except Cancelled as exc:
            self._measured(peers)
            raise hub_peers.Stopped(wanted.path) from exc
        except TooSlow as exc:
            self._measured(peers)           # the verified chunks stay for a later try
            self.notes.append(f"{wanted.name}: {exc}; using the Hub")
            return False
        except TransferError as exc:
            self._measured(peers)
            return self._gave_up(peers, stage, wanted, exc)
        self._measured(peers)
        phase("verifying")
        who = max(peers, key=lambda p: p.moved[0])
        try:
            accept(staged, final, Want(
                kind=sniff.expected_kind(wanted.name), sha256=pin, size=entry.size,
                purpose="peer download"), f"peer:{who.name}")
        except Blocked as exc:
            self._distrust(peers, wanted, f"the finished file was refused: {exc}")
            shutil.rmtree(stage, ignore_errors=True)
            return False
        shutil.rmtree(stage, ignore_errors=True)
        self._remember_origin(entry, pin, peers)
        return True

    def _measured(self, peers: list[PeerSource]) -> None:
        """Keep each peer's rate from this transfer for the next pull's ranking."""
        for peer in peers:
            moved, seconds = peer.moved
            if moved >= 1 << 20 and seconds > 0:
                self.book.record_rate(peer.name, moved / seconds)

    def _remember_origin(self, entry: Entry, pin: str, peers: list[PeerSource]) -> None:
        """Append who the file came from, and for a gated model whose acceptance of its licence
        this device relied on, to the local record `ml-stack-models list` shows."""
        giver = max(peers, key=lambda p: p.moved[0])
        row = next((r for r in self.book.rows() if r["name"] == giver.name), {})
        line: dict[str, Any] = {
            "file": entry.name, "repo": entry.repo, "size": entry.size, "sha256": pin,
            "peer": giver.name, "peer_fingerprint": str(row.get("fingerprint") or ""),
            "peers": sorted(p.name for p in peers if p.moved[0]), "sharing": entry.sharing}
        if entry.sharing == OWNER and entry.licence:
            line["relies_on"] = {"device": giver.name, "licence": entry.licence,
                                 "licence_url": entry.licence_url,
                                 "accepted_by": giver.accepted.get("by", ""),
                                 "accepted_at": giver.accepted.get("at", "")}
        try:
            origins.record(line, self.origins_path or self.book.path.with_name("peer-downloads.jsonl"))
        except OSError as exc:
            logger.warning("could not write the peer download record: %s", exc)

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
