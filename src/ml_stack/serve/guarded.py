"""What the Broker asks sentinel before a model server starts: a quarantined model or server is
refused, a pinned model must still hash to its pin, and an unpinned model is pinned on first use."""

from __future__ import annotations

import logging
import weakref
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

from ml_stack import sentinel
from ml_stack.files import sha256_file
from ml_stack.fleet.onboard.trusted import TrustedLists
from ml_stack.sentinel import Sentinel
from ml_stack.sentinel.adapters import broker_listener, serve_hooks
from ml_stack.sentinel.events import Event, Severity
from ml_stack.sentinel.findings import HIGH, finding
from ml_stack.sentinel.servers import unmanaged_findings
from ml_stack.sentinel.watch import Scanner
from ml_stack.serve import canaries, decoy
from ml_stack.serve.backend import ServerFailed
from ml_stack.serve.decoy import DecoyListener
from ml_stack.serve.leases import recorded_servers

logger = logging.getLogger(__name__)

__all__ = ["Armed", "SentinelRefused", "arm", "blocked", "caller_blocked", "file_of", "register",
           "report", "start", "unmanaged_seen", "verify"]


class SentinelRefused(ServerFailed):
    """Sentinel holds the model, or the model no longer matches its pin."""


_HOOKED: weakref.WeakKeyDictionary[Sentinel, set[str]] = weakref.WeakKeyDictionary()


def file_of(model: object) -> Path | None:
    """The model file ``model`` names, or None when it is not a file on this machine."""
    try:
        path = Path(str(model)).expanduser()
        return path if path.is_file() else None
    except (OSError, ValueError):
        return None


def blocked(model: object) -> str:
    """Why sentinel keeps ``model`` from being served, or an empty string. Reads the store
    only; nothing is hashed."""
    node = sentinel.default()
    if node.mode == sentinel.Mode.OFF:
        return ""
    path = Path(str(model)).expanduser()
    if node.store.blocked("model", str(path)):
        return (f"{path.name} is quarantined by sentinel; a person releases it with "
                f"`ml-stack-security release`")
    return ""


def report(event: str, **fields: object) -> None:
    """Tell sentinel what the Broker did (``lease``, ``refused``...), by ``caller``. Every event
    is logged and counts toward the caller's resource use; refusals add to its score."""
    node = sentinel.default()
    if node.mode != sentinel.Mode.OFF:
        broker_listener(node)(event, dict(fields))


def caller_blocked(caller: str) -> str:
    """Why sentinel keeps ``caller`` from leasing, or an empty string."""
    node = sentinel.default()
    if node.mode != sentinel.Mode.OFF and caller and node.store.blocked("caller", caller):
        return f"{caller} is quarantined by sentinel; a person releases it with `ml-stack-security release`"
    return ""


def unmanaged_seen(procs: Iterable[Mapping[str, Any]]) -> None:
    """Report listeners ml-stack did not start (watched, never acted on)."""
    node = sentinel.default()
    if node.mode != sentinel.Mode.OFF:
        node.handle_all(unmanaged_findings(procs))


def register(node: Sentinel, state_file: Path, stop: Callable[[int], object]) -> None:
    """Make a quarantined model or server stop the servers ml-stack started for it. Once per
    sentinel and lease file."""
    done = _HOOKED.setdefault(node, set())
    if str(state_file) in done:
        return
    done.add(str(state_file))
    serve_hooks(node, lambda: recorded_servers(state_file), stop)


class Armed:
    """What `arm` started: the scan loop (with the canary round) and the decoy listener."""

    def __init__(self, scanner: Scanner | None, decoy: DecoyListener | None) -> None:
        self.scanner, self.decoy = scanner, decoy

    def stop(self) -> None:
        if self.scanner is not None:
            self.scanner.stop()
        if self.decoy is not None:
            self.decoy.stop()


def arm(manager: Any, broker: Any = None) -> Armed:
    """What a process that runs the Broker arms: the decoys, the stop hooks for ``manager``'s
    servers, the periodic scan with the behavioural canary round (through ``broker``'s leases
    when given, else over the servers the lease file records) and the decoy endpoint. The caller
    calls `Armed.stop`."""
    node = sentinel.armed()
    register(node, manager.state_file, manager.reclaim)
    targets = broker.canary_targets if broker is not None \
        else canaries.lease_file_targets(manager.state_file)
    return start(node, targets)


def start(node: Sentinel, targets: Callable[[], list[canaries.Target]]) -> Armed:
    """The scan loop, canary round and decoy listener of ``node``; ``targets`` lists the
    served models the canaries ask."""
    rounds = []
    if (round_ := canaries.schedule(node, targets)) is not None:
        rounds.append(round_)
    return Armed(sentinel.arm_scan(node, rounds), decoy.arm(node))


def _by_manifest(node: Sentinel, path: Path, digest: str) -> str:
    """Check ``path`` (whose bytes hash to ``digest``) against the signed lists this machine
    has accepted. '' when no list names the file (or none can be read); ``verified by manifest
    serial N`` when one does and agrees; `SentinelRefused` after quarantining when lists name it
    and none lists these bytes."""
    try:
        listed = TrustedLists().lookup(path.name)
    except (OSError, ImportError, ValueError) as exc:
        logger.warning("signed lists not consulted for %s: %s", path.name, exc)
        return ""
    if not listed:
        return ""
    for one in listed:
        if one.entry.sha256 == digest:
            text = f"verified by manifest serial {one.serial}"
            node.bus.emit(Event("model.manifest_verified", Severity.INFO, "serve",
                                f"model:{path}", {"name": path.name, "serial": one.serial,
                                                  "key_id": one.key_id}, node.clock()))
            return text
    found = finding("integrity.manifest_mismatch", Severity.CRITICAL, ("model", str(path)), HIGH,
                    {"sha256": digest, "listed": sorted({o.entry.sha256 for o in listed}),
                     "serials": sorted({o.serial for o in listed})},
                    path=str(path), move="file")
    node.handle(found)
    if node.mode == sentinel.Mode.OBSERVE:
        return ""
    raise SentinelRefused(f"{path.name} is not what the signed manifest lists for it and is "
                          f"quarantined by sentinel; the file was moved aside")


def verify(model: object, *, state_file: Path, stop: Callable[[int], Any]) -> str:
    """Refuse to start a server for ``model`` when sentinel holds it, its bytes differ from
    the pin, or a signed manifest this machine accepted lists other bytes for it. A model with
    no pin is pinned now with source ``first-use`` (a model ml-stack pulled has its pin from
    the pull). Returns ``verified by manifest serial N`` when a manifest vouched for the file,
    else ''."""
    node = sentinel.armed()
    register(node, state_file, stop)
    if node.mode == sentinel.Mode.OFF:
        return ""
    if why := blocked(model):
        raise SentinelRefused(why)
    path = file_of(model)
    if path is None:
        return ""
    pin = node.manifest.pins().get(str(path))
    if pin is None:
        digest = sha256_file(path)
        said = _by_manifest(node, path, digest)
        node.manifest.pin_verified(path, "model", digest, source="first-use",
                                   digest_from="computed")
        logger.warning("%s was not pulled by ml-stack and had no pin: pinned on first use "
                       "(its bytes are trusted as they are now)", path.name)
        node.bus.emit(Event("model.pinned_first_use", Severity.NOTICE, "serve", f"model:{path}",
                            {"name": path.name, "manifest": said}, node.clock()))
        return said
    if not node.verify_before_load(path, cached=True):
        raise SentinelRefused(f"{path.name} does not match its pin and is quarantined by "
                              f"sentinel; the file was moved aside")
    return _by_manifest(node, path, pin.sha256)
