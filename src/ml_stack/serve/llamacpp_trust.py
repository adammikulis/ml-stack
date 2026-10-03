"""What sentinel holds for a managed llama-server: its pin, and the check before it is used."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from ml_stack import sentinel
from ml_stack.files import sha256_file
from ml_stack.sentinel.events import Severity
from ml_stack.sentinel.findings import HIGH, finding

logger = logging.getLogger(__name__)

__all__ = ["held", "pin", "problem", "unpin"]


def _recorded(binary: Path) -> dict:
    try:
        info = json.loads((binary.parent / "BUILD.json").read_text())
    except (OSError, ValueError):
        return {}
    return info if isinstance(info, dict) else {}


def held(binary: Path) -> bool:
    """Whether sentinel has quarantined ``binary``."""
    node = sentinel.default()
    return node.mode != sentinel.Mode.OFF and node.store.blocked("binary", str(binary))


def pin(binary: Path, sha256: str, *, source: str = "build") -> None:
    """Record ``sha256`` as the digest of ``binary``, a binary this machine just built."""
    node = sentinel.default()
    node.manifest.pin_verified(binary, "binary", sha256, source=source, digest_from="computed")


def unpin(binary: Path) -> None:
    sentinel.default().manifest.unpin(binary)


def problem(binary: Path) -> str:
    """Why ``binary`` may not be used, or an empty string.

    A binary sentinel holds is refused. A pinned binary must hash to its pin. A build this
    code made carries the digest it was installed with in ``BUILD.json``: with no pin, the
    file must hash to that digest and is pinned; otherwise it is quarantined. Any other binary
    is pinned on first use.
    """
    node = sentinel.default()
    if node.mode == sentinel.Mode.OFF:
        return ""
    if held(binary):
        return (f"{binary} is quarantined by sentinel; a person releases it with "
                f"`ml-stack-security release`")
    if not binary.is_file():
        return ""
    pins = node.manifest.pins()
    if str(binary) in pins:
        if node.verify_before_load(binary, cached=True):
            return ""
        return (f"{binary} does not match its pin and is quarantined by sentinel; the file "
                f"was moved aside. `ml-stack-serve llama-cpp rollback` switches to the previous build")
    recorded = str(_recorded(binary).get("sha256") or "")
    digest = sha256_file(binary)
    if not recorded or recorded == digest:
        node.manifest.pin_verified(binary, "binary", digest, source="build" if recorded else "first-use",
                                   digest_from="expected" if recorded else "computed")
        if not recorded:
            logger.warning("%s had no pin: pinned on first use", binary)
        return ""
    found = finding("integrity.binary_mismatch", Severity.CRITICAL, ("binary", str(binary)), HIGH,
                    {"sha256": digest, "recorded": recorded}, path=str(binary), move="file")
    node.handle(found)
    if node.mode == sentinel.Mode.OBSERVE:
        return ""
    return (f"{binary} is not the binary that was built (BUILD.json records another digest) and "
            f"is quarantined by sentinel; the file was moved aside")
