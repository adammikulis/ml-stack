"""What each device last showed for a project's tests, and which test files passed there.

A pass is kept per device: the key holds the device's fingerprint, its platform and Python and the
tier, besides the file's own content key, so a pass on Windows never satisfies macOS and a pass on
a device whose Python moved is a different pass.
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable
from pathlib import Path

from poolhouse.files import read_json, write_json

FILE = "remote-results.json"
MOST_PASSES = 4000
CLEAN_EXITS = (0, 1)
"""A run whose files can be trusted one by one: pytest passed, or failed some tests. A timeout or a cancel cannot."""


def key(device: dict, platform: dict, tier: str, file_key: str) -> str:
    """The reuse key of one test file on one device under one tier."""
    where = f"{device['fingerprint']}|{platform.get('system')}|{platform.get('wsl')}|{platform.get('python')}|{tier}|{file_key}"
    return hashlib.sha256(where.encode()).hexdigest()


def clean(counts: dict) -> bool:
    """Whether a file's counts show it ran and nothing failed."""
    return counts.get("passed", 0) > 0 and not counts.get("failed", 0) and not counts.get("error", 0)


class Ledger:
    """The file ``remote-results.json`` in a project's reuse folder."""

    def __init__(self, folder: Path) -> None:
        self.path = Path(folder) / FILE

    def _load(self) -> dict:
        got = read_json(self.path, {})
        return got if isinstance(got, dict) else {}

    def last(self, fingerprint: str) -> dict:
        """The most recent run recorded for a device: when, tier, files, exit, counts, tree; empty when none."""
        return self._load().get(fingerprint, {}).get("last", {})

    def passed(self, fingerprint: str, reuse_key: str) -> dict:
        """The pass kept under ``reuse_key`` for the device, or empty."""
        return self._load().get(fingerprint, {}).get("passes", {}).get(reuse_key, {})

    def record(self, device: dict, summary: dict, passes: dict[str, str]) -> None:
        """Keep a run's summary as the device's last, and ``passes`` (reuse key -> file) as passes."""
        everything = self._load()
        mine = everything.setdefault(device["fingerprint"], {})
        mine["name"] = device["name"]
        mine["last"] = {**summary, "at": time.time()}
        kept = mine.setdefault("passes", {})
        for reuse_key, file in passes.items():
            kept[reuse_key] = {"file": file, "at": time.time(), "tree": summary.get("tree", "")}
        for old in sorted(kept, key=lambda k: kept[k]["at"])[:-MOST_PASSES]:
            del kept[old]
        write_json(self.path, everything)

    def passes_of(self, device: dict, platform: dict, tier: str, result: dict,
                  file_key: Callable[[str], str]) -> dict[str, str]:
        """The files of a clean run that passed, as reuse keys; ``file_key`` is empty for a file that must never be reused."""
        if result.get("exit") not in CLEAN_EXITS:
            return {}
        found = {}
        for file, counts in result.get("files", {}).items():
            content = file_key(file) if clean(counts) else ""
            if content:
                found[key(device, platform, tier, content)] = file
        return found
