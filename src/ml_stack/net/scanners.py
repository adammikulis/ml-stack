"""The scanner backends: ClamAV, Windows Defender, the macOS notice, and an opt-in hash lookup."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from ml_stack.net.scan import Outcome, Scanner, ScanResult

__all__ = ["ClamAV", "HashLookup", "MacNotice", "WindowsDefender", "default_scanners"]

TIMEOUT_S = 900.0
STALE_DAYS = 7.0
FOUND = re.compile(r":\s*(.+?)\s+FOUND\s*$", re.MULTILINE)
CLAM_DATE = re.compile(r"ClamAV [\d.]+(?:-\S+)?/\d+/(.+)$")


class ClamAV:
    """`clamdscan` when its daemon answers, else `clamscan --no-summary`. Exit 0 is clean,
    1 a signature matched, 2 an error."""

    name = "clamav"

    def __init__(self, *, clamscan: str | None = None, clamdscan: str | None = None) -> None:
        self.clamscan = clamscan or shutil.which("clamscan")
        self.clamdscan = clamdscan or shutil.which("clamdscan")

    def available(self) -> bool:
        return bool(self.clamscan or self.clamdscan)

    def _run(self, argv: list[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.run(argv, capture_output=True, text=True, timeout=TIMEOUT_S, check=False)

    def freshness(self) -> str:
        """A warning when the signature database is older than a week or cannot be dated."""
        tool = self.clamscan or self.clamdscan
        try:
            out = self._run([str(tool), "--version"]).stdout.strip().splitlines()
        except (OSError, subprocess.SubprocessError):
            return ""
        found = CLAM_DATE.search(out[0]) if out else None
        if found is None:
            return "the signature database date is unknown"
        try:
            built = datetime.strptime(" ".join(found.group(1).split()), "%a %b %d %H:%M:%S %Y")
        except ValueError:
            return "the signature database date is unknown"
        age = (time.time() - built.replace(tzinfo=UTC).timestamp()) / 86400
        return f"the signature database is {age:.0f} days old; run freshclam" if age > STALE_DAYS else ""

    def scan(self, path: Path) -> ScanResult:
        attempts = []
        if self.clamdscan:
            attempts.append([self.clamdscan, "--no-summary", "--fdpass", str(path)])
        if self.clamscan:
            attempts.append([self.clamscan, "--no-summary", str(path)])
        last = ScanResult(self.name, Outcome.ERROR, "no clamav command ran")
        for argv in attempts:
            try:
                done = self._run(argv)
            except subprocess.TimeoutExpired:
                return ScanResult(self.name, Outcome.ERROR, "the scan timed out")
            except OSError as exc:
                last = ScanResult(self.name, Outcome.ERROR, f"{type(exc).__name__}: {exc}")
                continue
            text = (done.stdout + done.stderr).strip()
            warn = (self.freshness(),) if argv[0] == self.clamscan else ()
            warnings = tuple(w for w in warn if w)
            if done.returncode == 0:
                return ScanResult(self.name, Outcome.CLEAN, "no signature matched",
                                  warnings=warnings)
            if done.returncode == 1:
                hit = FOUND.search(done.stdout)
                return ScanResult(self.name, Outcome.INFECTED, text[:300],
                                  hit.group(1) if hit else "", warnings)
            last = ScanResult(self.name, Outcome.ERROR, text[:300] or f"exit {done.returncode}")
        return last


class WindowsDefender:
    """`MpCmdRun.exe -Scan -ScanType 3 -File` with remediation off, so a finding is reported
    and nothing is deleted. Exit 0 is clean and 2 is a threat."""

    name = "windows-defender"

    def __init__(self, exe: str | None = None) -> None:
        self.exe = exe or self._find()

    @staticmethod
    def _find() -> str | None:
        if sys.platform != "win32":
            return None
        roots = [os.environ.get("PROGRAMFILES", r"C:\Program Files"),
                 os.environ.get("PROGRAMDATA", r"C:\ProgramData")]
        fixed = Path(roots[0]) / "Windows Defender" / "MpCmdRun.exe"
        platform = Path(roots[1]) / "Microsoft" / "Windows Defender" / "Platform"
        found = [fixed] if fixed.is_file() else sorted(platform.glob("*/MpCmdRun.exe"))
        return str(found[-1]) if found else None

    def available(self) -> bool:
        return bool(self.exe)

    def scan(self, path: Path) -> ScanResult:
        argv = [str(self.exe), "-Scan", "-ScanType", "3", "-File", str(path),
                "-DisableRemediation"]
        try:
            done = subprocess.run(argv, capture_output=True, text=True, timeout=TIMEOUT_S,
                                  check=False)
        except (OSError, subprocess.SubprocessError) as exc:
            return ScanResult(self.name, Outcome.ERROR, f"{type(exc).__name__}: {exc}")
        text = (done.stdout + done.stderr).strip()[:300]
        if done.returncode == 0:
            return ScanResult(self.name, Outcome.CLEAN, "no threat found")
        if done.returncode == 2:
            return ScanResult(self.name, Outcome.INFECTED, text)
        return ScanResult(self.name, Outcome.ERROR, text or f"exit {done.returncode}")


class MacNotice:
    """macOS has no scanner this library may call: it answers `NO_SCANNER` and says so, with
    what Gatekeeper reports for a binary. XProtect does not scan arbitrary files on demand."""

    name = "macos"
    ADVICE = "no malware scanner on macOS; install ClamAV (brew install clamav) for scanning"

    def __init__(self, *, platform: str | None = None) -> None:
        self.platform = platform or sys.platform

    def available(self) -> bool:
        return self.platform == "darwin"

    def scan(self, path: Path) -> ScanResult:
        notes = [self.ADVICE]
        xattr = shutil.which("xattr")
        if xattr:
            done = subprocess.run([xattr, str(path)], capture_output=True, text=True,
                                  timeout=30, check=False)
            if "com.apple.quarantine" in done.stdout:
                notes.append("the file carries the com.apple.quarantine flag")
        spctl = shutil.which("spctl")
        if spctl and os.access(path, os.X_OK):
            done = subprocess.run([spctl, "--assess", "--type", "execute", str(path)],
                                  capture_output=True, text=True, timeout=60, check=False)
            verdict = (done.stderr or done.stdout).strip().splitlines()
            notes.append("Gatekeeper (not a malware scan): "
                         + (verdict[0] if verdict else f"exit {done.returncode}"))
        return ScanResult(self.name, Outcome.NO_SCANNER, "; ".join(notes))


class HashLookup:
    """Opt-in reputation lookup of a file's SHA-256 at a VirusTotal-style service. Only the
    digest leaves the machine. Off unless constructed with a key; an unknown hash is not clean."""

    name = "hash-reputation"

    def __init__(self, key: str, *, get: Callable[[str, dict[str, str]], tuple[int, bytes]],
                 base: str = "https://www.virustotal.com/api/v3/files/",
                 digest: Callable[[Path], str] | None = None) -> None:
        self.key, self.get, self.base = key, get, base
        self.digest = digest or _sha256

    def available(self) -> bool:
        return bool(self.key)

    def scan(self, path: Path) -> ScanResult:
        sha = self.digest(path)
        status, body = self.get(self.base + sha, {"x-apikey": self.key})
        if status == 404:
            return ScanResult(self.name, Outcome.NO_SCANNER, "the service does not know this hash")
        if status != 200:
            return ScanResult(self.name, Outcome.ERROR, f"the service answered {status}")
        try:
            stats = json.loads(body)["data"]["attributes"]["last_analysis_stats"]
            bad = int(stats.get("malicious", 0)) + int(stats.get("suspicious", 0))
        except (ValueError, KeyError, TypeError):
            return ScanResult(self.name, Outcome.ERROR, "the service's answer was not understood")
        if bad:
            return ScanResult(self.name, Outcome.INFECTED, f"{bad} engines flag this hash")
        return ScanResult(self.name, Outcome.CLEAN, "no engine flags this hash (reputation only)")


def _sha256(path: Path) -> str:
    from ml_stack.files import sha256_file

    return sha256_file(path)


def default_scanners() -> list[Scanner]:
    """The local backends in the order they are tried. The hash lookup is never in this list."""
    return [ClamAV(), WindowsDefender(), MacNotice()]
