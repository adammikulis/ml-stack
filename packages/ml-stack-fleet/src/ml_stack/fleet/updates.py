"""Checking for a newer release, downloading it, and putting it in place."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from collections.abc import Callable
from pathlib import Path
from typing import Any

__all__ = ["Release", "UpdateError", "apply_if_newer", "asset_for", "check",
           "current_version", "watch",
           "download", "install", "REPO"]

REPO = "adammikulis/ml-stack"
API = "https://api.github.com/repos/{repo}/releases/latest"
TIMEOUT = 30.0
CHUNK = 1 << 20


class UpdateError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class Release:
    version: str
    url: str
    notes: str
    assets: tuple[dict[str, Any], ...]
    checked_at: float

    def newer_than(self, version: str) -> bool:
        """False when the running version is unknown: nothing is newer than nothing."""
        if not version.strip():
            return False
        return _parse(self.version) > _parse(version)


def _parse(version: str) -> tuple[int, ...]:
    cleaned = version.strip().lstrip("vV").split("-")[0].split("+")[0]
    out = []
    for part in cleaned.split("."):
        digits = "".join(c for c in part if c.isdigit())
        out.append(int(digits) if digits else 0)
    return tuple(out) or (0,)


def current_version() -> str:
    """The running version, or empty when there is no way to tell."""
    try:
        from importlib.metadata import version
        return version("ml-stack-fleet")
    except Exception:                                 # noqa: BLE001
        pass
    told = os.environ.get("ML_STACK_VERSION", "").strip()
    if told:
        return told
    return _version_in_source() or ""


def _version_in_source() -> str:
    """The version in this checkout, when running from one rather than an install."""
    for parent in Path(__file__).resolve().parents:
        found = parent / "packages" / "ml-stack-fleet" / "pyproject.toml"
        if not found.is_file():
            continue
        for line in found.read_text().splitlines():
            if line.startswith("version"):
                _, _, value = line.partition("=")
                value, _, _ = value.partition("#")
                return value.strip().strip('"').strip("'")
    return ""


def platform_key() -> str:
    """The asset name fragment for this machine."""
    machine = platform.machine().lower()
    arch = "arm64" if machine in ("arm64", "aarch64") else "x86_64"
    if sys.platform == "darwin":
        return f"macos-{arch}"
    if sys.platform == "win32":
        return f"windows-{arch}"
    return f"linux-{arch}"


def check(repo: str = REPO, *, timeout: float = TIMEOUT) -> Release:
    """Ask GitHub for the newest release."""
    req = urllib.request.Request(API.format(repo=repo),
                                 headers={"Accept": "application/vnd.github+json",
                                          "User-Agent": "ml-stack"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            body = json.loads(r.read())
    except urllib.error.HTTPError as exc:
        raise UpdateError(f"could not reach GitHub: {exc.code}") from None
    except (urllib.error.URLError, OSError, ValueError) as exc:
        raise UpdateError(f"could not reach GitHub: {exc}") from None
    return Release(
        version=str(body.get("tag_name") or "").lstrip("v"),
        url=str(body.get("html_url") or ""),
        notes=str(body.get("body") or ""),
        assets=tuple(body.get("assets") or ()),
        checked_at=time.time(),
    )


def asset_for(release: Release, key: str = "") -> dict[str, Any] | None:
    """The download for this machine, or None if the release has none."""
    key = key or platform_key()
    for asset in release.assets:
        if key in str(asset.get("name", "")):
            return asset
    return None


def download(asset: dict[str, Any], into: Path | str,
             *, on_progress: Any = None, timeout: float = 600.0) -> Path:
    """Fetch one asset and check it against the digest GitHub reports for it.

    The digest is not a signature: it proves the bytes match what that release holds, not
    who built them. Trust here is the same as downloading it by hand -- TLS to github.com
    and the repository name below.
    """
    into = Path(into).expanduser()
    into.mkdir(parents=True, exist_ok=True)
    target = into / str(asset["name"])
    url = str(asset["browser_download_url"])
    total = int(asset.get("size") or 0)

    req = urllib.request.Request(url, headers={"User-Agent": "ml-stack"})
    digest = hashlib.sha256()
    done = 0
    partial = target.with_suffix(target.suffix + ".part")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r, partial.open("wb") as fh:
            while True:
                block = r.read(CHUNK)
                if not block:
                    break
                fh.write(block)
                digest.update(block)
                done += len(block)
                if on_progress:
                    on_progress(done, total)
    except (urllib.error.URLError, OSError) as exc:
        partial.unlink(missing_ok=True)
        raise UpdateError(f"download failed: {exc}") from None

    want = str(asset.get("digest") or "").removeprefix("sha256:").strip().lower()
    if want and digest.hexdigest() != want:
        partial.unlink(missing_ok=True)
        raise UpdateError("the download does not match the digest GitHub reports for it")
    if total and done != total:
        partial.unlink(missing_ok=True)
        raise UpdateError(f"got {done} of {total} bytes")
    os.replace(partial, target)
    return target


def install(archive: Path | str, *, app_path: Path | str | None = None) -> Path:
    """Unpack a downloaded release over the running one. Returns what it replaced.

    The replacement is atomic per item: the new copy is unpacked beside the old, and only
    then swapped in, so an interrupted install leaves the working copy alone.
    """
    archive = Path(archive).expanduser()
    target = Path(app_path).expanduser() if app_path else running_path()
    if target is None:
        raise UpdateError("cannot tell what to replace; install it by hand")

    staging = Path(tempfile.mkdtemp(prefix="ml-stack-update-", dir=str(target.parent)))
    try:
        with zipfile.ZipFile(archive) as zf:
            for member in zf.namelist():
                if member.startswith("/") or ".." in Path(member).parts:
                    raise UpdateError(f"refusing an archive entry named {member!r}")
            zf.extractall(staging)
        found = _pick(staging, target.name)
        if found is None:
            raise UpdateError(f"the download has no {target.name} in it")
        _restore_modes(found)

        backup = target.with_name(target.name + ".old")
        shutil.rmtree(backup, ignore_errors=True)
        backup.unlink(missing_ok=True)
        if target.exists():
            os.replace(target, backup)
        os.replace(found, target)
        shutil.rmtree(backup, ignore_errors=True)
        return target
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def _pick(staging: Path, name: str) -> Path | None:
    direct = staging / name
    if direct.exists():
        return direct
    for candidate in staging.rglob(name):
        return candidate
    return None


def _restore_modes(path: Path) -> None:
    """Zip does not carry the executable bit on every platform."""
    if path.is_file():
        path.chmod(path.stat().st_mode | 0o111)
        return
    for item in path.rglob("*"):
        if item.is_file() and (item.parent.name == "MacOS" or not item.suffix):
            item.chmod(item.stat().st_mode | 0o111)


def running_path() -> Path | None:
    """The app or binary currently executing, when it is a bundle."""
    if not getattr(sys, "frozen", False):
        return None
    exe = Path(sys.executable).resolve()
    for parent in exe.parents:
        if parent.suffix == ".app":
            return parent
    return exe


def relaunch(*, delay_s: float = 1.5, stop: bool = True) -> bool:
    """Start the replaced copy and stop this one. False when this is not a bundle.

    The wait is so an answer already being written reaches the browser first.
    """
    target = running_path()
    if target is None:
        return False

    def go() -> None:
        time.sleep(delay_s)
        try:
            if target.suffix == ".app":
                subprocess.Popen(["open", "-n", str(target)])
            else:
                subprocess.Popen([str(target)], start_new_session=True)
        finally:
            if stop:
                os._exit(0)

    threading.Thread(target=go, daemon=True, name="relaunch").start()
    return True


def apply_if_newer() -> dict[str, Any]:
    """Put the newest release in place, if there is one. Says what happened."""
    if running_path() is None:
        return {"ok": False, "installed": False,
                "error": "this copy was installed with pip; update it with pip"}
    now = current_version()
    try:
        release = check()
        if not release.newer_than(now):
            return {"ok": True, "installed": False, "version": now}
        asset = asset_for(release)
        if asset is None:
            return {"ok": False, "installed": False,
                    "error": f"release {release.version} has no download for "
                             "this machine"}
        archive = download(asset, tempfile.mkdtemp(prefix="ml-stack-update-"))
        install(archive)
    except UpdateError as exc:
        return {"ok": False, "installed": False, "error": str(exc)}
    return {"ok": True, "installed": True, "version": release.version}


def watch(*, wanted: "Callable[[], bool]", idle: "Callable[[], bool]",
          every_s: float = 24 * 3600, first_after_s: float = 300.0) -> threading.Thread:
    """Check for a newer release on a timer, and put it on when nothing is running.

    A machine part way through a training run is left alone until it is not.
    """
    def loop() -> None:
        time.sleep(first_after_s)
        while True:
            try:
                if wanted() and idle():
                    got = apply_if_newer()
                    if got.get("installed") and relaunch():
                        return
            except Exception:                         # noqa: BLE001
                pass
            time.sleep(every_s)

    thread = threading.Thread(target=loop, daemon=True, name="updates")
    thread.start()
    return thread
