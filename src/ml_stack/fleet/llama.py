"""Getting a llama.cpp server onto this machine."""

from __future__ import annotations

import json
import platform
import shutil
import sys
import tarfile
import tempfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

__all__ = ["LlamaError", "asset_for_this_machine", "cache_dir",
           "ensure_server", "find_server"]

REPO = "ggml-org/llama.cpp"
# The list, not /releases/latest: the binary builds are tagged bNNNNN and marked
# prerelease, and /releases/latest leaves prereleases out.
API = "https://api.github.com/repos/{repo}/releases?per_page={count}"
LOOK_BACK = 15
TIMEOUT = 15.0
ARCHIVES = (".zip", ".tar.gz")
SERVER = "llama-server.exe" if sys.platform == "win32" else "llama-server"


class LlamaError(RuntimeError):
    pass


def cache_dir(root: Path | str) -> Path:
    """Where a downloaded server is kept."""
    return Path(root).expanduser() / "llama"


def find_server(vendor: Path | str) -> Path | None:
    """A llama-server already on this machine, downloaded or installed by hand."""
    direct = Path(vendor).expanduser() / SERVER
    if direct.is_file():
        return direct.resolve()
    found = shutil.which(SERVER)
    return Path(found).resolve() if found else None


def _tokens() -> tuple[str, ...]:
    machine = platform.machine().lower()
    arm = machine in ("arm64", "aarch64")
    if sys.platform == "darwin":
        return ("macos-arm64",) if arm else ("macos-x64",)
    if sys.platform == "win32":
        return ("win-cpu-arm64", "win-arm64") if arm else ("win-cpu-x64", "win-x64")
    return ("ubuntu-arm64",) if arm else ("ubuntu-x64", "ubuntu-vulkan-x64")


def asset_for_this_machine(release: dict[str, Any]) -> dict[str, Any] | None:
    """The build of llama.cpp that runs here, or None if the release has none.

    Windows builds are zipped; macOS and Linux ones are tarred.
    """
    assets = [a for a in release.get("assets") or []
              if str(a.get("name", "")).lower().endswith(ARCHIVES)]
    for token in _tokens():
        for asset in assets:
            if token in str(asset["name"]).lower():
                return asset
    return None


def latest(repo: str = REPO, *, timeout: float = TIMEOUT,
           count: int = LOOK_BACK) -> dict[str, Any]:
    """The newest release carrying a build for this machine."""
    req = urllib.request.Request(API.format(repo=repo, count=count),
                                 headers={"User-Agent": "ml-stack",
                                          "Accept": "application/vnd.github+json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            found = json.loads(r.read())
    except (urllib.error.URLError, OSError, ValueError) as exc:
        raise LlamaError(f"could not reach the llama.cpp releases: {exc}") from None

    releases = found if isinstance(found, list) else [found]
    got = _first_with_a_build(releases)
    if got is None:
        raise LlamaError(
            f"no llama.cpp build for this machine ({'/'.join(_tokens())}) in the "
            f"last {len(releases)} releases")
    return got


def _first_with_a_build(releases: list[dict[str, Any]]) -> dict[str, Any] | None:
    for release in releases:
        if release.get("draft"):
            continue
        if asset_for_this_machine(release) is not None:
            return release
    return None


def ensure_server(root: Path | str, *, on_progress: Any = None,
                  repo: str = REPO) -> Path:
    """The llama-server binary, downloading it if this machine has none."""
    vendor = cache_dir(root)
    found = find_server(vendor)
    if found is not None:
        return found

    if on_progress:
        on_progress("Looking for a llama.cpp build")
    release = latest(repo)
    asset = asset_for_this_machine(release)
    if asset is None:
        raise LlamaError(
            f"llama.cpp {release.get('tag_name') or 'latest'} has no build for "
            f"this machine ({'/'.join(_tokens())})")

    from .updates import UpdateError, download
    vendor.mkdir(parents=True, exist_ok=True)
    if on_progress:
        on_progress(f"Downloading {asset['name']}")
    with tempfile.TemporaryDirectory() as tmp:
        try:
            archive = download(asset, tmp, timeout=1800.0)
        except UpdateError as exc:
            raise LlamaError(str(exc)) from None
        if on_progress:
            on_progress("Unpacking it")
        staging = Path(tmp) / "unpacked"
        _unpack(archive, staging)
        found = next(iter(staging.rglob(SERVER)), None)
        if found is None:
            raise LlamaError(f"{asset['name']} does not contain {SERVER}")
        _install(found.parent, vendor)

    got = find_server(vendor)
    if got is None:
        raise LlamaError("the downloaded server is not where it was expected")
    return got


def _unpack(archive: Path, into: Path) -> None:
    """Extract a .zip or a .tar.gz, refusing one that writes outside ``into``."""
    into.mkdir(parents=True, exist_ok=True)
    if archive.name.lower().endswith(".zip"):
        with zipfile.ZipFile(archive) as zf:
            _check(zf.namelist())
            zf.extractall(into)
        return
    with tarfile.open(archive) as tf:
        _check(tf.getnames())
        # 3.14 filters by default; asking for it keeps older versions the same.
        try:
            tf.extractall(into, filter="data")
        except TypeError:
            tf.extractall(into)


def _check(names: list[str]) -> None:
    for member in names:
        if member.startswith("/") or ".." in Path(member).parts:
            raise LlamaError("refusing an archive that escapes its directory")


def _install(source: Path, vendor: Path) -> None:
    """The server and the libraries it was built against, side by side."""
    for item in source.iterdir():
        if item.is_dir():
            continue
        target = vendor / item.name
        shutil.copy2(item, target)
        if item.name == SERVER or item.suffix in ("", ".so", ".dylib", ".dll"):
            target.chmod(target.stat().st_mode | 0o111)
