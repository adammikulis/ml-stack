"""Getting a llama.cpp server onto this machine."""

from __future__ import annotations

import platform
import re
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

from ml_stack import net
from ml_stack.http import ServerError
from ml_stack.hub.probe import machine_memory
from ml_stack.safenames import Unsafe, unpack
from ml_stack.serve.backend import help_of

from .updates import UpdateError, download

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
    if any(card.vendor == "nvidia" for card in machine_memory().gpus):
        return (f"ubuntu-cuda-{{version}}-{'arm64' if arm else 'x64'}",)
    return ("ubuntu-arm64",) if arm else ("ubuntu-x64", "ubuntu-vulkan-x64")


def asset_for_this_machine(release: dict[str, Any]) -> dict[str, Any] | None:
    """The build of llama.cpp that runs here, or None if the release has none.

    Windows builds are zipped; macOS and Linux ones are tarred.
    """
    assets = [a for a in release.get("assets") or []
              if str(a.get("name", "")).lower().endswith(ARCHIVES)
              and not str(a.get("name", "")).startswith("cudart-")]
    for token in _tokens():
        if "{version}" in token:
            pattern = re.escape(token).replace(r"\{version\}", r"\d+\.\d+")
            candidates = [a for a in assets if re.search(pattern, str(a["name"]))]
            if candidates:
                return min(candidates, key=_cuda_version)
            continue
        for asset in assets:
            if token in str(asset["name"]).lower():
                return asset
    return None


def _cuda_version(asset: dict[str, Any]) -> tuple[int, ...]:
    match = re.search(r"cuda-(\d+)\.(\d+)", str(asset["name"]))
    if match is None:
        raise LlamaError(f"{asset['name']} has no CUDA runtime version")
    return tuple(int(value) for value in match.groups())


def cuda_companion(asset: dict[str, Any], release: dict[str, Any]) -> dict[str, Any] | None:
    """The CUDA runtime belonging to a Linux CUDA server asset."""
    name = str(asset.get("name", ""))
    if "-ubuntu-cuda-" not in name:
        return None
    wanted = "cudart-" + name
    found = next((one for one in release.get("assets", []) if one.get("name") == wanted), None)
    if found is None:
        raise LlamaError(f"{name} has no matching CUDA runtime in this release")
    return found


def cuda_ready(binary: Path) -> bool:
    """Whether the server initializes at least one CUDA device."""
    return re.search(r"ggml_cuda_init: found [1-9]\d* CUDA devices?", help_of(binary)) is not None


def latest(repo: str = REPO, *, timeout: float = TIMEOUT,
           count: int = LOOK_BACK) -> dict[str, Any]:
    """The newest release carrying a build for this machine."""
    try:
        found = net.default().json(API.format(repo=repo, count=count), net.Ask(
            purpose="llama.cpp releases", tries=3,
            headers={"Accept": "application/vnd.github+json"}))
    except (ServerError, OSError, ValueError) as exc:
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
    needs_cuda = any("cuda-" in token for token in _tokens())
    if found is not None and (not needs_cuda or cuda_ready(found)):
        return found

    if on_progress:
        on_progress("Looking for a llama.cpp build")
    release = latest(repo)
    asset = asset_for_this_machine(release)
    if asset is None:
        raise LlamaError(
            f"llama.cpp {release.get('tag_name') or 'latest'} has no build for "
            f"this machine ({'/'.join(_tokens())})")

    companion = cuda_companion(asset, release)
    vendor.mkdir(parents=True, exist_ok=True)
    if on_progress:
        on_progress(f"Downloading {asset['name']}")
    with tempfile.TemporaryDirectory() as tmp:
        try:
            archive = download(asset, tmp)
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
        if companion is not None:
            try:
                runtime = download(companion, tmp)
            except UpdateError as exc:
                raise LlamaError(str(exc)) from None
            runtime_staging = Path(tmp) / "runtime"
            _unpack(runtime, runtime_staging)
            for library in runtime_staging.rglob("*.so*"):
                shutil.copy2(library, vendor / library.name)

    got = find_server(vendor)
    if got is None:
        raise LlamaError("the downloaded server is not where it was expected")
    if needs_cuda and not cuda_ready(got):
        raise LlamaError("the installed llama-server cannot initialize a CUDA device; check the NVIDIA driver")
    return got


def _unpack(archive: Path, into: Path) -> None:
    """Extract a .zip or a .tar.gz, refusing one that writes outside ``into``."""
    try:
        unpack(archive, into)
    except Unsafe as exc:
        raise LlamaError(f"refusing {archive.name}: {exc}") from None


def _install(source: Path, vendor: Path) -> None:
    """The server and the libraries it was built against, side by side."""
    for item in source.iterdir():
        if item.is_dir():
            continue
        target = vendor / item.name
        shutil.copy2(item, target)
        if item.name == SERVER or item.suffix in ("", ".so", ".dylib", ".dll"):
            target.chmod(target.stat().st_mode | 0o111)
