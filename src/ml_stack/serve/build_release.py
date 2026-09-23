"""Downloading a release for a machine with no compiler: the newest GitHub release with an
asset for this platform, ggml-org/llama.cpp's own or a fork's, unpacked and installed flat."""

from __future__ import annotations

import fnmatch
import json
import platform
import re
import shutil
import tempfile
from pathlib import Path

from ml_stack.files import promote
from ml_stack.fleet import updates as gh_updates
from ml_stack.http import ServerError, request_json
from ml_stack.log import say
from ml_stack.serve.binary import is_windows
from ml_stack.serve.build_paths import BuildFailed, builds_dir, named_dest, slug
from ml_stack.serve.build_platform import (
    arches_at,
    now_iso,
    server_name,
    version_of,
    vulkan_available,
)

__all__ = ["build_from_release", "build_from_release_named"]

# `llama-server-...-win-cuda-12.4-x64.zip` -- the CUDA version a companion `cudart-llama-*`
# asset is keyed off.
_CUDA_VER = re.compile(r"win-cuda-([\d.]+)-x64")


def _releases_for(repo: str, per_page: int = 5, timeout: float = 30.0) -> list[dict]:
    try:
        return request_json(
            f"https://api.github.com/repos/{repo}/releases?per_page={per_page}",
            method="GET", timeout=timeout, tries=3,
            headers={"Accept": "application/vnd.github+json"})
    except (ServerError, OSError, ValueError) as exc:
        raise BuildFailed(f"could not reach {repo}'s GitHub releases: {exc}") from None


def _llama_releases(per_page: int = 5, timeout: float = 30.0) -> list[dict]:
    return _releases_for("ggml-org/llama.cpp", per_page=per_page, timeout=timeout)


def _release_asset_globs(*, fork: bool = False) -> list[str]:
    """Release asset name patterns for this machine, best first.

    Mainline (ggml-org/llama.cpp) ships a plain llama-server build: Windows a ``.zip``,
    macOS and Linux a ``.tar.gz``. A fork built the ``unslothai/llama.cpp`` way (``fork``)
    bundles llama-server inside an ``app-*`` zip on Windows and Linux instead, and only
    matches mainline's own naming on macOS. Read off each repository's actual release
    assets on 2026-09-01 (``curl -s
    https://api.github.com/repos/<repo>/releases?per_page=3``). The CUDA build
    additionally needs a ``cudart-llama-*`` zip's DLLs alongside it; that is fetched
    separately by ``_cudart_companion``, keyed off the CUDA version named here.
    """
    machine = platform.machine().lower()
    arch = "arm64" if machine in ("arm64", "aarch64") else "x64"
    system = platform.system()
    if system == "Windows":
        globs = []
        if arch == "x64" and shutil.which("nvcc"):
            globs.append("app-*-windows-x64-cuda12-newer.zip" if fork
                         else "llama-*-bin-win-cuda-12.4-x64.zip")
        if vulkan_available():
            globs.append(f"app-*-windows-{arch}-vulkan.zip" if fork
                         else f"llama-*-bin-win-vulkan-{arch}.zip")
        globs.append(f"app-*-windows-{arch}-cpu.zip" if fork
                     else f"llama-*-bin-win-cpu-{arch}.zip")
        return globs
    if system == "Darwin":
        return [f"llama-*-bin-macos-{arch}.tar.gz"]
    return [f"app-*-linux-{arch}-cpu.tar.gz" if fork else f"llama-*-bin-ubuntu-{arch}.tar.gz"]


def _cudart_companion(asset_name: str, assets: dict[str, dict]) -> dict | None:
    """The matching ``cudart-llama-*`` asset a Windows CUDA build's DLLs need, if any."""
    match = _CUDA_VER.search(asset_name)
    if not match:
        return None
    return assets.get(f"cudart-llama-bin-win-cuda-{match.group(1)}-x64.zip")


def _extract(archive: Path, into: Path) -> None:
    if archive.name.endswith(".zip"):
        import zipfile

        with zipfile.ZipFile(archive) as zf:
            for member in zf.namelist():
                if member.startswith("/") or ".." in Path(member).parts:
                    raise BuildFailed(f"refusing an archive entry named {member!r}")
            zf.extractall(into)
    else:
        import tarfile

        with tarfile.open(archive) as tf:
            for member in tf.getmembers():
                if member.name.startswith("/") or ".." in Path(member.name).parts:
                    raise BuildFailed(f"refusing an archive entry named {member.name!r}")
            tf.extractall(into, filter="data")


def _release_install(dest: Path, archive: Path, *, extra: Path | None = None) -> Path:
    staging = dest.parent / f".{dest.name}.staging"
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    try:
        _extract(archive, staging)
        if extra is not None:
            _extract(extra, staging)
        # A release archive is either flat already or one folder deep; either is walked
        # the same way `~/.local/llama-next` is laid out, so only the top is copied.
        entries = list(staging.iterdir())
        root = entries[0] if len(entries) == 1 and entries[0].is_dir() else staging
        dest.mkdir(parents=True, exist_ok=True)
        for item in root.iterdir():
            target = dest / item.name
            if target.exists() or target.is_symlink():
                if target.is_dir() and not target.is_symlink():
                    shutil.rmtree(target)
                else:
                    target.unlink()
            promote(item, target)
    finally:
        shutil.rmtree(staging, ignore_errors=True)

    binary = dest / server_name()
    if not binary.is_file():
        raise BuildFailed(f"no {server_name()} in the downloaded release")
    if not is_windows():
        binary.chmod(binary.stat().st_mode | 0o111)
    return binary


def build_from_release(args) -> tuple[Path, str]:
    say("checking ggml-org/llama.cpp's releases")
    releases = _llama_releases()
    globs = _release_asset_globs()

    for release in releases:
        tag = str(release.get("tag_name", ""))
        assets = {str(a.get("name")): a for a in (release.get("assets") or [])}
        for pattern in globs:
            match = next((n for n in assets if fnmatch.fnmatch(n, pattern)), None)
            if match is None:
                continue
            dest = builds_dir() / tag
            if dest.is_dir() and (dest / "BUILD.json").is_file() and not args.force:
                say(f"{tag} is already installed at {dest} -- pass --force to redo it")
                return dest, tag

            say(f"downloading {match} from {tag}")
            with tempfile.TemporaryDirectory(prefix="ml-stack-llama-release-") as tmp:
                tmp_path = Path(tmp)
                archive = gh_updates.download(assets[match], tmp_path)
                extra = None
                companion = _cudart_companion(match, assets)
                if companion is not None:
                    say(f"downloading {companion['name']} (CUDA runtime)")
                    extra = gh_updates.download(companion, tmp_path)
                say("installing")
                binary = _release_install(dest, archive, extra=extra)

            version = version_of(binary)
            (dest / "BUILD.json").write_text(json.dumps(
                {"commit": tag, "built_at": now_iso(), "version": version,
                 "source": "release", "asset": match, "patches": [],
                 "arches": sorted(arches_at(tag))}, indent=2))
            return dest, tag

    raise BuildFailed(
        f"none of the last {len(releases)} ggml-org/llama.cpp releases had an asset "
        f"matching {globs} for this machine")


def build_from_release_named(args) -> tuple[Path, str]:
    """Download a fork's own release, kept at ``builds/<name>-<tag>/`` and linked from
    ``named/<name>`` rather than replacing ``current`` -- the release-download twin of
    ``build_from_source_named``, for a fork that ships binaries and a machine with no
    compiler, or simply to skip a compile when a matching asset already exists."""
    say(f"checking {args.repo}'s releases")
    releases = _releases_for(args.repo)
    globs = _release_asset_globs(fork=True)
    wanted_tag = args.tag

    for release in releases:
        tag = str(release.get("tag_name", ""))
        if wanted_tag and tag != wanted_tag:
            continue
        assets = {str(a.get("name")): a for a in (release.get("assets") or [])}
        for pattern in globs:
            match = next((n for n in assets if fnmatch.fnmatch(n, pattern)), None)
            if match is None:
                continue
            dest = named_dest(args.name, slug(tag))
            if dest.is_dir() and (dest / "BUILD.json").is_file() and not args.force:
                say(f"{tag} is already installed at {dest} -- pass --force to redo it")
                return dest, tag

            say(f"downloading {match} from {tag}")
            with tempfile.TemporaryDirectory(prefix="ml-stack-llama-release-") as tmp:
                tmp_path = Path(tmp)
                archive = gh_updates.download(assets[match], tmp_path)
                say("installing")
                binary = _release_install(dest, archive)

            version = version_of(binary)
            (dest / "BUILD.json").write_text(json.dumps(
                {"commit": tag, "built_at": now_iso(), "version": version,
                 "source": "release", "asset": match, "repo": args.repo,
                 "ref": wanted_tag or tag, "name": args.name, "patches": []}, indent=2))
            return dest, tag
        if wanted_tag:
            break

    raise BuildFailed(
        (f"{args.repo}'s release {wanted_tag!r}" if wanted_tag else
         f"none of the last {len(releases)} {args.repo} releases") +
        f" had an asset matching {globs} for this machine")
