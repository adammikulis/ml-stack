"""Build llama-server from its own master, since a release lags it by an architecture or two.

Measured on one machine the day this was written: the newest homebrew bottle
(``llama.cpp 0.3.0``, `brew outdated` empty) read ``gemma4`` and ``qwen3moe`` but not
``qwen4exp`` -- Qwen3.8-Flash-Next's architecture -- and lacked ``--kv-unified-per-slot``
besides. A hand-built binary in one person's ``~/.local/llama-next`` fixed that for one
machine, selected only by one application setting ``LLAMA_CPP_SERVER`` for one model name.
Every other bench run, on every other machine, kept loading the stale bottle without saying
so.

``ml-stack-serve build`` clones or fast-forwards llama.cpp's own master into a managed
directory and builds it (``--from source``), or downloads the newest GitHub release with an
asset for this machine (``--from release``, the fallback when no compiler is on PATH -- the
only path on a machine with no toolchain, such as most Windows installs). Either way, the
new binary is installed into its own flat, versioned directory and is trusted only once it
answers ``--help`` and reads every architecture the previous build did; only then does
``current`` -- a symlink (a junction on Windows, where a symlink needs a privilege a plain
account may not have) -- point at it. A build that fails verification leaves ``current``
untouched, which is what makes ``--persist``'s weekly, unattended rerun safe.
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from ml_stack import home
from ml_stack.http import ServerError, request_json
from ml_stack.log import say, warn
from ml_stack.serve.binary import (
    child_env,
    find_binary,
    is_windows,
    managed_current,
    managed_named,
    managed_root,
)

__all__ = [
    "BuildFailed", "PERSIST_PLIST", "PERSIST_TASK", "WEEK_SECONDS", "builds_dir",
    "cmd_build", "current_link", "named_dir", "named_src_dir", "patch_files",
    "patch_stamp", "patches_dir", "root", "src_dir",
]

REPO_URL = "https://github.com/ggml-org/llama.cpp"
LLAMA_RELEASES_API = "https://api.github.com/repos/ggml-org/llama.cpp/releases"
CMAKE_TARGET = "llama-server"          # the cmake target name, the same on every platform



def root() -> Path:
    """Where a managed build lives."""
    return managed_root()


def src_dir() -> Path:
    """The llama.cpp checkout `ml-stack-serve build` builds from."""
    return root() / "src"


def builds_dir() -> Path:
    """The directory holding one directory per build made."""
    return root() / "builds"


def current_link() -> Path:
    """The link naming the build `find_binary` prefers."""
    return managed_current()


def named_dir() -> Path:
    """The directory of links to builds kept beside ``current``."""
    return managed_named()


def named_src_dir() -> Path:
    """The directory holding one checkout per named build."""
    return root() / "named-src"


LIB_GLOBS = ("lib*.dylib", "lib*.so", "*.dll")

PERSIST_LABEL = "com.ml-stack.llama-build"
PERSIST_PLIST = home.user_home() / "Library" / "LaunchAgents" / f"{PERSIST_LABEL}.plist"
PERSIST_TASK = "MLStackLlamaBuild"
WEEK_SECONDS = 7 * 24 * 60 * 60

_ARCH_NAME = re.compile(r'LLM_ARCH_\w+,\s*"([a-z0-9_]+)"')
_CUDA_VER = re.compile(r"win-cuda-([\d.]+)-x64")
_COMMIT_IN_VERSION = re.compile(r"commit[ :=]?\s*([0-9a-f]{7,40})", re.IGNORECASE)


class BuildFailed(RuntimeError):
    """A step of the build did not do what the next step needs.

    Raised instead of letting the next step fail on a symptom -- a missing binary, a build
    directory with no ``bin/`` -- so the error names what actually went wrong.
    """


# -- small platform facts -------------------------------------------------
def _server_name() -> str:
    return "llama-server.exe" if is_windows() else "llama-server"


def _vulkan_available() -> bool:
    return bool(os.environ.get("VULKAN_SDK")) or shutil.which("vulkaninfo") is not None


def _can_build_from_source() -> bool:
    """Whether this machine has a compiler at all -- the source path needs one, the release
    path does not, and a machine with neither should not be told to try compiling."""
    if shutil.which("cmake") is None:
        return False
    if platform.system() == "Windows":
        return any(shutil.which(c) for c in ("cl", "gcc", "clang"))
    return any(shutil.which(c) for c in ("cc", "gcc", "clang"))


def _cmake_flags() -> list[str]:
    """Which GPU backend to build, decided the same way on every platform: a CUDA compiler
    beats a Vulkan SDK beats neither. Metal is unconditional on macOS -- every Mac since the
    architecture this reads GGUF for has one."""
    if platform.system() == "Darwin":
        return ["-DGGML_METAL=ON", "-DLLAMA_CURL=ON"]
    if shutil.which("nvcc"):
        return ["-DGGML_CUDA=ON"]
    if _vulkan_available():
        return ["-DGGML_VULKAN=ON"]
    return []


def _platform_asset_globs() -> list[str]:
    """Release asset name patterns for this machine, best first.

    Read off ggml-org/llama.cpp's actual release assets on 2026-09-01 (``curl -s
    https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=3``): Windows ships
    ``.zip``, macOS and Linux ship ``.tar.gz`` -- an easy mismatch to hardcode wrong. The
    CUDA build additionally needs a ``cudart-llama-*`` zip's DLLs alongside it; that is
    fetched separately by ``_cudart_companion``, keyed off the CUDA version named here.
    """
    machine = platform.machine().lower()
    arch = "arm64" if machine in ("arm64", "aarch64") else "x64"
    system = platform.system()
    if system == "Windows":
        globs = []
        if arch == "x64" and shutil.which("nvcc"):
            globs.append("llama-*-bin-win-cuda-12.4-x64.zip")
        if _vulkan_available():
            globs.append(f"llama-*-bin-win-vulkan-{arch}.zip")
        globs.append(f"llama-*-bin-win-cpu-{arch}.zip")
        return globs
    if system == "Darwin":
        return [f"llama-*-bin-macos-{arch}.tar.gz"]
    return [f"llama-*-bin-ubuntu-{arch}.tar.gz"]


def _cudart_companion(asset_name: str, assets: dict[str, dict]) -> dict | None:
    """The matching ``cudart-llama-*`` asset a Windows CUDA build's DLLs need, if any."""
    match = _CUDA_VER.search(asset_name)
    if not match:
        return None
    return assets.get(f"cudart-llama-bin-win-cuda-{match.group(1)}-x64.zip")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# -- git ---------------------------------------------------------------
def _git(*args: str, cwd: Path | None = None) -> subprocess.CompletedProcess:
    which = shutil.which("git")
    if which is None:
        raise BuildFailed("git is not on PATH")
    done = subprocess.run([which, *args], cwd=cwd, capture_output=True, text=True)
    if done.returncode != 0:
        raise BuildFailed(f"git {' '.join(args)} failed: {(done.stderr or '').strip()}")
    return done


def _sync_source(source: Path) -> None:
    """Only the tip of master, not its history -- llama.cpp's full history is a large,
    slow clone that buys nothing here, the same reason ``ensure_converter`` elsewhere in
    this codebase already clones ``--depth 1``. Measured the way it was found: a first run
    against a bare repo was still cloning several minutes in and past 190MB."""
    if (source / ".git").is_dir():
        say(f"  fetching {REPO_URL} into {source}")
        _git("fetch", "--depth", "1", "origin", "master", cwd=source)
        _git("checkout", "master", cwd=source)
        _git("reset", "--hard", "origin/master", cwd=source)
    else:
        say(f"  cloning {REPO_URL} into {source} (--depth 1)")
        source.parent.mkdir(parents=True, exist_ok=True)
        _git("clone", "--depth", "1", "--branch", "master", REPO_URL, str(source))


def _checkout_commit(source: Path, commit: str) -> None:
    """A shallow clone has only master's tip, so a specific commit is fetched by name
    before it can be checked out -- GitHub serves an arbitrary reachable commit SHA this
    way without needing the rest of history either."""
    say(f"  fetching and checking out {commit}")
    _git("fetch", "--depth", "1", "origin", commit, cwd=source)
    _git("checkout", "FETCH_HEAD", cwd=source)


def _short_commit(source: Path) -> str:
    return _git("rev-parse", "--short", "HEAD", cwd=source).stdout.strip()


# -- patches carried on top of upstream ----------------------------------------------
def patches_dir() -> Path:
    """The directory the llama.cpp patches are read from."""
    override = os.environ.get("MLSTACK_LLAMA_PATCHES", "")
    if override:
        return Path(override).expanduser()
    packaged = Path(__file__).resolve().parent / "_patches" / "llama.cpp"
    if packaged.is_dir():
        return packaged
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "patches" / "llama.cpp"
        if candidate.is_dir():
            return candidate
    return packaged


def patch_files() -> list[Path]:
    """Every patch a source build applies, in name order."""
    where = patches_dir()
    return sorted(where.glob("*.patch")) if where.is_dir() else []


def patch_stamp(files: list[Path] | None = None) -> str:
    """A short digest of the patch set, empty when there are none."""
    files = patch_files() if files is None else files
    if not files:
        return ""
    digest = hashlib.sha256()
    for item in files:
        digest.update(item.name.encode())
        digest.update(item.read_bytes())
    return "p" + digest.hexdigest()[:7]


def _applies_cleanly(source: Path, patch: Path, *, reverse: bool = False) -> bool:
    """Whether ``patch`` applies to ``source``, in reverse when asked."""
    args = ["apply", "--check", *(["--reverse"] if reverse else []), str(patch)]
    try:
        _git(*args, cwd=source)
    except BuildFailed:
        return False
    return True


def _apply_patches(source: Path) -> list[str]:
    """Apply every patch onto ``source``, returning their names."""
    files = patch_files()
    applied: list[str] = []
    for item in files:
        if _applies_cleanly(source, item, reverse=True):
            say(f"  {item.name} is already applied")
            applied.append(item.name)
            continue
        say(f"  applying {item.name}")
        try:
            _git("apply", "--3way", str(item), cwd=source)
        except BuildFailed as exc:
            raise BuildFailed(
                f"{item.name} does not apply to this checkout: {exc}") from None
        applied.append(item.name)
    return applied


# -- a fork, kept beside `current` rather than replacing it ---------------------------
def _named_source_dir(name: str) -> Path:
    return named_src_dir() / name


def _named_dest(name: str, commit: str) -> Path:
    return builds_dir() / f"{name}-{commit}"


def _sync_named_source(source: Path, repo: str, ref: str) -> None:
    """Clone or fetch ``repo`` into ``source``, then land on ``ref`` -- a tag, branch or
    SHA -- the same way ``_checkout_commit`` lands master's checkout on a specific commit.
    Left at the default branch's tip when ``ref`` is empty."""
    url = f"https://github.com/{repo}"
    if (source / ".git").is_dir():
        say(f"  fetching {url} into {source}")
        _git("fetch", "--depth", "1", "origin", cwd=source)
    else:
        say(f"  cloning {url} into {source} (--depth 1)")
        source.parent.mkdir(parents=True, exist_ok=True)
        _git("clone", "--depth", "1", url, str(source))
    if ref:
        _checkout_commit(source, ref)


def _arches_from_source(source: Path) -> set[str]:
    """Every architecture name master's own source reads, for ``--check`` to compare against."""
    arch_file = source / "src" / "llama-arch.cpp"
    if not arch_file.is_file():
        return set()
    text = arch_file.read_text(encoding="utf-8", errors="replace")
    return set(_ARCH_NAME.findall(text))


# -- cmake ---------------------------------------------------------------
def _configure(source: Path) -> None:
    cmake = shutil.which("cmake")
    if cmake is None:
        raise BuildFailed("cmake is not on PATH")
    done = subprocess.run([cmake, "-B", "build", *_cmake_flags()], cwd=source,
                          capture_output=True, text=True)
    if done.returncode != 0:
        raise BuildFailed(f"cmake configure failed:\n{(done.stderr or '').strip()[-4000:]}")


def _compile(source: Path, jobs: int) -> None:
    cmake = shutil.which("cmake")
    if cmake is None:
        raise BuildFailed("cmake is not on PATH")
    done = subprocess.run(
        [cmake, "--build", "build", "--config", "Release", "--target", CMAKE_TARGET,
         "-j", str(jobs)],
        cwd=source, capture_output=True, text=True)
    if done.returncode != 0:
        raise BuildFailed(f"build failed:\n{(done.stderr or '').strip()[-4000:]}")


# -- installing a flat, self-contained build ------------------------------
def _copy_flat(src_dir: Path, dest: Path) -> list[str]:
    """The server binary and every library beside it, flattened into ``dest`` -- the same
    shape as the hand-built ``~/.local/llama-next`` this replaces, so `child_env` finds the
    libraries by putting the binary's own directory on PATH."""
    copied: list[str] = []
    seen: set[str] = set()
    for pattern in (_server_name(), *LIB_GLOBS):
        for item in sorted(src_dir.glob(pattern)):
            if item.name in seen:
                continue
            seen.add(item.name)
            target = dest / item.name
            if target.exists() or target.is_symlink():
                target.unlink()
            if item.is_symlink():
                target.symlink_to(os.readlink(item))
            else:
                shutil.copy2(item, target)
            copied.append(item.name)
    return copied


def _version_of(binary: Path) -> str:
    try:
        done = subprocess.run([str(binary), "--version"], capture_output=True, text=True,
                              timeout=20, env=child_env(binary))
    except (OSError, subprocess.SubprocessError):
        return ""
    text = ((done.stdout or "") + (done.stderr or "")).strip()
    return text.splitlines()[0] if text else ""


def _install_source_build(build_dir: Path, dest: Path, commit: str, *,
                          extra: dict | None = None) -> Path:
    bin_dir = build_dir / "bin"
    binary_name = _server_name()
    if not (bin_dir / binary_name).is_file() and not (bin_dir / binary_name).is_symlink():
        raise BuildFailed(f"no {binary_name} in {bin_dir} -- the build did not produce one")
    dest.mkdir(parents=True, exist_ok=True)
    copied = set(_copy_flat(bin_dir, dest))
    lib_dir = build_dir / "lib"
    if lib_dir.is_dir():
        copied |= set(_copy_flat(lib_dir, dest))
    binary = dest / binary_name
    if not is_windows():
        binary.chmod(binary.stat().st_mode | 0o111)
    version = _version_of(binary)
    info = {"commit": commit, "built_at": _now_iso(), "version": version, "source": "source",
            "patches": []}
    if extra:
        info.update(extra)
    (dest / "BUILD.json").write_text(json.dumps(info, indent=2))
    return binary


def _build_from_source(args) -> tuple[Path, str]:
    source = Path(args.source).expanduser() if args.source else src_dir()
    if not args.source:
        _sync_source(source)
    if args.commit:
        _checkout_commit(source, args.commit)

    commit = _short_commit(source)
    stamp = patch_stamp()
    tag = f"{commit}-{stamp}" if stamp else commit
    dest = builds_dir() / tag
    if dest.is_dir() and (dest / "BUILD.json").is_file() and not args.force:
        say(f"{tag} is already built at {dest} -- pass --force to rebuild")
        return dest, tag

    applied = _apply_patches(source)
    jobs = args.jobs or (os.cpu_count() or 4)
    say(f"configuring {tag} ({', '.join(_cmake_flags()) or 'CPU only'})")
    _configure(source)
    say(f"building ({jobs} jobs) -- this takes several minutes")
    _compile(source, jobs)
    say("installing")
    _install_source_build(source / "build", dest, tag, extra={"patches": applied})
    return dest, tag


def _build_from_source_named(args) -> tuple[Path, str]:
    """Build a fork's own ref, kept at ``builds/<name>-<commit>/`` beside master's builds
    and linked from ``named/<name>`` rather than replacing ``current``."""
    source = _named_source_dir(args.name)
    _sync_named_source(source, args.repo, args.ref)

    commit = _short_commit(source)
    stamp = patch_stamp()
    commit = f"{commit}-{stamp}" if stamp else commit
    dest = _named_dest(args.name, commit)
    if dest.is_dir() and (dest / "BUILD.json").is_file() and not args.force:
        say(f"{args.name}-{commit} is already built at {dest} -- pass --force to rebuild")
        return dest, commit

    applied = _apply_patches(source)
    jobs = args.jobs or (os.cpu_count() or 4)
    say(f"configuring {args.name}-{commit} ({', '.join(_cmake_flags()) or 'CPU only'})")
    _configure(source)
    say(f"building ({jobs} jobs) -- this takes several minutes")
    _compile(source, jobs)
    say("installing")
    _install_source_build(source / "build", dest, commit,
                          extra={"repo": args.repo, "ref": args.ref or commit,
                                 "name": args.name, "patches": applied})
    return dest, commit


# -- downloading a release, for a machine with no compiler ----------------
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


def _named_release_asset_globs() -> list[str]:
    """Release asset name patterns for this machine, matching an ``unslothai/llama.cpp``
    -shaped fork -- a *different* naming convention from ``ggml-org/llama.cpp`` itself, so
    reusing ``_platform_asset_globs`` silently matched nothing there the first time this was
    tried.

    Read off unslothai/llama.cpp's actual release assets on 2026-09-01 (``curl -s
    https://api.github.com/repos/unslothai/llama.cpp/releases?per_page=3``): macOS ships the
    same shape mainline does, ``llama-<tag>-bin-macos-<arch>.tar.gz`` (both ``arm64`` and
    ``x64``) -- so the mainline glob works unchanged there. Windows and Linux ship no plain
    ``llama-server`` build at all, only an ``app-<tag>-<os>-<arch>-<backend>.zip`` bundle
    (``cpu``, ``vulkan``, a ``cuda12``/``cuda13`` variant, or ROCm) -- a different shape
    entirely, encoded here rather than assumed to match.
    """
    machine = platform.machine().lower()
    arch = "arm64" if machine in ("arm64", "aarch64") else "x64"
    system = platform.system()
    if system == "Windows":
        globs = []
        if arch == "x64" and shutil.which("nvcc"):
            globs.append("app-*-windows-x64-cuda12-newer.zip")
        if _vulkan_available():
            globs.append(f"app-*-windows-{arch}-vulkan.zip")
        globs.append(f"app-*-windows-{arch}-cpu.zip")
        return globs
    if system == "Darwin":
        return [f"llama-*-bin-macos-{arch}.tar.gz"]
    return [f"app-*-linux-{arch}-cpu.tar.gz"]


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
            shutil.move(str(item), str(target))
    finally:
        shutil.rmtree(staging, ignore_errors=True)

    binary = dest / _server_name()
    if not binary.is_file():
        raise BuildFailed(f"no {_server_name()} in the downloaded release")
    if not is_windows():
        binary.chmod(binary.stat().st_mode | 0o111)
    return binary


def _build_from_release(args) -> tuple[Path, str]:
    from ml_stack.fleet import updates as gh_updates

    say("checking ggml-org/llama.cpp's releases")
    releases = _llama_releases()
    globs = _platform_asset_globs()

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

            version = _version_of(binary)
            (dest / "BUILD.json").write_text(json.dumps(
                {"commit": tag, "built_at": _now_iso(), "version": version,
                 "source": "release", "asset": match, "patches": []}, indent=2))
            return dest, tag

    raise BuildFailed(
        f"none of the last {len(releases)} ggml-org/llama.cpp releases had an asset "
        f"matching {globs} for this machine")


def _build_from_release_named(args) -> tuple[Path, str]:
    """Download a fork's own release, kept at ``builds/<name>-<tag>/`` and linked from
    ``named/<name>`` rather than replacing ``current`` -- the release-download twin of
    ``_build_from_source_named``, for a fork that ships binaries and a machine with no
    compiler, or simply to skip a compile when a matching asset already exists."""
    from ml_stack.fleet import updates as gh_updates

    say(f"checking {args.repo}'s releases")
    releases = _releases_for(args.repo)
    globs = _named_release_asset_globs()
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
            dest = _named_dest(args.name, _slug(tag))
            if dest.is_dir() and (dest / "BUILD.json").is_file() and not args.force:
                say(f"{tag} is already installed at {dest} -- pass --force to redo it")
                return dest, tag

            say(f"downloading {match} from {tag}")
            with tempfile.TemporaryDirectory(prefix="ml-stack-llama-release-") as tmp:
                tmp_path = Path(tmp)
                archive = gh_updates.download(assets[match], tmp_path)
                say("installing")
                binary = _release_install(dest, archive)

            version = _version_of(binary)
            (dest / "BUILD.json").write_text(json.dumps(
                {"commit": tag, "built_at": _now_iso(), "version": version,
                 "source": "release", "asset": match, "repo": args.repo,
                 "ref": wanted_tag or tag, "name": args.name, "patches": []}, indent=2))
            return dest, tag
        if wanted_tag:
            break

    raise BuildFailed(
        (f"{args.repo}'s release {wanted_tag!r}" if wanted_tag else
         f"none of the last {len(releases)} {args.repo} releases") +
        f" had an asset matching {globs} for this machine")


# -- adopting a build that already exists ----------------------------------
def _commit_from_version(version: str) -> str:
    """The short commit ``--version`` names, when it names one -- llama-server prints
    ``0.3.0-dev (build 1, commit 62acc89)``, and that hash is a better directory name than
    anything derived from the rest of the string."""
    match = _COMMIT_IN_VERSION.search(version)
    return match.group(1) if match else ""


def _slug(text: str) -> str:
    cleaned = "".join(c if c.isalnum() else "-" for c in text.strip().lower())
    while "--" in cleaned:
        cleaned = cleaned.replace("--", "-")
    return cleaned.strip("-")[:40]


def _adopt(source_str: str) -> tuple[Path, str]:
    """Register a flat build directory that already exists -- a hand-built binary like
    ``~/.local/llama-next``, or a release zip someone unpacked by hand -- as a managed
    build, without compiling or downloading anything.

    Copied rather than linked in place: the source directory is somebody else's, outside
    ``~/.ml-stack``, and a rollback or a later ``build --force`` must not reach back into it.
    """
    source = Path(source_str).expanduser().resolve()
    if not source.is_dir():
        raise BuildFailed(f"{source} is not a directory")
    binary_name = _server_name()
    if not (source / binary_name).is_file():
        raise BuildFailed(f"no {binary_name} in {source}")

    version = _version_of(source / binary_name)
    if not version:
        raise BuildFailed(f"{source / binary_name} did not answer --version")
    commit = _commit_from_version(version) or _slug(version) or "adopted"

    dest = builds_dir() / commit
    dest.mkdir(parents=True, exist_ok=True)
    _copy_flat(source, dest)
    binary = dest / binary_name
    if not binary.is_file():
        raise BuildFailed(f"no {binary_name} in {source} after copying")
    if not is_windows():
        binary.chmod(binary.stat().st_mode | 0o111)
    (dest / "BUILD.json").write_text(json.dumps(
        {"commit": commit, "built_at": _now_iso(), "version": version,
         "source": f"adopted from {source}"}, indent=2))
    return dest, commit


def _cmd_adopt(args) -> int:
    try:
        dest, commit = _adopt(args.adopt)
        say(f"adopted {args.adopt} -> {dest} ({commit})")
        say("verifying")
        _verify_and_switch(dest, commit)
    except BuildFailed as exc:
        warn(f"error: {exc}")
        return 2
    return 0


# -- verifying, and only then switching -----------------------------------
def _relink(link: Path, target: Path) -> None:
    """Point ``link`` at ``target``: a symlink where the OS allows one without asking, a
    junction on Windows when it does not -- a symlink there needs Developer Mode or an
    administrator, a junction needs neither."""
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.is_symlink() or link.exists():
        if link.is_dir() and not link.is_symlink():
            shutil.rmtree(link)
        else:
            link.unlink()
    try:
        link.symlink_to(target, target_is_directory=True)
        return
    except OSError:
        if not is_windows():
            raise
    done = subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(target)],
                          capture_output=True, text=True)
    if done.returncode != 0:
        raise BuildFailed(f"could not point {link} at {target}: "
                          f"{(done.stderr or done.stdout or '').strip()}")


def _point_current(dest: Path) -> None:
    _relink(current_link(), dest)


def _verify_and_switch(dest: Path, commit: str, *, named: str | None = None) -> None:
    """Trust ``dest`` only once it answers ``--help``, then switch to it.

    ``named`` is the one thing that differs for a named build: it links ``named/<named>``
    instead of repointing ``current``, and a missing architecture is *reported*, never a
    reason to refuse -- a fork may read fewer architectures than master on purpose (or by
    being younger), and that is a fact about the fork, not a regression the way it would be
    for the default build. ``current`` is never touched when ``named`` is given.
    """
    from ml_stack.serve.backend import flags_of

    binary = dest / _server_name()
    help_flags = flags_of(binary)
    if not help_flags:
        raise BuildFailed(f"{binary} did not answer --help; leaving current alone")

    import ml_stack.setup as setup_module

    # Restricted to the real architecture names when a source checkout is around to read
    # them from -- otherwise a dylib string that merely shares a family prefix with an
    # architecture (a chat-template or vision-projector-type name: "phi4" names a chat
    # template, not an LLM_ARCH_PHI4 that does not exist) reads as one being lost. Empty
    # (a checkout with no readable llama-arch.cpp) is "could not read it", not "master has
    # none" -- restricting to nothing would make every comparison vacuously pass, which is
    # worse than the imprecise guess it would otherwise fall back to.
    source = src_dir()
    known = _arches_from_source(source) or None if source.is_dir() else None
    new_arches = setup_module._arches(dest, known=known)
    baseline = find_binary("llama-server")
    old_arches = setup_module._arches(str(baseline), known=known) if baseline else set()
    missing = old_arches - new_arches

    if named:
        say(f"  {binary} answers --help ({len(help_flags)} flags) and reads "
            f"{len(new_arches)} architectures"
            + (f"; missing {', '.join(sorted(missing))} that the current build reads"
                 if missing else ""))
        named_dir().mkdir(parents=True, exist_ok=True)
        _relink(named_dir() / named, dest)
        say(f"  named build {named!r} -> {dest} ({commit})")
        return

    if missing:
        raise BuildFailed(
            "the new build is missing " + ", ".join(sorted(missing)) +
            ", which the current build reads; leaving current alone")

    say(f"  {binary} answers --help ({len(help_flags)} flags) and reads "
        f"{len(new_arches)} architectures"
        + (f", a superset of the current {len(old_arches)}" if old_arches else ""))
    _point_current(dest)
    say(f"  current -> {dest} ({commit})")


# -- rollback and --check -------------------------------------------------
def _do_rollback() -> int:
    entries: list[tuple[str, Path]] = []
    for manifest in sorted(builds_dir().glob("*/BUILD.json")):
        try:
            info = json.loads(manifest.read_text())
        except (OSError, ValueError):
            continue
        entries.append((str(info.get("built_at", "")), manifest.parent))
    entries.sort()

    link = current_link()
    current = link.resolve() if link.is_symlink() or link.exists() else None
    for _, build_dir in reversed(entries):
        if build_dir != current:
            _point_current(build_dir)
            say(f"current -> {build_dir}")
            return 0
    warn("no earlier build to roll back to")
    return 2


def _report(args) -> int:
    target: Path | None = None
    link = current_link()
    if link.is_symlink() or link.exists():
        target = link / _server_name()
    else:
        found = find_binary("llama-server")
        target = Path(found) if found else None
    if target is None or not target.is_file():
        say("no llama-server build found")
        return 1

    build_json = target.parent / "BUILD.json"
    if build_json.is_file():
        try:
            info = json.loads(build_json.read_text())
        except (OSError, ValueError):
            info = {}
        say(f"{target}")
        say(f"  commit {info.get('commit', '?')} ({info.get('source', '?')}), "
            f"built {info.get('built_at', '?')}, {info.get('version', '?')}")
    else:
        say(f"{target}  {_version_of(target) or 'version unknown'} (not a managed build)")

    source = src_dir()
    if source.is_dir():
        import ml_stack.setup as setup_module

        master = _arches_from_source(source)
        if not master:
            say(f"could not read architecture names out of {source} -- "
                "src/llama-arch.cpp may have moved or renamed its table")
        else:
            mine = setup_module._arches(str(target), known=master)
            lacking = master - mine
            if lacking:
                say("master reads architectures this build lacks: "
                    + ", ".join(sorted(lacking)))
            else:
                say("reads every architecture master's own source does")
    else:
        say(f"no source checkout at {source} to compare architectures against "
            "-- ml-stack-serve build --from source clones one")
    return 0


# -- --list: current, and every named build alongside it -------------------
def _named_builds() -> list[tuple[str, Path]]:
    """Every named build's link, sorted by name. Not every entry is trustworthy on its own
    -- a link is only ever written once ``_verify_and_switch`` has already run for it -- but
    a link that no longer resolves (its build directory was removed by hand) is skipped
    rather than reported as a build that is not there."""
    named = named_dir()
    if not named.is_dir():
        return []
    out = []
    for link in sorted(named.iterdir()):
        if (link.is_symlink() or link.is_dir()) and (link / _server_name()).exists():
            out.append((link.name, link))
    return out


def _manifest_of(build_dir: Path) -> dict:
    manifest = build_dir / "BUILD.json"
    if not manifest.is_file():
        return {}
    try:
        return json.loads(manifest.read_text())
    except (OSError, ValueError):
        return {}


def _cmd_list() -> int:
    from ml_stack.setup import _age

    def _line(label: str, build_dir: Path) -> str:
        info = _manifest_of(build_dir)
        age = _age(str(info.get("built_at", ""))) or "?"
        repo = info.get("repo", "ggml-org/llama.cpp")
        return f"{label:14} {info.get('commit', '?'):12} {age:>4} old  {repo}"

    link = current_link()
    if link.is_symlink() or link.exists():
        say(_line("current", link))
    else:
        say("current        not built yet -- ml-stack-serve build")

    named = _named_builds()
    for name, link in named:
        say(_line(name, link))
    if not named:
        say("no named builds -- ml-stack-serve build --repo OWNER/REPO --ref REF --name NAME")
    return 0


# -- keeping it fresh on its own -------------------------------------------
def _persist_argv() -> list[str]:
    found = shutil.which("ml-stack-serve")
    if found:
        return [found, "build"]
    return [sys.executable, "-m", "ml_stack.serve.cli", "build"]


def _install_persist_macos(*, every: int = WEEK_SECONDS) -> Path:
    import plistlib

    log_dir = root() / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    plist = {
        "Label": PERSIST_LABEL,
        "ProgramArguments": _persist_argv(),
        "StartInterval": every,
        "StandardOutPath": str(log_dir / "build.log"),
        "StandardErrorPath": str(log_dir / "build.log"),
        "EnvironmentVariables": {"PATH": os.environ.get("PATH", "/usr/bin:/bin")},
    }
    PERSIST_PLIST.parent.mkdir(parents=True, exist_ok=True)
    PERSIST_PLIST.write_bytes(plistlib.dumps(plist))
    subprocess.run(["launchctl", "unload", str(PERSIST_PLIST)],
                   capture_output=True, check=False)
    done = subprocess.run(["launchctl", "load", "-w", str(PERSIST_PLIST)],
                          capture_output=True, text=True, check=False)
    if done.returncode != 0:
        raise BuildFailed(f"launchctl refused the job: {(done.stderr or '').strip()}")
    return PERSIST_PLIST


def _install_persist_windows() -> str:
    """A weekly Scheduled Task, since Windows has no LaunchAgent. Written against a faked
    ``subprocess.run`` only -- report to Adam that this is untested on a real Windows
    machine before relying on it."""
    argv = _persist_argv()
    quoted = " ".join(f'"{a}"' if " " in a else a for a in argv)
    done = subprocess.run(
        ["schtasks", "/Create", "/F", "/TN", PERSIST_TASK, "/TR", quoted,
         "/SC", "WEEKLY", "/RL", "LIMITED"],
        capture_output=True, text=True)
    if done.returncode != 0:
        raise BuildFailed(f"schtasks refused the job: {(done.stderr or '').strip()}")
    return PERSIST_TASK


def _cmd_persist() -> int:
    try:
        system = platform.system()
        if system == "Windows":
            name = _install_persist_windows()
            say(f"installed the scheduled task {name!r}")
        elif system == "Darwin":
            path = _install_persist_macos()
            say(f"installed {path}")
        else:
            warn("no scheduled-refresh install for this platform yet; run "
                 "'ml-stack-serve build' from cron or a timer of your own")
            return 1
    except BuildFailed as exc:
        warn(f"error: {exc}")
        return 2
    say(f"  ml-stack-serve build will run every {WEEK_SECONDS // 86400} days -- a "
        "refresh that fails verification changes nothing, which is what makes this safe "
        "to leave unattended")
    return 0


# -- the command -----------------------------------------------------------
def cmd_build(args) -> int:
    if args.check:
        return _report(args)
    if getattr(args, "list", False):
        return _cmd_list()
    if args.rollback:
        return _do_rollback()
    if args.persist:
        return _cmd_persist()
    if getattr(args, "adopt", ""):
        return _cmd_adopt(args)

    name = getattr(args, "name", "") or ""
    if name and not getattr(args, "repo", ""):
        warn("error: --name requires --repo OWNER/REPO")
        return 2

    kind = args.source_kind or ("source" if _can_build_from_source() else "release")
    if kind == "release" and patch_files():
        warn(f"a downloaded release carries none of the {len(patch_files())} patches in "
             f"{patches_dir()}; build --from source for those")
    try:
        if name:
            if kind == "release":
                dest, commit = _build_from_release_named(args)
            else:
                dest, commit = _build_from_source_named(args)
            say("verifying")
            _verify_and_switch(dest, commit, named=name)
        else:
            if kind == "release":
                dest, commit = _build_from_release(args)
            else:
                dest, commit = _build_from_source(args)
            say("verifying")
            _verify_and_switch(dest, commit)
    except BuildFailed as exc:
        warn(f"error: {exc}")
        return 2
    return 0
