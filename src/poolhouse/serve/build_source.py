"""Building llama-server from source: llama.cpp's own master, or a fork's ref, cloned or
fetched, patched, configured with cmake, compiled, and installed into a flat directory."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

from poolhouse.httpguard import Refused
from poolhouse.log import say
from poolhouse.net import git as netgit
from poolhouse.serve.binary import is_windows
from poolhouse.serve.build_patches import check_patches, verified_upstream
from poolhouse.serve.build_paths import (
    BuildFailed,
    builds_dir,
    named_dest,
    named_src_dir,
    patch_files,
    patch_stamp,
    src_dir,
)
from poolhouse.serve.build_platform import cmake_flags, copy_flat, now_iso, server_name, version_of

__all__ = ["build_from_source", "build_from_source_named"]

REPO_URL = "https://github.com/ggml-org/llama.cpp"
CMAKE_TARGET = "llama-server"          # the cmake target name, the same on every platform


# -- git ---------------------------------------------------------------
def _git(*args: str, cwd: Path | None = None, url: str = "") -> subprocess.CompletedProcess:
    try:
        return netgit.run(args, cwd=cwd, url=url)
    except (netgit.GitFailed, Refused) as exc:
        raise BuildFailed(str(exc)) from exc


def _fetch_master(source: Path) -> None:
    """Only the tip of master, not its history -- llama.cpp's full history is a large,
    slow clone that buys nothing here, the same reason ``ensure_converter`` elsewhere in
    this codebase already clones ``--depth 1``. Measured the way it was found: a first run
    against a bare repo was still cloning several minutes in and past 190MB. An existing
    checkout only gets new objects: its files and HEAD stay as they were until
    `_land_master` or `_checkout_commit` moves them, after the patches are known to apply."""
    if (source / ".git").is_dir():
        say(f"  fetching {REPO_URL} into {source}")
        _git("fetch", "--depth", "1", "origin", "master", cwd=source, url=REPO_URL)
    else:
        say(f"  cloning {REPO_URL} into {source} (--depth 1)")
        source.parent.mkdir(parents=True, exist_ok=True)
        _git("clone", "--depth", "1", "--branch", "master", REPO_URL, str(source), url=REPO_URL)


def _land_master(source: Path) -> None:
    _git("checkout", "master", cwd=source)
    _git("reset", "--hard", "origin/master", cwd=source)


def _fetch_commit(source: Path, commit: str, url: str = REPO_URL) -> None:
    """A shallow clone has only master's tip, so a specific commit is fetched by name
    before it can be checked out -- GitHub serves an arbitrary reachable commit SHA this
    way without needing the rest of history either."""
    say(f"  fetching {commit}")
    _git("fetch", "--depth", "1", "origin", commit, cwd=source, url=url)


def _checkout_commit(source: Path, commit: str, url: str = REPO_URL) -> None:
    _fetch_commit(source, commit, url)
    _git("checkout", "FETCH_HEAD", cwd=source)


def _short_commit(source: Path, rev: str = "HEAD") -> str:
    return _git("rev-parse", "--short", rev, cwd=source).stdout.strip()


def _full_commit(source: Path, rev: str) -> str:
    return _git("rev-parse", rev, cwd=source).stdout.strip()


# -- patches carried on top of upstream ----------------------------------------------
def _apply_patches(source: Path) -> list[str]:
    """Apply every patch onto ``source`` at the commit checked out, returning their names.

    The tree is reset to HEAD first, so a patch set that changed since the last build does
    not land on top of the one before it.
    """
    files = patch_files()
    if not files:
        return []
    say(f"  resetting {source.name} to HEAD before patching")
    _git("reset", "--hard", "HEAD", cwd=source)
    applied: list[str] = []
    for item in files:
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


def _sync_named_source(source: Path, repo: str, ref: str) -> None:
    """Clone or fetch ``repo`` into ``source``, then land on ``ref`` -- a tag, branch or
    SHA -- the same way ``_checkout_commit`` lands master's checkout on a specific commit.
    Left at the default branch's tip when ``ref`` is empty."""
    url = f"https://github.com/{repo}"
    if (source / ".git").is_dir():
        say(f"  fetching {url} into {source}")
        _git("fetch", "--depth", "1", "origin", cwd=source, url=url)
    else:
        say(f"  cloning {url} into {source} (--depth 1)")
        source.parent.mkdir(parents=True, exist_ok=True)
        _git("clone", "--depth", "1", url, str(source), url=url)
    if ref:
        _checkout_commit(source, ref, url)


# -- cmake ---------------------------------------------------------------
def _configure(source: Path) -> None:
    cmake = shutil.which("cmake")
    if cmake is None:
        raise BuildFailed("cmake is not on PATH")
    done = subprocess.run([cmake, "-B", "build", *cmake_flags()], cwd=source,
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
def _install_source_build(build_dir: Path, dest: Path, commit: str, *,
                          extra: dict | None = None) -> Path:
    bin_dir = build_dir / "bin"
    binary_name = server_name()
    if not (bin_dir / binary_name).is_file() and not (bin_dir / binary_name).is_symlink():
        raise BuildFailed(f"no {binary_name} in {bin_dir} -- the build did not produce one")
    dest.mkdir(parents=True, exist_ok=True)
    copied = set(copy_flat(bin_dir, dest))
    lib_dir = build_dir / "lib"
    if lib_dir.is_dir():
        copied |= set(copy_flat(lib_dir, dest))
    binary = dest / binary_name
    if not is_windows():
        binary.chmod(binary.stat().st_mode | 0o111)
    version = version_of(binary)
    info = {"commit": commit, "built_at": now_iso(), "version": version, "source": "source",
            "patches": []}
    if extra:
        info.update(extra)
    (dest / "BUILD.json").write_text(json.dumps(info, indent=2))
    return binary


def _wanted_commit(args) -> str:
    """The upstream commit to build: the one named, else the newest the patches are verified
    against, else (``--upstream-head``, or nothing verified) master's tip, which is ``""``."""
    if args.commit:
        return args.commit
    if getattr(args, "upstream_head", False):
        say("  --upstream-head: building upstream's head, which the patches may not apply to")
        return ""
    pinned = verified_upstream()
    if pinned:
        say(f"  building {pinned[:9]}, the upstream commit the patches were last verified "
            "against (--upstream-head builds master's tip instead)")
    return pinned


def build_from_source(args) -> tuple[Path, str]:
    source = Path(args.source).expanduser() if args.source else src_dir()
    want = _wanted_commit(args)
    if not args.source:
        _fetch_master(source)
    if want:
        _fetch_commit(source, want)
        rev = "FETCH_HEAD"
    else:
        rev = "HEAD" if args.source else "origin/master"

    commit = _short_commit(source, rev)
    stamp = patch_stamp()
    tag = f"{commit}-{stamp}" if stamp else commit
    dest = builds_dir() / tag
    if dest.is_dir() and (dest / "BUILD.json").is_file() and not args.force:
        say(f"{tag} is already built at {dest} -- pass --force to rebuild")
        return dest, tag

    check_patches(source, _full_commit(source, rev), _git)
    if want:
        _git("checkout", "FETCH_HEAD", cwd=source)
    elif not args.source:
        _land_master(source)
    applied = _apply_patches(source)
    jobs = args.jobs or (os.cpu_count() or 4)
    say(f"configuring {tag} ({', '.join(cmake_flags()) or 'CPU only'})")
    _configure(source)
    say(f"building ({jobs} jobs) -- this takes several minutes")
    _compile(source, jobs)
    say("installing")
    _install_source_build(source / "build", dest, tag, extra={"patches": applied})
    return dest, tag


def build_from_source_named(args) -> tuple[Path, str]:
    """Build a fork's own ref, kept at ``builds/<name>-<commit>/`` beside master's builds
    and linked from ``named/<name>`` rather than replacing ``current``."""
    source = _named_source_dir(args.name)
    _sync_named_source(source, args.repo, args.ref)

    commit = _short_commit(source)
    stamp = patch_stamp()
    commit = f"{commit}-{stamp}" if stamp else commit
    dest = named_dest(args.name, commit)
    if dest.is_dir() and (dest / "BUILD.json").is_file() and not args.force:
        say(f"{args.name}-{commit} is already built at {dest} -- pass --force to rebuild")
        return dest, commit

    check_patches(source, _full_commit(source, "HEAD"), _git)
    applied = _apply_patches(source)
    jobs = args.jobs or (os.cpu_count() or 4)
    say(f"configuring {args.name}-{commit} ({', '.join(cmake_flags()) or 'CPU only'})")
    _configure(source)
    say(f"building ({jobs} jobs) -- this takes several minutes")
    _compile(source, jobs)
    say("installing")
    _install_source_build(source / "build", dest, commit,
                          extra={"repo": args.repo, "ref": args.ref or commit,
                                 "name": args.name, "patches": applied})
    return dest, commit
