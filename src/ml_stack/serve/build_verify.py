"""Adopting a build that already exists, and trusting a build only once it answers
``--help`` and reads every architecture the previous one did -- then, and only then,
switching ``current`` (or a named build's own link) to point at it."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import ml_stack.setup as setup_module
from ml_stack.log import say, warn
from ml_stack.serve.backend import flags_of
from ml_stack.serve.binary import is_windows
from ml_stack.serve.build_paths import (
    BuildFailed,
    builds_dir,
    current_link,
    named_dir,
    slug,
    src_dir,
)
from ml_stack.serve.build_platform import (
    arches_from_source,
    copy_flat,
    now_iso,
    server_name,
    version_of,
)

__all__ = ["adopt", "cmd_adopt", "point_current", "relink", "verify_and_switch"]

# llama-server prints `0.3.0-dev (build 1, commit 62acc89)`; the hash it names is a better
# directory name than anything derived from the rest of the string.
_COMMIT_IN_VERSION = re.compile(r"commit[ :=]?\s*([0-9a-f]{7,40})", re.IGNORECASE)


def _commit_from_version(version: str) -> str:
    match = _COMMIT_IN_VERSION.search(version)
    return match.group(1) if match else ""


def adopt(source_str: str) -> tuple[Path, str]:
    """Register a flat build directory that already exists -- a hand-built binary like
    ``~/.local/llama-next``, or a release zip someone unpacked by hand -- as a managed
    build, without compiling or downloading anything.

    Copied rather than linked in place: the source directory is somebody else's, outside
    ``~/.ml-stack``, and a rollback or a later ``build --force`` must not reach back into it.
    """
    source = Path(source_str).expanduser().resolve()
    if not source.is_dir():
        raise BuildFailed(f"{source} is not a directory")
    binary_name = server_name()
    if not (source / binary_name).is_file():
        raise BuildFailed(f"no {binary_name} in {source}")

    version = version_of(source / binary_name)
    if not version:
        raise BuildFailed(f"{source / binary_name} did not answer --version")
    commit = _commit_from_version(version) or slug(version) or "adopted"

    dest = builds_dir() / commit
    dest.mkdir(parents=True, exist_ok=True)
    copy_flat(source, dest)
    binary = dest / binary_name
    if not binary.is_file():
        raise BuildFailed(f"no {binary_name} in {source} after copying")
    if not is_windows():
        binary.chmod(binary.stat().st_mode | 0o111)
    (dest / "BUILD.json").write_text(json.dumps(
        {"commit": commit, "built_at": now_iso(), "version": version,
         "source": f"adopted from {source}"}, indent=2))
    return dest, commit


def cmd_adopt(args) -> int:
    try:
        dest, commit = adopt(args.adopt)
        say(f"adopted {args.adopt} -> {dest} ({commit})")
        say("verifying")
        verify_and_switch(dest, commit)
    except BuildFailed as exc:
        warn(f"error: {exc}")
        return 2
    return 0


def relink(link: Path, target: Path) -> None:
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


def point_current(dest: Path) -> None:
    relink(current_link(), dest)


def _current_server() -> Path | None:
    """The server binary ``current`` points at, or None before the first build."""
    binary = current_link() / server_name()
    return binary if binary.is_file() else None


def _recorded_arches(build: Path) -> set[str]:
    try:
        info = json.loads((build / "BUILD.json").read_text())
    except (OSError, ValueError):
        return set()
    return set(info.get("arches") or []) if isinstance(info, dict) else set()


def _known_arches(dest: Path, current: Path | None) -> set[str] | None:
    """The architecture names llama.cpp's source defines for either build; None when unread.

    A libllama string sharing a family prefix (``phi4`` is a chat template) is not an
    architecture, so the strings are read against these names when there are any.
    """
    names = arches_from_source(src_dir()) if src_dir().is_dir() else set()
    names |= _recorded_arches(dest)
    if current is not None:
        names |= _recorded_arches(current.resolve().parent)
    return names or None


def verify_and_switch(dest: Path, commit: str, *, named: str | None = None) -> None:
    """Trust ``dest`` only once it answers ``--help``, then switch to it.

    ``named`` is the one thing that differs for a named build: it links ``named/<named>``
    instead of repointing ``current``, and a missing architecture is *reported*, never a
    reason to refuse -- a fork may read fewer architectures than master on purpose (or by
    being younger), and that is a fact about the fork, not a regression the way it would be
    for the default build. ``current`` is never touched when ``named`` is given.
    """
    binary = dest / server_name()
    help_flags = flags_of(binary)
    if not help_flags:
        raise BuildFailed(f"{binary} did not answer --help; leaving current alone")

    current = _current_server()
    known = _known_arches(dest, current)
    new_arches = setup_module._arches(dest, known=known)
    old_arches = setup_module._arches(str(current), known=known) if current else set()
    missing = old_arches - new_arches

    if named:
        say(f"  {binary} answers --help ({len(help_flags)} flags) and reads "
            f"{len(new_arches)} architectures"
            + (f"; missing {', '.join(sorted(missing))} that the current build reads"
                 if missing else ""))
        named_dir().mkdir(parents=True, exist_ok=True)
        relink(named_dir() / named, dest)
        say(f"  named build {named!r} -> {dest} ({commit})")
        return

    if missing:
        raise BuildFailed(
            "the new build is missing " + ", ".join(sorted(missing)) +
            ", which the current build reads; leaving current alone")

    say(f"  {binary} answers --help ({len(help_flags)} flags) and reads "
        f"{len(new_arches)} architectures"
        + (f", a superset of the current {len(old_arches)}" if old_arches else ""))
    point_current(dest)
    say(f"  current -> {dest} ({commit})")
