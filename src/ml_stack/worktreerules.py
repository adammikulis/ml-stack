"""Which paths and shell commands belong to the primary checkout, shared by the Claude Code
hooks, the harness hook and the git hooks. Standard library only: the hook scripts load it by file."""

from __future__ import annotations

import os
import re
import shlex
import subprocess
import sys
from collections.abc import Iterable
from pathlib import Path

__all__ = ["AGENT_MARKERS", "bash_refusal", "checkouts", "commit_refusal", "edit_refusal",
           "patch_paths", "switched_off"]

AGENT_MARKERS = ("CLAUDECODE", "ML_STACK_AGENT", "ML_STACK_NONINTERACTIVE")
CHANGES_THE_TREE = frozenset({"add", "commit", "checkout", "switch", "reset", "restore", "stash",
                              "rebase", "cherry-pick", "am", "apply", "merge", "mv", "rm", "clean"})

RUN = r"^[\s({!]*((\w+=\S*|exec|then|do|time|sudo)\s+)*"
GIT = RUN + r"git\s+((-C\s+\S+|-c\s+\S+|--no-pager)\s+)*(?P<verb>[a-z-]+)"
INSTALL = RUN + r"(uv\s+)?(python3?\s+-m\s+)?pip3?\s+install\b"

USE_A_WORKTREE = (
    "The primary checkout is what runs; it changes only by landing a branch. Work in your own "
    "worktree: git worktree add -b <branch> ../ml-stack-<branch> "
    '"$(git -C {primary} branch --show-current)" -- then edit and commit there.')


def switched_off() -> bool:
    """True when MLSTACK_GUARD=off."""
    return os.environ.get("MLSTACK_GUARD") == "off"


def checkouts(directory: str | Path) -> tuple[Path, Path] | None:
    """(top of the checkout holding `directory`, the primary checkout), None outside a repository."""
    here = Path(directory)
    while not here.exists() and here != here.parent:
        here = here.parent
    done = subprocess.run(
        ["git", "-C", str(here), "rev-parse", "--path-format=absolute", "--show-toplevel",
         "--git-common-dir"], capture_output=True, text=True, timeout=5, check=False)
    lines = done.stdout.split("\n")
    if done.returncode != 0 or len(lines) < 2:
        return None
    top, common = Path(lines[0]).resolve(), Path(lines[1]).resolve()
    return top, common.parent if common.name == ".git" else common


def in_primary(directory: str | Path) -> Path | None:
    """The primary checkout when `directory` is inside it and not inside a linked worktree."""
    found = checkouts(directory)
    return found[1] if found and found[0] == found[1] and _enforced(found[1]) else None


def _enforced(primary: Path) -> bool:
    return (primary / "scripts" / "hooks" / "primary-only").exists()


def _unquote(word: str) -> str:
    return re.sub(r"^~(?=/|$)", os.environ.get("HOME", "~"), word.strip("'\""))


def edit_refusal(paths: Iterable[str], cwd: str) -> str:
    """Why a write to any of `paths` is refused, empty when none is in the primary checkout."""
    if switched_off():
        return ""
    base = Path(cwd or Path.cwd())
    for raw in filter(None, paths):
        target = base / _unquote(raw)
        primary = in_primary(target.parent)
        if primary:
            return f"{target} is in the primary checkout. " + USE_A_WORKTREE.format(primary=primary)
    return ""


def patch_paths(patch: str) -> list[str]:
    """The files an `apply_patch` body names."""
    return re.findall(r"^\*\*\* (?:Add|Update|Delete) File: (.+)$", patch, re.M)


QUOTED_OR_REDIRECT = re.compile(
    r"""('[^']*'|"[^"]*")|(?:(?<=\s)|^)(?:\d*|&)(?:>>|>\||>|<<<|<<-?|<)&?\s*[^\s;&|]*""")


def _without_redirects(segment: str) -> str:
    return QUOTED_OR_REDIRECT.sub(lambda found: found.group(1) or "", segment).strip()


def _segments(command: str) -> list[str]:
    parts = (_without_redirects(part) for part in re.split(r"&&|\|\||[;|\n]", command))
    return [part for part in parts if part]


def _cd(segment: str, here: Path) -> Path | None:
    found = re.match(RUN + r"cd\s+(\S+)", segment)
    return here / _unquote(found.group(found.lastindex)) if found else None


def _landing(segment: str) -> bool:
    try:
        words = shlex.split(segment)
    except ValueError:
        words = segment.split()
    after = words[words.index("merge") + 1:] if "merge" in words else []
    names = [one for one in after if not one.startswith("-")]
    return ("--ff-only" in after and "--no-ff" not in after and len(names) == 1
            and ":" not in names[0] and ".." not in names[0])


def _installs(segment: str, here: Path) -> list[Path]:
    if not re.match(INSTALL, segment):
        return []
    words, found = segment.split(), []
    for i, word in enumerate(words):
        if word in ("-e", "--editable") and i + 1 < len(words):
            raw = words[i + 1]
        elif word.startswith("--editable="):
            raw = word.split("=", 1)[1]
        else:
            continue
        found.append((here / re.sub(r"\[[^\]]*\]$", "", _unquote(raw))).resolve())
    return found


def bash_refusal(command: str, cwd: str) -> str:
    """Why `command` may not run from `cwd`, empty when it may."""
    if switched_off():
        return ""
    here = Path(cwd or Path.cwd())
    for segment in _segments(command):
        moved = _cd(segment, here)
        if moved is not None:
            here = moved
            continue
        git = re.match(GIT, segment)
        if git and git["verb"] in CHANGES_THE_TREE:
            given = re.search(r"\s-C\s+(\S+)", segment)
            where = here / _unquote(given.group(1)) if given else here
            primary = in_primary(where)
            if primary and not (git["verb"] == "merge" and _landing(segment)):
                what = ("`git merge` in the primary checkout lands one branch: "
                        "`git merge --ff-only <branch>`" if git["verb"] == "merge"
                        else f"`git {git['verb']}` changes the primary checkout")
                return f"{what}. " + USE_A_WORKTREE.format(primary=primary)
        for target in _installs(segment, here):
            found = checkouts(target)
            if found and target != found[1] and _enforced(found[1]):
                return (f"`pip install -e {target}` repoints the machine's install at that tree and "
                        f"every `ml-stack-*` command then runs it. The install points at the primary "
                        f"checkout, {found[1]}, which changes only by landing a branch; run a "
                        f"worktree's code with `PYTHONPATH=src`.")
    return ""


def _acting_as_agent(environ: dict[str, str]) -> bool:
    return any(environ.get(name) for name in AGENT_MARKERS) or not (
        sys.stdin.isatty() and sys.stdout.isatty())


def commit_refusal(cwd: str, environ: dict[str, str] | None = None) -> str:
    """Why an agent may not commit from `cwd`, empty for a person at a terminal."""
    env = dict(os.environ if environ is None else environ)
    if switched_off() or not _acting_as_agent(env):
        return ""
    found = checkouts(cwd)
    if not found:
        return ""
    top, primary = found
    if not _enforced(primary):
        return ""
    if top == primary:
        return "a commit in the primary checkout. " + USE_A_WORKTREE.format(primary=primary)
    dev = subprocess.run(["git", "-C", str(primary), "branch", "--show-current"],
                         capture_output=True, text=True, check=False).stdout.strip()
    here = subprocess.run(["git", "-C", str(top), "branch", "--show-current"],
                          capture_output=True, text=True, check=False).stdout.strip()
    if dev and here == dev:
        return (f"a commit on {dev}, the development branch. Commit on your own branch; it "
                f"lands with `git merge --ff-only` from the primary checkout.")
    return ""
