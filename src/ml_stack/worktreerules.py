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
    "Resolve destructive changes and conflicts in your own "
    "worktree: git worktree add -b <branch> ../ml-stack-<branch> "
    '"$(git -C {primary} branch --show-current)" -- then edit and commit there.')


def switched_off() -> bool:
    """True when MLSTACK_GUARD=off."""
    return os.environ.get("MLSTACK_GUARD") == "off"


def _git_environment() -> dict[str, str]:
    selectors = {"GIT_DIR", "GIT_WORK_TREE", "GIT_COMMON_DIR", "GIT_INDEX_FILE", "GIT_PREFIX", "GIT_NAMESPACE",
                 "GIT_OBJECT_DIRECTORY", "GIT_ALTERNATE_OBJECT_DIRECTORIES"}
    return {name: value for name, value in os.environ.items() if name not in selectors}


def checkouts(directory: str | Path) -> tuple[Path, Path] | None:
    """(top of the checkout holding `directory`, the primary checkout), None outside a repository."""
    here = Path(directory)
    while not here.exists() and here != here.parent:
        here = here.parent
    done = subprocess.run(
        ["git", "-C", str(here), "rev-parse", "--path-format=absolute", "--show-toplevel",
         "--git-common-dir"], capture_output=True, text=True, timeout=5, check=False, env=_git_environment())
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
    return re.sub(r"^~(?=/|$)", lambda _: os.environ.get("HOME", "~"), word.strip("'\""))


def _branch(directory: Path) -> str:
    return subprocess.run(["git", "-C", str(directory), "branch", "--show-current"],
                          capture_output=True, text=True, check=False,
                          env=_git_environment()).stdout.strip()


def edit_refusal(paths: Iterable[str], cwd: str) -> str:
    """Why a write to any of `paths` is refused, empty for permitted development changes."""
    if switched_off():
        return ""
    base = Path(cwd or Path.cwd())
    for raw in filter(None, paths):
        target = base / _unquote(raw)
        found = checkouts(target.parent)
        if found and _enforced(found[1]) and _branch(found[0]) == "main":
            return f"{target} is on main; development changes require a development branch."
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



def _worktree_target(segment: str) -> str | None:
    try:
        words = [_unquote(word) for word in shlex.split(segment, posix=os.name != "nt")]
    except ValueError:
        return None
    if "worktree" not in words:
        return None
    index = words.index("worktree") + 1
    if words[index:index + 1] != ["add"]:
        return None
    index += 1
    while index < len(words):
        word = words[index]
        if word in {"-b", "-B", "--reason"}:
            index += 2
        elif word == "--":
            return words[index + 1] if index + 1 < len(words) else None
        elif word.startswith("-"):
            index += 1
        else:
            return word
    return None

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
        if git and git["verb"] == "worktree" and (target := _worktree_target(segment)):
            given = re.search(r"\s-C\s+(\S+)", segment)
            where = here / _unquote(given.group(1)) if given else here
            found = checkouts(where)
            if found and _enforced(found[1]) and (why := worktree_refusal(where / target, where)):
                return why
        if git and git["verb"] in CHANGES_THE_TREE:
            given = re.search(r"\s-C\s+(\S+)", segment)
            where = here / _unquote(given.group(1)) if given else here
            primary = in_primary(where)
            found = checkouts(where)
            if found and _enforced(found[1]) and git["verb"] in {"add", "commit"}:
                if _branch(found[0]) == "main":
                    return "An agent may not stage or commit on main."
                continue
            if primary and not (git["verb"] == "merge" and _landing(segment)):
                what = ("`git merge` in the primary checkout lands one branch: "
                        "`git merge --ff-only <branch>`" if git["verb"] == "merge"
                        else f"`git {git['verb']}` changes the primary checkout")
                return f"{what}. " + USE_A_WORKTREE.format(primary=primary)
        for target in _installs(segment, here):
            found = checkouts(target)
            if found and _enforced(found[1]):
                return (f"`pip install -e {target}` is forbidden; use an immutable built wheel "
                        "for runtimes or `PYTHONPATH=src` for development checks.")
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
    here = _branch(top)
    if here == "main":
        return "a commit on main; use a development branch."
    if top == primary:
        return ""
    dev = _branch(primary)
    if dev and here == dev:
        return (f"a commit on {dev}, the development branch. Commit on your own branch; it "
                f"lands with `git merge --ff-only` from the primary checkout.")
    return ""


def worktree_refusal(target: str | Path, source: str | Path, *, registered: bool = False) -> str:
    """Why a worktree path is nested inside an existing repository checkout."""
    found = checkouts(source)
    if not found:
        return "The source must belong to a Git checkout"
    primary = found[1]
    listed = subprocess.run(["git", "-C", str(primary), "worktree", "list", "--porcelain"],
                            capture_output=True, text=True, timeout=5, check=False, env=_git_environment())
    if listed.returncode:
        return "Git could not verify existing checkout paths"
    path = Path(target).resolve()
    for line in listed.stdout.splitlines():
        if line.startswith("worktree "):
            checkout = Path(line.removeprefix("worktree ")).resolve()
            if (path == checkout and not registered) or checkout in path.parents:
                return f"{path} is inside checkout {checkout}; create the task worktree beside the checkout"
    return ""
