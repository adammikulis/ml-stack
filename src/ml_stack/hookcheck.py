"""Whether git will run this repository's hooks: the hooks directory, the hook files, the repository config and the guard scripts."""

from __future__ import annotations

import hashlib
import os
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path

from ml_stack import home
from ml_stack.files import read_json, write_json

__all__ = ["HOOKS", "Problem", "context", "digest", "inspect", "line", "remember", "remembered"]

HOOKS = ("pre-commit", "commit-msg", "pre-push", "post-merge")
SCRIPTS = "scripts/hooks"
VERSION = 1
LINE_LIMIT = 190
RUNS_COMMANDS = {"core.fsmonitor": "false", "core.sshcommand": None}


@dataclass(frozen=True, slots=True)
class Problem:
    """One reason git may not run the hooks: what is wrong, how a person repairs it, and its kind (install, config or tampered)."""

    what: str
    repair: str = ""
    kind: str = "install"


def _git(repo: Path, *words: str) -> str | None:
    done = subprocess.run(["git", "-C", str(repo), *words], capture_output=True, text=True, timeout=30)
    return done.stdout.strip() if done.returncode == 0 else None


def _entries(repo: Path, scope: str) -> list[tuple[str, str]]:
    done = subprocess.run(["git", "-C", str(repo), "config", f"--{scope}", "--list", "-z"], capture_output=True, text=True, timeout=30)
    if done.returncode:
        return []
    pairs = (item.partition("\n") for item in done.stdout.split("\0") if item)
    return [(key, value) for key, _, value in pairs]


def _repair(top: Path, scope: str, key: str) -> str:
    return f"git -C {shlex.quote(str(top))} config --{scope} --unset-all {shlex.quote(key)}"


def _surprise(key: str, value: str, default: Path, top: Path) -> str:
    name = key.lower()
    if name == "core.hookspath":
        target = home.expand(value)
        return "" if (target if target.is_absolute() else top / target).resolve() == default.resolve() else f"core.hooksPath sends hooks to {value}"
    if name in RUNS_COMMANDS:
        return "" if value == RUNS_COMMANDS[name] else f"{key} runs {value}"
    if name.startswith("alias.") and value.startswith("!"):
        return f"{key} runs a shell command"
    if name.startswith("url.") and name.endswith(("insteadof", "pushinsteadof")):
        return f"{key} rewrites remote addresses"
    if name.startswith("remote.") and name.endswith(".pushurl"):
        return f"{key} sends pushes to {value}"
    return ""


def _config(top: Path, common: Path) -> list[Problem]:
    found: list[Problem] = []
    local = _entries(top, "local")
    for scope, entries in (("local", local), ("worktree", [e for e in _entries(top, "worktree") if e not in local])):
        for key, value in entries:
            said = _surprise(key, value, common / "hooks", top)
            if said:
                found.append(Problem(f"{said} (in the {scope} repository config)", _repair(top, scope, key), "config"))
    return found


def _ours(hook: Path, roots: list[Path]) -> bool:
    if hook.is_symlink():
        return any(hook.resolve().is_relative_to(root) for root in roots)
    try:
        text = hook.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False
    return f"{SCRIPTS}/" in text and "exec" in text


def _hooks(top: Path, directory: Path, common: Path, install: str) -> list[Problem]:
    if not directory.is_dir():
        return [Problem(f"git looks for hooks in {directory}, which does not exist", install)]
    roots = [(top / SCRIPTS).resolve(), (common.parent / SCRIPTS).resolve()]
    absent: list[str] = []
    found: list[Problem] = []
    for name in (h for h in HOOKS if (top / SCRIPTS / h).is_file()):
        hook = directory / name
        if not hook.exists() or not _ours(hook, roots):
            absent.append(name)
        elif not os.access(hook, os.X_OK):
            found.append(Problem(f"{name} in {directory} is not executable", f"chmod +x {shlex.quote(str(hook))}"))
    return [Problem(f"not installed: {', '.join(absent)}", install), *found] if absent else found


def _scripts(top: Path) -> list[Problem]:
    found: list[Problem] = []
    for row in (_git(top, "ls-tree", "-r", "HEAD", "--", SCRIPTS) or "").splitlines():
        meta, _, path = row.partition("\t")
        mode, _, blob, *_ = meta.split()
        if mode == "120000":
            continue
        actual = _git(top, "hash-object", "--", path) if (top / path).is_file() else None
        if actual != blob:
            what = "is missing" if actual is None else "differs from HEAD"
            found.append(Problem(f"{path} {what}", f"git -C {shlex.quote(str(top))} restore --source=HEAD -- {shlex.quote(path)}", "tampered"))
    return found


def inspect(repo: Path) -> list[Problem]:
    """Every reason git may not run the hooks of the repository holding `repo`; empty when it will."""
    top = _git(repo, "rev-parse", "--show-toplevel")
    directory = _git(repo, "rev-parse", "--path-format=absolute", "--git-path", "hooks")
    common = _git(repo, "rev-parse", "--path-format=absolute", "--git-common-dir")
    if top is None or directory is None or common is None:
        return [Problem(f"{repo} is not a git checkout", kind="config")]
    configured = _config(Path(top), Path(common))
    installer = Path(top) / "scripts" / "install-hooks.sh"
    install = f"sh {shlex.quote(str(installer))}" if installer.is_file() and not any(p.what.startswith("core.hooksPath") for p in configured) else ""
    return [*configured, *_hooks(Path(top), Path(directory), Path(common), install), *_scripts(Path(top))]


def digest(problems: list[Problem]) -> str:
    """A short fingerprint of the problems, empty when there are none."""
    return hashlib.sha256("\n".join(p.what for p in problems).encode()).hexdigest()[:16] if problems else ""


def line(repo: Path, problems: list[Problem]) -> str:
    """One board line naming the first problem and how many more there are."""
    more = f" (+{len(problems) - 1} more)" if len(problems) > 1 else ""
    return f"git hooks in {repo.name}: {problems[0].what}{more}"[:LINE_LIMIT]


def context(repo: Path, problems: list[Problem]) -> str:
    """The text an agent is shown: each problem and the command a person runs to repair it."""
    rows = [f"- {p.what}" + (f"\n  a person repairs it with: {p.repair}" if p.repair else "") for p in problems]
    return f"Git hooks in {repo} may not run (an agent does not edit git config):\n" + "\n".join(rows)


def _record(repo: Path) -> Path:
    return home.state("hookcheck", hashlib.sha256(str(repo.resolve()).encode()).hexdigest()[:16] + ".json")


def remembered(repo: Path) -> str:
    """The fingerprint last announced for this repository."""
    row = read_json(_record(repo), {})
    return str(row.get("digest", "")) if isinstance(row, dict) else ""


def remember(repo: Path, fingerprint: str) -> None:
    """Record the fingerprint announced for this repository."""
    write_json(_record(repo), {"version": VERSION, "digest": fingerprint})
