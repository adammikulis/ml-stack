"""Which commit a runtime deploy may build, who may redirect it, and the audit record every deploy command leaves."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from ml_stack import authority, home, person, runtime_store
from ml_stack.activity import writer
from ml_stack.runtime_deploy import DeployError, floor_of
from ml_stack.workspace import tokens

KIND = "runtime.deploy"
GATE = "runtime.deploy"
DETAIL_LIMIT = 150


def is_person() -> bool:
    """Whether stdin and stdout are terminals and no agent marker is set."""
    return not person.marked() and person.is_terminal(sys.stdin) and person.is_terminal(sys.stdout)


def common_dir(path: Path) -> Path:
    """The git directory shared by every worktree of the repository at `path`, read from files without running git."""
    link = path / ".git"
    if link.is_dir():
        return link.resolve()
    try:
        text = link.read_text(encoding="utf-8").strip() if link.is_file() else ""
    except (OSError, ValueError):
        text = ""
    if not text.startswith("gitdir:"):
        raise DeployError(f"{path} is not a git checkout")
    gitdir = Path(text.split(":", 1)[1].strip())
    gitdir = gitdir if gitdir.is_absolute() else path / gitdir
    pointer = gitdir / "commondir"
    if pointer.is_file():
        target = Path(pointer.read_text(encoding="utf-8").strip())
        gitdir = target if target.is_absolute() else gitdir / target
    return gitdir.resolve()


def main_worktree(path: Path) -> Path:
    """The primary checkout of the repository that holds `path`."""
    shared = common_dir(path)
    if shared.name != ".git":
        raise DeployError(f"{path} belongs to a repository with no primary checkout")
    return shared.parent


def recorded_primary() -> Path | None:
    """The primary checkout of the source recorded by an earlier deploy, or None."""
    named = runtime_store.read_state().get("checkout")
    if not isinstance(named, str) or not named:
        return None
    try:
        return main_worktree(home.expand(named).resolve())
    except DeployError:
        return None


def primary_for(supplied: Path, *, deploying: bool = True) -> Path:
    """The primary checkout to deploy from: the recorded one when `supplied` is a worktree of it.

    A person may name another repository; any other process that deploys must use the one an earlier deploy recorded.
    """
    recorded = recorded_primary()
    if recorded is not None and common_dir(supplied) == common_dir(recorded):
        return recorded
    if is_person() or not deploying:
        return main_worktree(supplied)
    if recorded is None:
        raise DeployError("no source repository is recorded: a person runs `ml-stack runtime ensure --checkout PATH` once at a terminal")
    raise DeployError(f"{supplied} is not a worktree of the recorded repository {recorded}; only a person at a terminal may deploy another repository")


def _git(primary: Path, *words: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(primary), *words], capture_output=True, text=True, timeout=60)


def tip(primary: Path) -> str:
    """The commit the primary checkout's development branch points at."""
    if _git(primary, "symbolic-ref", "-q", "HEAD").returncode:
        raise DeployError(f"{primary} is on a detached HEAD; check out the development branch")
    done = _git(primary, "rev-parse", "HEAD")
    if done.returncode:
        raise DeployError(f"git rev-parse HEAD failed in {primary}: {done.stderr.strip()[:200]}")
    return done.stdout.strip()


def admit(primary: Path, commit: str, *, allow_unmerged: bool = False) -> None:
    """Raise DeployError unless `commit` is the development branch tip or an ancestor of it (or `allow_unmerged`)."""
    head = tip(primary)
    if commit == head or allow_unmerged or _git(primary, "merge-base", "--is-ancestor", commit, head).returncode == 0:
        return
    raise DeployError(f"{commit[:7]} is not on the development branch ({head[:7]}); a person may deploy it with --allow-unmerged")


def floor(primary: Path) -> int:
    """The runtime epoch floor declared at the development branch tip."""
    return floor_of(primary, tip(primary))


def ignore_redirects() -> list[str]:
    """Remove the variables that move the state or cache root from an agent-started process; returns their names."""
    if not person.marked():
        return []
    names = [home.ROOT_ENV, home.CACHE_ENV, *home.OVERRIDES.values()]
    return [name for name in names if os.environ.pop(name, None)]


def authorize(command: str) -> str:
    """Who may run a deploy command here: "person" for a process no agent started, "delegated" for a lead agent
    while the owner has delegated the `runtime.deploy` gate; HumanRequired for an agent while it is the person's.

    A launcher or login unit carries no agent marker, so it is not gated; the `authority.use` row this writes for a
    delegated pass names the agent.
    """
    if not person.marked():
        return authority.PERSON
    return authority.require(GATE, f"runtime {command}")


def audit(command: str, commit: str, result: str, *, agent: str = "", detail: str = "", **passed: str) -> bool:
    """Append a sealed activity record for one deploy command; True when it was written."""
    who = agent or os.environ.get(tokens.AGENT_ENV, "")
    return writer.record(KIND, actor=f"agent:{who}" if who else None, subject=commit, outcome=result,
                         meta={"command": command, "agent": who, "person": is_person(), "authority": passed.get("via", ""),
                               "detail": detail[:DETAIL_LIMIT]})
