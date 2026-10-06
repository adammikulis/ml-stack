"""Token files under the workspace state directory, and how a process finds its own token."""

from __future__ import annotations

import os
import stat
from collections.abc import Mapping
from pathlib import Path

from ml_stack.files import writing
from ml_stack.sentinel import human
from ml_stack.workspace.identity import PREFIX, TOKEN_ENV, Denied, valid_id
from ml_stack.workspace.windows_tokens import problem as windows_problem, restrict

__all__ = ["AGENT_ENV", "OWNER_FILE", "directory", "inside_repo", "load", "prepare", "problem",
           "read_file", "resolve", "store"]

AGENT_ENV = "ML_STACK_WORKSPACE_AGENT"
OWNER_FILE = ".owner"
"""The person's own token file; not an agent name, so no ``--agent`` reaches it."""


def directory(base: Path) -> Path:
    """The token directory under ``base``, registered with sentinel so no tool call names it."""
    path = base / "tokens"
    human.protect(path)
    return path


def inside_repo(path: Path) -> Path | None:
    real = path.resolve()
    return next((p for p in (real, *real.parents) if (p / ".git").exists()), None)


def prepare(base: Path) -> Path:
    """The token directory, made owner-only; ValueError when it is a symlink or sits inside a
    git work tree."""
    path = directory(base)
    try:
        info = path.lstat()
    except FileNotFoundError:
        pass
    else:
        if _redirected(info):
            raise ValueError(f"{path} is a symlink or Windows reparse point")
    repo = inside_repo(path)
    if repo is not None:
        raise ValueError(f"{path} resolves into the git work tree {repo}; "
                         f"move ML_STACK_HOME out of it")
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    if os.name == "nt":
        if _redirected(base.lstat()):
            raise ValueError(f"{base} is a symlink or Windows reparse point")
        restrict(base)
        restrict(path)
    else:
        path.chmod(0o700)
    return path


def _redirected(info) -> bool:
    return (stat.S_ISLNK(info.st_mode) or (os.name == "nt"
            and bool(info.st_file_attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT)))


def problem(path: Path) -> str:
    """Why ``path`` may not hold a token (or be the directory of them), or an empty string."""
    try:
        info = path.lstat()
    except FileNotFoundError:
        return "missing"
    if _redirected(info):
        return "is a symlink or Windows reparse point"
    if not (stat.S_ISREG(info.st_mode) or stat.S_ISDIR(info.st_mode)):
        return "is not a plain file"
    if os.name == "nt":
        return windows_problem(path)
    if info.st_uid != os.getuid():
        return "belongs to another user"
    if info.st_mode & 0o077:
        return f"mode {info.st_mode & 0o777:o} lets others read it; chmod {'700' if stat.S_ISDIR(info.st_mode) else '600'}"
    return ""


def store(base: Path, name: str, token: str) -> Path:
    """Write ``token`` to ``name``'s file, atomically and readable by this user only."""
    if name != OWNER_FILE and not valid_id(name):
        raise ValueError(f"{name!r} is not a usable agent id")
    target = prepare(base) / name.replace("/", "~")
    with writing(target) as tmp:
        if os.name == "nt":
            restrict(tmp)
        else:
            tmp.chmod(0o600)
        tmp.write_text(token + "\n", encoding="utf-8")
    return target


def read_file(path: Path) -> str:
    """The token in ``path``; `Denied` when the file is missing or readable by others."""
    why = problem(path)
    if why:
        raise Denied(f"token file {path}: {why}")
    return path.read_text(encoding="utf-8").strip()


def load(base: Path, name: str) -> str:
    """``name``'s token; `Denied` when its file or directory is unsafe or holds another agent's."""
    if not valid_id(name):
        raise Denied(f"{name!r} is not a usable agent id")
    why = problem(directory(base))
    if why:
        raise Denied(f"token directory {directory(base)}: {why}")
    token = read_file(directory(base) / name.replace("/", "~"))
    if not token.startswith(f"{PREFIX}{name}."):
        raise Denied(f"the token file for {name} holds a token for another agent")
    return token


def resolve(base: Path, *, token_file: str = "", agent: str = "",
            env: Mapping[str, str] | None = None) -> str:
    """The token to act with: ``token_file``, else ``agent``'s file, else the token variable,
    else the file of the agent the environment names."""
    env = os.environ if env is None else env
    if token_file:
        return read_file(Path(token_file).expanduser())
    if agent:
        return load(base, agent)
    if env.get(TOKEN_ENV, "").strip():
        return env[TOKEN_ENV].strip()
    named = env.get(AGENT_ENV, "").strip()
    return load(base, named) if named else ""
