"""Token files under the workspace state directory, and how a process finds its own token."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

from poolhouse.files import writing
from poolhouse.private_path import problem, redirected, windows_mount
from poolhouse.sentinel import human
from poolhouse.windows_private import restrict, validate
from poolhouse.workspace.identity import PREFIX, TOKEN_ENV, Denied, valid_id

__all__ = ["AGENT_ENV", "OWNER_FILE", "directory", "inside_repo", "load", "prepare",
           "read_file", "resolve", "store"]

AGENT_ENV = "POOLHOUSE_WORKSPACE_AGENT"
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
    if os.name == "nt":
        validate(base)
        validate(path)
    try:
        info = path.lstat()
    except FileNotFoundError:
        pass
    else:
        if redirected(info):
            raise ValueError(f"{path} is a symlink or Windows reparse point")
    repo = inside_repo(path)
    if repo is not None:
        raise ValueError(f"{path} resolves into the git work tree {repo}; "
                         f"move POOLHOUSE_HOME out of it")
    if windows_mount(path):
        raise ValueError(f"{path} is on a Windows-mounted filesystem; keep workspace tokens "
                         "under the WSL home directory")
    if os.name != "nt":
        why = problem(base)
        if why not in {"", "missing"} and not why.startswith("mode "):
            raise ValueError(f"{base}: {why}")
    base.mkdir(parents=True, exist_ok=True, mode=0o700)
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    if os.name == "nt":
        if redirected(base.lstat()):
            raise ValueError(f"{base} is a symlink or Windows reparse point")
        restrict(base)
        restrict(path)
    else:
        path.chmod(0o700)
    return path


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
        # the temporary file is already this user's alone: written through its descriptor
        fd = os.open(tmp, os.O_WRONLY | os.O_TRUNC)
        try:
            pending = memoryview((token + "\n").encode("utf-8"))
            while pending:
                pending = pending[os.write(fd, pending):]
        finally:
            os.close(fd)
    return target


def read_file(path: Path) -> str:
    """The token in ``path``; `Denied` when the file is missing or readable by others."""
    why = problem(path)
    if why:
        raise Denied(f"token file {path}: {why}")
    return path.read_text(encoding="utf-8").strip()


class ActorMismatch(Denied):
    """A private identity slot contains a credential for a different actor."""


def load(base: Path, name: str) -> str:
    """``name``'s token; `Denied` when its file or directory is unsafe or holds another agent's."""
    if not valid_id(name):
        raise Denied(f"{name!r} is not a usable agent id")
    why = problem(directory(base))
    if why:
        raise Denied(f"token directory {directory(base)}: {why}")
    token = read_file(directory(base) / name.replace("/", "~"))
    if not token.startswith(f"{PREFIX}{name}."):
        raise ActorMismatch(f"the token file for {name} holds a token for another agent")
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
