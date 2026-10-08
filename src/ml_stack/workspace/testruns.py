"""The test runner's view of the workspace: who is running, which project's store, what to tell the board."""

from __future__ import annotations

import argparse
import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ml_stack.workspace import project, tokens
from ml_stack.workspace.identity import TOKEN_ENV, Denied

__all__ = ["Acting", "acting", "scope"]

LABEL_ENV = "ML_STACK_WORKSPACE_LABEL"


@dataclass
class Acting:
    """An authenticated workspace session: the workspace, its token and the identity it stands for."""

    ws: Any
    token: str
    identity: dict[str, str]


def scope(root: Path) -> str:
    """A short stable name for the project that ``root`` belongs to; ``local`` outside any project."""
    found = project.describe(start=root)
    return hashlib.sha256(found["key"].encode()).hexdigest()[:16] if found.get("key") else "local"


def acting(agent: str = "", label: str = "") -> Acting | None:
    """The session for ``--agent``/``--label`` or the environment, or None when none is configured."""
    from ml_stack.workspace import cli

    named = agent or os.environ.get(tokens.AGENT_ENV, "")
    if not named and not os.environ.get(TOKEN_ENV):
        return None
    args = argparse.Namespace(agent=named, label=label or os.environ.get(LABEL_ENV, ""), token_file="")
    try:
        ws, token = cli._context(args)
        who = ws.auth(token)
    except (Denied, OSError, ValueError, SystemExit):
        return None
    return Acting(ws, token, {"id": who.id, "label": args.label, "parent": who.parent or "",
                              "source": "workspace-session"})
