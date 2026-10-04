"""What the workspace allows: size caps, rates, retention, and where its files live."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

from ml_stack import home
from ml_stack.files import read_json, write_json

__all__ = ["DENYLIST_ENV", "Limits", "denylist_path", "load", "root"]

ROOT_ENV = "ML_STACK_WORKSPACE_HOME"
DENYLIST_ENV = "ML_STACK_WORKSPACE_DENYLIST"
VERSION = 1


@dataclass(slots=True)
class Limits:
    """The numbers every write is held to; ``limits.json`` in the workspace overrides them."""

    body_bytes: int = 16_384
    subject_chars: int = 200
    note_body_bytes: int = 8_192
    sends_per_window: int = 30
    window_s: float = 60.0
    inbox_pending: int = 500
    notes_per_agent: int = 500
    retention_s: float = 7 * 86_400.0
    token_ttl_s: float = 86_400.0
    claim_ttl_s: float = 900.0
    scratch_bytes: int = 256 * 1024 * 1024
    scratch_ttl_s: float = 3 * 86_400.0
    scratch_folders: int = 16
    max_children: int = 8
    child_sends_per_window: int = 10
    child_ttl_s: float = 28_800.0
    invite_ttl_s: float = 600.0
    shared_invite_ttl_s: float = 3_600.0
    shared_invite_uses: int = 10
    invite_failures: int = 5
    verify_timeout_s: float = 60.0
    verify_allow: list[list[str]] = field(default_factory=list)


def root() -> Path:
    """The workspace state directory, created owner-only."""
    named = os.environ.get(ROOT_ENV)
    path = Path(named).expanduser() if named else home.state("workspace")
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    path.chmod(0o700)
    return path


def denylist_path(base: Path) -> Path:
    """The file of private terms, one per line: the environment's, else ``private-terms``."""
    named = os.environ.get(DENYLIST_ENV)
    return Path(named).expanduser() if named else base / "private-terms"


def load(base: Path) -> Limits:
    """The limits in force: defaults, then ``limits.json`` for the keys it names."""
    data = read_json(base / "limits.json", {})
    wanted = {f.name for f in fields(Limits)}
    return Limits(**{k: v for k, v in data.items() if k in wanted}) if isinstance(data, dict) \
        else Limits()


def save(base: Path, limits: Limits) -> None:
    """Write ``limits`` as ``limits.json``."""
    body: dict[str, Any] = {"version": VERSION, **asdict(limits)}
    write_json(base / "limits.json", body)
