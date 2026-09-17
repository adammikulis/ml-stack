"""Questions put to a model while it trains: JSON files dropped in a run's inbox."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ml_stack.train.trainer import Hook

INBOX = "probes"
"""The directory under a run's ``--out`` that probe files are read from."""


def answer(inbox: Path, predict: Callable[[Any], Any]) -> list[dict[str, Any]]:
    """Answer and delete every ``*.json`` in ``inbox``, in name order; one note per file."""
    notes: list[dict[str, Any]] = []
    for path in sorted(inbox.glob("*.json")) if inbox.is_dir() else []:
        probe_id = path.stem
        try:
            asked = json.loads(path.read_text(encoding="utf-8"))
            probe_id = str(asked.get("id") or probe_id)
            notes.append({"id": probe_id, "output": predict(asked.get("inputs"))})
        except Exception as exc:                      # noqa: BLE001 - a failed probe is a note
            notes.append({"id": probe_id, "error": f"{type(exc).__name__}: {exc}"})
        path.unlink(missing_ok=True)
    return notes


def probe_hook(out: Path, predict: Callable[[Any], Any], every: int) -> Hook:
    """A hook answering ``out/probes`` every ``every`` steps, written as ``probe`` notes."""
    inbox = Path(out) / INBOX
    return Hook(name="probe", every=every, run=lambda _step: answer(inbox, predict) or None)
