"""Holding flagged text out of sight until a person releases it."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ml_stack import sentinel
from ml_stack.sentinel.store import Holding
from ml_stack.workspace.chain import ChainLog
from ml_stack.workspace.identity import HUMAN, Denied, Identity

__all__ = ["PLACEHOLDER", "Quarantine"]

logger = logging.getLogger("ml_stack.workspace")
logger.addHandler(logging.NullHandler())


def _sentinel_hold(kind: str, key: str, reason: str, text: str) -> bool:
    """Also hold ``text`` in the sentinel's quarantine."""
    try:
        sentinel.default().store.quarantine((kind, key), reason, {"source": "workspace"},
                                            Holding(text=text), actor="workspace")
    except (AttributeError, OSError, RuntimeError, TypeError, ValueError) as err:
        logger.warning("sentinel refused the hold: %s", err)
        return False
    return True


PLACEHOLDER = "[held in quarantine as {qid}: {why}. A person releases it; until then it is not shown]"


class Quarantine:
    """Flagged messages and notes: held here, and in the sentinel when there is one."""

    def __init__(self, base: Path, clock: Callable[[], float] = time.time) -> None:
        self.log = ChainLog(base / "quarantine.jsonl", clock)

    def hold(self, kind: str, subject: str, reasons: list[str], text: str, sender: str) -> str:
        """Keep ``text`` and return the id that stands in for it."""
        row = self.log.append({"op": "hold", "kind": kind, "subject": subject,
                               "reasons": reasons, "sender": sender, "text": text})
        qid = f"q{row['seq']}"
        _sentinel_hold("message" if kind == "message" else "memory", f"workspace:{subject}",
                       f"workspace {kind} flagged: {', '.join(reasons)}", text)
        return qid

    def _folded(self) -> dict[str, dict[str, Any]]:
        held: dict[str, dict[str, Any]] = {}
        for row in self.log.rows():
            if row["op"] == "hold":
                held[f"q{row['seq']}"] = {**row, "state": "quarantined"}
            elif row["op"] == "release" and row["qid"] in held:
                held[row["qid"]]["state"] = "released"
        return held

    def state(self, qid: str) -> str:
        """``quarantined``, ``released`` or an empty string for an id never held."""
        return str(self._folded().get(qid, {}).get("state", ""))

    def items(self) -> list[dict[str, Any]]:
        """Every held item without its text."""
        return [{k: v for k, v in item.items() if k not in {"text", "prev", "hash", "v"}}
                | {"qid": qid, "chars": len(item["text"])}
                for qid, item in self._folded().items()]

    def release(self, by: Identity, qid: str) -> str:
        """Let a held item be delivered (still fenced) and return its text. Needs a human token."""
        if by.role != HUMAN:
            raise Denied("only a human token releases a quarantined item")
        held = self._folded().get(qid)
        if held is None:
            raise ValueError(f"nothing is held as {qid}")
        self.log.append({"op": "release", "qid": qid, "by": by.id})
        return str(held["text"])

    def text(self, by: Identity, qid: str) -> str:
        """The text held as ``qid``, for a human token."""
        if by.role != HUMAN:
            raise Denied("only a human token reads a quarantined item")
        held = self._folded().get(qid)
        if held is None:
            raise ValueError(f"nothing is held as {qid}")
        return str(held["text"])
