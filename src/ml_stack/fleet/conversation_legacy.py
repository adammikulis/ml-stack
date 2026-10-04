"""Read saved JSON transcripts for import into the conversation graph."""
from __future__ import annotations

import json
from pathlib import Path

from ml_stack.fleet.conversation_settings import checked
from ml_stack.fleet.conversation_types import Conversation, Message, safe


def read_legacy(path: Path) -> Conversation | None:
    try:
        raw = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    if (not isinstance(raw, dict) or not safe(str(raw.get("id") or ""))
            or raw["id"] != path.stem or raw.get("version", 1) != 1):
        return None
    messages = []
    for row in raw.get("messages") or []:
        try:
            messages.append(Message(role=str(row["role"]),
                                    content=str(row["content"]),
                                    at=float(row.get("at") or 0)))
        except (KeyError, TypeError, ValueError):
            continue
    try:
        return Conversation(id=str(raw["id"]), title=str(raw.get("title") or ""),
                            model=str(raw.get("model") or ""),
                            created=float(raw.get("created") or 0),
                            messages=messages, settings=checked(raw.get("settings")))
    except (TypeError, ValueError):
        return None
