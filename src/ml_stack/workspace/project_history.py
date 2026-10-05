"""Explicit bounded adoption of one historical project board."""

import hashlib
import json
from pathlib import Path

from ml_stack.files import read_json, write_json
from ml_stack.workspace.screen import fence


def adopt(base: Path, history: dict) -> dict:
    """Keep selected historical messages as fenced data without identities or permissions."""
    rows = history.get("messages")
    if not isinstance(rows, list) or len(rows) > 200 or len(json.dumps(history).encode()) > 1 << 20:
        raise ValueError("select at most 200 historical messages under one MiB")
    board = str(history.get("board") or "")
    if not board.startswith("#") or len(board) > 48:
        raise ValueError("choose the original project's board explicitly")
    selected = []
    for row in rows:
        if not isinstance(row, dict) or row.get("board", board) != board:
            raise ValueError("history must contain only the selected project board")
        text = str(row.get("text") or "")
        if len(text.encode()) > 16_384:
            raise ValueError("historical message exceeds the size limit")
        selected.append({"seq": str(row.get("seq") or "")[:24],
                         "from": str(row.get("from") or "unknown")[:80],
                         "text": fence(text, "project-history", "historical agent data").text,
                         "trust": "historical-data", "authority": "none", "state": "imported"})
    digest = hashlib.sha256(json.dumps(selected, sort_keys=True).encode()).hexdigest()
    path = base / "adopted-history.json"
    previous = read_json(path, {})
    if previous and previous.get("digest") != digest:
        raise ValueError("this canonical board already has different adopted history")
    write_json(path, {"board": board, "digest": digest, "messages": selected})
    return {"messages": len(selected), "digest": digest, "authority": "none"}
