"""Putting a server on the lease registry the way a manager would have."""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import EllipsisType
from typing import Any

__all__ = ["record_server"]


def record_server(state_file: Path, port: int, *, model: str = "model.gguf",
                  pid: int | None | EllipsisType = ..., owner_pid: int | None = None,
                  **fields: Any) -> dict[str, Any]:
    """Put a server on the lease registry in ``state_file`` as one this machine started:
    running as this process and held by it, unless ``pid`` (``None`` for a start that has
    no process yet) or ``owner_pid`` say otherwise."""
    entry = {"port": port, "pid": os.getpid() if pid is ... else pid, "backend": "llama.cpp", "model": model,
             "owner_pid": owner_pid or os.getpid(),
             "base_url": f"http://127.0.0.1:{port}", **fields}
    state_file.parent.mkdir(parents=True, exist_ok=True)
    held = json.loads(state_file.read_text()) if state_file.exists() else {}
    held[str(port)] = entry
    state_file.write_text(json.dumps(held))
    return entry
