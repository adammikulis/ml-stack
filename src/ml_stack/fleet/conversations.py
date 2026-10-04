"""Chats held on this machine, kept between runs."""

from __future__ import annotations

import json
import re
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ml_stack.files import write_json
from ml_stack.fleet.conversation_settings import checked

__all__ = ["Conversation", "Conversations", "Message"]

TITLE_CHARS = 60
ROLES = ("system", "user", "assistant")


@dataclass
class Message:
    role: str
    content: str
    at: float = field(default_factory=time.time)

    def public(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Conversation:
    id: str
    title: str = ""
    model: str = ""
    created: float = field(default_factory=time.time)
    messages: list[Message] = field(default_factory=list)
    settings: dict[str, Any] = field(default_factory=checked)

    def public(self, *, full: bool = True) -> dict[str, Any]:
        out: dict[str, Any] = {"version": 1, "id": self.id, "title": self.title,
                               "settings": dict(self.settings),
                               "model": self.model, "created": self.created,
                               "count": len(self.messages)}
        if full:
            out["messages"] = [m.public() for m in self.messages]
        return out


class Conversations:
    """One JSON file per chat, under ``root``."""

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root).expanduser()

    def _path(self, cid: str) -> Path:
        return self.root / f"{cid}.json"

    def all(self) -> list[Conversation]:
        """Newest first."""
        if not self.root.exists():
            return []
        out = []
        for path in self.root.glob("*.json"):
            found = self._read(path)
            if found is not None:
                out.append(found)
        out.sort(key=lambda c: c.created, reverse=True)
        return out

    def get(self, cid: str) -> Conversation | None:
        if not _safe(cid):
            return None
        return self._read(self._path(cid))

    def start(self, model: str = "", title: str = "", settings: dict | None = None) -> Conversation:
        made = Conversation(id=uuid.uuid4().hex[:12], title=title, model=model, settings=checked(settings))
        self._write(made)
        return made

    def append(self, cid: str, role: str, content: str) -> Conversation:
        found = self.get(cid) or self.start()
        if role not in ROLES:
            raise ValueError(f"a message is from {' or '.join(ROLES)}, not {role!r}")
        found.messages.append(Message(role=role, content=content))
        if not found.title and role == "user":
            found.title = _title(content)
        self._write(found)
        return found

    def update(self, cid: str, *, model: str | None = None, settings: dict | None = None, title: str | None = None) -> Conversation | None:
        found = self.get(cid)
        if found is None:
            return None
        if settings is not None:
            if not isinstance(settings, dict):
                raise ValueError("conversation settings must be an object")
            found.settings = checked({**found.settings, **settings})
        if model is not None:
            if not isinstance(model, str):
                raise ValueError("model must be text")
            found.model = model
        if title is not None:
            if not isinstance(title, str):
                raise ValueError("title must be text")
            found.title = title.strip()[:TITLE_CHARS]
        self._write(found)
        return found

    def remove(self, cid: str) -> bool:
        if not _safe(cid):
            return False
        path = self._path(cid)
        if not path.exists():
            return False
        path.unlink()
        return True

    def search(self, needle: str) -> list[Conversation]:
        """Every chat whose title or messages mention ``needle``."""
        want = needle.strip().lower()
        if not want:
            return self.all()
        return [c for c in self.all()
                if want in c.title.lower()
                or any(want in m.content.lower() for m in c.messages)]

    def _read(self, path: Path) -> Conversation | None:
        try:
            raw = json.loads(path.read_text())
        except (OSError, ValueError):
            return None
        if (not isinstance(raw, dict) or not _safe(str(raw.get("id") or ""))
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

    def _write(self, chat: Conversation) -> None:
        write_json(self._path(chat.id), chat.public())


def _safe(cid: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z0-9_-]{1,64}", cid or ""))


def _title(text: str) -> str:
    line = " ".join(text.split())
    return line[:TITLE_CHARS].rstrip() or "New chat"
