"""Saved conversations with transactional messages and model/project relationships."""
from __future__ import annotations

import uuid
from pathlib import Path

from poolhouse.fleet import conversation_graph as graph
from poolhouse.fleet.conversation_settings import checked
from poolhouse.fleet.conversation_types import (
    ROLES,
    TITLE_CHARS,
    Conversation,
    Message,
    safe,
    title,
)

__all__ = ["Conversation", "Conversations", "Message"]


class Conversations:
    """One conversation graph under the daemon's chat root."""

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root).expanduser()

    def all(self) -> list[Conversation]:
        with graph.opened(self.root) as store:
            found = [conversation for cid in graph.all_ids(store) if (conversation := graph.get(store, cid)) is not None]
        return sorted(found, key=lambda conversation: conversation.created, reverse=True)

    def get(self, cid: str) -> Conversation | None:
        if not safe(cid):
            return None
        with graph.opened(self.root) as store:
            return graph.get(store, cid)

    def start(self, model: str = "", title: str = "", settings: dict | None = None) -> Conversation:
        made = Conversation(id=uuid.uuid4().hex[:12], title=title, model=model, settings=checked(settings))
        with graph.opened(self.root) as store, store.transaction():
            graph.save(store, made)
        return made

    def append(self, cid: str, role: str, content: str, *,
               reasoning: str = "", status: str = "complete") -> Conversation:
        if role not in ROLES:
            raise ValueError(f"a message is from {' or '.join(ROLES)}, not {role!r}")
        if not safe(cid):
            raise ValueError("invalid conversation id")
        with graph.opened(self.root) as store, store.transaction():
            found = graph.get(store, cid)
            if found is None:
                raise ValueError("no such chat")
            message = Message(role=role, content=content, reasoning=reasoning, status=status)
            graph.append(store, found, message, len(found.messages))
            found.messages.append(message)
            if not found.title and role == "user":
                found.title = title(content)
                graph.save(store, found)
        return found

    def update(self, cid: str, *, model: str | None = None, settings: dict | None = None, title: str | None = None) -> Conversation | None:
        if not safe(cid):
            return None
        with graph.opened(self.root) as store, store.transaction():
            found = graph.get(store, cid)
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
            graph.save(store, found)
        return found

    def remove(self, cid: str) -> bool:
        if not safe(cid):
            return False
        with graph.opened(self.root) as store, store.transaction():
            return graph.remove(store, cid)

    def search(self, needle: str) -> list[Conversation]:
        want = needle.strip().casefold()
        return [conversation for conversation in self.all()
                if not want or want in conversation.title.casefold()
                or any(want in message.content.casefold() for message in conversation.messages)]
