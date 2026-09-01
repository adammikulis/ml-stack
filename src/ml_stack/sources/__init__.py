"""Readers from the formats products export back to one `list[Message]`.

Slack's export directory, an mbox of mail, a Microsoft Graph `chatMessage` dump and the
rows a Slack scraper writes all come back as `ml_stack.world.Message`, so a graph built
from one is built from any of them with the same code. `read` looks at a path and picks the
reader; each reader is also there by name. Given the world's people (`id -> {"label",
"email"?, "handle"?}`, the mapping `ml_stack.world.emit` writes from), a reader puts the
`person:` ids back; without it the product's own id stays in `sender` and
`attrs["sender_kind"]` says which product's it is.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ml_stack.world import Message
from ml_stack.world.emit import directory, slack_user_id, teams_user_id

__all__ = ["People", "Message", "read", "sniff"]


class People:
    """The world's people, indexed by every id a product could have given them.

    A Slack `U0…`, an address, a Graph uuid, a handle or a display name each find the
    `person:` id, minted the same way the emitters mint them. Missing the mapping, every
    lookup answers None and the reader keeps the product's id.
    """

    def __init__(self, people: Mapping[str, Mapping[str, Any]] | None,
                 domain: str = "example.com") -> None:
        self.book = directory(people or {}, domain)
        self.by_slack: dict[str, str] = {}
        self.by_email: dict[str, str] = {}
        self.by_teams: dict[str, str] = {}
        self.by_label: dict[str, str] = {}
        self.by_handle: dict[str, str] = {}
        for pid, p in self.book.items():
            self.by_slack[slack_user_id(pid)] = pid
            self.by_email[p["email"].lower()] = pid
            self.by_teams[teams_user_id(pid)] = pid
            self.by_label.setdefault(p["label"].casefold(), pid)
            self.by_handle.setdefault(p["handle"].casefold(), pid)

    def slack(self, user_id: str, *, email: str = "", name: str = "", handle: str = ""
              ) -> str | None:
        return (self.by_slack.get(user_id) or self.by_email.get(email.lower())
                or self.by_handle.get(handle.casefold()) or self.by_label.get(name.casefold()))

    def email(self, address: str, *, name: str = "") -> str | None:
        return self.by_email.get(address.lower()) or self.by_label.get(name.casefold())

    def teams(self, user_id: str, *, name: str = "") -> str | None:
        return self.by_teams.get(user_id) or self.by_label.get(name.casefold())

    def label(self, name: str) -> str | None:
        return self.by_label.get(name.casefold())


def sniff(path: str | Path) -> str:
    """Which format a path holds: "slack_export", "mbox", "teams" or "rows"."""
    p = Path(path)
    if p.is_dir():
        if (p / "channels.json").exists() or (p / "users.json").exists():
            return "slack_export"
        raise ValueError(f"{p}: a directory, but not a Slack export (no channels.json)")
    with p.open("rb") as fh:
        head = fh.read(4096)
    if head.startswith(b"From "):
        return "mbox"
    stripped = head.lstrip()
    if p.suffix == ".jsonl" or (stripped.startswith(b"{") and b"\n{" in stripped):
        return "rows"
    if stripped.startswith((b"{", b"[")):
        try:
            doc = json.loads(p.read_text(encoding="utf-8"))
        except ValueError as exc:
            raise ValueError(f"{p}: not JSON ({exc})") from exc
        items = doc.get("value") if isinstance(doc, dict) else doc
        if isinstance(doc, dict) and isinstance(items, list):
            return "teams"
        if isinstance(items, list):
            first = next((i for i in items if isinstance(i, dict)), {})
            if "createdDateTime" in first:
                return "teams"
            if "ts" in first:
                return "rows"
    raise ValueError(f"{p}: not a Slack export, an mbox, a Teams dump or scraper rows")


def read(path: str | Path, people: Mapping[str, Mapping[str, Any]] | None = None, *,
         domain: str = "example.com") -> list[Message]:
    """Read whatever `path` is -- `sniff` says which -- back to messages."""
    from ml_stack.sources import mbox, rows, slack_export, teams

    kind = sniff(path)
    reader = {"slack_export": slack_export.read, "mbox": mbox.read,
              "teams": teams.read, "rows": rows.read}[kind]
    return reader(path, people, domain=domain)
