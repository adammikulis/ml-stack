"""A Microsoft Graph `chatMessage` dump, read back to messages.

Reads what `ml_stack.world.emit.teams` writes and what `GET /teams/{id}/channels/{id}/messages`
or `GET /chats/{id}/messages` return: either a bare list of `chatMessage` objects or the
`{"value": [...]}` envelope Graph wraps them in. When `channels` and `chats` sit beside
`value` their display names and members are used; otherwise the `19:…@thread` id is the
channel.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

from ml_stack.files import read_json
from ml_stack.sources import People
from ml_stack.world import Message
from ml_stack.world.emit import ts_of

__all__ = ["read"]


def _ts(iso: str) -> str:
    text = str(iso or "").strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        return ts_of(datetime.fromisoformat(text))
    except ValueError:
        return "0.000000"


def read(path: str | Path, people: Mapping[str, Mapping[str, Any]] | None = None, *,
         domain: str = "example.com") -> list[Message]:
    """Every chatMessage in the file, in the order it holds them.

    The id is Graph's `id`, `thread` its `replyToId`, `ts` its `createdDateTime`. A channel
    message's `channel` is the channel's display name when the file lists it, else the
    channel id; a chat's is `dm:<a>,<b>` over the chat's members when listed, else the chat
    id, with the other members as `recipients`. Each entry in `reactions` becomes a
    `kind="reaction"` message with `attrs["to"]` naming the message it is on. Senders are
    found by Graph user id, then by display name. `attrs["teams"]` keeps Graph's ids.
    """
    who = People(people, domain)
    doc = read_json(Path(path), None)
    if isinstance(doc, dict):
        value = doc.get("value") or []
        channels = {c.get("id"): c for c in doc.get("channels") or [] if isinstance(c, dict)}
        chats = {c.get("id"): c for c in doc.get("chats") or [] if isinstance(c, dict)}
    elif isinstance(doc, list):
        value, channels, chats = doc, {}, {}
    else:
        raise ValueError(f"{path}: not a Teams dump")

    def person(user: Mapping[str, Any] | None) -> tuple[str, dict[str, Any]]:
        user = user or {}
        uid = str(user.get("id") or user.get("userId") or "")
        name = str(user.get("displayName") or "")
        pid = who.teams(uid, name=name)
        if pid:
            return pid, {"sender_kind": "person", "sender_name": name}
        return uid or name, {"sender_kind": "teams_user_id", "sender_name": name}

    members_of: dict[str, list[str]] = {}
    names: dict[str, str] = {cid: str(c.get("displayName") or cid) for cid, c in channels.items()}
    for chat_id, chat in chats.items():
        members = sorted(person(m)[0] for m in chat.get("members") or [] if isinstance(m, dict))
        members_of[chat_id] = members
        names[chat_id] = "dm:" + ",".join(members) if members else str(chat.get("topic") or chat_id)

    out: list[Message] = []
    for row in value:
        if not isinstance(row, dict) or not row.get("id"):
            continue
        identity = row.get("channelIdentity") or {}
        channel_id = str(identity.get("channelId") or row.get("chatId") or "")
        channel = names.get(channel_id, channel_id)
        sender, note = person((row.get("from") or {}).get("user"))
        mid = str(row["id"])
        thread = str(row["replyToId"]) if row.get("replyToId") else None
        ts = _ts(row.get("createdDateTime"))
        body = row.get("body") or {}
        recipients = tuple(p for p in members_of.get(channel_id, []) if p != sender)
        attrs: dict[str, Any] = {
            "teams": {"user": str(((row.get("from") or {}).get("user") or {}).get("id") or ""),
                      "channel": channel_id},
            "content_type": str(body.get("contentType") or "text"), **note}
        out.append(Message(id=mid, source="teams", channel=channel, sender=sender, ts=ts,
                           text=str(body.get("content") or ""), recipients=recipients,
                           thread=thread, kind="reply" if thread else "message", attrs=attrs))
        for reaction in row.get("reactions") or []:
            if not isinstance(reaction, dict):
                continue
            rsender, rnote = person((reaction.get("user") or {}).get("user"))
            emoji = str(reaction.get("reactionType") or "")
            out.append(Message(
                id=f"{mid}:reaction:{emoji}:{rsender}", source="teams", channel=channel,
                sender=rsender, ts=_ts(reaction.get("createdDateTime")) or ts, text=emoji,
                recipients=(), thread=thread or mid, kind="reaction",
                attrs={"to": mid, "teams": {"channel": channel_id}, **rnote}))
    return out
