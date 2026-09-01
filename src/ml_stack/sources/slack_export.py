"""A Slack export directory, read back to messages.

Reads what `ml_stack.world.emit.slack_export` writes and what Slack's own export gives a
workspace owner: `users.json`, `channels.json`, `dms.json` (when present), and a
`YYYY-MM-DD.json` array per day in a directory per channel. Direct-message directories are
looked for under `dms/<D0…>` and, as Slack lays them out, at the top level by their id.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ml_stack.files import read_json
from ml_stack.jsonl import ts_key
from ml_stack.sources import People
from ml_stack.world import Message
from ml_stack.world.emit import message_id

__all__ = ["read"]

_DAY = re.compile(r"^\d{4}-\d{2}-\d{2}\.json$")


def _days(folder: Path) -> list[Path]:
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.iterdir() if _DAY.match(p.name))


def read(path: str | Path, people: Mapping[str, Mapping[str, Any]] | None = None, *,
         domain: str = "example.com") -> list[Message]:
    """Every message, reply and reaction in the export, by channel and then by ts.

    A message's id is its `client_msg_id` when the export has one, else `<channelId>-<ts>`;
    a reply's `thread` is the id of the message its `thread_ts` names; a reaction becomes a
    `kind="reaction"` message whose text is the emoji name, whose `attrs["to"]` is the
    message it sits on, and whose id is derived from those. A direct message's channel is
    `dm:<a>,<b>` over the members' ids, sorted, and its recipients the members other than
    the sender. `attrs["slack"]` keeps the `U0…` and `C0…` the export used.
    """
    root = Path(path)
    who = People(people, domain)
    users: dict[str, dict[str, Any]] = {u["id"]: u for u in read_json(root / "users.json", [])
                                        if isinstance(u, dict) and u.get("id")}

    def person(uid: str) -> tuple[str, dict[str, Any]]:
        u = users.get(uid, {})
        profile = u.get("profile") or {}
        name = str(u.get("real_name") or profile.get("real_name") or u.get("name") or "")
        pid = who.slack(uid, email=str(profile.get("email") or ""), name=name,
                        handle=str(u.get("name") or ""))
        if pid:
            return pid, {"sender_kind": "person", "sender_name": name}
        return uid, {"sender_kind": "slack_user_id", "sender_name": name}

    folders: list[tuple[str, str, Path, list[str]]] = []  # (name, channel id, folder, members)
    for ch in read_json(root / "channels.json", []):
        if isinstance(ch, dict) and ch.get("id"):
            name = str(ch.get("name") or ch["id"])
            folders.append((name, ch["id"], root / name, list(ch.get("members") or [])))
    for dm in read_json(root / "dms.json", []):
        if isinstance(dm, dict) and dm.get("id"):
            members = [person(u)[0] for u in dm.get("members") or []]
            name = "dm:" + ",".join(sorted(members))
            folder = root / "dms" / dm["id"]
            if not folder.is_dir():
                folder = root / dm["id"]
            folders.append((name, dm["id"], folder, members))
    listed = {f for _, _, f, _ in folders}
    for extra in sorted(p for p in root.iterdir() if p.is_dir() and p not in listed
                        and p.name != "dms" and _days(p)):
        folders.append((extra.name, extra.name, extra, []))

    out: list[Message] = []
    for name, cid, folder, members in folders:
        rows_: list[dict[str, Any]] = []
        for day in _days(folder):
            rows_.extend(r for r in read_json(day, []) if isinstance(r, dict) and r.get("ts"))
        rows_.sort(key=lambda r: ts_key(str(r["ts"])) or (0, 0))
        ids_by_ts = {str(r["ts"]): str(r.get("client_msg_id") or message_id(cid, str(r["ts"])))
                     for r in rows_}
        for r in rows_:
            ts = str(r["ts"])
            uid = str(r.get("user") or r.get("bot_id") or "")
            sender, note = person(uid)
            mid = ids_by_ts[ts]
            thread_ts = str(r.get("thread_ts") or "")
            thread = ids_by_ts.get(thread_ts) if thread_ts and thread_ts != ts else None
            recipients = tuple(p for p in members if p != sender) if name.startswith("dm:") else ()
            attrs: dict[str, Any] = {"slack": {"user": uid, "channel": cid}, **note}
            if r.get("subtype"):
                attrs["subtype"] = r["subtype"]
            out.append(Message(id=mid, source="slack", channel=name, sender=sender, ts=ts,
                               text=str(r.get("text") or ""), recipients=recipients,
                               thread=thread, kind="reply" if thread else "message",
                               attrs=attrs))
            for reaction in r.get("reactions") or []:
                if not isinstance(reaction, dict):
                    continue
                emoji = str(reaction.get("name") or "")
                for ruid in reaction.get("users") or []:
                    rsender, rnote = person(str(ruid))
                    out.append(Message(
                        id=f"{mid}:reaction:{emoji}:{rsender}", source="slack", channel=name,
                        sender=rsender, ts=ts, text=emoji, recipients=(), thread=thread or mid,
                        kind="reaction",
                        attrs={"to": mid, "slack": {"user": str(ruid), "channel": cid}, **rnote}))
    return out
