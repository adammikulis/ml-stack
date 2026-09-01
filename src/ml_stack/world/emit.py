"""What the invented company said, written the way each product exports it.

A corpus is one `list[Message]`; a demo needs it on disk looking like what Slack, a mail
client and Microsoft Teams actually hand over, so that a reader written for real exports
(`ml_stack.sources`) reads the invented ones unchanged, and so that a pipeline built on a
scraper's rows (`rows`) needs no adapter. Each emitter writes the product's own ids -- a
Slack `U0…`, an address, a Graph user uuid -- minted deterministically from the world's
`person:` ids, so the same world always exports the same files and a reader with the
world's people can map every id back.

The world's message id survives in the one slot each product has for it: Slack's
`client_msg_id`, Teams' `id`, and an `X-World-Id` header in mail (with `X-World-Ts` beside
it, since `Date:` cannot hold Slack's fraction of a second). Scraper rows have no such slot,
so `ml_stack.sources.rows` mints `<channelId>-<ts>` the way that pipeline does.
"""

from __future__ import annotations

import hashlib
import mailbox
import re
import time
import unicodedata
import uuid
from collections import defaultdict
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from email.message import EmailMessage
from email.utils import formataddr, format_datetime
from pathlib import Path
from typing import Any

from ml_stack.files import write_json
from ml_stack.jsonl import ts_key
from ml_stack.world import Message

__all__ = ["directory", "mbox", "message_id", "rows", "slack_channel_id", "slack_dm_id",
           "slack_export", "slack_user_id", "teams", "teams_channel_id", "teams_chat_id",
           "teams_user_id", "when", "ts_of", "dm_members", "is_dm", "msgid"]

DEFAULT_DOMAIN = "example.com"
_ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
_TEAMS_NAMESPACE = uuid.UUID("6f1b2a3c-4d5e-4f60-8172-839405a6b7c8")
_ATOM = re.compile(r"^[A-Za-z0-9!#$%&'*+/=?^_`{|}~.-]+$")


# --- people -------------------------------------------------------------------------------

def _words(label: str) -> list[str]:
    text = unicodedata.normalize("NFKD", label).encode("ascii", "ignore").decode()
    return [w for w in re.split(r"[^a-z0-9]+", text.lower()) if w]


def directory(people: Mapping[str, Mapping[str, Any]], domain: str = DEFAULT_DOMAIN
              ) -> dict[str, dict[str, str]]:
    """Every person with a label, an address and a handle, derived where not given.

    `people` maps a world id to `{"label", "email"?, "handle"?}`. The address is
    `first.last@<domain>` and the handle `first.last`, both from the label; a second person
    who would get the same ones gets a number after the name, so two Ada Lovelaces do not
    share a mailbox. The result is what every emitter and reader keys on.
    """
    out: dict[str, dict[str, str]] = {}
    used_handles: set[str] = set()
    used_emails: set[str] = set()
    for pid, given in people.items():
        label = str(given.get("label") or pid.split(":", 1)[-1].replace("-", " ").title())
        words = _words(label) or _words(pid) or ["someone"]
        base = words[0] if len(words) == 1 else f"{words[0]}.{words[-1]}"
        handle = str(given.get("handle") or "")
        if not handle:
            handle, n = base, 1
            while handle in used_handles:
                n += 1
                handle = f"{base}{n}"
        email = str(given.get("email") or "")
        if not email:
            email, n = f"{base}@{domain}", 1
            while email in used_emails:
                n += 1
                email = f"{base}{n}@{domain}"
        used_handles.add(handle)
        used_emails.add(email)
        out[pid] = {"id": pid, "label": label, "email": email, "handle": handle}
    return out


def _mint(prefix: str, key: str, length: int) -> str:
    n = int(hashlib.sha256(key.encode("utf-8")).hexdigest(), 16)
    out = []
    for _ in range(length):
        out.append(_ALPHABET[n % 36])
        n //= 36
    return prefix + "".join(out)


def slack_user_id(person_id: str) -> str:
    """The `U0…` Slack would give this person: nine characters, the same every time."""
    return _mint("U0", f"slack-user:{person_id}", 7)


def slack_channel_id(channel: str) -> str:
    """The `C0…` of a channel name."""
    return _mint("C0", f"slack-channel:{channel}", 7)


def slack_dm_id(channel: str) -> str:
    """The `D0…` of a direct message, from its `dm:<a>,<b>` name."""
    return _mint("D0", f"slack-dm:{channel}", 7)


def teams_user_id(person_id: str) -> str:
    """The Entra user uuid Graph would put in `from.user.id`."""
    return str(uuid.uuid5(_TEAMS_NAMESPACE, f"user:{person_id}"))


def teams_team_id(domain: str) -> str:
    return str(uuid.uuid5(_TEAMS_NAMESPACE, f"team:{domain}"))


def teams_channel_id(channel: str) -> str:
    """A Teams channel id, `19:<hex>@thread.tacv2`, from its name."""
    return f"19:{hashlib.sha256(f'teams-channel:{channel}'.encode()).hexdigest()[:32]}@thread.tacv2"


def teams_chat_id(channel: str) -> str:
    """A Teams chat id, `19:<hex>@thread.v2`, from a `dm:<a>,<b>` name."""
    return f"19:{hashlib.sha256(f'teams-chat:{channel}'.encode()).hexdigest()[:32]}@thread.v2"


def message_id(channel_id: str, ts: str) -> str:
    """The id a scraped row gets when nothing carries the world's: `<channelId>-<ts>`."""
    return f"{channel_id}-{ts}"


# --- time ---------------------------------------------------------------------------------

def when(ts: str) -> datetime:
    """A Slack `ts` ("1725148800.000100") as an aware UTC datetime."""
    sec, _, frac = str(ts).partition(".")
    micro = int((frac or "0")[:6].ljust(6, "0"))
    return datetime.fromtimestamp(int(sec), UTC).replace(microsecond=micro)


def ts_of(moment: datetime) -> str:
    """A datetime back as a Slack `ts`, six digits of fraction."""
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=UTC)
    return f"{int(moment.timestamp())}.{moment.microsecond:06d}"


def _iso(ts: str) -> str:
    return when(ts).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _day(ts: str) -> str:
    return when(ts).strftime("%Y-%m-%d")


def _order(m: Message) -> tuple[int, int]:
    return ts_key(m.ts) or (0, 0)


# --- channels -----------------------------------------------------------------------------

def is_dm(channel: str) -> bool:
    return channel.startswith("dm:")


def dm_members(channel: str) -> list[str]:
    """The ids named in a `dm:<a>,<b>` channel, sorted."""
    return sorted(p for p in channel[3:].split(",") if p)


def _pick(messages: Iterable[Message], source: str | None) -> list[Message]:
    return [m for m in messages if source is None or m.source == source]


def _split(messages: list[Message]) -> tuple[list[Message], dict[str, list[Message]]]:
    """Messages and replies in one list, reactions grouped by the message they are on."""
    said = [m for m in messages if m.kind != "reaction"]
    on: dict[str, list[Message]] = defaultdict(list)
    for m in messages:
        if m.kind == "reaction":
            target = str(m.attrs.get("to") or m.thread or "")
            if target:
                on[target].append(m)
    return said, on


def _entry(book: dict[str, dict[str, str]], pid: str) -> dict[str, str]:
    """A person's entry, invented on the spot for a sender the world never listed."""
    if pid not in book:
        book.update(directory({pid: {}}))
    return book[pid]


# --- Slack --------------------------------------------------------------------------------

def slack_export(messages: Iterable[Message], people: Mapping[str, Mapping[str, Any]],
                 out_dir: str | Path, *, source: str | None = "slack",
                 domain: str = DEFAULT_DOMAIN) -> Path:
    """Write a Slack export directory and say where it is.

    The layout is the one Slack's workspace export has: `users.json`, `channels.json`,
    `dms.json`, and one directory per channel holding a `YYYY-MM-DD.json` array per day
    (days cut at midnight UTC), each message `{"type": "message", "user", "text", "ts",
    "client_msg_id", ...}` with `thread_ts`/`reply_count`/`replies` on a thread's root,
    `thread_ts`/`parent_user_id` on its replies and `reactions` where there were any. Direct
    messages, which Slack lists in `dms.json` and stores in directories named by their
    `D0…` id, are kept under `dms/<D0…>/` here so the channel directories stay readable at
    the top. The world's message id goes in `client_msg_id`.

    The Slack ids minted for each person and channel are written back into every message's
    `attrs["slack"]` (`user`, `channel`), so the world keeps what the export said.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    book = directory(people, domain)
    team = _mint("T0", f"slack-team:{domain}", 7)
    said, reactions_on = _split(_pick(messages, source))
    said.sort(key=_order)
    by_id = {m.id: m for m in said}

    channels: dict[str, list[Message]] = defaultdict(list)
    for m in said:
        channels[m.channel].append(m)

    speakers: dict[str, set[str]] = defaultdict(set)
    for m in said:
        _entry(book, m.sender)
        speakers[m.channel].add(m.sender)
        for r in m.recipients:
            _entry(book, r)
            speakers[m.channel].add(r)
    for target, reacts in reactions_on.items():
        for r in reacts:
            _entry(book, r.sender)

    users = [{
        "id": slack_user_id(pid), "team_id": team, "name": p["handle"], "deleted": False,
        "real_name": p["label"],
        "profile": {"real_name": p["label"], "display_name": p["handle"],
                    "email": p["email"], "title": ""},
        "is_bot": False, "is_admin": False,
    } for pid, p in book.items()]
    write_json(out / "users.json", users, indent=4)

    channel_ids: dict[str, str] = {}
    listed_channels, listed_dms = [], []
    for name, rows_ in channels.items():
        created = int(when(rows_[0].ts).timestamp())
        if is_dm(name):
            members = sorted(set(dm_members(name)) | speakers[name])
            cid = slack_dm_id(name)
            listed_dms.append({"id": cid, "created": created,
                               "members": [slack_user_id(p) for p in members]})
        else:
            cid = slack_channel_id(name)
            first = slack_user_id(rows_[0].sender)
            listed_channels.append({
                "id": cid, "name": name, "created": created, "creator": first,
                "is_archived": False, "is_general": name == "general",
                "members": [slack_user_id(p) for p in sorted(speakers[name])],
                "topic": {"value": "", "creator": "", "last_set": 0},
                "purpose": {"value": "", "creator": "", "last_set": 0},
            })
        channel_ids[name] = cid
    write_json(out / "channels.json", listed_channels, indent=4)
    write_json(out / "dms.json", listed_dms, indent=4)

    replies_of: dict[str, list[Message]] = defaultdict(list)
    for m in said:
        if m.thread and m.thread in by_id:
            replies_of[m.thread].append(m)

    for name, rows_ in channels.items():
        cid = channel_ids[name]
        where = out / "dms" / cid if is_dm(name) else out / name
        days: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for m in rows_:
            uid = slack_user_id(m.sender)
            m.attrs.setdefault("slack", {}).update({"user": uid, "channel": cid})
            person = book[m.sender]
            row: dict[str, Any] = {
                "client_msg_id": m.id, "type": "message", "text": m.text, "user": uid,
                "ts": m.ts, "team": team,
                "user_profile": {"real_name": person["label"], "display_name": person["handle"],
                                 "name": person["handle"], "team": team},
            }
            if m.thread and m.thread in by_id:
                root = by_id[m.thread]
                row["thread_ts"] = root.ts
                row["parent_user_id"] = slack_user_id(root.sender)
            elif m.id in replies_of:
                under = sorted(replies_of[m.id], key=_order)
                row["thread_ts"] = m.ts
                row["reply_count"] = len(under)
                row["reply_users_count"] = len({r.sender for r in under})
                row["latest_reply"] = under[-1].ts
                row["reply_users"] = sorted({slack_user_id(r.sender) for r in under})
                row["replies"] = [{"user": slack_user_id(r.sender), "ts": r.ts} for r in under]
                row["subscribed"] = False
            if m.id in reactions_on:
                grouped: dict[str, list[str]] = defaultdict(list)
                for r in reactions_on[m.id]:
                    grouped[r.text].append(slack_user_id(r.sender))
                row["reactions"] = [{"name": k, "users": v, "count": len(v)}
                                    for k, v in grouped.items()]
            days[_day(m.ts)].append(row)
        for day, items in days.items():
            write_json(where / f"{day}.json", items, indent=4)
    return out


# --- email --------------------------------------------------------------------------------

def msgid(message_id_: str, domain: str) -> str:
    """The `Message-ID` for a world id; punctuation RFC 5322 refuses becomes a dot, with a
    hash appended so two ids that differ only in punctuation stay two."""
    if _ATOM.match(message_id_) and ".." not in message_id_ and not message_id_.startswith("."):
        return f"<{message_id_}@{domain}>"
    safe = re.sub(r"[^A-Za-z0-9._+-]+", ".", message_id_).strip(".")
    tag = hashlib.sha256(message_id_.encode("utf-8")).hexdigest()[:8]
    return f"<{safe}.{tag}@{domain}>"


def _addr(p: dict[str, str]) -> str:
    return formataddr((p["label"], p["email"]))


def mbox(messages: Iterable[Message], people: Mapping[str, Mapping[str, Any]],
         out_path: str | Path, *, source: str | None = "email",
         domain: str = DEFAULT_DOMAIN) -> Path:
    """Write the messages as one mbox of RFC 5322 mail and say where it is.

    Each message has From/To/Cc/Date/Subject/Message-ID and, for a reply, In-Reply-To and
    References naming the root, so any mail client threads it; the body is plain text. The
    subject is the message's `channel` (a reply gets "Re: "), recipients are `recipients`,
    and `attrs["cc"]` goes on Cc. Reactions have no place in mail and are left out. Two
    headers carry what mail cannot: `X-World-Id`, the world's message id, and
    `X-World-Ts`, the exact Slack-shaped timestamp -- the way a Takeout carries
    `X-GM-THRID` -- and the reader falls back to Message-ID and Date without them.
    """
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    book = directory(people, domain)
    said, _ = _split(_pick(messages, source))
    said.sort(key=_order)
    box = mailbox.mbox(out)
    try:
        box.lock()
        for m in said:
            sender = _entry(book, m.sender)
            em = EmailMessage()
            em["From"] = _addr(sender)
            if m.recipients:
                em["To"] = ", ".join(_addr(_entry(book, r)) for r in m.recipients)
            cc = [str(c) for c in (m.attrs.get("cc") or [])]
            if cc:
                em["Cc"] = ", ".join(_addr(_entry(book, c)) for c in cc)
            moment = when(m.ts)
            em["Date"] = format_datetime(moment)
            subject = m.channel
            if m.thread and not re.match(r"^\s*re:", subject, re.I):
                subject = f"Re: {subject}"
            em["Subject"] = subject
            em["Message-ID"] = msgid(m.id, domain)
            if m.thread:
                root = msgid(m.thread, domain)
                em["In-Reply-To"] = root
                em["References"] = root
            em["X-World-Id"] = m.id
            em["X-World-Ts"] = m.ts
            em.set_content(m.text)
            boxed = mailbox.mboxMessage(em)
            boxed.set_from(sender["email"], time.gmtime(moment.timestamp()))
            box.add(boxed)
        box.flush()
    finally:
        try:
            box.unlock()
        finally:
            box.close()
    return out


# --- Teams --------------------------------------------------------------------------------

def _graph_user(p: dict[str, str], pid: str) -> dict[str, Any]:
    return {"id": teams_user_id(pid), "displayName": p["label"], "userIdentityType": "aadUser"}


def teams(messages: Iterable[Message], people: Mapping[str, Mapping[str, Any]],
          out_path: str | Path, *, source: str | None = "teams",
          domain: str = DEFAULT_DOMAIN) -> Path:
    """Write the messages as Microsoft Graph `chatMessage` JSON and say where it is.

    The file is one object the way a Graph list comes back -- `{"@odata.context", "value":
    [chatMessage, ...]}` -- with each message carrying `id`, `replyToId`, `messageType`,
    `createdDateTime`, `from.user.{id,displayName,userIdentityType}`,
    `body.{contentType,content}`, and either `channelIdentity.{teamId,channelId}` for a
    channel or `chatId` for a direct message, plus `reactions` in Graph's shape. Beside
    `value` sit `channels` and `chats` (Graph's `channel` and `chat` resources, members
    inline) so a reader can put display names back on the `19:…@thread` ids; Graph itself
    would make you fetch those separately. The `id` is the world's, where Graph mints an
    epoch in milliseconds.
    """
    out = Path(out_path)
    book = directory(people, domain)
    team_id = teams_team_id(domain)
    said, reactions_on = _split(_pick(messages, source))
    said.sort(key=_order)

    channels: dict[str, dict[str, Any]] = {}
    chats: dict[str, dict[str, Any]] = {}
    members_of: dict[str, set[str]] = defaultdict(set)
    for m in said:
        _entry(book, m.sender)
        for r in m.recipients:
            _entry(book, r)
        members_of[m.channel].add(m.sender)
        members_of[m.channel].update(m.recipients)
    for reacts in reactions_on.values():
        for r in reacts:
            _entry(book, r.sender)

    value: list[dict[str, Any]] = []
    for m in said:
        person = book[m.sender]
        row: dict[str, Any] = {
            "id": m.id, "replyToId": m.thread, "etag": m.id, "messageType": "message",
            "createdDateTime": _iso(m.ts), "lastModifiedDateTime": _iso(m.ts),
            "deletedDateTime": None, "subject": None, "summary": None,
            "chatId": None, "importance": "normal", "locale": "en-us", "webUrl": None,
            "from": {"application": None, "device": None, "user": _graph_user(person, m.sender)},
            "body": {"contentType": "text", "content": m.text},
            "channelIdentity": None, "attachments": [], "mentions": [], "reactions": [],
        }
        if is_dm(m.channel):
            chat_id = teams_chat_id(m.channel)
            row["chatId"] = chat_id
            if chat_id not in chats:
                members = sorted(set(dm_members(m.channel)) | members_of[m.channel])
                for p in members:
                    _entry(book, p)
                chats[chat_id] = {
                    "id": chat_id, "chatType": "oneOnOne" if len(members) == 2 else "group",
                    "topic": None,
                    "members": [{"@odata.type": "#microsoft.graph.aadUserConversationMember",
                                 "userId": teams_user_id(p), "displayName": book[p]["label"],
                                 "email": book[p]["email"]} for p in members],
                }
        else:
            channel_id = teams_channel_id(m.channel)
            row["channelIdentity"] = {"teamId": team_id, "channelId": channel_id}
            channels.setdefault(channel_id, {"id": channel_id, "displayName": m.channel,
                                             "membershipType": "standard"})
        for r in sorted(reactions_on.get(m.id, []), key=_order):
            row["reactions"].append({
                "reactionType": r.text, "createdDateTime": _iso(r.ts),
                "user": {"application": None, "device": None,
                         "user": _graph_user(book[r.sender], r.sender)},
            })
        value.append(row)

    write_json(out, {
        "@odata.context": "https://graph.microsoft.com/v1.0/$metadata#Collection(chatMessage)",
        "team": {"id": team_id},
        "channels": list(channels.values()),
        "chats": list(chats.values()),
        "value": value,
    }, indent=2)
    return out


# --- scraper rows -------------------------------------------------------------------------

def rows(messages: Iterable[Message], people: Mapping[str, Mapping[str, Any]], *,
         source: str | None = "slack", domain: str = DEFAULT_DOMAIN,
         export: str = "slack-export", scraped_at: str | None = None) -> list[dict[str, Any]]:
    """The messages as the rows a Slack scraper writes, one dict each.

    This is the Slack scraper's row -- `channel`, `channelId`, `ts`, `sender` (the display
    name), `text`, `replies` on a channel row, `threadTs` on a reply, `scrapedAt` -- and it
    is here so a demo profile drops straight into a pipeline that reads those rows, with no
    adapter between them. `permalink` points into the export `slack_export` writes:
    `<export>/<channel>/<YYYY-MM-DD>.json#p<ts>`. Reactions are not rows; a scraper does not
    read them. Rows come back sorted by channel and then ts, the way that pipeline sorts.
    """
    book = directory(people, domain)
    said, _ = _split(_pick(messages, source))
    said.sort(key=lambda m: (m.channel, _order(m)))
    by_id = {m.id: m for m in said}
    reply_count: dict[str, int] = defaultdict(int)
    for m in said:
        if m.thread and m.thread in by_id:
            reply_count[m.thread] += 1
    stamp = scraped_at or datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.000Z")
    out: list[dict[str, Any]] = []
    for m in said:
        cid = slack_dm_id(m.channel) if is_dm(m.channel) else slack_channel_id(m.channel)
        m.attrs.setdefault("slack", {}).update({"user": slack_user_id(m.sender), "channel": cid})
        folder = f"dms/{cid}" if is_dm(m.channel) else m.channel
        row: dict[str, Any] = {"channel": m.channel, "channelId": cid, "ts": m.ts,
                               "sender": _entry(book, m.sender)["label"], "text": m.text}
        if m.thread and m.thread in by_id:
            row["threadTs"] = by_id[m.thread].ts
        else:
            row["replies"] = reply_count.get(m.id, 0)
        row["scrapedAt"] = stamp
        row["permalink"] = f"{export}/{folder}/{_day(m.ts)}.json#p{m.ts.replace('.', '')}"
        out.append(row)
    return out
