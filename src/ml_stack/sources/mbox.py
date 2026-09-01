"""An mbox of mail, read back to messages.

Reads what `ml_stack.world.emit.mbox` writes and what a mail client or a Takeout hands
over: one RFC 5322 message after another, threaded by Message-ID, In-Reply-To and
References. The plain-text part is the text; an HTML-only message is read as its HTML.
"""

from __future__ import annotations

import email
import email.policy
import mailbox
import re
from collections.abc import Mapping
from email.utils import getaddresses, parsedate_to_datetime
from pathlib import Path
from typing import Any

from ml_stack.sources import People
from ml_stack.world import Message
from ml_stack.world.emit import ts_of

__all__ = ["read"]

_REPLY_PREFIX = re.compile(r"^\s*(?:(?:re|fw|fwd|aw|sv)\s*:\s*)+", re.I)


def _factory(fh: Any) -> email.message.EmailMessage:
    return email.message_from_binary_file(fh, policy=email.policy.default)  # type: ignore[return-value]


def _addresses(msg: email.message.EmailMessage, header: str) -> list[tuple[str, str]]:
    values = msg.get_all(header, [])
    return [(name, address) for name, address in getaddresses(values) if address]


def _body(msg: email.message.EmailMessage) -> str:
    part = msg.get_body(preferencelist=("plain", "html"))
    if part is None:
        return ""
    try:
        text = part.get_content()
    except (KeyError, LookupError):
        payload = part.get_payload(decode=True) or b""
        text = payload.decode("utf-8", "replace")
    return text[:-1] if text.endswith("\n") else text


def read(path: str | Path, people: Mapping[str, Mapping[str, Any]] | None = None, *,
         domain: str = "example.com") -> list[Message]:
    """Every mail in the mbox, in the order it holds them.

    The id is `X-World-Id` when present, else the Message-ID without its brackets; `ts` is
    `X-World-Ts`, else the Date. `channel` is the subject with any "Re:"/"Fwd:" stripped,
    so a thread is one channel; `thread` is the message In-Reply-To (or the first of
    References) names, when it is in the mailbox. `recipients` are To; Cc goes to
    `attrs["cc"]`. Senders are found by address, then by display name.
    """
    who = People(people, domain)
    box = mailbox.mbox(Path(path), factory=_factory, create=False)
    try:
        mails = list(box)
    finally:
        box.close()

    def world_id(msg: email.message.EmailMessage) -> str:
        given = str(msg.get("X-World-Id") or "").strip()
        return given or str(msg.get("Message-ID") or "").strip().strip("<>")

    by_msgid = {str(m.get("Message-ID") or "").strip(): world_id(m) for m in mails}

    def resolve(name: str, address: str) -> tuple[str, dict[str, Any]]:
        pid = who.email(address, name=name)
        if pid:
            return pid, {"sender_kind": "person", "sender_name": name}
        return address, {"sender_kind": "email", "sender_name": name}

    out: list[Message] = []
    for msg in mails:
        senders = _addresses(msg, "From")
        name, address = senders[0] if senders else ("", "")
        sender, note = resolve(name, address)
        ts = str(msg.get("X-World-Ts") or "").strip()
        if not ts:
            try:
                ts = ts_of(parsedate_to_datetime(str(msg.get("Date") or "")))
            except (TypeError, ValueError):
                ts = "0.000000"
        parent = str(msg.get("In-Reply-To") or "").strip()
        if not parent:
            refs = str(msg.get("References") or "").split()
            parent = refs[0] if refs else ""
        thread = by_msgid.get(parent)
        subject = str(msg.get("Subject") or "")
        channel = _REPLY_PREFIX.sub("", subject).strip()
        recipients = tuple(resolve(n, a)[0] for n, a in _addresses(msg, "To"))
        attrs: dict[str, Any] = {"message_id": str(msg.get("Message-ID") or "").strip(),
                                 "subject": subject, **note}
        cc = [resolve(n, a)[0] for n, a in _addresses(msg, "Cc")]
        if cc:
            attrs["cc"] = cc
        out.append(Message(id=world_id(msg), source="email", channel=channel, sender=sender,
                           ts=ts, text=_body(msg), recipients=recipients, thread=thread,
                           kind="reply" if thread else "message", attrs=attrs))
    return out
