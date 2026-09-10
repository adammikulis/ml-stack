"""One thing somebody said, and the product ids and timestamps that go with it.

`Message` is the shape every emitter writes and every reader returns. Beside it are the
ids each product mints for a person, a channel and a message, derived from a `person:` id
so the same person always gets the same `U0…`, address and Entra uuid, and the two
conversions between a Slack `ts` and a datetime.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

__all__ = ["DEFAULT_DOMAIN", "Message", "directory", "dm_members", "is_dm", "message_id",
           "slack_channel_id", "slack_dm_id", "slack_team_id", "slack_user_id",
           "teams_channel_id", "teams_chat_id", "teams_team_id", "teams_user_id",
           "ts_of", "when"]

DEFAULT_DOMAIN = "example.com"
_ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
_TEAMS_NAMESPACE = uuid.UUID("6f1b2a3c-4d5e-4f60-8172-839405a6b7c8")


@dataclass(frozen=True)
class Message:
    """One thing somebody said, in whichever product they said it.

    The one shape every emitter writes and every reader returns, so a corpus of Slack,
    email and Teams is one list. Ids are the world's (`person:<slug>`); `ts` is unix seconds
    as Slack writes it ("1725148800.000100") and emitters convert. `channel` is a Slack
    channel name, `dm:<a>,<b>` for a direct message, an email subject line, or a Teams chat
    id. `thread` is the id of the root message, or None for a root. `kind` is "message",
    "reply" or "reaction". `attrs` carries what one product has and the others do not.
    """

    id: str
    source: str  # "slack" | "email" | "teams"
    channel: str
    sender: str
    ts: str
    text: str
    recipients: tuple[str, ...] = ()
    thread: str | None = None
    kind: str = "message"
    attrs: dict[str, Any] = field(default_factory=dict)


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


def slack_team_id(domain: str) -> str:
    """The `T0…` of a workspace."""
    return _mint("T0", f"slack-team:{domain}", 7)


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


# --- channels -----------------------------------------------------------------------------

def is_dm(channel: str) -> bool:
    return channel.startswith("dm:")


def dm_members(channel: str) -> list[str]:
    """The ids named in a `dm:<a>,<b>` channel, sorted."""
    return sorted(p for p in channel[3:].split(",") if p)
