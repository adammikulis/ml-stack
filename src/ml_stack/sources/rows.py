"""The rows a Slack scraper writes, read back to messages.

The row is the one `ml_stack.world.emit.rows` describes -- `channel`, `channelId`, `ts`,
`sender` as a display name, `text`, `replies` or `threadTs`, `scrapedAt` -- either as a
JSON-lines log where every poll appended every row it saw, or already loaded as a list.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from ml_stack.jsonl import ts_key
from ml_stack.sources import People
from ml_stack.world import Message
from ml_stack.world.emit import message_id

__all__ = ["read"]


def _rows(path_or_rows: str | Path | Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if isinstance(path_or_rows, (str, Path)):
        text = Path(path_or_rows).read_text(encoding="utf-8")
        stripped = text.lstrip()
        if stripped.startswith("["):
            loaded = json.loads(text)
            return [r for r in loaded if isinstance(r, dict)]
        out = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if isinstance(row, dict):
                out.append(row)
        return out
    return [dict(r) for r in path_or_rows]


def read(path_or_rows: str | Path | Iterable[Mapping[str, Any]],
         people: Mapping[str, Mapping[str, Any]] | None = None, *,
         domain: str = "example.com") -> list[Message]:
    """Every row with a ts, the last version of each, by channel and then by ts.

    The id is `<channelId>-<ts>`, the way the scraper's pipeline names a message, and a
    reply's `thread` is `<channelId>-<threadTs>`. The sender is found by display name;
    without a match the display name stays in `sender` and `attrs["sender_kind"]` says
    "display_name". A row marked `degraded` (the scraper could not read the page) is
    skipped.
    """
    who = People(people, domain)
    latest: dict[str, dict[str, Any]] = {}
    for row in _rows(path_or_rows):
        if not row.get("ts") or row.get("degraded"):
            continue
        cid = str(row.get("channelId") or row.get("channel") or "x")
        latest[message_id(cid, str(row["ts"]))] = row
    out: list[Message] = []
    for mid, row in sorted(latest.items(),
                           key=lambda kv: (str(kv[1].get("channelId") or ""),
                                           ts_key(str(kv[1]["ts"])) or (0, 0))):
        cid = str(row.get("channelId") or row.get("channel") or "x")
        name = str(row.get("sender") or "")
        pid = who.label(name)
        sender = pid or name
        note = {"sender_kind": "person" if pid else "display_name", "sender_name": name}
        thread = message_id(cid, str(row["threadTs"])) if row.get("threadTs") else None
        attrs: dict[str, Any] = {"slack": {"channel": cid}, **note}
        for key in ("scrapedAt", "permalink", "replies"):
            if key in row:
                attrs[key] = row[key]
        out.append(Message(id=mid, source="slack", channel=str(row.get("channel") or cid),
                           sender=sender, ts=str(row["ts"]), text=str(row.get("text") or ""),
                           recipients=(), thread=thread,
                           kind="reply" if thread else "message", attrs=attrs))
    return out
