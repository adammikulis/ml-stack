"""The Board: named scopes for messages, members, threads, direct conversations, mentions,
subscriptions and digests, each checked here against the caller's identity."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from ml_stack.workspace import plain
from ml_stack.workspace.boards import GENERAL, MODES, STYPES, Boards
from ml_stack.workspace.bus import TYPES
from ml_stack.workspace.identity import AGENT, HUMAN, Denied, Identity, valid_id
from ml_stack.workspace.screen import NEUTRAL, Refused, fence

if TYPE_CHECKING:
    from ml_stack.workspace.service import Workspace

__all__ = ["BoardApi"]

MENTION = re.compile(r"(?<![\w/@])@([a-z0-9][a-z0-9._-]{0,47}(?:/[a-z0-9][a-z0-9._-]{0,47})?)")


def data_line(value: object, width: int = 120) -> str:
    """``value`` as one plain line with fence tags and chat markup neutralised."""
    out = plain.line(value, width)
    for pattern, repl in NEUTRAL:
        out = pattern.sub(repl, out)
    return out


class BoardApi:
    """Board operations for a `Workspace`; every read and write names who is asking."""

    def __init__(self, ws: Workspace) -> None:
        self.ws = ws
        self.store = Boards(ws.base, ws.clock)

    # -- who may --------------------------------------------------------------------------
    def _who(self, token: str) -> Identity:
        who = self.ws.auth(token)
        self.ws._may(who, "read")
        return who

    @staticmethod
    def _root(who: Identity) -> str:
        return who.parent or who.id

    def _top(self, who: Identity, what: str) -> None:
        if who.parent:
            raise Denied(f"a delegated identity cannot {what}")

    def can_read(self, who: Identity, name: str, boards: dict[str, Any] | None = None) -> bool:
        """Whether ``who`` may read board ``name``: a member, or a person or lead (read-only)."""
        found = (boards or self.store.state()[0]).get(name)
        if found is None:
            return False
        return who.role != AGENT or name == GENERAL or self._root(who) in found["members"]

    def require_read(self, who: Identity, name: str) -> None:
        """`Denied` unless ``who`` may read ``name``; says the same for a board that is absent."""
        if not self.can_read(who, name):
            self.ws.audit("board.denied", who.id, board=plain.line(name, 48))
            raise Denied(f"no board {plain.line(name, 48)} that {who.id} belongs to")

    def require_post(self, who: Identity, name: str) -> None:
        """`Denied` unless ``who`` is a member of ``name``."""
        boards = self.store.state()[0]
        found = boards.get(name)
        if found is None or not (name == GENERAL or self._root(who) in found["members"]):
            self.ws.audit("board.denied", who.id, board=plain.line(name, 48), act="post")
            raise Denied(f"no board {plain.line(name, 48)} that {who.id} belongs to; "
                         f"`join-board` first")

    def prepare(self, who: Identity, to: str, body: str, reply_to: int) -> list[str]:
        """Checks a board post; returns the registered identities its text mentions."""
        self.require_post(who, to)
        parent = self.ws.bus.get(reply_to) if reply_to else None
        if parent is not None and parent["to"] != to:
            raise ValueError(f"message {reply_to} is not on {to}")
        found = dict.fromkeys(m for m in MENTION.findall(body) if self.ws.registry.role_of(m))
        return list(found)[:10]

    # -- reading helpers ------------------------------------------------------------------
    def _rows(self, name: str = "") -> list[dict[str, Any]]:
        return [r for r in self.ws.bus.log.rows() if r["kind"] == "msg"
                and r["to"].startswith("#") and (not name or r["to"] == name)
                and self.ws.bus.live(r)]

    def _show(self, row: dict[str, Any]) -> dict[str, Any]:
        cap = self.ws.limits.board_message_chars
        cut = {**row, "body": plain.text(row["body"], cap)[0],
               "subject": plain.line(row["subject"], self.ws.limits.subject_chars)}
        out = self.ws.deliver(cut)
        out["board"] = row["to"]
        out["mentions"] = list(row.get("mentions", []))
        out["truncated"] = len(row["body"]) > cap
        return out

    def _plain(self, row: dict[str, Any]) -> dict[str, Any]:
        body, cut = plain.text(row["body"], self.ws.limits.board_message_chars)
        return {"seq": row["seq"], "type": row["type"], "from": row["from"],
                "label": plain.line(row.get("label", ""), 48), "role": row["role"],
                "to": row["to"], "ts": row["ts"], "thread": row.get("thread") or row["seq"],
                "reply_to": row.get("reply_to", 0), "mentions": list(row.get("mentions", [])),
                "subject": plain.line(row["subject"], self.ws.limits.subject_chars), "body": body,
                "held": bool(row["held"]), "truncated": cut}

    # -- boards ---------------------------------------------------------------------------
    def list(self, token: str) -> list[dict[str, Any]]:
        """The boards ``token`` can see with membership, activity and unread counts."""
        who = self._who(token)
        boards = self.store.state()[0]
        marks, rows = self.store.marks(who.id), self._rows()
        out = []
        for name, b in sorted(boards.items()):
            if not self.can_read(who, name, boards):
                continue
            mine = [r for r in rows if r["to"] == name]
            out.append({
                "name": name, "title": data_line(b["title"], 60), "project": bool(b["project"]),
                "open": b["open"], "created_by": b["by"],
                "members": "everyone" if name == GENERAL else sorted(b["members"]),
                "member": name == GENERAL or self._root(who) in b["members"],
                "posts": len(mine), "last": mine[-1]["ts"] if mine else b["created"],
                "unread": sum(1 for r in mine if r["seq"] > marks.get(name, 0)
                              and r["from"] != who.id)})
        return out

    def create(self, token: str, name: str, *, private: bool = False,
               title: str = "") -> dict[str, Any]:
        """Make board ``name`` with the caller as its first member."""
        who = self._who(token)
        self._top(who, "create boards")
        lim = self.ws.limits
        if not plain.name_ok(name) or name == GENERAL:
            raise ValueError("a board name is # then lowercase letters, digits, . _ - (up to 40)")
        self.ws._check(who, "the board title", 400, title)
        with self.store.locked():
            boards = self.store.state()[0]
            if name in boards:
                raise ValueError(f"{name} exists already")
            made = sum(1 for b in boards.values() if b["by"] == who.id)
            if made >= lim.boards_created or len(boards) >= lim.boards_total:
                self.ws.audit("write.refused", who.id, what="board", why="cap")
                raise Refused(f"{who.id} made {made} boards already; the limit is "
                              f"{lim.boards_created} ({lim.boards_total} in all)")
            self.store.append({"kind": "board", "op": "create", "name": name, "by": who.id,
                               "open": not private, "project": "",
                               "title": plain.line(title, 60)})
        self.ws.audit("board.create", who.id, board=name)
        self._subscribe(who, "board", name, "inbox", quiet=True)
        return {"board": name, "open": not private}

    def _admit(self, who: Identity, name: str, member: str, boards: dict[str, Any]) -> None:
        lim = self.ws.limits
        b = boards[name]
        if member in b["members"]:
            return
        if len(b["members"]) >= lim.board_members:
            raise Refused(f"{name} has {lim.board_members} members already")
        joined = sum(1 for k, v in boards.items() if k != GENERAL and member in v["members"])
        if joined >= lim.boards_joined:
            raise Refused(f"{member} belongs to {joined} boards already")
        self.store.append({"kind": "board", "op": "join", "name": name, "who": member,
                           "by": who.id})

    def join(self, token: str, name: str) -> dict[str, Any]:
        """Join an open board; a person may join any."""
        who = self._who(token)
        self._top(who, "join boards")
        with self.store.locked():
            boards = self.store.state()[0]
            found = boards.get(name)
            if found is None or name == GENERAL or not (found["open"] or who.role == HUMAN):
                self.ws.audit("board.denied", who.id, board=plain.line(name, 48), act="join")
                raise Denied(f"{plain.line(name, 48)} is not a board {who.id} can join; ask a "
                             f"member or the person to add you")
            self._admit(who, name, who.id, boards)
        self.ws.audit("board.join", who.id, board=name)
        self._subscribe(who, "board", name, "inbox", quiet=True)
        return {"joined": name}

    def leave(self, token: str, name: str) -> dict[str, Any]:
        """Leave a board; its subscription for you goes with it."""
        who = self._who(token)
        self._top(who, "leave boards")
        with self.store.locked():
            boards = self.store.state()[0]
            if name == GENERAL or name not in boards or who.id not in boards[name]["members"]:
                raise ValueError(f"{who.id} is not a member of {plain.line(name, 48)}")
            self.store.append({"kind": "board", "op": "leave", "name": name, "who": who.id})
            if ("board", name) in self.store.state()[1].get(who.id, {}):
                self.store.append({"kind": "sub", "op": "del", "who": who.id,
                                   "stype": "board", "target": name})
        self.ws.audit("board.leave", who.id, board=name)
        return {"left": name}

    def add(self, token: str, name: str, member: str) -> dict[str, Any]:
        """Put ``member`` on a board; the person, a lead or the board's maker may."""
        who = self._who(token)
        self._top(who, "add members")
        with self.store.locked():
            boards = self.store.state()[0]
            found = boards.get(name)
            if found is None or name == GENERAL or not (
                    who.role != AGENT or found["by"] == who.id):
                raise Denied(f"{who.id} cannot add members to {plain.line(name, 48)}")
            if "/" in member or not self.ws.registry.role_of(member):
                raise ValueError(f"no agent called {plain.line(member, 48)}")
            self._admit(who, name, member, boards)
        self.ws.audit("board.add", who.id, board=name, member=member)
        return {"added": member, "to": name}

    def place(self, member: str, project: dict[str, str]) -> list[str]:
        """Put a newly joined agent on its project's board and subscribe it as the person's
        setup would: the project board in full, `#general` for mentions."""
        who = Identity(member, AGENT)
        names = []
        if project:
            name = self.store.project_board(project)
            with self.store.locked():
                boards = self.store.state()[0]
                try:
                    self._admit(who, name, member, boards)
                except Refused:
                    return names
            self._subscribe(who, "board", name, "inbox", quiet=True)
            names.append(name)
        self._subscribe(who, "mentions", "", "inbox", quiet=True)
        return names

    # -- threads and reads ----------------------------------------------------------------
    def threads(self, token: str, name: str, limit: int = 50) -> list[dict[str, Any]]:
        """Threads on ``name``, latest activity first: root, subject, replies, unread."""
        who = self._who(token)
        self.require_read(who, name)
        mark, found = self.store.marks(who.id).get(name, 0), {}
        for r in self._rows(name):
            root = int(r.get("thread") or r["seq"])
            t = found.setdefault(root, {"root": root, "subject": "(earlier message pruned)",
                                        "from": "", "started": 0.0, "last": 0.0,
                                        "replies": 0, "unread": 0})
            if r["seq"] == root:
                t.update(subject=data_line(r["subject"] or r["body"], 120), started=r["ts"],
                         **{"from": r["from"]})
            else:
                t["replies"] += 1
            t["last"] = max(t["last"], r["ts"])
            t["unread"] += r["seq"] > mark and r["from"] != who.id
        ranked = sorted(found.values(), key=lambda t: (t["last"], t["root"]), reverse=True)
        return [{**t, "authority": "none"} for t in ranked[:max(limit, 0)]]

    def read(self, token: str, name: str, limit: int = 50, after: int = 0,
             mark: bool = True) -> list[dict[str, Any]]:
        """Messages on ``name`` after ``after``, oldest first, fenced; marks the board read."""
        who = self._who(token)
        self.require_read(who, name)
        rows = [r for r in self._rows(name) if r["seq"] > after][-max(limit, 0):]
        out, used = [], 0
        for r in rows:
            shown = self._show(r)
            used += len(shown["text"])
            if used > self.ws.limits.board_read_chars:
                break
            out.append(shown)
        if mark and out:
            self.store.mark(who.id, name, out[-1]["seq"])
        return out

    def ui_read(self, token: str, name: str, after: int = 0, limit: int = 100) -> dict[str, Any]:
        """Plain, bounded messages of ``name`` for a page; reading marks nothing."""
        who = self._who(token)
        self.require_read(who, name)
        rows = [r for r in self._rows(name) if r["seq"] > after][-max(limit, 0):]
        return {"board": name, "messages": [self._plain(r) for r in rows]}

    def ui_thread(self, token: str, root: int) -> dict[str, Any]:
        """Plain, bounded messages of one thread for a page."""
        who = self._who(token)
        rows = self.ws.bus.thread(root)
        if not rows:
            raise ValueError(f"no thread {root}")
        if rows[0]["to"].startswith("#"):
            self.require_read(who, rows[0]["to"])
        elif who.role == AGENT:
            self._pair_ok(who, rows[0]["from"], rows[0]["to"])
        return {"root": root, "messages": [self._plain(r) for r in rows[:200]]}

    # -- direct conversations -------------------------------------------------------------
    @staticmethod
    def _pair_ok(who: Identity, a: str, b: str) -> None:
        if who.role == AGENT and who.id not in (a, b):
            raise Denied(f"{who.id} is not part of that conversation")

    def _pair_rows(self, a: str, b: str) -> list[dict[str, Any]]:
        return [r for r in self.ws.bus.log.rows() if r["kind"] == "msg" and self.ws.bus.live(r)
                and {r["from"], r["to"]} == {a, b} and not r["to"].startswith("#")
                and r["to"] != "*" and a != b]

    def dm(self, token: str, other: str, between: str = "", limit: int = 50,
           mark: bool = True) -> list[dict[str, Any]]:
        """The conversation of the caller with ``other`` (or of ``between`` and ``other``,
        for a person or lead), both directions, oldest first."""
        who = self._who(token)
        a = between or who.id
        if not (valid_id(other) and valid_id(a)):
            raise ValueError("name an agent id")
        self._pair_ok(who, a, other)
        out = []
        for r in self._pair_rows(a, other)[-max(limit, 0):]:
            shown = self._show(r)
            shown["direction"] = "sent" if r["from"] == a else "received"
            out.append(shown)
        if mark and out and a == who.id:
            self.store.mark(who.id, f"dm:{other}", out[-1]["seq"])
        return out

    def ui_dm(self, token: str, a: str, b: str, limit: int = 100) -> list[dict[str, Any]]:
        """Plain, bounded messages between ``a`` and ``b`` for a page; a person or lead only."""
        who = self._who(token)
        if not (valid_id(a) and valid_id(b)):
            raise ValueError("name two agent ids")
        self._pair_ok(who, a, b)
        return [{**self._plain(r), "direction": "sent" if r["from"] == a else "received"}
                for r in self._pair_rows(a, b)[-max(limit, 0):]]

    def dm_list(self, token: str) -> list[dict[str, Any]]:
        """The conversations the caller is in, or every pair for a person or lead."""
        who = self._who(token)
        marks, pairs = self.store.marks(who.id), {}
        for r in self.ws.bus.log.rows():
            if r["kind"] != "msg" or r["to"].startswith("#") or r["to"] == "*" \
                    or not self.ws.bus.live(r):
                continue
            if who.role == AGENT and who.id not in (r["from"], r["to"]):
                continue
            key = tuple(sorted((r["from"], r["to"])))
            p = pairs.setdefault(key, {"a": key[0], "b": key[1], "messages": 0, "last": 0.0,
                                       "unread": 0})
            p["messages"] += 1
            p["last"] = r["ts"]
            other = r["from"]
            p["unread"] += r["to"] == who.id and r["seq"] > marks.get(f"dm:{other}", 0)
        return sorted(pairs.values(), key=lambda p: p["last"], reverse=True)

    def mentions(self, token: str, limit: int = 50) -> list[dict[str, Any]]:
        """Recent board messages that mention the caller, on boards the caller can read."""
        who = self._who(token)
        boards = self.store.state()[0]
        found = [r for r in self._rows() if who.id in r.get("mentions", []) and r["from"] != who.id
                 and self.can_read(who, r["to"], boards)]
        return [self._show(r) for r in found[-max(limit, 0):]]

    # -- subscriptions --------------------------------------------------------------------
    def _target_ok(self, who: Identity, stype: str, target: str) -> str:
        if stype not in STYPES:
            raise ValueError(f"subscribe to one of {', '.join(STYPES)}")
        if stype == "mentions":
            return ""
        if stype == "board":
            self.require_read(who, target)
        elif stype == "agent":
            if not valid_id(target) or not self.ws.registry.role_of(target):
                raise ValueError(f"no agent called {plain.line(target, 48)}")
        elif stype == "kind":
            if target not in TYPES:
                raise ValueError(f"kind must be one of {', '.join(TYPES)}")
        else:
            root = self.ws.bus.get(int(target)) if target.isdigit() else None
            if root is None:
                raise ValueError(f"no message {plain.line(target, 20)}")
            if not root["to"].startswith("#"):
                self.ws.audit("board.denied", who.id, act="subscribe-dm")
                raise Denied("direct messages are never subscribed to; they reach your inbox")
            self.require_read(who, root["to"])
            target = str(int(root.get("thread") or root["seq"]))
        return target

    def _subscribe(self, who: Identity, stype: str, target: str, mode: str,
                   quiet: bool = False) -> dict[str, Any]:
        with self.store.locked():
            mine = self.store.state()[1].get(who.id, {})
            key = (stype, target)
            if key not in mine and len(mine) >= self.ws.limits.subs_per_identity:
                if quiet:
                    return {}
                self.ws.audit("write.refused", who.id, what="subscription", why="cap")
                raise Refused(f"{who.id} has {len(mine)} subscriptions; the limit is "
                              f"{self.ws.limits.subs_per_identity}")
            if mine.get(key) != mode:
                self.store.append({"kind": "sub", "op": "set", "who": who.id, "stype": stype,
                                   "target": target, "mode": mode})
        return {"type": stype, "target": target, "mode": mode}

    def subscribe(self, token: str, stype: str, target: str = "",
                  mode: str = "inbox") -> dict[str, Any]:
        """Subscribe the caller to a board, thread, agent, message kind or mentions."""
        who = self._who(token)
        self._top(who, "subscribe")
        if mode not in MODES:
            raise ValueError(f"mode must be one of {', '.join(MODES)}")
        made = self._subscribe(who, stype, self._target_ok(who, stype, target), mode)
        self.ws.audit("board.subscribe", who.id, type=stype, mode=mode)
        return made

    def unsubscribe(self, token: str, stype: str, target: str = "") -> dict[str, Any]:
        """Drop one subscription of the caller; the messages stay."""
        who = self._who(token)
        self._top(who, "subscribe")
        if stype not in STYPES:
            raise ValueError(f"unsubscribe from one of {', '.join(STYPES)}")
        key = (stype, "" if stype == "mentions" else target)
        with self.store.locked():
            if key not in self.store.state()[1].get(who.id, {}):
                raise ValueError("no such subscription")
            self.store.append({"kind": "sub", "op": "del", "who": who.id, "stype": key[0],
                               "target": key[1]})
        self.ws.audit("board.unsubscribe", who.id, type=stype)
        return {"removed": {"type": stype, "target": key[1]}}

    def subs(self, token: str) -> list[dict[str, Any]]:
        """The caller's own subscriptions."""
        who = self._who(token)
        mine = self.store.state()[1].get(who.id, {})
        return [{"type": t, "target": data_line(v, 60), "mode": m}
                for (t, v), m in sorted(mine.items())]

    # -- delivery -------------------------------------------------------------------------
    @staticmethod
    def _mode(mine: dict[tuple[str, str], str], row: dict[str, Any], me: str) -> str:
        thread = str(int(row.get("thread") or row["seq"]))
        for key in (("thread", thread), ("mentions", "") if me in row["mentions"] else None,
                    ("board", row["to"])):
            if key in mine:
                return mine[key]
        for (kind, value), mode in mine.items():
            if kind == "agent" and (row["from"] == value or row["from"].startswith(value + "/")):
                return mode
        return mine.get(("kind", row["type"]), "")

    def routed(self, who: Identity, after: int, mode: str) -> list[dict[str, Any]]:
        """Live board messages after ``after`` that ``who``'s subscriptions deliver as ``mode``."""
        boards, subs = self.store.state()
        mine = subs.get(who.id)
        if who.parent or not mine:
            return []
        return [r for r in self._rows() if r["seq"] > after and r["from"] != who.id
                and self.can_read(who, r["to"], boards)
                and self._mode(mine, {**r, "mentions": r.get("mentions", [])}, who.id) == mode]

    def listeners(self, row: dict[str, Any]) -> list[str]:
        """The identities whose subscriptions deliver ``row`` to their inbox."""
        boards, subs = self.store.state()
        full = {**row, "mentions": row.get("mentions", [])}
        return [m for m, mine in subs.items() if m != row["from"] and self._mode(mine, full, m)
                == "inbox" and self.can_read(Identity(m, AGENT), row["to"], boards)]

    # -- digests --------------------------------------------------------------------------
    def digest(self, token: str, ack: bool = False, thread: int = 0) -> dict[str, Any]:
        """A bounded summary of what digest subscriptions collected, or of one thread."""
        who = self._who(token)
        if thread:
            return self._digest_thread(who, thread)
        rows = self.routed(who, self.store.marks(who.id).get("digest", 0), "digest")
        groups: dict[tuple[str, int], list[dict[str, Any]]] = {}
        for r in rows:
            groups.setdefault((r["to"], int(r.get("thread") or r["seq"])), []).append(r)
        lines = []
        for (board, root), items in sorted(groups.items(), key=lambda g: g[1][-1]["seq"]):
            head = self.ws.bus.get(root)
            last = items[-1]
            lines.append(f"{board} thread {root} {data_line(head['subject'] if head else '', 60)!r}: "
                         f"{len(items)} new, last from {last['from']}: "
                         f"{data_line(last['body'], 100)}")
        cap = self.ws.limits.digest_lines
        shown = lines[-cap:]
        if len(lines) > cap:
            shown.insert(0, f"({len(lines) - cap} older threads omitted)")
        if ack and rows:
            self.store.mark(who.id, "digest", rows[-1]["seq"])
        return self._digested(shown or ["(nothing new)"], "digest", {"messages": len(rows),
                                                                      "threads": len(lines)})

    def _digested(self, lines: list[str], what: str, extra: dict[str, Any]) -> dict[str, Any]:
        screened = fence("\n".join(lines), f"workspace:{what}", "summary of agent messages")
        return {**extra, "text": screened.text, "flags": list(screened.markers),
                "authority": "none"}

    def _digest_thread(self, who: Identity, root: int) -> dict[str, Any]:
        rows = self.ws.bus.thread(root)
        if not rows:
            raise ValueError(f"no thread {root}")
        if rows[0]["to"].startswith("#"):
            self.require_read(who, rows[0]["to"])
        elif who.role == AGENT:
            self._pair_ok(who, rows[0]["from"], rows[0]["to"])
        head, replies = rows[0], rows[1:]
        recent = replies[-5:]
        lines = [f"[{head['seq']}] {head['from']}: "
                 f"{data_line(head['subject'] or head['body'], 300)}"]
        if len(replies) > len(recent):
            lines.append(f"({len(replies) - len(recent)} earlier replies omitted)")
        lines += [f"[{r['seq']}] {r['from']}: {data_line(r['body'], 200)}" for r in recent]
        return self._digested(lines, f"thread#{root}", {"messages": len(rows)})

    # -- the summary ----------------------------------------------------------------------
    def summary(self, token: str) -> dict[str, Any]:
        """The caller's boards and unread counts, and unread direct messages."""
        listing = self.list(token)
        return {"boards": [{"name": b["name"], "unread": b["unread"], "member": b["member"]}
                           for b in listing],
                "dms_unread": sum(p["unread"] for p in self.dm_list(token))}
