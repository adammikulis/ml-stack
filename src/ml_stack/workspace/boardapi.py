"""The Board: named scopes for messages, members, threads, direct conversations, mentions,
subscriptions and digests, each checked here against the caller's identity."""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ml_stack.workspace import board_graph_merge, coordination_access, plain
from ml_stack.workspace.board_pages import page
from ml_stack.workspace.boards import ANNOUNCE, ANNOUNCE_MARK, GENERAL, MODES, STYPES, Boards
from ml_stack.workspace.bus import TYPES
from ml_stack.workspace.identity import AGENT, HUMAN, Denied, Identity, valid_id
from ml_stack.workspace.screen import NEUTRAL, Refused, fence

if TYPE_CHECKING:
    from ml_stack.workspace.service import Workspace

__all__ = ["BoardApi", "Follow"]

FOLLOWERS = (".follow", ".chat", ".web")
MENTION = re.compile(r"(?<![\w/@])@([a-z0-9][a-z0-9._-]{0,47}(?:/[a-z0-9][a-z0-9._-]{0,47})?)")


def data_line(value: object, width: int = 120) -> str:
    """``value`` as one plain line with fence tags and chat markup neutralised."""
    out = plain.line(value, width)
    for pattern, repl in NEUTRAL:
        out = pattern.sub(repl, out)
    return out


class Held(list):  # type: ignore[type-arg]
    """A list of results that also says how many more were held back by the default caps."""

    held: int = 0


@dataclass(slots=True)
class Follow:
    """What to follow: one of ``board``, ``thread`` or ``dm`` (with ``between`` for a pair a person
    reads), the messages after ``after`` (negative: from now, after the last ``backlog``), at most
    ``limit`` of them, how long `Workspace.follow` waits and which wake pipe it sleeps on."""

    board: str = ""
    thread: int = 0
    dm: str = ""
    between: str = ""
    after: int = -1
    backlog: int = 0
    limit: int = 50
    plain_text: bool = False
    timeout_s: float = 0.0
    suffix: str = ".follow"


class BoardApi:
    """Board operations for a `Workspace`; every read and write names who is asking."""

    def __init__(self, ws: Workspace) -> None:
        self.ws = ws
        self.store = Boards(ws.base, ws.clock)

    def export_graph(self, token: str) -> dict[str, Any]:
        """Export this authority's message graph for an authorized replica."""
        who = self._who(token)
        if who.role != HUMAN or who.parent:
            raise Denied("only the workspace person exports the board graph")
        return board_graph_merge.export(self.store.log.graph)

    def combine_graph(self, token: str, payload: dict[str, Any]) -> int:
        """Combine same-authority messages without importing memberships or credentials."""
        who = self._who(token)
        if who.role != HUMAN or who.parent:
            raise Denied("only the workspace person combines board graphs")
        count = board_graph_merge.combine(self.store.log.graph, payload)
        self.ws.audit("board.graph.combine", who.id, messages=count)
        return count

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
        """Whether a reader belongs to the board or its recorded project."""
        if not plain.name_ok(name):
            return False
        found = (self.store.state()[0] if boards is None else boards).get(name)
        if found is None:
            return False
        return (who.role != AGENT or name in (GENERAL, ANNOUNCE)
                or self._root(who) in found["members"]
                or bool(found["project"] and found["project"]
                        == coordination_access.project_key(self.ws, who.id)))

    def require_read(self, who: Identity, name: str) -> None:
        """`Denied` unless ``who`` may read ``name``; says the same for a board that is absent."""
        if not self.can_read(who, name):
            self.ws.audit("board.denied", who.id, board=plain.line(name, 48))
            raise Denied(f"no board {plain.line(name, 48)} that {who.id} belongs to")

    def can_post(self, who: Identity, name: str) -> bool:
        """Whether ``who`` is a member of ``name`` and so may post to it."""
        if not plain.name_ok(name):
            return False
        found = self.store.state()[0].get(name)
        return found is not None and (name == GENERAL or self._root(who) in found["members"])

    def require_post(self, who: Identity, name: str) -> None:
        """`Denied` unless ``who`` is a member of ``name``."""
        if not plain.name_ok(name):
            raise ValueError("invalid board name")
        boards = self.store.state()[0]
        found = boards.get(name)
        if found is None or not (name == GENERAL or self._root(who) in found["members"]):
            self.ws.audit("board.denied", who.id, board=plain.line(name, 48), act="post")
            raise Denied(f"no board {plain.line(name, 48)} that {who.id} belongs to; "
                         f"`join-board` first")

    def prepare(self, who: Identity, to: str, body: str, reply_to: int) -> list[str]:
        """Checks a board post; returns the registered identities its text mentions."""
        if to == ANNOUNCE:
            raise Refused(f"{ANNOUNCE} takes only `announce KIND TEXT` (joined, milestone, done, "
                          f"blocked; one line); to answer one, send the poster a direct message")
        self.require_post(who, to)
        parent = self.ws.bus.get(reply_to) if reply_to else None
        if parent is not None and parent["to"] != to:
            raise ValueError(f"message {reply_to} is not on {to}")
        found = dict.fromkeys(m for m in MENTION.findall(body) if self.ws.registry.role_of(m))
        return list(found)[:10]

    # -- reading helpers ------------------------------------------------------------------
    def _rows(self, name: str = "", after: int = 0) -> list[dict[str, Any]]:
        return [r for r in self.ws.bus.log.after(after) if r["kind"] == "msg"
                and r["to"].startswith("#") and (not name or r["to"] == name)
                and self.ws.bus.live(r)]

    def _show(self, row: dict[str, Any], short: int = 0, who: Identity | None = None) -> dict[str, Any]:
        cap = self.ws.limits.board_message_chars
        cut = {**row, "body": plain.text(row["body"], cap)[0],
               "subject": plain.line(row["subject"], self.ws.limits.subject_chars)}
        out = self.ws.deliver(cut, cap=short, reader=who)
        out["board"] = row["to"]
        out["mentions"] = list(row.get("mentions", []))
        out["truncated"] = len(row["body"]) > cap
        return out

    def _plain(self, row: dict[str, Any]) -> dict[str, Any]:
        body, cut = plain.text(row["body"], self.ws.limits.board_message_chars)
        from ml_stack.workspace.agent_display import metadata
        return {"seq": row["seq"], "type": row["type"], "from": row["from"],
                **metadata(self.ws.registry, row["from"]),
                "role": row["role"],
                "to": row["to"], "ts": row["ts"], "thread": row.get("thread") or row["seq"],
                "reply_to": row.get("reply_to", 0), "mentions": list(row.get("mentions", [])),
                "subject": plain.line(row["subject"], self.ws.limits.subject_chars), "body": body,
                "held": bool(row["held"]), "truncated": cut,
                **({"file": {k: row["file"][k] for k in ("id", "name", "size", "type", "text")}}
                   if row.get("file") else {}),
                "model": "" if row["role"] == "human" else row.get("model", ""),
                "model_state": "" if row["role"] == "human" else row.get("model_state", "")}

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
        if not plain.name_ok(name) or name in (GENERAL, ANNOUNCE):
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
        self._subscribe(who, "board", name, "digest", "quiet")
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
        if not plain.name_ok(name):
            raise ValueError("invalid board name")
        with self.store.locked():
            boards = self.store.state()[0]
            found = boards.get(name)
            if found is None or name == GENERAL or not (found["open"] or who.role == HUMAN):
                self.ws.audit("board.denied", who.id, board=plain.line(name, 48), act="join")
                raise Denied(f"{plain.line(name, 48)} is not a board {who.id} can join; ask a "
                             f"member or the person to add you")
            self._admit(who, name, who.id, boards)
        self.ws.audit("board.join", who.id, board=name)
        self._subscribe(who, "board", name, "digest", "quiet")
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
        if not plain.name_ok(name):
            raise ValueError("invalid board name")
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

    def place(self, member: str, project: dict[str, str], *, initial_only: bool = False) -> list[str]:
        """Put a newly joined agent on its project's board. It is subscribed to nothing: its
        inbox carries direct messages and mentions, and `#announcements` arrives as a roll-up."""
        who = Identity(member, AGENT)
        names = []
        if project:
            name = self.store.project_board(project)
            with self.store.locked():
                boards = self.store.state()[0]
                if initial_only and any(row.get("kind") == "board"
                                        and row.get("op") == "leave"
                                        and row.get("name") == name
                                        and row.get("who") == member
                                        for row in self.store.log.rows()):
                    return names
                try:
                    self._admit(who, name, member, boards)
                except Refused:
                    return names
            names.append(name)
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
                         **{"from": r["from"], "display_name": self._plain(r)["display_name"]})
            else:
                t["replies"] += 1
            t["last"] = max(t["last"], r["ts"])
            t["unread"] += r["seq"] > mark and r["from"] != who.id
        ranked = sorted(found.values(), key=lambda t: (t["last"], t["root"]), reverse=True)
        return [{**t, "authority": "none"} for t in ranked[:max(limit, 0)]]

    def read(self, token: str, name: str, limit: int = 0, after: int = 0,
             mark: bool = True) -> list[dict[str, Any]]:
        """Messages on ``name`` after ``after``, oldest first, fenced; marks the board read.
        By default the newest few, each cut short; a ``limit`` widens it. The result's ``held``
        says how many more there were."""
        who = self._who(token)
        self.require_read(who, name)
        rows = [r for r in self._rows(name) if r["seq"] > after]
        lim = self.ws.limits
        newest = rows[-(limit if limit > 0 else lim.read_items):]
        out, used = Held(), 0
        for r in newest:
            shown = self._show(r, 0 if limit > 0 else lim.read_item_chars, who)
            used += len(shown["text"])
            if used > (lim.board_read_chars if limit > 0 else lim.read_total_chars) and out:
                break
            out.append(shown)
        out.held = len(rows) - len(out)
        if mark and out:
            self.store.mark(who.id, name, out[-1]["seq"])
        return out

    def ui_read(self, token: str, name: str, after: int = 0, limit: int = 100,
                before: int = 0) -> dict[str, Any]:
        """A cursor and byte bounded channel page; reading marks nothing."""
        who = self._who(token)
        self.require_read(who, name)
        rows = self._rows(name) if limit else []
        return {"board": name, **page(rows, self._plain, limit, after, before)}

    def ui_thread(self, token: str, root: int, limit: int = 100,
                  after: int = 0, before: int = 0) -> dict[str, Any]:
        """A cursor and byte bounded thread page."""
        who = self._who(token)
        rows = self.ws.bus.thread(root)
        if not rows:
            raise ValueError(f"no thread {root}")
        self._thread_access(who, rows)
        return {"root": root, **page(rows, self._plain, limit, after, before)}

    # -- direct conversations -------------------------------------------------------------
    def _pair_ok(self, who: Identity, a: str, b: str) -> None:
        if (who.role == AGENT and who.id not in (a, b)
                and not coordination_access.shared_pair(self.ws, who, a, b)):
            raise Denied(f"{who.id} is not part of that conversation")

    def _pair_rows(self, a: str, b: str) -> list[dict[str, Any]]:
        return [r for r in self.ws.bus.log.rows() if r["kind"] == "msg" and self.ws.bus.live(r)
                and {r["from"], r["to"]} == {a, b} and not r["to"].startswith("#")
                and r["to"] != "*" and a != b]

    def dm(self, token: str, other: str, between: str = "", limit: int = 50,
           mark: bool = True) -> list[dict[str, Any]]:
        """An authorized conversation in both directions, oldest first."""
        who = self._who(token)
        a = between or who.id
        if not (valid_id(other) and valid_id(a)):
            raise ValueError("name an agent id")
        self._pair_ok(who, a, other)
        out = []
        for r in self._pair_rows(a, other)[-max(limit, 0):]:
            shown = self._show(r, 0, who)
            shown["direction"] = "sent" if r["from"] == a else "received"
            out.append(shown)
        if mark and out and a == who.id:
            self.store.mark(who.id, f"dm:{other}", out[-1]["seq"])
        return out

    def ui_dm(self, token: str, a: str, b: str, limit: int = 100) -> list[dict[str, Any]]:
        """Plain, bounded messages of an authorized conversation for a page."""
        who = self._who(token)
        if not (valid_id(a) and valid_id(b)):
            raise ValueError("name two agent ids")
        self._pair_ok(who, a, b)
        return [{**self._plain(r), "direction": "sent" if r["from"] == a else "received"}
                for r in self._pair_rows(a, b)[-max(limit, 0):]]

    def ui_dm_page(self, token: str, a: str, b: str, window: tuple[int, int, int] = (100, 0, 0)) -> dict[str, Any]:
        """A cursor and byte bounded direct conversation page."""
        who = self._who(token)
        if not (valid_id(a) and valid_id(b)):
            raise ValueError("name two agent ids")
        self._pair_ok(who, a, b)
        limit, after, before = window
        rows = self._pair_rows(a, b) if limit else []
        return {"a": a, "b": b, **page(rows, self._plain, limit, after, before)}

    def dm_list(self, token: str) -> list[dict[str, Any]]:
        """The caller's discoverable addressed and shared project conversations."""
        who = self._who(token)
        marks, pairs = self.store.marks(who.id), {}
        for r in self.ws.bus.log.rows():
            if r["kind"] != "msg" or r["to"].startswith("#") or r["to"] == "*" \
                    or not self.ws.bus.live(r):
                continue
            if not coordination_access.can_read_row(self.ws, who, r):
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
        return [self._show(r, 0, who) for r in found[-max(limit, 0):]]

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
            if root["to"].startswith("#"):
                self.require_read(who, root["to"])
            elif not coordination_access.shared_pair(self.ws, who, root["from"], root["to"]):
                self.ws.audit("board.denied", who.id, act="subscribe-dm")
                raise Denied("subscribe to direct coordination within your project")
            self._thread_access(who, self.ws.bus.thread(root["seq"]))
            target = str(int(root.get("thread") or root["seq"]))
        return target

    def _head(self) -> int:
        rows = self.ws.bus.log.rows()
        return int(rows[-1]["seq"]) if rows else 0

    def _subscribe(self, who: Identity, stype: str, target: str, mode: str,
                   how: str = "") -> dict[str, Any]:
        quiet, force = how == "quiet", how == "force"
        lim = self.ws.limits
        with self.store.locked():
            mine = self.store.state()[1].get(who.id, {})
            key = (stype, target)
            if key not in mine and len(mine) >= lim.subs_per_identity:
                if quiet:
                    return {}
                self.ws.audit("write.refused", who.id, what="subscription", why="cap")
                raise Refused(f"{who.id} has {len(mine)} subscriptions; the limit is "
                              f"{lim.subs_per_identity}")
            cost = ""
            loud = sum(1 for (t, _), m in mine.items() if m == "inbox" and t != "mentions")
            if mode == "inbox" and stype != "mentions" and mine.get(key) != "inbox":
                if loud >= lim.inbox_subs_free and not force:
                    self.ws.audit("write.refused", who.id, what="subscription", why="loud")
                    raise Refused(
                        f"{who.id} already has {loud} inbox subscriptions; every message on "
                        f"{stype} {plain.line(target, 48) or '(any)'} would reach your inbox "
                        f"and your context, and wake `wait`. Use --mode digest (a roll-up you "
                        f"read when you choose) or add --force to accept the cost")
                if loud >= lim.inbox_subs_free:
                    cost = "forced: each new message here now enters your inbox and context"
            if mine.get(key) != mode:
                self.store.append({"kind": "sub", "op": "set", "who": who.id, "stype": stype,
                                   "target": target, "mode": mode, "since": self._head()})
        out = {"type": stype, "target": target, "mode": mode}
        if cost:
            out["cost"] = cost
        return out

    def subscribe(self, token: str, stype: str, target: str = "",
                  mode: str = "inbox", force: bool = False) -> dict[str, Any]:
        """Subscribe the caller to a board, thread, agent, message kind or mentions. Only
        messages after this moment are delivered; the backlog is `board read`."""
        who = self._who(token)
        self._top(who, "subscribe")
        if mode not in MODES:
            raise ValueError(f"mode must be one of {', '.join(MODES)}")
        target = self._target_ok(who, stype, target)
        if (stype, target) == ("board", ANNOUNCE):
            if who.role != AGENT:
                raise Denied(f"{ANNOUNCE} always reaches a lead or a person")
            if mode == "inbox":
                raise ValueError(f"{ANNOUNCE} is a roll-up, never inbox: use digest or silent")
        made = self._subscribe(who, stype, target, mode, "force" if force else "")
        self.ws.audit("board.subscribe", who.id, type=stype, mode=mode)
        return made

    def unsubscribe(self, token: str, stype: str, target: str = "") -> dict[str, Any]:
        """Drop one subscription of the caller; the messages stay."""
        who = self._who(token)
        self._top(who, "subscribe")
        if stype not in STYPES:
            raise ValueError(f"unsubscribe from one of {', '.join(STYPES)}")
        key = (stype, "" if stype == "mentions" else target)
        if key == ("board", ANNOUNCE):
            if who.role != AGENT:
                raise Denied(f"{ANNOUNCE} always reaches a lead or a person")
            return self.subscribe(token, "board", ANNOUNCE, "silent")
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
    def _match(mine: dict[tuple[str, str], str], row: dict[str, Any],
               me: str) -> tuple[tuple[str, str] | None, str]:
        """The subscription that decides how ``row`` reaches ``me``, and its mode. Defaults with
        no subscription: a mention reaches the inbox, `#announcements` a roll-up, the rest
        nothing."""
        if row["to"] == ANNOUNCE:
            key = ("board", ANNOUNCE)
            return key, "silent" if mine.get(key) == "silent" else "digest"
        thread = str(int(row.get("thread") or row["seq"]))
        if ("thread", thread) in mine:
            return ("thread", thread), mine[("thread", thread)]
        if me in row["mentions"]:
            return ("mentions", ""), mine.get(("mentions", ""), "inbox")
        if ("board", row["to"]) in mine:
            return ("board", row["to"]), mine[("board", row["to"])]
        for (kind, value), mode in mine.items():
            if kind == "agent" and (row["from"] == value or row["from"].startswith(value + "/")):
                return (kind, value), mode
        key = ("kind", row["type"])
        return (key, mine[key]) if key in mine else (None, "")

    def routed(self, who: Identity, after: int, mode: str) -> list[dict[str, Any]]:
        """Live authorized messages after ``after`` subscribed in ``mode``;
        a subscription delivers nothing from before it was made."""
        if who.parent:
            return []
        boards, subs = self.store.state()
        mine, since = subs.get(who.id, {}), self.store.sinces().get(who.id, {})
        found = []
        for r in self.ws.bus.log.after(after):
            if (r["kind"] != "msg" or not self.ws.bus.live(r) or r["seq"] <= after
                    or r["from"] == who.id
                    or (mode == "inbox" and r["to"] in (who.id, "*"))
                    or not coordination_access.can_read_row(self.ws, who, r, boards)):
                continue
            key, got = self._match(mine, {**r, "mentions": r.get("mentions", [])}, who.id)
            if got == mode and (key is None or key[0] == "mentions"
                                or r["seq"] > since.get(key, 0)):
                found.append(r)
        return found

    def listeners(self, row: dict[str, Any]) -> list[str]:
        """The identities whose subscriptions (or a mention) deliver ``row`` to their inbox."""
        boards, subs = self.store.state()
        full = {**row, "mentions": row.get("mentions", [])}
        who = dict.fromkeys([*subs, *full["mentions"]])
        return [m for m in who if m != row["from"]
                and self._match(subs.get(m, {}), full, m)[1] == "inbox"
                and coordination_access.can_read_row(self.ws, Identity(m, AGENT), row, boards)]

    def wake_names(self, row: dict[str, Any]) -> list[str]:
        """The wake-pipe names to signal for ``row``: its recipient and subscribers by plain id,
        and everyone who may be following its board or conversation by each follower name."""
        direct = set(self.listeners(row))
        if not row["to"].startswith("#") and row["to"] != "*":
            direct.add(row["to"])
        boards = self.store.state()[0] if row["to"].startswith("#") else {}
        found = boards.get(row["to"])
        following = set(self.ws.registry.ids() if row["to"] == GENERAL else
                        found["members"] if found else ()) | set(self.ws.registry.readers())
        following |= {row["from"], *direct}
        following |= {name for name in self.ws.registry.ids()
                      if self.ws.registry.role_of(name)
                      and coordination_access.can_read_row(self.ws, Identity(name, AGENT), row, boards)}
        return [*direct, *(f"{n}{s}" for n in following for s in FOLLOWERS)]

    # -- following -------------------------------------------------------------------------
    def follow(self, token: str, spec: Follow) -> dict[str, Any]:
        """Messages of one board, thread or conversation newer than ``after`` (negative: from now,
        after the last ``backlog``), and the log's newest sequence number. A person may ask for
        plain text; everyone else gets it fenced."""
        who = self._who(token)
        board, thread, dm, between = spec.board, spec.thread, spec.dm, spec.between
        after, backlog, limit, plain_text = spec.after, spec.backlog, spec.limit, spec.plain_text
        if sum(bool(x) for x in (board, thread, dm)) != 1:
            raise ValueError("follow one of a board, a thread or a conversation")
        if plain_text and who.role != HUMAN:
            raise Denied("only the person's token reads messages unfenced")
        member = self._scope(who, board, thread, dm, between)
        lim = self.ws.limits
        if not plain_text:  # the same default caps as every other agent-facing read
            limit, backlog = min(limit, lim.read_items), min(backlog, lim.read_items)
        rows = self.ws.bus.log.after(0 if after < 0 else after)
        found = [r for r in rows if r["kind"] == "msg" and self.ws.bus.live(r) and member(r)
                 and coordination_access.can_read_row(self.ws, who, r)]
        found = found[-backlog:] if after < 0 and backlog > 0 else [] if after < 0 else found[:limit]
        newest = rows[-1]["seq"] if rows else max(after, 0)
        if after >= 0 and len(found) == limit:
            newest = found[-1]["seq"]
        return {"messages": [self._plain(r) if plain_text else self._show(r, lim.read_item_chars, who)
                             for r in found],
                "seq": newest}

    def _scope(self, who: Identity, board: str, thread: int, dm: str,
               between: str) -> Callable[[dict[str, Any]], bool]:
        if board:
            self.require_read(who, board)
            return lambda r: r["to"] == board
        if thread:
            head = self.ws.bus.thread(thread)
            if not head:
                raise ValueError(f"no thread {thread}")
            self._thread_access(who, head)
            return lambda r: thread in (r["seq"], r.get("thread"))
        a = between or who.id
        if not (valid_id(dm) and valid_id(a)):
            raise ValueError("name an agent id")
        self._pair_ok(who, a, dm)
        return lambda r: ({r["from"], r["to"]} == {a, dm} and a != dm
                          and not r["to"].startswith("#") and r["to"] != "*")

    def _thread_access(self, who: Identity, rows: list[dict[str, Any]]) -> None:
        for row in rows:
            if row["to"].startswith("#"):
                self.require_read(who, row["to"])
            elif not coordination_access.can_read_row(self.ws, who, row):
                raise Denied(f"{who.id} cannot read that coordination thread")

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
            if head and not coordination_access.can_read_row(self.ws, who, head):
                head = None
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
        self._thread_access(who, rows)
        head, replies = rows[0], rows[1:]
        recent = replies[-5:]
        lines = [f"[{head['seq']}] {head['from']}: "
                 f"{data_line(head['subject'] or head['body'], 300)}"]
        if len(replies) > len(recent):
            lines.append(f"({len(replies) - len(recent)} earlier replies omitted)")
        lines += [f"[{r['seq']}] {r['from']}: {data_line(r['body'], 200)}" for r in recent]
        return self._digested(lines, f"thread#{root}", {"messages": len(rows)})

    def rollup(self, token: str, ack: bool = False) -> dict[str, Any] | None:
        """The newest few unseen `#announcements` as one-liners, oldest first, with how many
        more there are; None when there is nothing. Never wakes anything; ``ack`` marks them
        all seen."""
        who = self._who(token)
        mine = self.store.state()[1].get(who.id, {})
        if mine.get(("board", ANNOUNCE)) == "silent":
            return None
        seen = self.store.marks(who.id).get(ANNOUNCE_MARK, 0)
        rows = [r for r in self._rows(ANNOUNCE) if r["seq"] > seen and r["from"] != who.id]
        if not rows:
            return None
        top = self.ws.limits.announce_rollup
        lines = [f"[{r['seq']}] {r['type']} {r['from']}: {data_line(r['body'], 200)}"
                 for r in rows[-top:]]
        more = len(rows) - len(lines)
        if more:
            lines.insert(0, f"(+{more} older announcements; `digest` lists them)")
        if ack:
            self.store.mark(who.id, ANNOUNCE_MARK, rows[-1]["seq"])
        return self._digested(lines, "announcements", {"messages": len(rows), "more": more})

    # -- the summary ----------------------------------------------------------------------
    def summary(self, token: str) -> dict[str, Any]:
        """The caller's boards and unread counts, and unread direct messages."""
        listing = self.list(token)
        return {"boards": [{"name": b["name"], "unread": b["unread"], "member": b["member"]}
                           for b in listing],
                "unread_lines": [f"{b['unread']} unread on {b['name']}" for b in listing
                                 if b["unread"]],
                "dms_unread": sum(p["unread"] for p in self.dm_list(token))}
