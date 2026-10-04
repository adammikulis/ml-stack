"""Files on the board: post, read, list, search and delete, each checked against who may read
the message that carries the file."""

from __future__ import annotations

import hashlib
import os
import re
import stat
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ml_stack.net import scan as netscan, sniff
from ml_stack.net.scanners import default_scanners
from ml_stack.workspace import plain
from ml_stack.workspace.boardapi import Held, data_line
from ml_stack.workspace.boards import ANNOUNCE
from ml_stack.workspace.chain import ChainLog
from ml_stack.workspace.filestore import FileStore, Unavailable, handle_of
from ml_stack.workspace.identity import AGENT, HUMAN, Denied, Identity, valid_id
from ml_stack.workspace.screen import Refused, fence, injection_markers, refusals

if TYPE_CHECKING:
    from ml_stack.workspace.service import Workspace

__all__ = ["REF", "Attachment", "FileApi", "Where", "human_size", "valid_handle"]

REF = re.compile(r"(?<![A-Za-z0-9_])file:([0-9a-f]{12})(?![A-Za-z0-9_])")
HANDLE = re.compile(r"[0-9a-f]{12}")
BLOCKED_KINDS = ("executable", "zip", "gzip", "tar", "gguf", "safetensors")
BLOCKED_SUFFIXES = (*sniff.ARCHIVE_SUFFIXES, *sniff.PICKLE_SUFFIXES, ".exe", ".dll", ".so",
                    ".dylib", ".app", ".dmg", ".pkg", ".msi", ".sh", ".bat", ".cmd", ".ps1",
                    ".command", ".py", ".js", ".jar", ".gguf", ".safetensors")
PICKLE_MAGIC = re.compile(rb"\x80[\x02-\x05]")
HOUR_S = 3600.0
NOT_AVAILABLE = "file {} (not available to you)"
LISTED = 50


@dataclass(frozen=True, slots=True)
class Attachment:
    """What is said about a file being posted besides its bytes."""

    name: str = ""
    note: str = ""
    reply_to: int = 0
    derived_from: str = ""


@dataclass(frozen=True, slots=True)
class Where:
    """Conditions on which files a listing or search covers."""

    board: str = ""
    project: str = ""
    by: str = ""
    derived_from: str = ""


PLAIN = Attachment()
EVERYWHERE = Where()


@dataclass(frozen=True, slots=True)
class Posting:
    """A file ready to be recorded in the store and graph."""

    who: Identity
    to: str
    reply_to: int
    meta: dict[str, Any]
    digest: str
    data: bytes
    text: str
    note: str
    source: str


def human_size(n: int) -> str:
    """``n`` bytes as ``812 B``, ``12 KB`` or ``1.5 MB``."""
    if n < 1024:
        return f"{n} B"
    if n < 1024 * 1024:
        return f"{n // 1024} KB"
    return f"{n / (1024 * 1024):.1f} MB"


def valid_handle(text: str) -> bool:
    """Whether ``text`` is a file handle: 12 lowercase hex characters."""
    return bool(HANDLE.fullmatch(text))


def clean_name(name: str, width: int) -> str:
    """``name`` as a plain file name: no separators, control or bidirectional characters, cut
    to ``width``; ``file`` when nothing is left."""
    base = re.split(r"[\\/]", name.replace("\x00", " "))[-1]
    cleaned = plain.line(re.sub(r"[\\/:*?\"<>|]", "_", base), width).strip(". ")
    return cleaned or "file"


class FileApi:
    """File operations for a `Workspace`; the content and the graph live in a `FileStore`."""

    def __init__(self, ws: Workspace, store: FileStore | None = None,
                 scanners: Sequence[netscan.Scanner] | None = None) -> None:
        self.ws = ws
        self.store = store or FileStore(ws.base)
        self.scanners = scanners
        self.removed = ChainLog(ws.base / "files-deleted.jsonl", ws.clock)

    # -- access ----------------------------------------------------------------------
    def _carriers(self) -> list[dict[str, Any]]:
        return [r for r in self.ws.bus.log.rows() if r["kind"] == "msg" and r.get("file")
                and self.ws.bus.live(r)]

    def _row_ok(self, who: Identity, row: dict[str, Any], boards: dict[str, Any]) -> bool:
        if row["to"].startswith("#"):
            return self.ws.board.can_read(who, row["to"], boards)
        return who.role != AGENT or who.id in (row["from"], row["to"])

    def _readable(self, who: Identity, rows: list[dict[str, Any]] | None = None
                  ) -> dict[str, dict[str, Any]]:
        """Handle to the first carrying message ``who`` may read, for every file they may."""
        boards = self.ws.board.store.state()[0]
        found: dict[str, dict[str, Any]] = {}
        for r in rows if rows is not None else self._carriers():
            if r["file"]["id"] not in found and self._row_ok(who, r, boards):
                found[r["file"]["id"]] = r
        return found

    def _gone(self) -> set[str]:
        return {str(r["id"]) for r in self.removed.rows() if r.get("op") == "delete"}

    def _reach(self, who: Identity, handle: str) -> dict[str, Any]:
        """The carrying message that lets ``who`` read ``handle``; `Denied` with one wording
        for a file that is unknown, deleted or not theirs."""
        found = self._readable(who).get(handle) if valid_handle(handle) else None
        if found is None or handle in self._gone():
            self.ws.audit("file.denied", who.id, file=plain.line(handle, 16))
            raise Denied(NOT_AVAILABLE.format(plain.line(handle, 16)))
        return found

    def _who(self, token: str) -> Identity:
        who = self.ws.auth(token)
        self.ws._may(who, "read")
        return who

    # -- referring to files in text --------------------------------------------------
    def render(self, who: Identity, text: str) -> str:
        """``text`` with each ``file:HANDLE`` shown as its name and size when ``who`` may read
        it, else as ``file HANDLE (not available to you)``. Nothing is fetched or expanded."""
        if "file:" not in text:
            return text
        reach = self._readable(who)
        gone = self._gone()

        def one(m: re.Match[str]) -> str:
            h = m.group(1)
            if h not in reach or h in gone:
                return NOT_AVAILABLE.format(h)
            meta = reach[h]["file"]
            return f"file:{h} ({data_line(meta['name'], 60)}, {human_size(int(meta['size']))})"
        return REF.sub(one, text)

    # -- posting ---------------------------------------------------------------------
    def _destination(self, who: Identity, to: str, reply_to: int) -> tuple[str, int]:
        if to.startswith("thread:"):
            root = to[7:]
            head = self.ws.bus.get(int(root)) if root.isdigit() else None
            if head is None:
                raise ValueError(f"no thread {plain.line(root, 20)}")
            self.ws.board._thread_access(who, [head])
            if head["to"].startswith("#"):
                return head["to"], int(root)
            return (head["to"] if head["from"] == who.id else head["from"]), int(root)
        if to == ANNOUNCE or not (to.startswith("#") or valid_id(to)):
            raise ValueError("attach to a board such as #general, an agent id, or thread:SEQ")
        return to, reply_to

    def _kind(self, name: str, data: bytes) -> tuple[str, bool]:
        """``(kind, is_text)``; `Refused` for what is never kept."""
        shown = sniff.detect(data[:sniff.HEAD])
        low = name.lower()
        if low.endswith(BLOCKED_SUFFIXES) or shown in BLOCKED_KINDS or PICKLE_MAGIC.match(data[:2]):
            raise Refused(f"{plain.line(name, 40)} is not accepted: archives, executables, scripts, "
                          f"pickles and model weights are never posted ({shown or 'by its name'})")
        if shown == "text" or shown == "json":
            try:
                data.decode("utf-8")
            except UnicodeDecodeError:
                return "binary", False
            return shown, b"\x00" not in data
        return shown or "binary", False

    def _scan(self, who: Identity, name: str, kind: str, data: bytes) -> None:
        scanners = self.scanners if self.scanners is not None else default_scanners()
        tmp = self.store.directory / "scan"
        tmp.mkdir(parents=True, exist_ok=True, mode=0o700)
        path = tmp / f"{hashlib.sha256(data).hexdigest()[:16]}.bin"
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        try:
            with os.fdopen(fd, "wb") as out:
                out.write(data)
            summary = netscan.scan_file(path, scanners)
            keep, why = netscan.ScanPolicy().decide(netscan.category(kind), summary)
        finally:
            path.unlink(missing_ok=True)
        if not keep:
            qid = self.ws.quarantine.hold("file", f"{who.id}:{plain.line(name, 40)}",
                                          ["scanner"], f"file {plain.line(name, 40)} held: {why}", who.id)
            self.ws.audit("file.refused", who.id, why="scan", qid=qid)
            raise Refused(f"the file was refused and held as {qid}: {why}")

    def _caps(self, who: Identity, to: str, size: int) -> None:
        lim, now = self.ws.limits, self.ws.clock()
        if size > lim.file_bytes:
            self.ws.audit("write.refused", who.id, what="file", why="size", bytes=size)
            raise Refused(f"the file is {size} bytes; the limit is {lim.file_bytes}")
        rows = self._carriers()
        recent = sum(int(r["file"]["size"]) for r in rows
                     if r["from"] == who.id and r["ts"] > now - HOUR_S)
        if recent + size > lim.file_bytes_per_hour:
            self.ws.audit("write.refused", who.id, what="file", why="hourly")
            raise Refused(f"{who.id} posted {recent} bytes of files in the last hour; this one is "
                          f"{size} and the limit is {lim.file_bytes_per_hour} an hour")
        here = sum(1 for r in rows if r["to"] == to) if to.startswith("#") else 0
        if here >= lim.files_per_board:
            self.ws.audit("write.refused", who.id, what="file", why="board")
            raise Refused(f"{to} holds {here} files; the limit is {lim.files_per_board}")

    def attach(self, token: str, to: str, data: bytes,
               given: Attachment = PLAIN) -> dict[str, Any]:
        """Post ``data`` as a file message on a board, to an agent or into a thread. The
        message carries the handle, never the content."""
        who = self.ws.auth(token)
        self.ws._may(who, "send")
        lim = self.ws.limits
        note, reply_to = given.note.strip(), given.reply_to
        if "\n" in note or len(note) > lim.file_note_chars:
            raise Refused(f"the note is one sentence of at most {lim.file_note_chars} characters; "
                          f"put detail in the file and keep the note to a sentence")
        if not data:
            raise Refused("the file is empty")
        name = clean_name(given.name, lim.file_name_chars)
        to, reply_to = self._destination(who, to, reply_to)
        if to.startswith("#"):
            self.ws.board.require_post(who, to)
        elif not self.ws._known(to):
            raise ValueError(f"no agent called {plain.line(to, 48)}")
        source = self._source(who, given.derived_from)
        self._caps(who, to, len(data))
        kind, is_text = self._kind(name, data)
        text = data.decode("utf-8") if is_text else ""
        if is_text and (why := refusals(text, self.ws.denylist)):
            self.ws.audit("write.refused", who.id, what="file", why="screen")
            raise Refused(f"the file was not stored: {'; '.join(why)}. Remove it and attach again.")
        if not is_text:
            self._scan(who, name, kind, data)
        flags = injection_markers(text) if is_text else []
        digest, handle = handle_of(data)
        qid = self.ws.quarantine.hold(
            "file", f"{who.id}->{to}", flags,
            f"file {handle} {name} ({len(data)} bytes) held for: {', '.join(flags)}", who.id) if flags else ""
        meta = {"id": handle, "sha": digest[:16], "name": name, "size": len(data), "type": kind, "text": is_text,
                "held": qid}
        try:
            self._record(Posting(who, to, reply_to, meta, digest, data, text, note, source))
        except Unavailable as err:
            self.ws.audit("write.refused", who.id, what="file", why="store")
            raise Refused(f"the file store is not available, so nothing was posted: {err}") from err
        line = (f"file: {name} {human_size(len(data))} sha:{digest[:4]}… (file {handle})"
                + (f" held as {qid}" if qid else ""))
        sent = self.ws.post(who, to, "file", f"{line}\n{note}" if note else line,
                            reply_to=reply_to, file=meta)
        self.ws.audit("file.post", who.id, file=handle, name_hash=hashlib.sha256(name.encode()).hexdigest()[:12],
                      size=len(data), to=to, held=qid)
        return {"seq": sent["seq"], "file": handle, "handle": f"file:{handle}", "name": name,
                "size": len(data), "type": kind, "held": qid, "to": to, "authority": "none"}

    def _source(self, who: Identity, ref: str) -> str:
        """The graph node a ``--derived-from`` reference names: a file handle or a message
        number the caller can read; `Denied` otherwise."""
        if not ref:
            return ""
        ref = ref.removeprefix("file:")
        if valid_handle(ref):
            self._reach(who, ref)
            return f"file:{ref}"
        row = self.ws.bus.get(int(ref)) if ref.isdigit() else None
        if row is None or not self._row_ok(who, row, self.ws.board.store.state()[0]):
            raise Denied(f"{plain.line(ref, 20)} is not available to you")
        return f"msg:{row['seq']}"

    def _record(self, p: Posting) -> None:
        who, meta, digest, text, note, source = p.who, p.meta, p.digest, p.text, p.note, p.source
        self.store.put(p.data)
        project = self.ws.registry.info(who.id).get("project", {})
        lim = self.ws.limits
        indexed = text[:lim.file_index_chars] if text and not meta["held"] else ""

        def change(g: Any) -> None:
            node = f"file:{meta['id']}"
            if not g.has(node):
                g.upsert_node({"id": node, "kind": "file", "label": meta["name"], "attrs": {
                    "sha256": digest, "size": meta["size"], "ftype": meta["type"],
                    "name": meta["name"], "note": note, "text": indexed, "state": "clear",
                    "qid": meta["held"]}})
            g.upsert_node({"id": f"agent:{who.id}", "kind": "agent", "label": who.id})
            g.upsert_edge({"source": node, "target": f"agent:{who.id}", "rel": "posted_by",
                           "model": "agent-claimed"})
            self._place(g, node, p, project)
            if source:
                if not g.has(source):
                    g.upsert_node({"id": source, "kind": "message", "label": source})
                g.upsert_edge({"source": node, "target": source, "rel": "derived_from"})
        self.store.edit(change)

    @staticmethod
    def _place(g: Any, node: str, p: Posting, project: dict[str, str]) -> None:
        to, reply_to, who = p.to, p.reply_to, p.who.id
        if to.startswith("#"):
            place, rel = f"board:{to}", "in_board"
        else:
            pair = "|".join(sorted((who, to)))
            place, rel = f"board:dm:{pair}", "in_board"
        g.upsert_node({"id": place, "kind": "board", "label": place[6:]})
        g.upsert_edge({"source": node, "target": place, "rel": rel})
        if reply_to:
            for kind, target, name in (("thread", f"thread:{reply_to}", "in_thread"),
                                       ("message", f"msg:{reply_to}", "reply_to")):
                g.upsert_node({"id": target, "kind": kind, "label": target})
                g.upsert_edge({"source": node, "target": target, "rel": name})
        if project.get("key"):
            pid = f"project:{project['key']}"
            g.upsert_node({"id": pid, "kind": "project", "label": project.get("name", ""),
                           "attrs": {"key": project["key"]}})
            g.upsert_edge({"source": node, "target": pid, "rel": "in_project"})

    # -- reading ---------------------------------------------------------------------
    def _line(self, row: dict[str, Any]) -> str:
        meta = row["file"]
        sha = f" sha:{meta['sha'][:4]}…"
        return f"file: {data_line(meta['name'], 80)} {human_size(int(meta['size']))}{sha} (file {meta['id']})"

    def meta(self, token: str, handle: str) -> dict[str, Any]:
        """What the board says about a file the caller may read: no content."""
        who = self._who(token)
        row = self._reach(who, handle)
        meta = row["file"]
        state = self.ws.quarantine.state(meta.get("held", "")) if meta.get("held") else ""
        return {"id": handle, "handle": f"file:{handle}", "name": data_line(meta["name"], 80),
                "size": meta["size"], "type": meta["type"], "is_text": bool(meta["text"]),
                "by": row["from"], "in": row["to"] if row["to"].startswith("#") else "a conversation",
                "message": row["seq"], "state": state or ("clear" if not meta.get("held") else "held"),
                "line": self._line(row), "authority": "none"}

    def read_text(self, token: str, handle: str, limit: int = 0, widen: bool = False) -> dict[str, Any]:
        """The text of a text file, fenced as data and cut to a bounded length; ``held`` says how
        many characters were left out. A held file is read by the person only until released."""
        who = self._who(token)
        row = self._reach(who, handle)
        meta = row["file"]
        if not meta["text"]:
            raise Refused(f"file {handle} is binary: only --out writes it, --meta describes it")
        self._unheld(who, meta)
        text = self._content(handle).decode("utf-8")
        shown, cut = plain.text(text, len(text) if widen else limit if limit > 0
                                else self.ws.limits.file_text_chars)
        screened = fence(shown, f"workspace:file:{handle}", f"file posted by {row['from']}")
        self.ws.audit("file.read", who.id, file=handle, size=meta["size"], what="text",
                      name_hash=hashlib.sha256(meta["name"].encode()).hexdigest()[:12])
        return {"id": handle, "name": data_line(meta["name"], 80), "size": meta["size"],
                "text": screened.text, "flags": list(screened.markers), "authority": "none",
                "held": max(len(text) - len(shown), 0) if cut else 0}

    def _unheld(self, who: Identity, meta: dict[str, Any]) -> None:
        qid = meta.get("held", "")
        if qid and who.role != HUMAN and self.ws.quarantine.state(qid) != "released":
            raise Denied(f"file {meta['id']} is held as {qid} until a person releases it")

    def _content(self, handle: str) -> bytes:
        digest = self._digest(handle)
        try:
            return self.store.get(digest)
        except Unavailable as err:
            raise Refused(f"file {handle} cannot be read: {err}") from err

    def _digest(self, handle: str) -> str:
        try:
            node = self.store.node(f"file:{handle}")
        except Unavailable as err:
            raise Refused(f"file {handle} cannot be read: {err}") from err
        if node is None:
            raise Refused(f"file {handle} has no stored content")
        return str(node["attrs"]["sha256"])

    def save(self, token: str, handle: str, target: str, roots: Sequence[Path]) -> dict[str, Any]:
        """Write the file's bytes to ``target``: a path inside one of ``roots``, with no ``..``
        and no symlink on the way, that does not exist yet; created with mode 0600."""
        who = self._who(token)
        row = self._reach(who, handle)
        self._unheld(who, row["file"])
        path = self._out_path(target, roots)
        data = self._content(handle)
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600)
        with os.fdopen(fd, "wb") as out:
            out.write(data)
        self.ws.audit("file.read", who.id, file=handle, size=len(data), what="out")
        return {"id": handle, "wrote": str(path), "size": len(data), "authority": "none"}

    @staticmethod
    def _out_path(target: str, roots: Sequence[Path]) -> Path:
        raw = Path(target)
        if not target or "\x00" in target or ".." in raw.parts:
            raise Refused("--out is a path inside your worktree, with no ..")
        for root in roots:
            base = root.resolve()
            path = raw if raw.is_absolute() else base / raw
            try:
                inside = path.relative_to(base)
            except ValueError:
                continue
            walk = base
            for part in inside.parts:
                walk = walk / part
                if walk.is_symlink():
                    raise Refused("--out never goes through a symlink")
            if not path.parent.is_dir():
                raise Refused("--out needs its folder to exist")
            if path.exists():
                raise Refused("--out never overwrites a file")
            if not stat.S_ISDIR(path.parent.stat().st_mode):
                raise Refused("--out needs its folder to exist")
            return path
        raise Refused("--out is a path inside your worktree")

    # -- listing and searching -------------------------------------------------------
    def _wanted(self, who: Identity, w: Where) -> dict[str, dict[str, Any]]:
        reach = self._readable(who)
        for gone in self._gone():
            reach.pop(gone, None)
        if w.board or w.project or w.by or w.derived_from:
            try:
                inside = self.store.where(by=w.by, project=w.project, board=w.board,
                                          derived_from=w.derived_from.removeprefix("file:"))
            except Unavailable as err:
                raise Refused(f"the file index is not available: {err}") from err
            reach = {h: r for h, r in reach.items() if f"file:{h}" in inside}
        return reach

    def _item(self, row: dict[str, Any], extra: str = "") -> dict[str, Any]:
        meta = row["file"]
        place = row["to"] if row["to"].startswith("#") else "a conversation"
        line = f"{self._line(row)} by {row['from']} in {data_line(place, 48)}" + (f": {extra}" if extra else "")
        return {"id": meta["id"], "handle": f"file:{meta['id']}", "name": data_line(meta["name"], 80),
                "size": meta["size"], "by": row["from"], "line": line,
                "text": fence(line, "workspace:files", "file names and notes written by agents").text,
                "authority": "none"}

    def list(self, token: str, where: Where = EVERYWHERE, limit: int = 0, widen: bool = False) -> Held:
        """Handles of files the caller may read, newest message first; a bounded few unless
        widened, ``.held`` counts the rest."""
        who = self._who(token)
        reach = self._wanted(who, where)
        ordered = sorted(reach.values(), key=lambda r: (-int(r["seq"]), r["file"]["id"]))
        take = len(ordered) if widen else limit if limit > 0 else self.ws.limits.file_search_results
        out = Held(self._item(r) for r in ordered[:take])
        out.held = len(ordered) - len(out)
        return out

    def search(self, token: str, query: str, where: Where = EVERYWHERE, limit: int = 0) -> Held:
        """Files whose name, note or text matches ``query``, ranked by the graph's search, ties
        by handle, each with a short snippet; only files the caller may read."""
        who = self._who(token)
        reach = self._wanted(who, where)
        try:
            ranked = self.store.search(query, {f"file:{h}" for h in reach})
            nodes = {n["id"]: n for n in self.store.nodes()} if ranked else {}
        except Unavailable as err:
            raise Refused(f"the file index is not available: {err}") from err
        take = min(limit if limit > 0 else self.ws.limits.file_search_results, LISTED)
        out = Held(self._item(reach[i[5:]], self._snippet(nodes[i], query)) for i, _ in ranked[:take])
        out.held = len(ranked) - len(out)
        return out

    def _snippet(self, node: dict[str, Any], query: str) -> str:
        attrs, width = node["attrs"], self.ws.limits.file_snippet_chars
        words = [w for w in query.casefold().split() if w]
        for field in ("text", "note", "name"):
            body = " ".join(str(attrs.get(field, "")).split())
            low = body.casefold()
            at = min((low.find(w) for w in words if w in low), default=-1)
            if at >= 0:
                start = max(at - width // 4, 0)
                return data_line(("…" if start else "") + body[start:start + width], width + 1)
        return ""

    # -- deleting --------------------------------------------------------------------
    def delete(self, token: str, handle: str) -> dict[str, Any]:
        """Remove a file: its content is shredded and its node becomes a tombstone. A person
        only."""
        who = self.ws.auth(token)
        if who.role != HUMAN:
            self.ws.audit("file.denied", who.id, act="delete")
            raise Denied("only a person's token deletes a file")
        if not valid_handle(handle):
            raise ValueError("a file handle is 12 hex characters")
        try:
            digest = self._digest(handle)
            shredded = self.store.shred(digest)

            def tombstone(g: Any) -> None:
                g.upsert_node({"id": f"file:{handle}", "kind": "file", "label": "(deleted)",
                               "attrs": {"sha256": digest, "state": "deleted"}})
            self.store.edit(tombstone)
        except Unavailable as err:
            raise Refused(f"the file store is not available: {err}") from err
        self.removed.append({"op": "delete", "id": handle, "by": who.id})
        self.ws.audit("file.delete", who.id, file=handle, shredded=shredded)
        return {"deleted": handle, "shredded": shredded}

    def content_for_person(self, token: str, handle: str) -> tuple[dict[str, Any], bytes]:
        """For the person's page: the file's metadata and bytes; a person's token only."""
        who = self._who(token)
        if who.role != HUMAN:
            raise Denied("only a person's token reads a file unfenced")
        row = self._reach(who, handle)
        return row["file"], self._content(handle)

