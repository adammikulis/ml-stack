"""Shared notes: decisions, rules, facts and open questions, with a trust level each."""

from __future__ import annotations

import hashlib
import os
import shlex
import subprocess
import time
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from ml_stack.graph.search import lexical, rrf
from ml_stack.workspace.chain import ChainLog
from ml_stack.workspace.identity import AGENT, HUMAN, Denied, Identity

__all__ = ["KINDS", "RANK", "Notes", "allowed_command"]

KINDS = ("decision", "rule", "fact", "question")
RANK = {"agent-claimed": 1, "test-verified": 2, "human": 3}
OUTPUT_BYTES = 65_536


def allowed_command(argv: list[str], allow: Iterable[Iterable[str]]) -> bool:
    """Whether ``argv`` starts with every word of one allow-list entry."""
    return any(entry and argv[:len(entry)] == entry for entry in (list(e) for e in allow))


class Notes:
    """Notes kept as chained add and verify rows, folded into their current state."""

    def __init__(self, base: Path, clock: Callable[[], float] = time.time) -> None:
        self.log = ChainLog(base / "notes.jsonl", clock)
        self.clock = clock

    def add(self, who: Identity, fields: dict[str, Any]) -> dict[str, Any]:
        """Append a note authored by ``who``; its trust never comes from ``fields``."""
        row = self.log.append({"op": "add", "author": who.id, "role": who.role, **fields})
        return self.view(self._folded()[row["seq"]])

    def _folded(self) -> dict[int, dict[str, Any]]:
        notes: dict[int, dict[str, Any]] = {}
        for row in self.log.rows():
            if row["op"] == "add":
                notes[row["seq"]] = {**row, "id": row["seq"], "superseded_by": 0, "verified": None}
                for old in row.get("supersedes", []):
                    if old in notes and not notes[old]["superseded_by"]:
                        notes[old]["superseded_by"] = row["seq"]
            elif row["op"] == "verify" and row["note"] in notes:
                notes[row["note"]]["verified"] = row
        return notes

    def trust_of(self, note: dict[str, Any]) -> str:
        """The trust level a note has now."""
        if note["role"] == HUMAN:
            return "human"
        proof = note["verified"]
        if proof and proof["exit"] == 0 and not self._rotted(note["ttl_s"], proof["ts"]):
            return "test-verified"
        return "agent-claimed"

    def _rotted(self, ttl_s: float, since: float) -> bool:
        return bool(ttl_s) and self.clock() - since > ttl_s

    def view(self, note: dict[str, Any]) -> dict[str, Any]:
        """A note as shown: trust, staleness and the fact that it binds nobody."""
        proof = note["verified"]
        stale = self._rotted(note["ttl_s"], proof["ts"] if proof else note["ts"])
        return {"id": note["id"], "kind": note["nkind"], "title": note["title"],
                "body": note["body"], "author": note["author"], "ts": note["ts"],
                "source": note["source"], "tags": note["tags"], "trust": self.trust_of(note),
                "stale": stale, "ttl_s": note["ttl_s"], "supersedes": note["supersedes"],
                "superseded_by": note["superseded_by"], "held": note["held"], "flags": note["flags"],
                "verify_cmd": note["verify_cmd"], "binding": False,
                "last_verified": None if not proof else {
                    "at": proof["ts"], "exit": proof["exit"], "by": proof["by"],
                    "output_sha256": proof["out_sha"]}}

    def get(self, note_id: int) -> dict[str, Any] | None:
        """The note ``note_id`` as shown, or None."""
        note = self._folded().get(note_id)
        return self.view(note) if note else None

    def count_by(self, author: str) -> int:
        """How many notes ``author`` wrote."""
        return sum(1 for n in self._folded().values() if n["author"] == author)

    def check_supersedes(self, who: Identity, targets: list[int]) -> None:
        """Refuse a supersede of a note that does not exist or outranks the author."""
        notes = self._folded()
        mine = RANK[who.trust]
        for target in targets:
            if target not in notes:
                raise ValueError(f"no note {target} to supersede")
            if RANK[self.trust_of(notes[target])] > mine:
                raise Denied(f"note {target} is {self.trust_of(notes[target])}; an "
                             f"{who.trust} note cannot replace it, add a question instead")

    def search(self, query: str, kind: str = "", include_old: bool = False,
               limit: int = 10) -> list[dict[str, Any]]:
        """Notes matching the words of ``query``, ranked with the graph's keyword search."""
        live = [n for n in self._folded().values()
                if (include_old or not n["superseded_by"]) and (not kind or n["nkind"] == kind)]
        graph = {"nodes": [{"id": str(n["id"]), "label": n["title"], "mentions": n["id"],
                            "attrs": {"body": n["body"], "tags": " ".join(n["tags"]),
                                      "source": n["source"]}} for n in live]}
        words = query.split()
        ranks = [lexical(graph, w, limit=len(live) or 1) for w in words]
        if not any(ranks):
            return []
        by_id = {str(n["id"]): n for n in live}
        fused = rrf(*ranks, limit=limit)
        every = set.intersection(*(set(r) for r in ranks))
        order = [i for i in fused if i in every] + [i for i in fused if i not in every]
        return [self.view(by_id[i]) for i in order[:limit]]

    def verify(self, who: Identity, note_id: int, cwd: Path, allow: list[list[str]],
               timeout_s: float) -> dict[str, Any]:
        """Run a note's re-derive command if the owner allow-listed it, and record the result."""
        if who.role == AGENT:
            raise Denied("only a lead or human token runs a note's command")
        note = self._folded().get(note_id)
        if note is None:
            raise ValueError(f"no note {note_id}")
        argv = shlex.split(note["verify_cmd"]) if note["verify_cmd"] else []
        if not argv:
            raise ValueError(f"note {note_id} has no re-derive command")
        if not allowed_command(argv, allow):
            raise Denied("that command is not on the owner's verify_allow list in limits.json")
        env = {k: os.environ[k] for k in ("PATH", "LANG", "TMPDIR") if k in os.environ}
        try:
            done = subprocess.run(argv, cwd=cwd, env=env, capture_output=True,
                                  timeout=timeout_s, check=False)
            code, out = done.returncode, done.stdout + done.stderr
        except subprocess.TimeoutExpired as late:
            code, out = 124, (late.stdout or b"") + b"[timed out]"
        except OSError as err:
            code, out = 127, str(err).encode()
        self.log.append({"op": "verify", "note": note_id, "exit": code, "by": who.id,
                         "cmd": note["verify_cmd"],
                         "out_sha": hashlib.sha256(out[:OUTPUT_BYTES]).hexdigest()})
        return self.get(note_id) or {}
