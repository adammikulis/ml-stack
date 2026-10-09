"""A guard-change diff as plain data: parsed, normalized, hashed and applied to text without a shell, so the hash the person approved is the hash of exactly the bytes that are written."""

from __future__ import annotations

import difflib
import hashlib
import re
from dataclasses import dataclass

__all__ = ["MAX_LINES", "PROTECTED", "FilePatch", "Patch", "Refused", "diff", "expected", "parse", "render"]

PROTECTED = re.compile(
    r"(^|/)(src/poolhouse/workspace/(person|guard)_\w+\.py"
    r"|scripts/hooks/(claude-user-prompt|person-consume|claude-bash-guard|claude-edit-guard|pre-push|rules_loader\.py)"
    r"|\.claude/settings(\.local)?\.json)$")
"""The code and settings that make the person record and its guards: a person approves any change to them."""

MAX_LINES = 600
HUNK = re.compile(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
MODE = re.compile(r"new file mode (100644|100755)")
SAFE_PATH = re.compile(r"[A-Za-z0-9_.\-]+(/[A-Za-z0-9_.\-]+)*")


class Refused(ValueError):
    """A diff that cannot be approved or applied, with the reason."""


@dataclass(frozen=True, slots=True)
class FilePatch:
    """The hunks for one path: ``new`` for a file the diff creates, ``mode`` its mode line when it has one."""

    path: str
    new: bool
    mode: str
    hunks: tuple[tuple[int, tuple[str, ...]], ...]
    added: int
    removed: int


@dataclass(frozen=True, slots=True)
class Patch:
    """A normalized diff, its SHA-256 and the files it changes."""

    text: str
    sha: str
    files: tuple[FilePatch, ...]


def _path(line: str, mark: str) -> str:
    named = line[4:].split("\t")[0].rstrip()
    if named == "/dev/null":
        return ""
    if not named.startswith(mark):
        raise Refused(f"unexpected path line: {line[:80]}")
    return named[len(mark):]


def _hunk(lines: list[str], at: int) -> tuple[int, tuple[str, ...], int, int, int]:
    found = HUNK.match(lines[at])
    if found is None:
        raise Refused(f"expected a hunk header, found: {lines[at][:80]}")
    old, new = int(found[2] or 1) if found[2] != "0" else 0, int(found[4] or 1) if found[4] != "0" else 0
    body, seen_old, seen_new, plus, minus = [], 0, 0, 0, 0
    at += 1
    while (seen_old < old or seen_new < new) and at < len(lines):
        row = lines[at]
        if row.startswith("\\"):
            raise Refused("a diff of a file with no newline at its end is not supported")
        kind = row[:1]
        if kind not in (" ", "+", "-"):
            raise Refused(f"a hunk line must start with a space, + or -: {row[:80]!r}")
        seen_old += kind != "+"
        seen_new += kind != "-"
        plus += kind == "+"
        minus += kind == "-"
        body.append(row)
        at += 1
    if seen_old != old or seen_new != new:
        raise Refused("a hunk is shorter than its header says")
    return at, (f"@@ -{found[1]},{old} +{found[3]},{new} @@", *body), int(found[1]), plus, minus


def _file(lines: list[str], at: int) -> tuple[int, FilePatch]:
    mode = ""
    while at < len(lines) and not lines[at].startswith("--- "):
        row = lines[at]
        if MODE.fullmatch(row):
            mode = row
        elif not (row.startswith(("diff --git ", "index ")) or row == ""):
            raise Refused(f"not allowed in a guard change (mode, rename, delete, binary or stray text): {row[:80]}")
        at += 1
    if at + 1 >= len(lines) or not lines[at + 1].startswith("+++ "):
        raise Refused("a file section needs --- and +++ lines")
    new = lines[at].split("\t")[0].rstrip() == "--- /dev/null"
    old_path = "" if new else _path(lines[at], "a/")
    path = _path(lines[at + 1], "b/")
    if lines[at + 1].startswith("+++ /dev/null") or not path or (old_path and old_path != path):
        raise Refused("a guard change creates or edits files; it never deletes, renames or copies one")
    if not SAFE_PATH.fullmatch(path) or ".." in path.split("/"):
        raise Refused(f"unsafe path: {path!r}")
    if mode and not new:
        raise Refused("a mode line is only allowed for a new file")
    at += 2
    hunks, plus, minus = [], 0, 0
    while at < len(lines) and lines[at].startswith("@@"):
        at, body, start, added, removed = _hunk(lines, at)
        hunks.append((start, body))
        plus, minus = plus + added, minus + removed
    if not hunks:
        raise Refused(f"{path} has no hunks")
    return at, FilePatch(path, new, mode, tuple(hunks), plus, minus)


def parse(raw: str) -> Patch:
    """The normalized `Patch` for the diff text ``raw``; raises `Refused` for anything outside a plain text diff of
    protected files. Timestamps, `diff --git` and `index` lines and hunk section headings are dropped."""
    if "\r" in raw or "\0" in raw:
        raise Refused("carriage returns and NUL bytes are not allowed in a guard change")
    lines = raw.split("\n")
    while lines and lines[-1] == "":
        lines.pop()
    if not lines:
        raise Refused("the diff is empty")
    files, at = [], 0
    while at < len(lines):
        at, one = _file(lines, at)
        files.append(one)
    paths = [f.path for f in files]
    if len(set(paths)) != len(paths):
        raise Refused("a path appears twice in the diff")
    outside = [p for p in paths if not PROTECTED.search(p)]
    if outside:
        raise Refused(f"not a protected file (change it the ordinary way): {', '.join(outside)}")
    text = "\n".join(_normal(f) for f in files) + "\n"
    if text.count("\n") > MAX_LINES:
        raise Refused(f"the diff is over {MAX_LINES} lines; split it so the person can read all of it")
    return Patch(text, hashlib.sha256(text.encode()).hexdigest(), tuple(files))


def _normal(one: FilePatch) -> str:
    head = [one.mode] if one.mode else []
    head += ["--- /dev/null" if one.new else f"--- a/{one.path}", f"+++ b/{one.path}"]
    return "\n".join([*head, *(row for _, body in one.hunks for row in body)])


def expected(one: FilePatch, current: str | None) -> str | None:
    """The file's text after the hunks, or None when they do not apply to ``current`` exactly where they say
    (None for ``current`` means the file does not exist)."""
    if (current is None) != one.new or (current is not None and not current.endswith("\n")):
        return None
    lines = (current or "").split("\n")[:-1] if current else []
    out, cursor = [], 0
    for start, body in one.hunks:
        before = [r[1:] for r in body[1:] if r[0] != "+"]
        after = [r[1:] for r in body[1:] if r[0] != "-"]
        first = max(start - 1, 0) if before else start
        if first < cursor or lines[first:first + len(before)] != before:
            return None
        out += lines[cursor:first] + after
        cursor = first + len(before)
    return "\n".join([*out, *lines[cursor:]]) + "\n"


def render(patch: Patch) -> str:
    """The question body a person reads: the hash, each file with its added and removed lines, and the whole diff."""
    names = "\n".join(f"  {'new ' if f.new else ''}{f.path}  +{f.added} -{f.removed}" for f in patch.files)
    return (f"Apply this change to guard files?\nDiff sha256: {patch.sha}\nFiles:\n{names}\n\n{patch.text.rstrip()}")


def diff(path: str, old: str | None, new: str) -> str:
    """The diff text that turns ``old`` (None for a file that does not exist) into ``new`` at repository ``path``."""
    before = (old or "").splitlines()
    return "\n".join(difflib.unified_diff(before, new.splitlines(), "/dev/null" if old is None else f"a/{path}",
                                          f"b/{path}", lineterm="", n=3)) + "\n"
