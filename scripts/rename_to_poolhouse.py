#!/usr/bin/env python3
"""Rename the project to Poolhouse: every tracked path and every tracked text file.

Default is a dry run that lists each path move and each file that would change, then the
names that two old spellings would collapse into. ``--apply`` moves the paths with ``git mv``,
rewrites the files and writes the changed paths to ``--list`` so they can be staged by name.
Running it again changes nothing.

The token rules are in ``rewrite``; the files it must leave alone are ``EXCLUDED`` (history,
the rename's own records) and ``PROTECTED`` (files only a person at a terminal edits: their
patch is made by ``--protected-patch``).
"""

from __future__ import annotations

import argparse
import difflib
import re
import subprocess
import sys
from pathlib import Path

KEPT = (
    "github.com/adammikulis/ml-stack",
    "githubusercontent.com/adammikulis/ml-stack",
    "adammikulis/ml-stack",
    "repos/ml-stack",
    "com.mlstack",
    "com/mlstack",
    "ml-stack/keystore/v1",
    "ml-stack/activity/v1",
    "ml-stack/workspace-files/v1",
    "ml-stack/requests/v1",
    "ml-stack/v1",
    "ml-stack-memory-key-id",
    "ml-stack-join-v1",
)
"""Spellings that survive. The repository URLs stay until the owner renames the repository; the
Android package is the installed app's identity; the last seven are bytes that key derivation and
authenticated encryption already used on data at rest, so renaming one makes that data unreadable."""

RULES = (
    (re.compile(r"X-ML-Stack-"), "X-Poolhouse-"),
    (re.compile(r"x-ml-stack-"), "x-poolhouse-"),
    (re.compile(r"ML[_-]?STACK"), "POOLHOUSE"),
    (re.compile(r"MLSTACK"), "POOLHOUSE"),
    (re.compile(r"ML [Ss]tack|ML-Stack|MLStack|MlStack"), "Poolhouse"),
    (re.compile(r"mlStack"), "poolhouse"),
    (re.compile(r"ml[_ -]stack|mlstack"), "poolhouse"),
    (re.compile(r"POOLSIDE"), "POOLHOUSE"),
    (re.compile(r"Poolside"), "Poolhouse"),
    (re.compile(r"poolside"), "poolhouse"),
)
"""Old spelling to new, one rule per casing, applied in order."""

PROSE = re.compile(r"(?<![\w./~$-])ml-stack(?![\w/.-]|\.\w)")
"""The project's name standing alone in running text of a document, which takes a capital."""

ARTICLE = re.compile(r"\b([Aa])n (poolhouse|Poolhouse|POOLHOUSE)\b")

SENTENCE = re.compile(r"""(["'])poolhouse (?=(?:is|will|needs|has|changed)\b)""")
"""A message that opens with the name and a verb: a sentence, so the name takes a capital."""

EXCLUDED = (
    "CHANGELOG.md",
    "LICENSE",
    "scripts/rename_to_poolhouse.py",
    "tests/test_rename_to_poolhouse.py",
    "src/poolhouse/legacy.py",
    "src/poolhouse/migrate.py",
    "tests/test_migrate.py",
    "tests/test_migrate_edges.py",
    "tests/migrate_support.py",
    "docs/cutover.md",
    "tests/test_hooks_rename_aware.py",
    "docs/rename-protected.patch",
)
"""Paths whose old names are history or the rename's own record; they are neither moved nor rewritten."""

PROTECTED = re.compile(
    r"(^|/)(src/(ml_stack|poolhouse)/workspace/person_\w+\.py|scripts/hooks/(claude-user-prompt|person-consume"
    r"|claude-bash-guard|claude-edit-guard|pre-push|rules_loader\.py)|\.claude/settings(\.local)?\.json)$")
"""Files that only a person at a terminal changes (the edit guard's own list): left as they are."""

PATH_RULES = (
    (re.compile(r"ml_stack"), "poolhouse"),
    (re.compile(r"ml-stack"), "poolhouse"),
    (re.compile(r"poolside"), "poolhouse"),
    (re.compile(r"Poolside"), "Poolhouse"),
)


def rewrite(text: str, prose: bool = False) -> str:
    """The text with the project renamed; ``prose`` also capitalises the name in running text."""
    for literal in sorted(KEPT, key=len, reverse=True):
        text = text.replace(literal, f"\x00K{KEPT.index(literal)}\x00")
    if prose:
        text = _capitalised(text)
    for pattern, new in RULES:
        text = pattern.sub(new, text)
    text = ARTICLE.sub(r"\1 \2", text)
    text = SENTENCE.sub(r"\1Poolhouse ", text)
    for index, literal in enumerate(KEPT):
        text = text.replace(f"\x00K{index}\x00", literal)
    return text


def _capitalised(text: str) -> str:
    """Capitalise the name where a document uses it as a word, leaving code spans and fences."""
    out, fenced = [], False
    for line in text.split("\n"):
        if line.lstrip().startswith("```"):
            fenced = not fenced
        elif not fenced:
            pieces = re.split(r"(`[^`]*`|\]\([^)]*\)|<[^>]*>)", line)
            line = "".join(p if i % 2 else PROSE.sub("Poolhouse", p) for i, p in enumerate(pieces))
        out.append(line)
    return "\n".join(out)


def target(path: str) -> str:
    """Where a tracked path goes."""
    if path in EXCLUDED:
        return path
    for pattern, new in PATH_RULES:
        path = pattern.sub(new, path)
    return path


def tracked(root: Path) -> list[str]:
    """Every tracked path, by git."""
    out = subprocess.run(["git", "ls-files", "-z"], cwd=root, capture_output=True, check=True).stdout
    return [name for name in out.decode().split("\0") if name]


def read(root: Path, path: str) -> str | None:
    """The file as text, or None when it is binary, a link or missing."""
    where = root / path
    if where.is_symlink() or not where.is_file():
        return None
    try:
        return where.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return None


def changed(root: Path, path: str) -> str | None:
    """The renamed text of a tracked file, or None when it must not or need not change."""
    if path in EXCLUDED or PROTECTED.search(path):
        return None
    text = read(root, path)
    if text is None:
        return None
    new = rewrite(text, prose=path.endswith(".md"))
    return new if new != text else None


def collisions(root: Path, paths: list[str]) -> list[str]:
    """Distinct old tokens that the rules turn into one new token, so a person can look at them."""
    seen: dict[str, set[str]] = {}
    token = re.compile(r"[A-Za-z0-9_.-]*(?:ml[_ -]?stack|poolside)[A-Za-z0-9_.-]*", re.I)
    for path in paths:
        text = None if path in EXCLUDED else read(root, path)
        for found in token.findall(text or ""):
            seen.setdefault(rewrite(found), set()).add(found)
    return [f"{new}: {', '.join(sorted(old))}" for new, old in sorted(seen.items())
            if len({o.lower().replace("_", "-") for o in old}) > 1]


def protected_patch(root: Path, paths: list[str]) -> str:
    """A patch that renames the protected files' contents, against their moved paths."""
    out = []
    for path in paths:
        if not PROTECTED.search(path):
            continue
        text = read(root, path)
        if text is None:
            continue
        new = rewrite(text)
        if new == text:
            continue
        where = target(path)
        out.extend(difflib.unified_diff(text.splitlines(True), new.splitlines(True), f"a/{where}", f"b/{where}"))
    return "".join(out)


def _prune(root: Path, directory: Path) -> None:
    """Remove ``directory`` and its parents while they are empty: ``git mv`` leaves them."""
    while directory != Path() and not any((root / directory).iterdir()):
        (root / directory).rmdir()
        directory = directory.parent


def main(argv: list[str] | None = None) -> int:
    ask = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ask.add_argument("--apply", action="store_true", help="move and rewrite; the default only lists")
    ask.add_argument("--root", type=Path, default=Path.cwd(), help="the checkout to rename")
    ask.add_argument("--list", type=Path, help="write every touched path (new names) here")
    ask.add_argument("--protected-patch", type=Path, help="write the protected files' patch here")
    args = ask.parse_args(argv)
    root = args.root.resolve()
    paths = tracked(root)
    moves = [(old, target(old)) for old in paths if target(old) != old]
    rewrites = [path for path in paths if (new := changed(root, path)) is not None]
    for old, new in moves:
        print(f"move    {old} -> {new}")
    for path in rewrites:
        print(f"rewrite {path}")
    for line in collisions(root, paths):
        print(f"collide {line}")
    print(f"{len(moves)} paths move, {len(rewrites)} files rewrite", file=sys.stderr)
    if args.protected_patch:
        args.protected_patch.write_text(protected_patch(root, paths), encoding="utf-8")
    if not args.apply:
        return 0
    for old, new in moves:
        (root / new).parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "mv", old, new], cwd=root, check=True)
        _prune(root, Path(old).parent)
    for path in rewrites:
        text = read(root, target(path))
        (root / target(path)).write_text(rewrite(text or "", prose=path.endswith(".md")), encoding="utf-8")
    if args.list:
        touched = sorted({new for _, new in moves} | {target(path) for path in rewrites})
        args.list.write_text("\n".join(touched) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
