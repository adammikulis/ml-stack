"""The lines a commit adds to a file, so the name check judges what the commit wrote and not what
the file already said: text committed earlier was accepted then, and a product name in an old
table must not block every later edit of its file."""

from __future__ import annotations

import re
import subprocess

HUNK = re.compile(r"^@@ -\S+ \+(\d+)(?:,(\d+))? @@", re.M)


SIMILARITY = "-M50%"
"""How alike two files must be for git to call one a rename of the other: the figure
scripts/hooks/renames.py uses, so every check reads the same pairs. A copy is never a rename."""


def _renamed_from(root: str | None, span: list[str], path: str) -> list[str]:
    """The path ``path`` was renamed from in ``span``, as a list that is empty when it was not."""
    done = subprocess.run(["git", "diff", "--name-status", "-z", SIMILARITY, *span], cwd=root or None,
                          capture_output=True, text=True, errors="replace", check=False)
    fields = done.stdout.split("\0")
    at = 0
    while at < len(fields) and fields[at]:
        width = 3 if fields[at][0] in "RC" else 2
        if fields[at][0] == "R" and fields[at + 2] == path:
            return [fields[at + 1]]
        at += width
    return []


def added_lines(root: str | None, path: str, against: str | None = None) -> set[int]:
    """The 1-based line numbers of ``path`` that the staged change adds, or, with ``against``, that
    ``HEAD`` adds over that revision. A new file adds every line. A pure deletion adds none."""
    if against and against.startswith("-"):
        raise ValueError(f"not a revision: {against!r}")
    span = [f"{against}...HEAD"] if against else ["--cached"]
    paths = [path, *_renamed_from(root, span, path)]
    done = subprocess.run(["git", "diff", "-U0", SIMILARITY, *span, "--", *paths], cwd=root or None,
                          capture_output=True, text=True, errors="replace", check=False)
    lines: set[int] = set()
    for start, count in HUNK.findall(done.stdout):
        first, length = int(start), 1 if count == "" else int(count)
        lines.update(range(first, first + length))
    return lines
