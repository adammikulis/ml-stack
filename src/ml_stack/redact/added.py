"""The lines a commit adds to a file, so the name check judges what the commit wrote and not what
the file already said: text committed earlier was accepted then, and a product name in an old
table must not block every later edit of its file."""

from __future__ import annotations

import re
import subprocess

HUNK = re.compile(r"^@@ -\S+ \+(\d+)(?:,(\d+))? @@", re.M)


def added_lines(root: str | None, path: str, against: str | None = None) -> set[int]:
    """The 1-based line numbers of ``path`` that the staged change adds, or, with ``against``, that
    ``HEAD`` adds over that revision. A new file adds every line. A pure deletion adds none."""
    command = (["diff", "-U0", "--no-renames", f"{against}...HEAD", "--", path] if against
               else ["diff", "--cached", "-U0", "--no-renames", "--", path])
    done = subprocess.run(["git", *command], cwd=root or None, capture_output=True, text=True,
                          errors="replace", check=False)
    lines: set[int] = set()
    for start, count in HUNK.findall(done.stdout):
        first, length = int(start), 1 if count == "" else int(count)
        lines.update(range(first, first + length))
    return lines
