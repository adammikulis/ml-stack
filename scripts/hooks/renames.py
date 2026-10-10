"""Rename detection the pre-commit checks share, so a file that only moved is judged against its old path.

A rename is a path git pairs with an earlier one at SIMILARITY or more; a copy is never one. The pair
lets a check compare what the file holds now with what it held, instead of calling every line new.
"""

from __future__ import annotations

SIMILARITY = "50%"
FLAG = f"-M{SIMILARITY}"


def pairs(name_status: str) -> list[tuple[str, str, str]]:
    """``(status, old, new)`` for each entry of ``git diff --name-status -z`` output.

    Only a status starting with ``R`` carries a different old path; a copy (``C``) is read as an
    addition, so a duplicated file is never excused by the file it was copied from.
    """
    fields = name_status.split("\0")
    out = []
    i = 0
    while i < len(fields) and fields[i]:
        status = fields[i]
        if status[0] in "RC":
            old, new = fields[i + 1], fields[i + 2]
            out.append(("R", old, new) if status[0] == "R" else ("A", new, new))
            i += 3
        else:
            out.append((status[0], fields[i + 1], fields[i + 1]))
            i += 2
    return out
