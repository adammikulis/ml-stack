"""A label read as a name: the form two spellings of one name share, the plurals among a
set of names, a name's singulars and plurals, the names a spelling away from one, and why a
label does not read as the name of a thing at all."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping

from ml_stack.entities.spelling import close

__all__ = ["TRAILING_PREPOSITION", "kin", "near", "plurals", "same_name", "suspect"]

_CLAUSE = re.compile(r"\b(that|which|who|aims? to|is an?|are|used to|in order to)\b", re.I)
TRAILING_PREPOSITION = re.compile(r"\b(of|in|to|for|by|with|on|at|from|into)$", re.I)
_GENERIC = frozenset({"thing", "things", "process", "form of science", "form", "type",
                      "concept", "part", "item", "object", "structure", "substance"})


def plurals(names: Iterable[str]) -> dict[str, str]:
    """``{plural (casefolded): singular}`` for every name whose singular is also a name."""
    by_lower = {str(name).casefold(): str(name) for name in names}
    out: dict[str, str] = {}
    for low, _name in by_lower.items():
        for ending, singular in (("ies", "y"), ("es", ""), ("s", "")):
            if low.endswith(ending) and len(low) > len(ending) + 2:
                stem = low[: -len(ending)] + singular
                if stem in by_lower and stem != low:
                    out[low] = by_lower[stem]
                    break
    return out


def suspect(label: str) -> str:
    """Why a label is doubtful as a *name*, or ``""`` when it reads as one."""
    text = " ".join(str(label or "").split())
    if not text:
        return "empty"
    words = text.split()
    if len(words) > 6:
        return f"a clause of {len(words)} words, not a name"
    if _CLAUSE.search(text):
        return "reads as a clause"
    if TRAILING_PREPOSITION.search(text):
        return "ends in a preposition"
    if text.casefold() in _GENERIC:
        return "too generic to be one thing"
    if re.fullmatch(r"[\d.,%-]+", text):
        return "a number"
    if len(text) == 1:
        return "a single letter"
    return ""


def same_name(label: str) -> str:
    """The form under which two labels are one name: casefolded, with spaces, hyphens and
    underscores collapsed -- "T-cell", "t cell" and "T_cell" are one; "atrium" and
    "Natrium" are not."""
    return re.sub(r"[\s_\-]+", "", str(label or "").casefold())


def kin(key: str) -> list[str]:
    """The singulars and plurals of one `same_name` key."""
    out: list[str] = []
    for ending, singular in (("ies", "y"), ("es", ""), ("s", "")):
        if key.endswith(ending) and len(key) > len(ending) + 2:
            out.append(key[: -len(ending)] + singular)
    if key.endswith("y") and len(key) > 3:
        out.append(key[:-1] + "ies")
    out += [key + "s", key + "es"]
    return [one for one in dict.fromkeys(out) if one and one != key]



def near(by_length: Mapping[int, list[str]], key: str) -> list[str]:
    """The store's names a spelling away from one incoming name; only lengths that could be."""
    out: list[str] = []
    for length in range(len(key) - 2, len(key) + 3):
        for other in by_length.get(length) or ():
            if other != key and close(key, other):
                out.append(other)
    return out
