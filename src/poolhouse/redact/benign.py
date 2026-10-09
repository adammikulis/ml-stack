"""Shapes that are never a person, judged by structure and not by a list of phrases.

A name is words made of letters, one of them capitalised. A model id, a path of drawing
numbers, a heading number and a lowercase phrase are not that, however a recogniser scores
them. A capitalised technical phrase that the repository's own documents already carry was
read when it was committed and is not a person now. Real names, emails and phone numbers
keep failing: none of these rules looks at a name's letters, only at what surrounds or
composes the match.
"""

from __future__ import annotations

import re
import subprocess

# one word of a name: a capital, then letters, apostrophes and hyphens (no digit, no inner
# digit-hyphen run such as `qwen3-30b`)
NAME_WORD = re.compile(r"^[A-Z][A-Za-z'\u2019]*(?:-[A-Z][A-Za-z'\u2019]*)*\.?$|^[A-Z]\.$")
# a quoted value made only of drawing-path commands and numbers, with at least one command
PATH_DATA = re.compile(r"""(["'])([MmLlHhVvCcSsQqTtAaZz0-9\s,.+\-eE]+)\1""")
PATH_COMMAND = re.compile(r"[MmLlHhVvCcSsQqTtAaZz]")
QUOTED_NAME = re.compile(r"^'(.*)' (?:reads as a person|is shaped like a name)")


def not_a_name(body: str) -> str | None:
    """Why a recogniser's PERSON hit cannot be a person, or None. The words must each look
    like a name's word and at least one must be capitalised and the first not a number."""
    words = body.split()
    if not any(w[:1].isupper() for w in words):
        return "no_capital: every word is lower case"
    if any(c.isdigit() for c in body):
        return "has_digit: a name holds no digit"
    if not all(NAME_WORD.match(w) or (w.islower() and w.isalpha() and 1 < len(w) < 4) for w in words):
        return "not_name_words: a word is not a capitalised word of letters"
    return None


PLACEHOLDER_LETTERS = frozenset("XY")


def placeholder(body: str) -> str | None:
    """Why a capitalised phrase is a placeholder and not a person, or None: it ends in X or Y with
    no full stop (`Land X`, `Branch Y`). Those letters stand for a variable in prose and are no
    surname's initial in practice; every other lone capital stays shaped like a name."""
    words = body.split()
    if len(words) > 1 and words[-1] in PLACEHOLDER_LETTERS:
        return "placeholder: a trailing X or Y stands for a variable"
    return None


def phone_is_code(line: str, start: int, end: int) -> str | None:
    """Why a phone-shaped run at ``line[start:end]`` is not a phone number, or None: it sits
    inside a longer identifier (a model id, a build tag) or inside quoted path data."""
    before = line[start - 1] if start else ""
    after = line[end] if end < len(line) else ""
    if before.isalpha() or before in "_-" or after.isalpha() or after == "_":
        return "inside_identifier"
    for found in PATH_DATA.finditer(line):
        if found.start(2) <= start and end <= found.end(2) and PATH_COMMAND.search(found.group(2)):
            return "inside_path_data"
    return None


def documented(root: str | None, phrase: str) -> bool:
    """Whether ``phrase`` is already in a markdown document at ``HEAD``: committed text passed
    this check when it went in, so the same capitalised phrase is the repository's own
    vocabulary. Only phrases of capitalised words qualify, so this never clears a name that
    is not already written down in the documents."""
    words = phrase.split()
    if len(words) < 2 or not all(w[:1].isupper() for w in words):
        return False
    done = subprocess.run(["git", "grep", "-q", "-F", "-e", phrase, "HEAD", "--", "*.md"],
                          cwd=root or None, capture_output=True, check=False)
    return done.returncode == 0


def quoted(why: str) -> str:
    """The phrase a finding's text is about, or the empty string."""
    found = QUOTED_NAME.match(why)
    return found.group(1) if found else ""
