"""One word written two ways.

Someone reading a thread knows that "pellrd" three messages down is the "Pellard" someone named
above, because they saw the word spelled once already. A per-message reader has no such luck.
What is here is the small part of that judgement a machine can make alone: whether two words
are close enough to be the same word typed twice, and which word in a piece of text a name is
closest to. The judgement about whether they *mean* the same thing belongs to whoever has the
context — a model reading the thread, or a person.
"""

from __future__ import annotations

import re

# a run of letters; digits and punctuation are not part of how a name is spelled
_WORD = re.compile(r"[^\W\d_]{2,}", re.UNICODE)


def distance(a: str, b: str, ceiling: int = 2) -> int:
    """Levenshtein distance, giving up once it is past ``ceiling``.

    Returns ``ceiling + 1`` for anything further apart, so a caller comparing against a budget
    never pays for the whole matrix on two words that are obviously unrelated.
    """
    a, b = a.casefold(), b.casefold()
    if a == b:
        return 0
    if abs(len(a) - len(b)) > ceiling:
        return ceiling + 1
    if len(a) > len(b):
        a, b = b, a
    row = list(range(len(a) + 1))
    for j, cb in enumerate(b, 1):
        prev, row[0] = row[0], j
        best = row[0]
        for i, ca in enumerate(a, 1):
            prev, row[i] = row[i], min(row[i] + 1, row[i - 1] + 1, prev + (ca != cb))
            best = min(best, row[i])
        if best > ceiling:
            return ceiling + 1
    return row[-1] if row[-1] <= ceiling else ceiling + 1


def budget(word: str) -> int:
    """How far apart two words may be and still be the same word.

    A short word cannot afford a substitution — "Neel" and "Neal" are two people, and at that
    length every neighbour is another word. A long one can: nobody types "responsibilities" the
    same way twice. Dropping or doubling a letter is handled separately, at any length.
    """
    n = len(word)
    if n <= 4:
        return 0
    if n <= 7:
        return 1
    return 2


def collapsed(word: str) -> str:
    """The word with every doubled letter written once: "Pellard" and "Pelard" both give "pelard"."""
    out = []
    for ch in word.casefold():
        if not out or out[-1] != ch:
            out.append(ch)
    return "".join(out)


def close(a: str, b: str) -> bool:
    """Whether two words are near enough to be one word spelled twice.

    Doubling is its own case, not a matter of distance: a doubled letter is the commonest way
    to write a name wrong, and it stays obvious at a length where an edit otherwise means a
    different word. Everything else is judged on distance against what the length affords.
    """
    if not a or not b:
        return False
    if collapsed(a) == collapsed(b):
        return True
    return distance(a, b, ceiling=2) <= min(budget(a), budget(b))


def nearest(name: str, text: str) -> tuple[str, int] | None:
    """The word in ``text`` that ``name`` is closest to, when it is close at all.

    Multi-word names are matched word by word: "Pelard Foundry" finds "Pellard". Returns the word
    and its distance, or None when nothing in the text is close.
    """
    words = set(_WORD.findall(text))
    best: tuple[str, int] | None = None
    for part in _WORD.findall(name):
        for word in words:
            if word.casefold() == part.casefold():
                return (word, 0)
            if not close(part, word):
                continue
            d = distance(part, word, ceiling=2)
            if best is None or d < best[1]:
                best = (word, d)
    return best


def spelled_in(name: str, text: str) -> bool:
    """Whether ``text`` writes ``name``, exactly or near enough to be a typo of it."""
    return nearest(name, text) is not None
