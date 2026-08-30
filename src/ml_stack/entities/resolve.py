"""Canonical names and near-duplicate folding for labelled records."""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any, Collection, Iterable, Mapping

STOPWORDS = frozenset({"the", "a", "an", "of", "and", "&", "for", "in", "on"})

# stripped repeatedly, longest first, so that a word and its inflections reduce to one form
_SUFFIXES = ("ment", "ance", "ence", "ity", "ing", "ion", "ical", "ic", "al", "er")
_MIN_STEM = 4

_WORD = re.compile(r"[a-z0-9]+")
_HANDLE = re.compile(r"[a-z0-9._-]+")
_LEADING_AT = re.compile(r"^@")


def stem(word: str) -> str:
    """'agentic', 'agents' -> 'agent'. Plural first, then derivational suffixes to a fixpoint."""
    w = word.casefold()
    if len(w) > _MIN_STEM and w.endswith("s") and not w.endswith("ss"):
        w = w[:-1]
    while True:
        for suffix in _SUFFIXES:
            if w.endswith(suffix) and len(w) - len(suffix) >= _MIN_STEM:
                w = w[: -len(suffix)]
                break
        else:
            return w


def fold_key(label: str, stopwords: Collection[str] | None = None) -> tuple[str, ...]:
    """The comparison key for a label: sorted stems of its content words."""
    stop = STOPWORDS if stopwords is None else frozenset(stopwords)
    return tuple(sorted({stem(w) for w in _WORD.findall(label.casefold()) if w not in stop}))


def canonical(name: str, aliases: Mapping[str, str]) -> str:
    """The alias target for a name, matched case-insensitively; the tidied name when there is none."""
    key = _LEADING_AT.sub("", " ".join(name.split())).strip().strip(".,;:")
    return aliases.get(key.casefold(), key)


def looks_like_handle(name: str) -> bool:
    """'night owl', 'a.person', 'katanhz96' -- not a written personal name."""
    n = name.strip()
    return not any(w[:1].isupper() for w in n.split()) or bool(_HANDLE.fullmatch(n))


def fold_duplicates(records: Iterable[Mapping[str, Any]], *, rank: Mapping[str, int],
                    weak_kinds: Collection[str] = (), stopwords: Collection[str] | None = None,
                    weight_key: str = "weight") -> dict[str, str]:
    """Map record id -> surviving id for near-duplicate labels. Deterministic, no model.

    Each record carries `id`, `kind`, `label` and a weight. A record of a weak kind may fold
    into a record of another kind; anything else folds only within its own kind. `rank` orders
    the kinds, lowest first, for choosing which of a group survives.

    - equal fold keys: the survivor is the best (rank, -weight, label)
    - weak-kind A whose key is a strict subset of weak-kind B: B folds into A

    No value in the map is itself a key: a record folds straight to the one that survives.
    """
    weak = frozenset(weak_kinds)
    items = list(records)
    fold: dict[str, str] = {}
    groups: dict[tuple[str, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for r in items:
        groups[fold_key(r["label"], stopwords)].append(r)
    for key, group in groups.items():
        if not key or len(group) < 2:
            continue
        keep = min(group, key=lambda r: (rank.get(r["kind"], len(rank)), -r.get(weight_key, 0), r["label"]))
        for r in group:
            if r["id"] != keep["id"] and (r["kind"] == keep["kind"] or r["kind"] in weak):
                fold[r["id"]] = keep["id"]
    loose = [r for r in items if r["kind"] in weak and r["id"] not in fold]
    keys = {r["id"]: set(fold_key(r["label"], stopwords)) for r in loose}
    for a in loose:
        ka = keys[a["id"]]
        if not ka:
            continue
        for b in loose:
            if b["id"] != a["id"] and b["id"] not in fold and ka < keys[b["id"]]:
                fold[b["id"]] = a["id"]
    # a subset of a subset can fold in two steps; callers resolve one, so close the map here
    for src in list(fold):
        seen, dst = {src}, fold[src]
        while dst in fold and dst not in seen:
            seen.add(dst)
            dst = fold[dst]
        fold[src] = dst
    return fold
