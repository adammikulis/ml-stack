"""One relationship, one word for it: folding a vocabulary a model invented as it went.

A model asked to read prose into (subject, relation, object) coins the relation as it goes,
so an open vocabulary drifts: ``works_at`` here, ``worksat`` there, ``worked_at`` next door.
Nothing is wrong with any of them and the graph is split three ways, so a question about who
works where is answered by a third of the edges that should have answered it.

Two things settle a drifted vocabulary, and they are not the same thing. A **written** map --
somebody deciding that ``mentors`` and ``advises`` are one relationship -- is a decision, and
is obeyed. A **fold** is a guess that two names are one word spelled twice, made by
:func:`ml_stack.entities.close`, and a guess is only safe while one of the spellings is rare:
once both carry :data:`ESTABLISHED` weight they are two names people keep choosing, and
merging them would rewrite relationships with nobody saying so. Every fold is logged and
returned, so it is something a reader or a test can point at rather than infer.

:func:`dead_keys` is the other half of keeping a written map honest: a key nothing produces
any more never fires, and fails silently for as long as nobody looks.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Set
from typing import Any

__all__ = ["ESTABLISHED", "dead_keys", "fold_edges", "fold_names"]

# How much use makes a name established. An automatic fold is a guess that two names are one
# word spelled twice; once both carry this much weight they are two names people keep
# choosing. A written entry is a decision, not a guess, and is exempt.
ESTABLISHED = 3


def fold_names(weight: Mapping[str, int], written: Mapping[str, str] | None = None, *,
               established: int = ESTABLISHED, log: Callable[[str], None] | None = None,
               label: str = "",
               settles: str = "a written entry settles which is right",
               ) -> tuple[dict[str, str], list[dict[str, Any]]]:
    """``{name: the name it folds into}`` and one record per name that folded.

    ``weight`` is how much each name is used -- edges, mentions, rows, whatever the caller
    counts -- and the heaviest spelling is the one that survives. ``written`` is the map
    somebody decided by hand, keyed casefolded; a name in it takes the value it names
    whatever its weight. Everything else folds into an already-kept name that
    :func:`ml_stack.entities.close` calls the same word, unless both are past
    ``established``, when neither folds and ``log`` is told why.

    Every name in ``weight`` gets an entry, mapping to itself when it folds into nothing, so
    the result can be used as a plain lookup. ``label`` prefixes the log lines with what
    vocabulary this is ("relations: ..."); ``settles`` is the sentence that says how a
    refusal is resolved, which is caller-specific because the written map is the caller's.
    """
    from ml_stack.entities import close

    said = {k.casefold(): v for k, v in (written or {}).items()}
    prefix = f"{label}: " if label else ""
    canonical: dict[str, str] = {}
    folds: list[dict[str, Any]] = []
    for name in sorted(weight, key=lambda n: (-weight[n], n)):
        if name in said:
            canonical[name] = said[name]
            if said[name] != name:
                folds.append({"from": name, "into": said[name], "written": True})
            continue
        for kept in list(canonical.values()):
            if kept == name or not close(name.replace("_", ""), kept.replace("_", "")):
                continue
            if weight[name] >= established and weight.get(kept, 0) >= established:
                if log:
                    log(f"{prefix}'{name}' ({weight[name]}) and '{kept}' "
                        f"({weight.get(kept, 0)}) are both established, so neither folds; "
                        f"{settles}")
                continue
            canonical[name] = kept
            folds.append({"from": name, "into": kept, "written": False})
            if log:
                log(f"{prefix}'{name}' ({weight[name]}) folded into "
                    f"'{kept}' ({weight.get(kept, 0)})")
            break
        else:
            canonical[name] = name
    return canonical, folds


def fold_edges(edges: Mapping[tuple[str, str, str], dict[str, Any]],
               written: Mapping[str, str] | None = None, *,
               established: int = ESTABLISHED, log: Callable[[str], None] | None = None,
               label: str = "", settles: str = "a written entry settles which is right",
               field: str = "rel", provenance: str = "messages",
               ) -> tuple[dict[tuple[str, str, str], dict[str, Any]], list[dict[str, Any]]]:
    """:func:`fold_names` over a graph's edges, keyed ``(source, relation, target)``.

    Each edge carries a ``weight`` (how often it was said) and ``provenance`` (a list of the
    records that said it); two edges that become the same triple are one edge whose weight
    is the sum and whose provenance is the union, in the order first seen. ``field`` is the
    key inside the edge that repeats the relation, rewritten to the name it folded into.
    Returns the folded edges and the fold records, which is what makes a fold reviewable.
    """
    weight: dict[str, int] = {}
    for (_, rel, _), e in edges.items():
        weight[rel] = weight.get(rel, 0) + int(e.get("weight") or 1)
    canonical, folds = fold_names(weight, written, established=established, log=log,
                                  label=label, settles=settles)
    out: dict[tuple[str, str, str], dict[str, Any]] = {}
    for (source, rel, target), e in edges.items():
        name = canonical.get(rel, rel)
        key = (source, name, target)
        if key in out:
            kept = out[key]
            kept["weight"] += e["weight"]
            kept[provenance] = list(dict.fromkeys(kept[provenance] + e[provenance]))
        else:
            out[key] = {**e, field: name}
    return out, folds


def dead_keys(maps: Mapping[str, tuple[Mapping[str, Any], Set[str]]]) -> dict[str, list[str]]:
    """Per map, the keys nothing produces any more -- sorted, and only the maps with any.

    A hand-written map is checked against what the extractor still says, never against the
    graph it produced: a key that fires is *renamed*, so it is missing from the graph's
    labels exactly like a dead one, and checking there reports every working entry as dead.
    Each map is ``(keys, alive)``; ``alive`` is casefolded by the caller, which knows which
    fields of its own records a key is allowed to match.
    """
    found = {what: sorted(k for k in keys if k.casefold() not in alive)
             for what, (keys, alive) in maps.items()}
    return {what: keys for what, keys in found.items() if keys}
