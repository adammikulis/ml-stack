"""Which downloaded model a local agent runs on, and whether this machine can hold it."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path

from ml_stack.workspace.identity import valid_name

__all__ = ["AUTO", "FETCH_QUERY", "Pick", "agent_name", "choose", "fetch_hint", "short_name"]

AUTO = "auto"
FETCH_QUERY = "Qwen3.6 35B A3B"
FLASH_NEXT = re.compile(r"flash[-_ .]?next", re.I)
MOE = re.compile(r"-a\d+(?:\.\d+)?b|moe", re.I)
QWEN = re.compile(r"qwen", re.I)
_QUANT = re.compile(r"[-_.](?:IQ|Q)\d\w*|[-_.](?:BF16|F16|F32)\b|-\d{5}-of-\d{5}", re.I)
FITS = ("green", "yellow", "none")


@dataclass(frozen=True, slots=True)
class Pick:
    """The model chosen: its reference, a short name, size and rating, and when there is
    none, ``problem`` in one line with ``hint`` the command that fixes it."""

    ref: str = ""
    name: str = ""
    size_bytes: int = 0
    verdict: str = ""
    note: str = ""
    problem: str = ""
    hint: str = ""

    @property
    def ok(self) -> bool:
        return not self.problem


def short_name(name: str) -> str:
    """``name`` as a short lower-case id part: no extension, quantisation or shard marks."""
    stem = _QUANT.sub("", Path(name).name.removesuffix(".gguf"))
    cleaned = re.sub(r"[^a-z0-9.]+", "-", stem.lower()).strip("-.")
    return cleaned[:28].strip("-.") or "model"


def agent_name(pick_name: str) -> str:
    """The workspace id a local agent on ``pick_name`` gets: ``local-`` and the short name."""
    name = f"local-{short_name(pick_name)}"
    if not valid_name(name):
        raise ValueError(f"{pick_name!r} does not make a usable agent name; pass --name")
    return name


def _is_moe(one: object) -> bool:
    return bool(MOE.search(f"{getattr(one, 'name', '')} {getattr(one, 'ref', '')} "
                           f"{getattr(one, 'architecture', '')}"))


def fetch_hint(*, search: bool = True) -> str:
    """The one command that downloads the model ``auto`` wants: the Hub's reference when it
    answers, otherwise the search that finds it."""
    find = f'ml-stack-models find "{FETCH_QUERY}"'
    if not search:
        return find
    from ml_stack.hub import remote
    from ml_stack.serve import suggest

    try:
        found = suggest.recommend(goal="agent", query=FETCH_QUERY, limit=6)
    except (OSError, remote.RemoteError):
        return find
    for row in found:
        one = row.choice.candidate
        if not row.installed and not FLASH_NEXT.search(f"{one.name} {row.ref}") and _is_moe(one) \
                and QWEN.search(one.name):
            return f"ml-stack-models fetch {row.ref}"
    return find


def _ranked(installed: Sequence[object], machine: object | None) -> list[object]:
    from ml_stack.serve import suggest

    return suggest.suggest_model(installed, machine, "agent")  # type: ignore[arg-type]


def _line(row: object) -> Pick:
    one = row.candidate  # type: ignore[attr-defined]
    return Pick(ref=one.ref or str(one.path or one.name), name=one.name, size_bytes=one.size_bytes,
                verdict=row.verdict, note=row.reason)  # type: ignore[attr-defined]


def _smaller(rows: Sequence[object], than: Pick) -> Pick | None:
    for row in rows:
        one = _line(row)
        if one.size_bytes < than.size_bytes and one.verdict in FITS:
            return one
    return None


def choose(asked: str = AUTO, *, installed: Sequence[object] | None = None,
           machine: object | None = None, search: bool = True) -> Pick:
    """The downloaded model ``asked`` names, or for ``auto`` the best downloaded mixture-of-experts
    Qwen model ranked for agent work on this machine; Flash-Next is never chosen. A model that
    is absent, or rated red here, comes back with ``problem`` and what to do about it."""
    from ml_stack import hub

    have = list(hub.discover(formats=("gguf",)) if installed is None else installed)
    rows = _ranked(have, machine)
    if asked and asked != AUTO:
        found = next((r for r in rows if asked in (r.candidate.ref, r.candidate.name,  # type: ignore[attr-defined]
                                                   Path(str(r.candidate.path or "")).name)), None)  # type: ignore[attr-defined]
        if found is None:
            return Pick(problem=f"{asked} is not downloaded",
                        hint=f'ml-stack-models find "{asked}"')
        return _checked(_line(found), rows)
    pool = [r for r in rows if not FLASH_NEXT.search(f"{r.candidate.name} {r.candidate.ref}")  # type: ignore[attr-defined]
            and _is_moe(r.candidate)]  # type: ignore[attr-defined]
    pool.sort(key=lambda r: not QWEN.search(r.candidate.name))  # type: ignore[attr-defined]
    qwen = [r for r in pool if QWEN.search(r.candidate.name)]  # type: ignore[attr-defined]
    if not qwen:
        return Pick(problem="no downloaded mixture-of-experts Qwen model for agent work",
                    hint=fetch_hint(search=search))
    return _checked(_line(qwen[0]), qwen)


def _checked(pick: Pick, rows: Sequence[object]) -> Pick:
    """``pick``, or the same model with the one-line reason it will not fit and the smaller choice."""
    if pick.verdict in FITS:
        return pick
    small = _smaller([r for r in rows if r.candidate.name != pick.name], pick)  # type: ignore[attr-defined]
    offer = f"; the smaller choice is `--model {small.name}`" if small else ""
    return replace(pick, problem=f"{pick.name} is rated {pick.verdict} for this machine's memory "
                                 f"({pick.note}){offer}")
