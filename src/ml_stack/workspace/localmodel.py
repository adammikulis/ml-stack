"""Which downloaded model a local agent runs on, and whether this machine can hold it."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path

from ml_stack import hub
from ml_stack.hub import remote
from ml_stack.serve import suggest
from ml_stack.workspace.identity import valid_name

__all__ = ["AUTO", "FETCH_QUERY", "Pick", "Selection", "agent_name", "choose", "fetch_hint",
           "short_name"]

AUTO = "auto"
FETCH_QUERY = "Qwen3.8"
FLASH_NEXT = re.compile(r"flash[-_ .]?next", re.I)
QWEN = re.compile(r"qwen", re.I)
NOT_AGENT = re.compile(r"vl\b|vision|guard|ocr|embed|coder", re.I)
_QUANT = re.compile(r"\s*\([^)]*\)|[-_.](?:IQ|Q)\d\w*|[-_.](?:BF16|F16|F32)\b|-\d{5}-of-\d{5}", re.I)
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


@dataclass(frozen=True, slots=True)
class Selection:
    """Machine and serving requirements for automatic model choice."""

    machine: object | None = None
    search: bool = True
    coding: bool = False
    context: int = 32768


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


def fetch_hint(*, search: bool = True) -> str:
    """The one command that downloads the model ``auto`` wants: the Hub's reference when it
    answers, otherwise the search that finds it."""
    find = f'ml-stack-models find "{FETCH_QUERY}"'
    if not search:
        return find
    try:
        found = suggest.recommend(goal="agent", query=FETCH_QUERY, limit=6)
    except (OSError, remote.RemoteError):
        return find
    for row in found:
        one = row.choice.candidate
        if not row.installed and not FLASH_NEXT.search(f"{one.name} {row.ref}") \
                and QWEN.search(one.name) and row.choice.verdict in FITS:
            return f"ml-stack-models fetch {row.ref}"
    return find


def _ranked(installed: Sequence[object], machine: object | None) -> list[object]:
    return suggest.suggest_model(installed, machine, "agent")  # type: ignore[arg-type]


def _line(row: object) -> Pick:
    one = row.candidate  # type: ignore[attr-defined]
    return Pick(ref=str(one.path or one.ref or one.name), name=one.name, size_bytes=one.size_bytes,
                verdict=row.verdict, note=row.reason)  # type: ignore[attr-defined]


def _smaller(rows: Sequence[object], than: Pick) -> Pick | None:
    for row in rows:
        one = _line(row)
        if one.size_bytes < than.size_bytes and one.verdict in FITS:
            return one
    return None


def _matches(one: object, asked: str) -> bool:
    path = getattr(one, "path", None)
    if Path(asked).is_absolute() or ("/" in asked and not asked.startswith("hf:")):
        try:
            return path is not None and Path(asked).resolve(strict=True) == Path(path).resolve(strict=True)
        except (OSError, RuntimeError):
            return False
    return asked in (getattr(one, "ref", ""), getattr(one, "name", ""), Path(str(path or "")).name)


def choose(asked: str = AUTO, *, installed: Sequence[object] | None = None,
           selection: Selection | None = None) -> Pick:
    """The downloaded model ``asked`` names, or the best fitting downloaded Qwen for this machine.
    Flash-Next is never chosen automatically."""
    selection = selection or Selection()
    have = list(hub.discover(formats=("gguf",)) if installed is None else installed)
    rows = _ranked(have, selection.machine)
    if asked and asked != AUTO:
        found = next((r for r in rows if _matches(r.candidate, asked)), None)  # type: ignore[attr-defined]
        if found is None:
            return Pick(problem=f"{asked} is not downloaded",
                        hint=f'ml-stack-models find "{asked}"')
        return _checked(_line(found), rows)
    if selection.coding:
        return _coding(rows, selection.search, selection.context)
    qwen = [r for r in rows if r.verdict in FITS  # type: ignore[attr-defined]
            and not FLASH_NEXT.search(f"{r.candidate.name} {r.candidate.ref}")
            and not NOT_AGENT.search(r.candidate.name)
            and QWEN.search(r.candidate.name)]
    if not qwen:
        return Pick(problem="no downloaded Qwen model fits this machine for agent work",
                    hint=fetch_hint(search=selection.search))
    for row in qwen:
        picked = _line(row)
        if _fits_context(picked, selection.context):
            return _checked(picked, qwen)
    return Pick(problem=f"no downloaded Qwen model fits at {selection.context // 1024}K context",
                hint=fetch_hint(search=selection.search))


def _fits_context(pick: Pick, context: int) -> bool:
    """Whether the memory planner can serve ``pick`` at ``context`` tokens."""
    if not pick.ref or not Path(pick.ref).is_file():
        return True
    try:
        plan = suggest.suggest(pick.ref, goal="long-context", max_verdict="yellow")
    except (OSError, ValueError):
        return True
    return plan.context >= context and plan.verdict in FITS


def _checked(pick: Pick, rows: Sequence[object]) -> Pick:
    """``pick``, or the same model with the one-line reason it will not fit and the smaller choice."""
    if pick.verdict in FITS:
        return pick
    small = _smaller([r for r in rows if r.candidate.name != pick.name], pick)  # type: ignore[attr-defined]
    offer = f"; the smaller choice is `--model {small.name}`" if small else ""
    return replace(pick, problem=f"{pick.name} is rated {pick.verdict} for this machine's memory "
                                 f"({pick.note}){offer}")


CODING_MODEL = re.compile(r"qwen\s*3\.8.*27b", re.I)
CODING_QUERY = "Qwen3.8 27B"


def _coding(rows: Sequence[object], search: bool, context: int) -> Pick:
    """The downloaded Qwen3.8-27B for a coding agent (Q4_K_XL first); Flash-Next only when named."""
    found = [r for r in rows if CODING_MODEL.search(r.candidate.name)  # type: ignore[attr-defined]
             and not FLASH_NEXT.search(r.candidate.name)]  # type: ignore[attr-defined]
    found.sort(key=lambda r: (r.verdict not in FITS, "Q4_K_XL" not in r.candidate.name))  # type: ignore[attr-defined]
    for row in found:
        pick = _line(row)
        if _fits_context(pick, context):
            return _checked(pick, found)
    if found:
        return Pick(problem=f"no downloaded Qwen3.8-27B fits at {context // 1024}K context",
                    hint=f'ml-stack-models find "{CODING_QUERY}"')
    else:
        return Pick(problem="no Qwen3.8-27B is downloaded for a coding agent",
                    hint=f'ml-stack-models find "{CODING_QUERY}"')
