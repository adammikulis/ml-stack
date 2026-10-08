"""The model tier table: which tier an exact model id belongs to and whether that tier may coordinate."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from fnmatch import fnmatchcase
from functools import cache
from pathlib import Path
from typing import Any

from ml_stack.workspace.modelid import VERIFIED

__all__ = ["NOT_LISTED", "NOT_VERIFIED", "LOWEST", "Tier", "TierTableError", "load_table", "tier_of"]

TABLE = Path(__file__).with_name("model_tiers.json")
NOT_LISTED = "model not in the tier table"
NOT_VERIFIED = "model not verified"
LOWEST = "lowest model tier"
WILDCARDS = frozenset("*?[")
TOP_KEYS = {"version", "retrieved", "source", "families"}
FAMILY_KEYS = {"family", "source", "tiers"}
TIER_KEYS = {"lowest", "id_globs"}


class TierTableError(ValueError):
    """The tier table is malformed."""


@dataclass(frozen=True)
class Tier:
    """A model's place in the table; ``index`` counts from 0 at the lowest tier, -1 when unknown."""

    family: str
    index: int
    lowest: bool
    coordinator_eligible: bool
    reason: str = ""


def _unknown(reason: str) -> Tier:
    return Tier("", -1, False, False, reason)


def _check_keys(shape: Mapping[str, Any], allowed: set[str], where: str) -> None:
    extra = sorted(set(shape) - allowed)
    if extra:
        raise TierTableError(f"{where} has unknown fields {extra}")


def _globs(tier: Any, where: str) -> list[str]:
    if not isinstance(tier, dict):
        raise TierTableError(f"{where} is not an object")
    _check_keys(tier, TIER_KEYS, where)
    globs = tier.get("id_globs")
    if not isinstance(tier.get("lowest"), bool):
        raise TierTableError(f"{where} needs a boolean lowest")
    if not isinstance(globs, list) or not globs or not all(isinstance(g, str) and g for g in globs):
        raise TierTableError(f"{where} needs a non-empty list of id_globs")
    return globs


def _family_rows(family: Any, where: str) -> list[tuple[str, int, bool, list[str]]]:
    if not isinstance(family, dict):
        raise TierTableError(f"{where} is not an object")
    _check_keys(family, FAMILY_KEYS, where)
    name, tiers = family.get("family"), family.get("tiers")
    if not isinstance(name, str) or not name:
        raise TierTableError(f"{where} needs a family name")
    if not isinstance(tiers, list) or not tiers:
        raise TierTableError(f"{where} needs a list of tiers")
    rows = [(name, i, tier.get("lowest") is True, _globs(tier, f"{where} tier {i}")) for i, tier in enumerate(tiers)]
    if [low for _, _, low, _ in rows].count(True) != 1 or not rows[0][2]:
        raise TierTableError(f"family {name} needs exactly one lowest tier, listed first")
    return rows


def _check_overlaps(rows: list[tuple[str, int, bool, list[str]]]) -> None:
    owners: dict[str, tuple[str, int]] = {}
    for name, index, _, globs in rows:
        for glob in globs:
            if glob in owners:
                raise TierTableError(f"id glob {glob!r} appears in {owners[glob]} and {(name, index)}")
            owners[glob] = (name, index)
    for glob, owner in owners.items():
        if WILDCARDS & set(glob):
            continue
        for other, other_owner in owners.items():
            if other != glob and other_owner != owner and fnmatchcase(glob, other):
                raise TierTableError(f"id {glob!r} in {owner} also matches {other!r} in {other_owner}")


def load_table(data: Mapping[str, Any]) -> tuple[tuple[str, int, bool, tuple[str, ...]], ...]:
    """The validated ``(family, tier index, lowest, id_globs)`` rows of a table; TierTableError when malformed."""
    if not isinstance(data, dict):
        raise TierTableError("the tier table is not an object")
    _check_keys(data, TOP_KEYS, "the tier table")
    if not isinstance(data.get("version"), int) or isinstance(data.get("version"), bool):
        raise TierTableError("the tier table needs an integer version")
    families = data.get("families")
    if not isinstance(families, list) or not families:
        raise TierTableError("the tier table needs a list of families")
    rows: list[tuple[str, int, bool, list[str]]] = []
    for i, family in enumerate(families):
        rows.extend(_family_rows(family, f"family {i}"))
    if len({r[0] for r in rows}) != len(families):
        raise TierTableError("a family name appears twice")
    _check_overlaps(rows)
    return tuple((n, i, low, tuple(g)) for n, i, low, g in rows)


@cache
def _shipped() -> tuple[tuple[str, int, bool, tuple[str, ...]], ...]:
    return load_table(json.loads(TABLE.read_text(encoding="utf-8")))


def tier_of(model_id: str, state: str, table: Mapping[str, Any] | None = None) -> Tier:
    """The tier of the exact id ``model_id`` recorded as ``state``; unknown unless the id is listed and verified."""
    if not model_id:
        return _unknown(NOT_LISTED)
    rows = _shipped() if table is None else load_table(table)
    for family, index, lowest, globs in rows:
        if any(fnmatchcase(model_id, glob) for glob in globs):
            if state != VERIFIED:
                return _unknown(NOT_VERIFIED)
            return Tier(family, index, lowest, not lowest, LOWEST if lowest else "")
    return _unknown(NOT_LISTED)
