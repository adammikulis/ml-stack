"""Locate and parse the contract files."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

_CACHE: dict[str, Any] = {}


class ContractError(RuntimeError):
    """A contract file is missing, unparseable, or missing a required key."""


def _candidates() -> list[Path]:
    found: list[Path] = []

    override = os.environ.get("ML_STACK_CONTRACTS")
    if override:
        found.append(Path(override).expanduser())

    found.append(Path(__file__).resolve().parent / "_data")

    # Walk up looking for a repo-root `contracts/`. Bounded: stop at the filesystem root.
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "contracts"
        if (candidate / "model_tiers.json").is_file():
            found.append(candidate)
            break

    return found


def contracts_dir() -> Path:
    """The directory the contracts are being read from."""
    cached = _CACHE.get("dir")
    if cached is not None:
        return cached

    tried = _candidates()
    for candidate in tried:
        if (candidate / "model_tiers.json").is_file():
            _CACHE["dir"] = candidate
            return candidate

    raise ContractError(
        "no contracts directory found (looked for model_tiers.json in: "
        + ", ".join(str(p) for p in tried)
        + "). Set ML_STACK_CONTRACTS to point at one."
    )


def load(name: str) -> Any:
    """Parse one contract file by name, e.g. ``load("model_tiers.json")``. Cached."""
    cached = _CACHE.get(name)
    if cached is not None:
        return cached

    path = contracts_dir() / name
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ContractError(f"cannot read contract {path}: {exc}") from exc

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ContractError(f"contract {path} is not valid JSON: {exc}") from exc

    _CACHE[name] = parsed
    return parsed


def sampling_schema() -> dict[str, Any]:
    """The sampler surface, shared with any non-Python host that reads the contracts."""
    return load("sampling.schema.json")


def grammar(name: str) -> str:
    """One GBNF grammar's source, by stem or filename."""
    stem = name[:-5] if name.endswith(".gbnf") else name
    path = contracts_dir() / "grammars" / f"{stem}.gbnf"
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ContractError(f"cannot read grammar {path}: {exc}") from exc


def recipes() -> list[dict[str, Any]]:
    """Every recipe contract, sorted by id."""
    folder = contracts_dir() / "recipes"
    if not folder.is_dir():
        return []
    out = []
    for path in sorted(folder.glob("*.json")):
        try:
            out.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError) as exc:
            raise ContractError(f"cannot read recipe {path}: {exc}") from exc
    return out


def recipe(recipe_id: str) -> dict[str, Any]:
    """One recipe contract by id."""
    for found in recipes():
        if found.get("id") == recipe_id:
            return found
    known = ", ".join(sorted(r.get("id", "?") for r in recipes())) or "none"
    raise ContractError(f"no recipe {recipe_id!r}; have {known}")


def reset_cache() -> None:
    """Drop the parse cache. For tests that point ML_STACK_CONTRACTS somewhere else."""
    _CACHE.clear()
