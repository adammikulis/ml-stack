"""Named training setups, built from a config a form can produce."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from ml_stack.contracts import ContractError, recipes
from ml_stack.train.checkpoint import find_latest, is_valid, load_state, load_tensors, tensor_reader
from ml_stack.train.recipes.built import Built, Hook, Phase
from ml_stack.train.recipes.classify_text import build_classifier
from ml_stack.train.recipes.text_lm import build_text_lm
from ml_stack.train.recipes.tool_calls import build_tool_caller
from ml_stack.train.step import step_for

__all__ = ["Builder", "Built", "Hook", "Phase", "build", "known", "load", "register", "spec",
           "specs", "validate"]

Builder = Callable[[dict[str, Any], dict[str, Any], "Path | None", str], Built]
"""``(spec, config, data or None, framework) -> Built``."""

FRAMEWORKS = ("torch", "mlx", "")
UNIVERSAL = {"size": "", "data": None, "out": None, "seed": None, "eval_every": None,
             "checkpoint_every": None, "framework": "", "max_minutes": 0.0}
"""Settings every recipe accepts, and their defaults; None is left out unless given."""

_BUILDERS: dict[str, Builder] = {
    "classify-text": build_classifier,
    "text-lm": build_text_lm,
    "tool-calls": build_tool_caller,
}
_REGISTERED: dict[str, dict[str, Any]] = {}


def specs() -> list[dict[str, Any]]:
    """Every recipe contract, shipped and registered, sorted by id."""
    return sorted([*recipes(), *_REGISTERED.values()], key=lambda s: s["id"])


def spec(recipe_id: str) -> dict[str, Any]:
    """One recipe contract by id."""
    for found in specs():
        if found["id"] == recipe_id:
            return found
    raise ContractError(f"no recipe {recipe_id!r}; have {', '.join(known()) or 'none'}")


def known() -> list[str]:
    return [s["id"] for s in specs()]


def register(spec: dict[str, Any], builder: Builder) -> None:
    """Add a recipe defined outside the library. Raises when its id is taken."""
    recipe_id = str(spec.get("id") or "")
    if not recipe_id:
        raise ValueError("a recipe spec needs an id")
    if recipe_id in known():
        raise ValueError(f"a recipe called {recipe_id!r} already exists")
    _REGISTERED[recipe_id] = dict(spec)
    _BUILDERS[recipe_id] = builder


def validate(recipe_id: str, config: dict[str, Any]) -> dict[str, Any]:
    """Config with defaults filled in. Raises on an unknown or out-of-range field."""
    found = spec(recipe_id)
    fields = {f["name"]: f for f in found.get("fields", [])}
    allowed = set(fields) | set(UNIVERSAL)
    unknown = sorted(set(config) - allowed)
    if unknown:
        raise ValueError(
            f"{recipe_id} has no setting called {unknown[0]!r}; "
            f"it accepts {sorted(allowed)}")

    out = {**{k: v for k, v in UNIVERSAL.items() if v is not None}, **config}
    if out["framework"] not in FRAMEWORKS:
        raise ValueError(f"framework must be torch or mlx, got {out['framework']!r}")
    out["max_minutes"] = float(out["max_minutes"] or 0.0)
    if out["max_minutes"] < 0:
        raise ValueError(f"max_minutes must be at least 0, got {out['max_minutes']}")
    size = out.get("size") or ""
    sizes = found.get("sizes", {})
    if size and size not in sizes:
        raise ValueError(f"{recipe_id} has no size {size!r}; it has {sorted(sizes)}")
    # A size's defaults fill in only where the caller said nothing, so --set always wins.
    by_size = dict(sizes.get(size, {}).get("defaults", {})) if size else {}

    for name, f in fields.items():
        if name not in out or out[name] is None:
            out[name] = by_size.get(name, f.get("default"))
            continue
        kind = f.get("type")
        if kind == "bool":
            out[name] = bool(out[name])
            continue
        if kind == "text":
            out[name] = str(out[name])
            continue
        value = float(out[name]) if kind == "float" else int(out[name])
        low, high = f.get("min"), f.get("max")
        if low is not None and value < low:
            raise ValueError(f"{name} must be at least {low}, got {value}")
        if high is not None and value > high:
            raise ValueError(f"{name} must be at most {high}, got {value}")
        out[name] = value

    return out


def build(recipe_id: str, config: dict[str, Any], data: Path | str | None,
          *, framework: str = "") -> Built:
    """The model, optimizer and loss for one recipe, and its batches when ``data`` is given."""
    config = validate(recipe_id, config)
    return _built(recipe_id, config, data, framework or config["framework"])


def _built(recipe_id: str, config: dict[str, Any], data: Path | str | None,
           framework: str) -> Built:
    if framework not in FRAMEWORKS:
        raise ValueError(f"framework must be torch or mlx, got {framework!r}")
    builder = _BUILDERS.get(recipe_id)
    if builder is None:
        raise ValueError(f"no builder for recipe {recipe_id!r}")
    return builder(spec(recipe_id), config, None if data is None else Path(data), framework)


def load(directory: Path | str, *, which: str = "latest") -> Built:
    """The recipe a run trained, rebuilt without data and holding ``which`` checkpoint's weights."""
    root = Path(directory).expanduser()
    where = find_latest(root) if which == "latest" else root / which
    if where is None or not is_valid(where):
        raise FileNotFoundError(f"no checkpoint {which!r} under {root}")
    recorded = dict(load_state(where).config)
    recipe_id, framework = recorded.get("recipe"), recorded.get("framework")
    if not recipe_id or framework not in ("torch", "mlx"):
        raise ValueError(f"{where} does not record the recipe and framework it was trained with")
    accepted = set(UNIVERSAL) | {f["name"] for f in spec(recipe_id).get("fields", [])}
    settings = validate(recipe_id, {k: v for k, v in recorded.items() if k in accepted})
    built = _built(recipe_id, {**recorded, **settings, "framework": framework}, None, framework)
    step = built.step or step_for(built.model, built.optimizer, built.loss)
    step.restore(load_tensors(where, read_tensors=tensor_reader(framework)), None)
    return built
