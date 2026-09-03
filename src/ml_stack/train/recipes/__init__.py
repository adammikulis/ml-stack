"""Named training setups, built from a config a form can produce."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = ["Built", "Recipe", "build", "known", "validate"]


@dataclass
class Built:
    """Everything Trainer needs, plus what the run should record."""

    model: Any
    optimizer: Any
    loss: Callable[[Any, Any], Any]
    batches: Callable[[int], Any]
    eval_batches: Callable[[int], Any] | None = None
    config: dict[str, Any] = field(default_factory=dict)
    step: Any = None
    """How to advance one step, when the framework's default is the wrong one. A LoRA's is
    `train.lora.LoraStep`, whose checkpoints hold the adapter rather than the frozen 16G
    base underneath it."""


Recipe = Callable[..., Built]


def known() -> list[str]:
    from ml_stack.contracts import recipes
    return [r["id"] for r in recipes()]


def validate(recipe_id: str, config: dict[str, Any]) -> dict[str, Any]:
    """Config with defaults filled in. Raises on an unknown or out-of-range field."""
    from ml_stack.contracts import recipe

    spec = recipe(recipe_id)
    fields = {f["name"]: f for f in spec.get("fields", [])}
    allowed = set(fields) | {"size", "data", "out", "seed", "eval_every",
                             "checkpoint_every"}
    unknown = sorted(set(config) - allowed)
    if unknown:
        raise ValueError(
            f"{recipe_id} has no setting called {unknown[0]!r}; "
            f"it accepts {sorted(allowed)}")

    out = dict(config)
    size = out.get("size") or ""
    sizes = spec.get("sizes", {})
    if size and size not in sizes:
        raise ValueError(f"{recipe_id} has no size {size!r}; it has {sorted(sizes)}")
    # A size may carry its own defaults: what fits and what is worth trying for *that*
    # model, rather than one set of numbers that suits a 270m and starves an 8B. They fill
    # in only where the caller said nothing, so --set always wins.
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


def build(recipe_id: str, config: dict[str, Any], data: Path | str,
          *, framework: str = "") -> Built:
    """Construct the model, optimizer, loss and batches for one recipe."""
    from ml_stack.contracts import recipe

    config = validate(recipe_id, config)
    spec = recipe(recipe_id)
    if not framework:
        from ml_stack.backend import detect_backend
        framework = detect_backend()

    if recipe_id == "text-lm":
        from ml_stack.train.recipes.text_lm import build_text_lm
        return build_text_lm(spec, config, Path(data), framework)
    if recipe_id == "classify-text":
        from ml_stack.train.recipes.classify_text import build_classifier
        return build_classifier(spec, config, Path(data), framework)
    if recipe_id == "tool-calls":
        from ml_stack.train.recipes.tool_calls import build_tool_caller
        return build_tool_caller(spec, config, Path(data), framework)
    raise ValueError(f"no builder for recipe {recipe_id!r}")
