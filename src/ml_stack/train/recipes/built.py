"""What a recipe's builder returns: the model and everything a run does with it."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ml_stack.train.trainer import Hook

__all__ = ["Built", "Hook", "Phase"]


@dataclass
class Phase:
    """A follow-on round after the main fit: ``prepare()`` changes the model and says how,
    or returns None to end the phases; then ``steps`` more steps at ``learning_rate``."""

    name: str
    prepare: Callable[[], str | None]
    steps: int
    learning_rate: float


@dataclass
class Built:
    """Everything Trainer needs, plus what the run should record."""

    model: Any
    optimizer: Any
    loss: Callable[[Any, Any], Any]
    batches: Callable[[int], Any] | None = None
    """None when the recipe was built without data."""
    eval_batches: Callable[[int], Any] | None = None
    config: dict[str, Any] = field(default_factory=dict)
    step: Any = None
    """The `train.step.Step` to advance with, when the framework's default is the wrong one."""
    predict: Callable[[Any], Any] | None = None
    """Inputs to a JSON-serialisable output, with the weights the model holds now."""
    hooks: list[Hook] = field(default_factory=list)
    phases: list[Phase] = field(default_factory=list)
    export: Callable[[Path], dict[str, Any]] | None = None
    """Writes what the recipe ships into the directory it is given, and says what it wrote."""
