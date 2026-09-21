"""How an extraction is put to a model, what its answer must pass, and where it is kept."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = ["Checking", "Kept", "Prompting"]


@dataclass(frozen=True, slots=True)
class Prompting:
    """How an extraction is asked: instructions over the text, a whole conversation in
    ``messages``, or a raw ``prompt`` under a grammar built from the schema."""

    instructions: str = ""
    messages: list[dict[str, Any]] | None = None
    prompt: str | None = None
    think: bool = False
    schema_name: str = "extraction"
    n_predict: int | None = None


@dataclass(frozen=True, slots=True)
class Checking:
    """What an extraction must pass, and how many calls it has to pass it."""

    check: Callable[[dict[str, Any]], list[str]] | None = None
    tries: int = 2

    def __post_init__(self) -> None:
        if self.tries < 1:
            raise ValueError(f"tries must be at least 1, got {self.tries}")


@dataclass(frozen=True, slots=True)
class Kept:
    """Where extractions are cached, keyed by ``version`` + schema + text + ``extra``."""

    root: str | Path
    version: str = ""
    extra: str = ""
