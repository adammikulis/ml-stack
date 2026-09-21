"""What a `Client` sends on every request, and how it reaches its server."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

SAMPLERS = ("temperature", "top_p", "top_k", "min_p")


@dataclass(frozen=True, slots=True)
class Request:
    """The settings every request carries. A sampler left None is left out of the body."""

    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    min_p: float | None = None
    n_predict: int = 16384               # a ceiling, not a budget
    spec_draft_max: int | None = None    # tokens the draft head guesses ahead
    slot: int | None = None
    context: int | None = None           # Ollama's num_ctx
    keep_alive: str | int | None = None  # Ollama's keep_alive

    def sampling(self) -> dict[str, Any]:
        """The sampler fields chosen, with temperature 0.0 when none was."""
        out = {name: getattr(self, name) for name in SAMPLERS
               if getattr(self, name) is not None}
        out.setdefault("temperature", 0.0)
        return out


@dataclass(frozen=True, slots=True)
class Transport:
    """How a `Client` reaches its server: the protocol, the key, the timeout, the tries."""

    api: str | None = None
    api_key: str | None = None
    timeout: float = 180.0
    tries: int = 1
