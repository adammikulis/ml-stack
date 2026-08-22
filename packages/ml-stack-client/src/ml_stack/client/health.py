"""Is a model server up, and what is it actually serving?

Two ideas here are load-bearing:

**Watch for death, do not just wait.** A bad model path makes a server exit in under a
second, and polling a dead port for two minutes turns "no such file" into "timed out",
which sends the reader looking for a slow load that never happened. Liveness arrives as a
callable so this module stays subprocess-free; ``ml_stack.serve`` passes
``lambda: proc.poll() is None``.

**Ask what ran, do not assume what you asked for.** The quantisation, context size and
sampler defaults a server actually came up with are frequently not the ones on the
command line, so ``serving_params`` reads ``/props`` back.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from ml_stack.client.http import ServerError, request_json

# Ordered by how much they tell you. /health is llama.cpp's cheap liveness bit; a server
# can answer it while still loading weights. /v1/models means the OpenAI surface is up,
# which is what a client actually needs.
HEALTH_PATHS = ("/health", "/v1/models", "/models")

# Sentinel seeds llama.cpp reports when none was pinned. Reading these back as a real
# seed makes a run look reproducible when it is not.
_SENTINEL_SEEDS = frozenset({-1, 0xFFFFFFFF, 4294967295})


@dataclass(frozen=True, slots=True)
class ServingParams:
    """What the server reports it is actually doing. All fields optional."""

    n_ctx: int | None = None
    model: str | None = None
    quant: str | None = None
    seed: int | None = None
    total_slots: int | None = None
    raw: dict[str, Any] = field(default_factory=dict, repr=False)


def is_healthy(base_url: str, *, timeout: float = 2.0, path: str | None = None) -> bool:
    """One probe. ``True`` if the server answers on any known health path."""
    for candidate in (path,) if path else HEALTH_PATHS:
        try:
            request_json(f"{base_url.rstrip('/')}{candidate}", timeout=timeout)
            return True
        except ServerError:
            continue
    return False


def wait_for_health(
    base_url: str,
    *,
    timeout: float = 120.0,
    is_alive: Callable[[], bool] | None = None,
    path: str | None = None,
    initial_delay: float = 0.25,
    max_delay: float = 2.0,
) -> bool:
    """Poll until the server answers, the process dies, or the deadline passes.

    Returns ``False`` on death or timeout; the caller distinguishes them, because the
    two need different messages and only the caller knows the exit code.

    Backoff is multiplicative and capped: a model that takes ninety seconds to load
    should not be probed four hundred times on the way.
    """
    deadline = time.monotonic() + timeout
    delay = initial_delay

    while time.monotonic() < deadline:
        if is_alive is not None and not is_alive():
            return False
        if is_healthy(base_url, path=path):
            return True
        time.sleep(delay)
        delay = min(delay * 1.5, max_delay)

    return False


def serving_params(base_url: str, *, timeout: float = 5.0) -> ServingParams | None:
    """Read ``/props`` to learn what the server actually came up with.

    ``None`` means no server answered -- deliberately distinguished from "answered, but
    the shape was unfamiliar", which returns a mostly-empty ``ServingParams`` instead.
    Collapsing those two turns a llama.cpp version bump into a silent loss of provenance.
    """
    try:
        props = request_json(f"{base_url.rstrip('/')}/props", timeout=timeout)
    except ServerError:
        return None

    if not isinstance(props, dict):
        return ServingParams()

    settings = props.get("default_generation_settings")
    settings = settings if isinstance(settings, dict) else {}

    seed = settings.get("seed")
    if isinstance(seed, int) and seed in _SENTINEL_SEEDS:
        seed = None

    model = props.get("model_path") or settings.get("model") or props.get("model")
    return ServingParams(
        n_ctx=settings.get("n_ctx") or props.get("n_ctx"),
        model=model,
        quant=quant_from_model_path(model) if isinstance(model, str) else None,
        seed=seed if isinstance(seed, int) else None,
        total_slots=props.get("total_slots"),
        raw=props,
    )


def reported_models(base_url: str, *, timeout: float = 5.0) -> list[str]:
    """Model ids the server admits to serving, via ``/v1/models``.

    Used to tell "a server is on this port" from "the server on this port is serving the
    model I want" -- the check that makes adopting an already-running server safe.
    """
    try:
        payload = request_json(f"{base_url.rstrip('/')}/v1/models", timeout=timeout)
    except ServerError:
        return []
    if not isinstance(payload, dict):
        return []
    return [
        entry["id"]
        for entry in payload.get("data", [])
        if isinstance(entry, dict) and isinstance(entry.get("id"), str)
    ]


def quant_from_model_path(path: str) -> str | None:
    """Pull ``Q4_K_M`` / ``Q8_0`` / ``BF16`` out of a GGUF filename.

    The server does not report quantisation as a field, and it is the single most
    important thing to record when comparing two benchmark runs.
    """
    stem = path.rsplit("/", 1)[-1]
    for part in reversed(stem.replace(".gguf", "").split("-")):
        upper = part.upper()
        if upper.startswith("Q") and any(c.isdigit() for c in upper):
            return upper
        if upper in ("BF16", "F16", "F32", "FP16", "FP32"):
            return upper
    return None
