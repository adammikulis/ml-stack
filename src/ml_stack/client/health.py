"""Is a model server up, and what is it actually serving?"""

from __future__ import annotations

import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from ml_stack.http import ServerError, request_json

HEALTH_PATHS = ("/health", "/v1/models", "/models", "/props")

_QUANT = re.compile(
    r"(?<![A-Za-z0-9])(MXFP4|IQ\d+_[A-Z]+(?:_[A-Z]+)?|Q\d+_K_[A-Z]+|Q\d+_K|Q\d+_\d+|BF16|FP16|FP32|F16|F32)"
    r"(?![A-Za-z0-9])",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class ServingParams:
    """What the server reports it is actually doing. All fields optional."""

    n_ctx: int | None = None
    model: str | None = None
    quant: str | None = None
    seed: int | None = None
    min_p: float | None = None
    repeat_penalty: float | None = None
    total_slots: int | None = None
    chat_template: str | None = None
    raw: dict[str, Any] = field(default_factory=dict, repr=False)


def is_healthy(base_url: str, *, timeout: float = 2.0, path: str | None = None) -> bool:
    """One probe. ``True`` if the server answers on any of `HEALTH_PATHS`."""
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
    """Poll until the server answers, the process dies, or the deadline passes."""
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
    """Read ``/props`` to learn what the server actually came up with."""
    try:
        props = request_json(f"{base_url.rstrip('/')}/props", timeout=timeout)
    except ServerError:
        return None

    if not isinstance(props, dict):
        return ServingParams()

    settings = props.get("default_generation_settings")
    settings = settings if isinstance(settings, dict) else {}
    nested = settings.get("params")
    nested = nested if isinstance(nested, dict) else {}

    def setting(name: str) -> Any:
        value = nested.get(name)
        return settings.get(name) if value is None else value

    seed = setting("seed")
    if not isinstance(seed, int) or seed < 0 or seed >= 0xFFFFFFFF:
        seed = None

    model = props.get("model_path") or settings.get("model") or props.get("model")
    return ServingParams(
        n_ctx=settings.get("n_ctx") or props.get("n_ctx"),
        model=model,
        quant=quant_from_model_path(model) if isinstance(model, str) else None,
        seed=seed,
        min_p=setting("min_p"),
        repeat_penalty=setting("repeat_penalty"),
        total_slots=props.get("total_slots") or props.get("n_parallel"),
        chat_template=(props.get("chat_template")
                       if isinstance(props.get("chat_template"), str) else None),
        raw=props,
    )


def reported_models(base_url: str, *, timeout: float = 5.0) -> list[str]:
    """Model ids the server admits to serving, via ``/v1/models``."""
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
    """The quantisation a GGUF filename names (``Q4_K_M``, ``IQ4_XS``, ``MXFP4``, ``BF16``)."""
    match = _QUANT.search(path.rsplit("/", 1)[-1])
    return match.group(1).upper() if match else None
