"""Talk to a local model server over HTTP. Standard library only.

Device tier. This package knows how to *talk* to a server; it does not know how to
*start* one -- that is ``ml_stack.serve``, which is host tier and may use psutil and
subprocess.

The seam between them is deliberate. ``wait_for_health`` takes an ``is_alive`` callable
rather than a ``Popen``, so the process-death watch that makes a bad model path fail in
under a second works without this package ever importing ``subprocess``.
"""

from __future__ import annotations

from ml_stack.client.chat import (
    Client,
    GrammarBudgetError,
    GrammarUnsupportedError,
    Reply,
    strip_thinking,
)
from ml_stack.client.embed import EmbeddingError, cosine, embed, top_k
from ml_stack.client.health import (
    HEALTH_PATHS,
    ServingParams,
    is_healthy,
    quant_from_model_path,
    reported_models,
    serving_params,
    wait_for_health,
)
from ml_stack.client.http import ServerError, ServerUnreachable, request_json
from ml_stack.client.tokens import estimate_tokens, heuristic_tokens, set_token_counter

__all__ = [
    "HEALTH_PATHS",
    "Client",
    "EmbeddingError",
    "GrammarBudgetError",
    "GrammarUnsupportedError",
    "Reply",
    "ServerError",
    "ServerUnreachable",
    "ServingParams",
    "cosine",
    "embed",
    "estimate_tokens",
    "heuristic_tokens",
    "is_healthy",
    "quant_from_model_path",
    "reported_models",
    "request_json",
    "serving_params",
    "set_token_counter",
    "strip_thinking",
    "top_k",
    "wait_for_health",
]
