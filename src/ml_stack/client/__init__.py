"""Talk to a local model server over HTTP. Standard library only."""

from __future__ import annotations

from ml_stack.client.chat import (
    Client,
    GrammarBudgetError,
    GrammarUnsupportedError,
    Reply,
    strip_thinking,
)
from ml_stack.client.embed import EmbeddingError, cosine, embed, rank_pairs, top_k
from ml_stack.client.health import (
    HEALTH_PATHS,
    ServingParams,
    is_healthy,
    quant_from_model_path,
    reported_models,
    serving_params,
    wait_for_health,
)
from ml_stack.client.http import ServerError, ServerUnreachable, request_json, request_stream
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
    "request_stream",
    "serving_params",
    "set_token_counter",
    "strip_thinking",
    "rank_pairs",
    "top_k",
    "wait_for_health",
]
