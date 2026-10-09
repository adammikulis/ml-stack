"""Talk to a local model server over HTTP. Standard library only."""

from __future__ import annotations

from poolhouse.client.chat import (
    Client,
    GrammarBudgetError,
    GrammarUnsupportedError,
    Reply,
    strip_thinking,
)
from poolhouse.client.embed import (
    EmbeddingError,
    VectorMismatch,
    cosine,
    embed,
    rank_pairs,
    top_k,
)
from poolhouse.client.families import (
    Family,
    by_name as family_by_name,
    for_model_id as family_for_model_id,
)
from poolhouse.client.health import (
    HEALTH_PATHS,
    ServingParams,
    is_healthy,
    quant_from_model_path,
    reported_models,
    serving_params,
    wait_for_health,
)
from poolhouse.client.settings import Request, Transport
from poolhouse.client.tokens import (
    CHARS_PER_TOKEN,
    estimate_tokens,
    heuristic_tokens,
    set_token_counter,
)
from poolhouse.http import ServerError, ServerUnreachable, request_json, request_stream

__all__ = [
    "CHARS_PER_TOKEN",
    "HEALTH_PATHS",
    "Client",
    "EmbeddingError",
    "Family",
    "GrammarBudgetError",
    "GrammarUnsupportedError",
    "Reply",
    "Request",
    "ServerError",
    "ServerUnreachable",
    "ServingParams",
    "Transport",
    "VectorMismatch",
    "cosine",
    "embed",
    "estimate_tokens",
    "family_by_name",
    "family_for_model_id",
    "heuristic_tokens",
    "is_healthy",
    "quant_from_model_path",
    "rank_pairs",
    "reported_models",
    "request_json",
    "request_stream",
    "serving_params",
    "set_token_counter",
    "strip_thinking",
    "top_k",
    "wait_for_health",
]
