"""The process `MlxTreeBackend` starts: an MLX model with a drafter, answering chat completions.

Run as ``python -m ml_stack.serve.mlx_tree_server MODEL`` by a lease, never by hand: the lease
is what records the process and reaps it.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import mlx.core as mx

from ml_stack.command import Group, flag, option
from ml_stack.graph.serve import Handler
from ml_stack.http import Server
from ml_stack.log import say
from ml_stack.serve.backend import DEFAULT_HOST
from ml_stack.serve.mlx_tree import PREFIX
from ml_stack.spec.accept import Rule
from ml_stack.spec.engine import DRAFTERS, Engine, EngineConfig, Reply, Request
from ml_stack.spec.sample import Sampling

__all__ = ["COMMANDS", "TreeCompleter", "timings_of"]

#: the relaxed rule measured on the 27B target as leaving task accuracy unchanged
DEFAULT_RULE = "ratio:theta=0.3"


def timings_of(reply: Reply) -> dict[str, Any]:
    """llama.cpp's ``timings`` for one reply: prompt, generation, and the draft and its passes."""
    decoded = reply.decoded
    drafting = sum(p.drafting for p in decoded.passes)
    return {"prompt_n": decoded.prompt_tokens - decoded.reused,
            "prompt_ms": 1000 * decoded.prefill_s, "cache_n": decoded.reused,
            "predicted_n": len(decoded.tokens), "predicted_ms": 1000 * decoded.seconds,
            "draft_n": sum(p.nodes - 1 for p in decoded.passes),
            "draft_n_accepted": sum(p.accepted for p in decoded.passes),
            "draft_ms": 1000 * drafting, "verify_n": len(decoded.passes),
            "verify_ms": 1000 * (decoded.seconds - drafting)}


class TreeCompleter:
    """An `Engine` as the completer `ml_stack.graph.completions` routes to."""

    def __init__(self, engine: Engine, *, name: str, context: int, rule: str) -> None:
        self.engine, self.name, self.context, self.rule = engine, name, context, rule
        self.program = {"program": "mlx-tree", "build_info": f"mlx {mx.__version__}",
                        "runtime": "mlx", "format": "safetensors"}

    def complete(self, messages: Sequence[Mapping[str, Any]], options: Mapping[str, Any],
                 on_text: Callable[[str, bool], None] | None,
                 stop: Callable[[], bool] | None) -> dict[str, Any]:
        thinking = bool(options.get("thinking", True))
        sampled = None
        if "temperature" in options:
            sampled = Sampling(float(options["temperature"]), int(options.get("top_k", 20)),
                               float(options.get("top_p", 1.0)), float(options.get("min_p", 0.0)))
        template = ({"reasoning_effort": options["reasoning_effort"]}
                    if options.get("reasoning_effort") else {})
        wanted = int(options.get("max_tokens", options.get("n_predict", -1)))
        request = Request(max_tokens=wanted if wanted > 0 else self.context,
                          sampling=sampled, thinking=thinking,
                          rule=Rule.parse(str(options.get("accept") or self.rule)),
                          seed=options.get("seed"), template=template,
                          tools=tuple(options.get("tools") or ()))
        reply = self.engine.chat(messages, request, on_text, stop)
        return {"content": reply.content, "reasoning": reply.reasoning,
                "finish": reply.decoded.finish, "prompt_tokens": reply.decoded.prompt_tokens,
                "cached_tokens": reply.decoded.reused,
                "completion_tokens": len(reply.decoded.tokens),
                "reasoning_tokens": reply.reasoning_tokens, "tool_calls": reply.tool_calls,
                "timings": timings_of(reply)}


def cmd_tree(args: argparse.Namespace) -> int:
    """Load, then answer until stopped."""
    model = str(args.model)
    engine = Engine(EngineConfig(model=model.removeprefix(PREFIX), drafter=args.drafter,
                                 drafter_model=args.drafter_model, max_nodes=args.max_nodes))
    completer = TreeCompleter(engine, name=model, context=int(args.context), rule=args.accept)
    handler = Handler.configured(name="TreeServer", completer=completer)
    httpd = Server((DEFAULT_HOST, int(args.port)), handler)
    say(f"serving {model} with the {args.drafter} drafter on "
        f"http://{DEFAULT_HOST}:{args.port}/v1/chat/completions", flush=True)
    try:
        httpd.serve_forever()
    finally:
        httpd.server_close()
    return 0


COMMANDS = Group(
    "ml_stack.serve.mlx_tree_server",
    "An MLX model served with tree speculative decoding, as a lease starts it.",
    options=(
        flag("model", help="mlx:owner/repo, or a directory of MLX weights"),
        option("port", default=8080),
        option("context", default=32768),
        flag("--drafter", choices=DRAFTERS, default="none",
             help="what proposes each tree (default: none, one token a pass)"),
        flag("--drafter-model", default="", metavar="REPO_OR_DIR",
             help="the DFlash or MTP head's weights"),
        flag("--max-nodes", type=int, default=32, metavar="N",
             help="the most tree nodes one pass verifies (default: 32)"),
        flag("--accept", default=DEFAULT_RULE, metavar="RULE",
             help=f"acceptance rule when sampling: lossless or ratio:theta=T "
                  f"(default: {DEFAULT_RULE})"),
    ),
    run=cmd_tree)

if __name__ == "__main__":
    raise SystemExit(COMMANDS.run())
