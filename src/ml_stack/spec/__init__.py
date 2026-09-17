"""Tree speculative decoding on MLX for hybrid DeltaNet/attention models.

A drafter proposes a tree of tokens; `TreeVerifier` scores every node in one pass of the
target; an acceptance rule keeps a path; the caches are advanced along it. `Engine` holds a
loaded target, its drafter and the session that reuses a prompt's prefix across calls.
"""

#: model_type -> the package that loads it, and the layout that verifies it as ``module:class``
LAYOUTS = {
    "qwen3_5": ("mlx_lm", "ml_stack.spec.layout:Qwen35Layout"),
    "qwen3_5_moe": ("mlx_lm", "ml_stack.spec.layout:Qwen35Layout"),
    "qwen4_exp": ("mlx_vlm", "ml_stack.spec.qwen4:Qwen4ExpLayout"),
}
