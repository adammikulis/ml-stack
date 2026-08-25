"""MLX's implementation of the pieces ``mlx.core`` does not already provide."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

Tensor = Any


def require_mlx() -> tuple[Any, Any]:
    """Import ``mlx.core`` and ``mlx.nn``, with a clear error if they are absent."""
    try:
        import mlx.core as mx
        import mlx.nn as nn
    except ImportError as exc:
        raise RuntimeError(
            "MLX is not available. It ships only for Apple silicon; on any other "
            "platform use the torch backend."
        ) from exc
    return mx, nn


def build_scatter_add(mx: Any) -> Callable[..., Tensor]:
    """``(target, index, src, axis=0) -> target + scattered src``. Never in-place."""

    def scatter_add(target: Tensor, index: Tensor, src: Tensor, axis: int = 0) -> Tensor:
        idx = index.astype(mx.int32).reshape(-1)

        if axis == 0:
            if src.shape[0] != idx.shape[0]:
                raise ValueError("index length must match the leading dimension of src")
            return target.at[idx].add(src)

        moved_target = mx.swapaxes(target, 0, axis)
        moved_src = mx.swapaxes(src, 0, axis)
        scattered = moved_target.at[idx].add(moved_src)
        return mx.swapaxes(scattered, 0, axis)

    return scatter_add


def build_segment_sum(mx: Any) -> Callable[..., Tensor]:
    """``(values, segment_ids, num_segments, axis=0) -> summed``."""
    scatter_add = build_scatter_add(mx)

    def segment_sum(values: Tensor, segment_ids: Tensor, num_segments: int, axis: int = 0) -> Tensor:
        axis_norm = axis if axis >= 0 else axis + values.ndim
        shape = list(values.shape)
        shape[axis_norm] = int(num_segments)
        out = mx.zeros(tuple(shape), dtype=values.dtype)
        return scatter_add(out, segment_ids, values, axis=axis_norm)

    return segment_sum


def build_make_linear(nn: Any) -> Callable[[int, int], Any]:
    def make_linear(d_in: int, d_out: int) -> Any:
        return nn.Linear(int(d_in), int(d_out))

    return make_linear
