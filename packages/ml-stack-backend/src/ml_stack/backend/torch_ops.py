"""PyTorch's implementation of ``ArrayOps``."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

Tensor = Any


def require_torch() -> tuple[Any, Any]:
    """Import torch and torch.nn, with a clear error if they are absent."""
    try:
        import torch
        import torch.nn as nn
    except ImportError as exc:
        raise RuntimeError("PyTorch is not installed. pip install torch") from exc
    return torch, nn


class TorchArrayOps:
    """``ArrayOps`` over ``torch.Tensor``."""

    def __init__(self, torch_module: Any) -> None:
        self._torch = torch_module
        self.int32 = torch_module.int32
        self.int64 = torch_module.int64
        self.float32 = torch_module.float32
        self._device: Any = None

    # ---------------------------------------------------------------- device

    def bind_device(self, reference: Tensor) -> None:
        self._device = getattr(reference, "device", None)

    def current_device(self) -> Any:
        return self._device

    # ---------------------------------------------------------------- factories

    def array(self, value: Any, dtype: Any = None, *, device: Any = None) -> Tensor:
        """Convert to a tensor. An existing tensor keeps the device it is already on."""
        torch = self._torch
        if isinstance(value, torch.Tensor):
            tensor, target = value, device
        else:
            target = device if device is not None else self._device
            tensor = torch.as_tensor(value, dtype=dtype, device=target)
        if dtype is not None and tensor.dtype != dtype:
            tensor = tensor.to(dtype=dtype)
        if target is not None and tensor.device != target:
            tensor = tensor.to(device=target)
        return tensor

    def zeros(self, shape: tuple[int, ...], dtype: Any = None, *, device: Any = None) -> Tensor:
        return self._torch.zeros(shape, dtype=dtype, device=device or self._device)

    def ones(self, shape: tuple[int, ...], dtype: Any = None, *, device: Any = None) -> Tensor:
        return self._torch.ones(shape, dtype=dtype, device=device or self._device)

    def zeros_like(self, ref: Tensor) -> Tensor:
        return self._torch.zeros_like(ref)

    def ones_like(self, ref: Tensor) -> Tensor:
        return self._torch.ones_like(ref)

    def arange(self, *args: Any, dtype: Any = None) -> Tensor:
        return self._torch.arange(*args, dtype=dtype, device=self._device)

    def eye(self, n: int) -> Tensor:
        return self._torch.eye(n, device=self._device)

    # ---------------------------------------------------------------- shape

    def reshape(self, x: Tensor, shape: tuple[int, ...]) -> Tensor:
        return x.reshape(shape)

    def stack(self, values: list[Tensor], axis: int = 0) -> Tensor:
        return self._torch.stack(list(values), dim=axis)

    def concatenate(self, values: list[Tensor], axis: int = -1) -> Tensor:
        return self._torch.cat(list(values), dim=axis)

    def tile(self, value: Tensor, reps: tuple[int, ...]) -> Tensor:
        return self._torch.tile(value, reps)

    def broadcast_to(self, x: Tensor, shape: tuple[int, ...]) -> Tensor:
        return self._torch.broadcast_to(x, shape)

    def swapaxes(self, x: Tensor, axis1: int, axis2: int) -> Tensor:
        return self._torch.swapaxes(x, axis1, axis2)

    # ---------------------------------------------------------------- indexing

    def take(self, x: Tensor, index: Tensor, axis: int = 0) -> Tensor:
        torch = self._torch
        idx = index
        if idx.ndim > 1:
            idx = idx.reshape(idx.shape[0])
        return torch.index_select(x, axis, idx.to(dtype=torch.int64, device=x.device))

    def argsort(self, x: Tensor, axis: int = -1) -> Tensor:
        return self._torch.argsort(x, dim=axis)

    def argmin(self, x: Tensor, axis: int = -1) -> Tensor:
        return self._torch.argmin(x, dim=axis)

    def take_along_axis(self, x: Tensor, indices: Tensor, axis: int = -1) -> Tensor:
        return self._torch.take_along_dim(x, indices.to(dtype=self._torch.int64), dim=axis)

    def where(self, condition: Tensor, a: Tensor, b: Tensor) -> Tensor:
        return self._torch.where(condition, a, b)

    def stop_gradient(self, value: Tensor) -> Tensor:
        return value.detach()

    # ---------------------------------------------------------------- elementwise

    def sigmoid(self, value: Tensor) -> Tensor:
        return self._torch.sigmoid(value)

    def tanh(self, value: Tensor) -> Tensor:
        return self._torch.tanh(value)

    def sin(self, value: Tensor) -> Tensor:
        return self._torch.sin(value)

    def cos(self, value: Tensor) -> Tensor:
        return self._torch.cos(value)

    def sqrt(self, value: Tensor) -> Tensor:
        return self._torch.sqrt(value)

    def rsqrt(self, value: Tensor) -> Tensor:
        return self._torch.rsqrt(value)

    def erf(self, value: Tensor) -> Tensor:
        return self._torch.erf(value)

    def exp(self, value: Tensor) -> Tensor:
        return self._torch.exp(value)

    def log(self, value: Tensor) -> Tensor:
        return self._torch.log(value)

    def log1p(self, value: Tensor) -> Tensor:
        return self._torch.log1p(value)

    def reciprocal(self, value: Tensor) -> Tensor:
        return self._torch.reciprocal(value)

    def abs(self, value: Tensor) -> Tensor:
        return self._torch.abs(value)

    def floor(self, value: Tensor) -> Tensor:
        return self._torch.floor(value)

    def round(self, value: Tensor) -> Tensor:
        return self._torch.round(value)

    def clip(self, value: Tensor, min_value: float, max_value: float) -> Tensor:
        return self._torch.clamp(value, min=min_value, max=max_value)

    def maximum(self, a: Tensor, b: Tensor) -> Tensor:
        return self._torch.maximum(a, b)

    def minimum(self, a: Tensor, b: Tensor) -> Tensor:
        return self._torch.minimum(a, b)

    def nan_to_num(self, value: Tensor) -> Tensor:
        return self._torch.nan_to_num(value)

    # ---------------------------------------------------------------- linalg

    def matmul(self, a: Tensor, b: Tensor) -> Tensor:
        return self._torch.matmul(a, b)

    def softmax(self, value: Tensor, axis: int = -1) -> Tensor:
        return self._torch.softmax(value, dim=axis)

    def einsum(self, spec: str, *operands: Tensor) -> Tensor:
        return self._torch.einsum(spec, *operands)

    def repeat(self, x: Tensor, repeats: int, axis: int) -> Tensor:
        return self._torch.repeat_interleave(x, repeats, dim=axis)

    # ---------------------------------------------------------------- reductions

    def sum(self, value: Tensor, axis: int | None = None, keepdims: bool = False) -> Tensor:
        if axis is None:
            return self._torch.sum(value)
        return self._torch.sum(value, dim=axis, keepdim=keepdims)

    def mean(self, value: Tensor, axis: int | None = None, keepdims: bool = False) -> Tensor:
        if axis is None:
            return self._torch.mean(value)
        return self._torch.mean(value, dim=axis, keepdim=keepdims)

    def max(self, value: Tensor, axis: int | None = None, keepdims: bool = False) -> Tensor:
        if axis is None:
            return self._torch.max(value)
        return self._torch.amax(value, dim=axis, keepdim=keepdims)

    def min(self, value: Tensor, axis: int | None = None, keepdims: bool = False) -> Tensor:
        if axis is None:
            return self._torch.min(value)
        return self._torch.amin(value, dim=axis, keepdim=keepdims)

    def logsumexp(self, value: Tensor, axis: int = -1) -> Tensor:
        return self._torch.logsumexp(value, dim=axis)


def _normalize_axis(axis: int, ndim: int) -> int:
    ax = int(axis)
    if ax < 0:
        ax += ndim
    if ax < 0 or ax >= ndim:
        raise ValueError(f"axis {axis} is out of bounds for a rank-{ndim} tensor")
    return ax


def _expand_index_like(torch: Any, index: Tensor, like_shape: tuple[int, ...], axis: int) -> Tensor:
    idx = index.to(torch.int64)
    shape = tuple(int(s) for s in like_shape)
    if not shape:
        raise ValueError("cannot scatter into a scalar tensor")
    if idx.ndim == len(shape) and tuple(int(s) for s in idx.shape) == shape:
        return idx
    if idx.ndim == 1 or idx.numel() == shape[axis]:
        if idx.numel() != shape[axis]:
            raise ValueError("index length must match the source dimension along the scatter axis")
        view = [1] * len(shape)
        view[axis] = shape[axis]
        return idx.reshape(view).expand(shape)
    raise ValueError("index must be 1-D, broadcastable, or the same rank as the source")


def build_scatter_add(torch: Any) -> Callable[..., Tensor]:
    """``(target, index, src, axis=0) -> target + scattered src``. Never in-place."""

    def scatter_add(target: Tensor, index: Tensor, src: Tensor, axis: int = 0) -> Tensor:
        out = target.clone()
        axis_norm = _normalize_axis(axis, out.ndim)
        idx = index.to(torch.int64)

        if axis_norm == 0 and idx.ndim == 1:
            if src.shape[0] != idx.shape[0]:
                raise ValueError("index length must match the leading dimension of src")
            return out.index_add(0, idx, src)

        return out.scatter_add(axis_norm, _expand_index_like(torch, idx, src.shape, axis_norm), src)

    return scatter_add


def build_segment_sum(torch: Any) -> Callable[..., Tensor]:
    """``(values, segment_ids, num_segments, axis=0) -> summed``."""
    scatter_add = build_scatter_add(torch)

    def segment_sum(values: Tensor, segment_ids: Tensor, num_segments: int, axis: int = 0) -> Tensor:
        axis_norm = _normalize_axis(axis, values.ndim)
        shape = [int(s) for s in values.shape]
        shape[axis_norm] = int(num_segments)
        out = torch.zeros(shape, dtype=values.dtype, device=values.device)
        idx = _expand_index_like(torch, segment_ids, values.shape, axis_norm)
        return scatter_add(out, idx, values, axis=axis_norm)

    return segment_sum
