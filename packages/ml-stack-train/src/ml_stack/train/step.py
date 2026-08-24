"""One training step, on whichever framework the model belongs to."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any, Protocol

__all__ = ["Step", "TorchStep", "MLXStep", "step_for"]

Batch = Any
Loss = Callable[[Any, Batch], Any]


class Step(Protocol):
    """How to advance one training step on a particular framework."""

    def learning_rate(self, lr: float) -> None:
        """Set the optimizer's learning rate. A plain float, outside any compiled"""

    def __call__(self, batch: Batch) -> tuple[float, bool]:
        """Forward, backward, update. Returns ``(loss, applied)``."""

    def parameters(self) -> dict[str, Any]:
        """Flat name -> tensor, for checkpointing."""

    def optimizer_state(self) -> dict[str, Any]:
        """Flat name -> tensor of optimizer state. Empty when there is none to save."""

    def restore(self, tensors: dict[str, Any], optimizer: dict[str, Any] | None) -> None:
        """Put a checkpoint's tensors back. Must restore everything or raise."""

    def eval_loss(self, batch: Batch) -> float:
        """Loss with no gradient and no update."""


class TorchStep:
    """A step on PyTorch, where gradients live on the module."""

    name = "torch"

    def __init__(self, model: Any, optimizer: Any, loss: Loss,
                 *, clip_grad_norm: float = 0.0) -> None:
        self.model = model
        self.opt = optimizer
        self.loss = loss
        self.clip = clip_grad_norm

    def learning_rate(self, lr: float) -> None:
        for group in self.opt.param_groups:
            group["lr"] = lr

    def __call__(self, batch: Batch) -> float:
        import torch

        self.model.train()
        self.opt.zero_grad(set_to_none=True)
        loss = self.loss(self.model, batch)
        value = float(loss.detach())
        if not is_finite(value):
            return value, False
        loss.backward()
        if self.clip:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.opt.step()
        return value, True

    def eval_loss(self, batch: Batch) -> float:
        import torch

        self.model.eval()
        with torch.no_grad():
            return float(self.loss(self.model, batch))

    def parameters(self) -> dict[str, Any]:
        return {k: v.detach().cpu() for k, v in self.model.state_dict().items()}

    def optimizer_state(self) -> dict[str, Any]:
        import torch

        out: dict[str, Any] = {}
        names = [n for n, _ in self.model.named_parameters()]
        params = list(self.model.parameters())
        index = {id(p): n for n, p in zip(names, params)}
        for param, state in self.opt.state.items():
            base = index.get(id(param))
            if base is None:
                continue
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    out[f"{base}.{key}"] = value.detach().cpu()
                else:
                    out[f"{base}.{key}"] = torch.tensor(value)
        return out

    def restore(self, tensors: dict[str, Any], optimizer: dict[str, Any] | None) -> None:
        self.model.load_state_dict(tensors)
        if not optimizer:
            return
        names = [n for n, _ in self.model.named_parameters()]
        params = list(self.model.parameters())
        by_name = dict(zip(names, params))
        for flat, value in optimizer.items():
            base, _, key = flat.rpartition(".")
            param = by_name.get(base)
            if param is None:
                continue
            self.opt.state.setdefault(param, {})[key] = value


class MLXStep:
    """A step on MLX, where gradients are a returned value and nothing is eager."""

    name = "mlx"

    def __init__(self, model: Any, optimizer: Any, loss: Loss,
                 *, clip_grad_norm: float = 0.0) -> None:
        import mlx.core as mx
        import mlx.nn as nn                            # noqa: F401

        self.mx = mx
        self.model = model
        self.opt = optimizer
        self.loss = loss
        self.clip = clip_grad_norm
        self._value_and_grad = mx.value_and_grad(
            lambda m, b: self.loss(m, b))

    def learning_rate(self, lr: float) -> None:
        self.opt.learning_rate = lr

    def __call__(self, batch: Batch) -> tuple[float, bool]:
        loss, grads = self._value_and_grad(self.model, batch)
        self.mx.eval(loss)
        value = float(loss)
        if not is_finite(value):
            return value, False
        if self.clip:
            import mlx.optimizers as optim
            grads, _ = optim.clip_grad_norm(grads, self.clip)
        self.opt.update(self.model, grads)
        # Without this the graph grows without bound and nothing is ever computed.
        self.mx.eval(self.model.parameters(), self.opt.state)
        return value, True

    def eval_loss(self, batch: Batch) -> float:
        loss = self.loss(self.model, batch)
        self.mx.eval(loss)
        return float(loss)

    def parameters(self) -> dict[str, Any]:
        from mlx.utils import tree_flatten

        return dict(tree_flatten(self.model.parameters()))

    def optimizer_state(self) -> dict[str, Any]:
        from mlx.utils import tree_flatten

        out = {}
        for key, value in tree_flatten(self.opt.state):
            if isinstance(value, self.mx.array):
                out[key] = value
        return out

    def restore(self, tensors: dict[str, Any], optimizer: dict[str, Any] | None) -> None:
        from mlx.utils import tree_unflatten

        self.model.update(tree_unflatten(list(tensors.items())))
        if optimizer:
            self.opt.state = tree_unflatten(list(optimizer.items()))
        self.mx.eval(self.model.parameters())


def step_for(model: Any, optimizer: Any, loss: Loss,
             *, clip_grad_norm: float = 0.0) -> Step:
    """Pick the step that matches the model, by asking the model what it is."""
    module = type(model).__module__
    if module.startswith("torch") or _is_torch_module(model):
        return TorchStep(model, optimizer, loss, clip_grad_norm=clip_grad_norm)
    if module.startswith("mlx") or _is_mlx_module(model):
        return MLXStep(model, optimizer, loss, clip_grad_norm=clip_grad_norm)
    raise TypeError(
        f"cannot tell which framework {type(model).__name__} belongs to; "
        "pass a torch.nn.Module or an mlx.nn.Module, or build the Step yourself")


def _is_torch_module(model: Any) -> bool:
    try:
        import torch
        return isinstance(model, torch.nn.Module)
    except ImportError:
        return False


def _is_mlx_module(model: Any) -> bool:
    try:
        import mlx.nn as nn
        return isinstance(model, nn.Module)
    except ImportError:
        return False


def is_finite(value: float) -> bool:
    return not (math.isnan(value) or math.isinf(value))
