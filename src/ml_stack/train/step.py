"""One training step, on whichever framework the model belongs to."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any, Protocol

__all__ = ["Step", "TorchStep", "MLXStep", "step_for",
           "tied_names", "state_once", "load_state_once"]

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
        return {k: v.detach().cpu() for k, v in state_once(self.model).items()}

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
        load_state_once(self.model, tensors)
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


def tied_names(model: Any) -> dict[str, str]:
    """Each state-dict name that is a second name for a storage listed earlier, and the first.

    Measured on a Gemma checkpoint: ``lm_head.weight`` and ``model.embed_tokens.weight`` are
    one storage under two names, and safetensors refuses to write a shared tensor twice, so
    a state dict taken as it comes fails at the first checkpoint. An empty tensor shares
    nothing, whatever its pointer says.
    """
    seen: dict[int, str] = {}
    tied: dict[str, str] = {}
    for name, tensor in model.state_dict().items():
        if not tensor.numel():
            continue
        key = tensor.data_ptr()
        if key in seen:
            tied[name] = seen[key]
        else:
            seen[key] = name
    return tied


def state_once(model: Any) -> dict[str, Any]:
    """A torch module's state dict with every storage under one name, as safetensors wants it."""
    tied = tied_names(model)
    return {k: v for k, v in model.state_dict().items() if k not in tied}


def load_state_once(model: Any, tensors: dict[str, Any]) -> None:
    """Put a ``state_once`` dict back into the module, re-tying what it left out.

    Strict in every other way: a tensor the module has and the file lacks, or the reverse,
    is a checkpoint that does not fit, and nothing of it is kept.
    """
    from ml_stack.train.checkpoint import CheckpointError

    tied = tied_names(model)
    missing = [k for k in state_once(model) if k not in tensors]
    unexpected = [k for k in tensors if k not in model.state_dict()]
    if missing or unexpected:
        raise CheckpointError(
            f"checkpoint does not fit the model: missing {missing}, unexpected {unexpected}")
    model.load_state_dict(dict(tensors), strict=False)
    # Copying into one name of a shared Parameter filled the other; a model that ties
    # explicitly (a Hugging Face one) is asked to as well, in case a load untied it.
    if any(name not in tensors for name in tied) and callable(getattr(model, "tie_weights", None)):
        model.tie_weights()


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
