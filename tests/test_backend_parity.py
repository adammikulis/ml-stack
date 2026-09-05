"""Every declared array operation, and a whole training step, on both backends at once."""

from __future__ import annotations

import numpy as np
import pytest
from ml_stack.testing import (
    ParityError,
    assert_grad_parity,
    copy_torch_weights_to_mlx,
    inputs,
    mlx_grad_norms,
    needs_both,
    run_pair,
    torch_grad_norms,
)
from ml_stack.train.backend import get_backend
from ml_stack.train.backend.parity import ATOL, CASES, check_all, check_op, table
from ml_stack.train.backend.ops import ArrayOps
from ml_stack.train.step import step_for

STEP_ATOL = 1e-5
"""Loss and gradient-norm agreement over several optimiser steps."""


# --------------------------------------------------------------------------- coverage


def test_every_protocol_operation_has_a_case():
    """Fails when the protocol grows a method no case covers."""
    declared = {
        attr for attr in dir(ArrayOps)
        if not attr.startswith("_") and callable(getattr(ArrayOps, attr, None))
    }
    covered = {name.partition("(")[0] for name in CASES}
    assert not declared - covered, f"no parity case for {sorted(declared - covered)}"


def test_every_backend_field_has_a_case():
    covered = {name.partition("(")[0] for name in CASES}
    for field in ("scatter_add", "segment_sum", "cumsum", "cumprod", "rfft_abs"):
        assert field in covered, f"no parity case for ArrayBackend.{field}"


def test_axis_arguments_are_compared_on_more_than_one_axis():
    """Every operation taking an axis is compared on more than its default axis."""
    for op in ("stack", "concatenate", "softmax", "sum", "mean", "max", "min",
               "logsumexp", "argsort", "argmin", "take", "scatter_add", "segment_sum"):
        axes = [n for n in CASES if n.partition("(")[0] == op]
        assert len(axes) > 1, f"{op} takes an axis but is compared on only {axes}"


# --------------------------------------------------------------------------- operations


@needs_both
@pytest.mark.parametrize("name", list(CASES))
def test_operation_agrees_across_backends(name):
    result = check_op(name, get_backend("torch"), get_backend("mlx"))
    assert result.ok, f"{name}: max |Δ|={result.diff:.3e} (atol={ATOL:g}) {result.detail}"


@needs_both
def test_the_table_names_every_case_and_says_how_many_agree():
    results = check_all(get_backend("torch"), get_backend("mlx"))
    rendered = table(results)
    assert len(results) == len(CASES)
    for name in CASES:
        assert name in rendered
    assert f"{len(results)}/{len(results)} agree" in rendered


class Stub:
    """A backend that returns whatever it was handed."""

    ops = None

    def __init__(self, name: str, value) -> None:
        self.name = name
        self.value = value


def test_a_disagreeing_operation_is_reported_not_raised(monkeypatch):
    """The command prints a table rather than stopping at the first failure."""
    from ml_stack.train.backend import parity

    monkeypatch.setitem(parity.CASES, "invented", lambda b, o: b.value)
    result = parity.check_op("invented", Stub("torch", np.zeros(2)), Stub("mlx", np.ones(2)))

    assert not result.ok
    assert result.diff == pytest.approx(1.0)
    rendered = table([result])
    assert "invented" in rendered and "FAIL" in rendered


def test_an_operation_that_raises_is_a_failure_not_a_crash(monkeypatch):
    from ml_stack.train.backend import parity

    monkeypatch.setitem(parity.CASES, "explodes", lambda b, o: 1 / 0)
    result = parity.check_op("explodes", Stub("torch", None), Stub("mlx", None))

    assert not result.ok
    assert "ZeroDivisionError" in result.detail


def test_a_shape_mismatch_is_a_failure(monkeypatch):
    from ml_stack.train.backend import parity

    monkeypatch.setitem(parity.CASES, "reshaped", lambda b, o: b.value)
    result = parity.check_op("reshaped", Stub("torch", np.zeros((2, 3))),
                             Stub("mlx", np.zeros((3, 2))))

    assert not result.ok
    assert "shapes differ" in result.detail


# --------------------------------------------------------------------------- a model


def build_torch_mlp():
    import torch
    import torch.nn as nn

    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(4, 6), nn.Tanh(), nn.Linear(6, 2))


def build_mlx_mlp():
    import mlx.nn as nn

    return nn.Sequential(nn.Linear(4, 6), nn.Tanh(), nn.Linear(6, 2))


@needs_both
def test_a_module_agrees_forward_and_backward():
    """``run_pair``: one set of weights, both forwards, both backwards, compared."""
    import mlx.core as mx
    import torch

    report = run_pair(
        build_torch_mlp,
        build_mlx_mlp,
        lambda m, x: m(torch.tensor(x)),
        lambda m, x: m(mx.array(x)),
        (8, 4),
    )
    assert report.checked_parameters == 4
    assert report.max_forward_diff < STEP_ATOL
    assert set(report.grad_norms) == {"0.weight", "0.bias", "2.weight", "2.bias"}


# --------------------------------------------------------------------------- a step


def torch_loss(model, batch):
    import torch

    x, y = batch
    return ((model(torch.tensor(x)) - torch.tensor(y)) ** 2).mean()


def mlx_loss(model, batch):
    import mlx.core as mx

    x, y = batch
    return mx.mean((model(mx.array(x)) - mx.array(y)) ** 2)


def batches(count: int = 6):
    return [(inputs((8, 4), seed=i), inputs((8, 2), seed=100 + i)) for i in range(count)]


def _paired_steps(make_torch_opt, make_mlx_opt):
    """``(torch step, mlx step, the two models)`` from one set of initial weights."""
    torch_model, mlx_model = build_torch_mlp(), build_mlx_mlp()
    copy_torch_weights_to_mlx(torch_model, mlx_model)
    return (
        step_for(torch_model, make_torch_opt(torch_model), torch_loss),
        step_for(mlx_model, make_mlx_opt(), mlx_loss),
        (torch_model, mlx_model),
    )


def _grad_norms(torch_model, mlx_model, batch):
    import mlx.core as mx

    torch_model.zero_grad(set_to_none=True)
    torch_loss(torch_model, batch).backward()
    _value, grads = mx.value_and_grad(lambda m: mlx_loss(m, batch))(mlx_model)
    mx.eval(grads)
    return torch_grad_norms(torch_model), mlx_grad_norms(grads)


@needs_both
def test_sgd_steps_track_across_backends():
    """Six SGD steps through TorchStep and MLXStep from one set of weights."""
    import mlx.optimizers as optim
    import torch

    torch_step, mlx_step, models = _paired_steps(
        lambda m: torch.optim.SGD(m.parameters(), lr=0.1),
        lambda: optim.SGD(learning_rate=0.1),
    )

    for batch in batches():
        torch_value, torch_applied = torch_step(batch)
        mlx_value, mlx_applied = mlx_step(batch)
        assert torch_applied and mlx_applied
        assert torch_value == pytest.approx(mlx_value, abs=STEP_ATOL)

    first, second = _grad_norms(*models, batches()[0])
    assert assert_grad_parity(first, second, rtol=1e-4)


@needs_both
def test_adamw_steps_track_across_backends():
    """MLX's AdamW leaves bias correction off by default; torch's cannot be turned off."""
    import mlx.optimizers as optim
    import torch

    torch_step, mlx_step, models = _paired_steps(
        lambda m: torch.optim.AdamW(m.parameters(), lr=0.01),
        lambda: optim.AdamW(learning_rate=0.01, bias_correction=True),
    )

    for batch in batches():
        torch_value, _ = torch_step(batch)
        mlx_value, _ = mlx_step(batch)
        assert torch_value == pytest.approx(mlx_value, abs=STEP_ATOL)

    first, second = _grad_norms(*models, batches()[0])
    assert assert_grad_parity(first, second, rtol=1e-4)


def test_the_recipes_ask_mlx_for_the_optimizer_torch_gives():
    """Every MLX Adam the recipes build asks for bias correction, so a recipe trains the
    same on both backends."""
    import re
    from pathlib import Path

    import ml_stack.train.recipes as recipes

    call = re.compile(r"(?<![\w.])optim\.Adam[W]?\([^)]*\)", re.S)
    found = 0
    for path in sorted(Path(recipes.__file__).parent.glob("*.py")):
        for match in call.finditer(path.read_text(encoding="utf-8")):
            found += 1
            assert "bias_correction=True" in match.group(0), (
                f"{path.name}: {match.group(0)} trains differently from the torch "
                "branch beside it")
    assert found, "no MLX optimizer found in the recipes; has the scan gone stale?"


@needs_both
def test_divergent_gradients_are_caught_rather_than_averaged_away():
    first = {"w": 1.0, "b": 2.0}
    with pytest.raises(ParityError, match="'b'"):
        assert_grad_parity(first, {"w": 1.0, "b": 2.4}, rtol=1e-4)


# --------------------------------------------------------------------------- the command


@needs_both
def test_the_command_prints_a_row_per_operation_and_exits_zero():
    from ml_stack.train.run import parity

    said: list[str] = []
    code = parity(say=said.append)

    printed = "\n".join(said)
    assert code == 0, printed
    assert "operation" in printed and "max |Δ|" in printed
    for name in CASES:
        assert name in printed
    assert f"{len(CASES)}/{len(CASES)} agree" in printed


@needs_both
def test_the_command_exits_one_when_an_operation_disagrees(monkeypatch):
    from ml_stack.train.backend import parity as checks
    from ml_stack.train.run import parity

    monkeypatch.setitem(checks.CASES, "invented",
                        lambda b, o: o.array(np.zeros(2, dtype=np.float32) if b.name == "torch"
                                             else np.ones(2, dtype=np.float32)))
    said: list[str] = []

    assert parity(say=said.append) == 1
    printed = "\n".join(said)
    assert "invented" in printed and "FAIL" in printed
    assert "disagreeing: invented" in printed


def test_the_command_says_so_when_a_backend_is_missing():
    """Naming a backend that cannot be built here exits 2 and says which."""
    from ml_stack.train.run import parity

    said: list[str] = []
    assert parity(say=said.append, second="jax") == 2
    assert "jax is not usable here" in "\n".join(said)


def test_parity_is_one_of_the_words_the_command_takes_instead_of_flags():
    from ml_stack.train.run import WORDS

    assert "parity" in WORDS
