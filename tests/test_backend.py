"""The array seam, and a real MLX-vs-PyTorch parity check on shared math.

The parity test at the bottom is the point of the whole package: one function, written
once against ``ArrayOps``, run on both frameworks, compared forward and backward. If that
passes, the seam is doing its job.
"""

from __future__ import annotations

import numpy as np
import pytest
from ml_stack.backend import (
    ArrayBackend,
    BackendUnavailable,
    available,
    detect_backend,
    get_backend,
    set_seeds,
)
from ml_stack.backend.registry import reset
from ml_stack.testing import assert_forward_parity, needs_both, needs_mlx, needs_torch

BACKENDS = available()
each_backend = pytest.mark.parametrize("name", BACKENDS)


# --------------------------------------------------------------------------- registry


def test_at_least_one_backend_is_available():
    assert BACKENDS, "neither MLX nor PyTorch imports; the lab tier cannot be tested"


def test_detect_returns_something_importable():
    assert detect_backend() in BACKENDS


def test_env_override_is_honoured(monkeypatch):
    """So a comparison run can pin a backend without editing code."""
    if "torch" not in BACKENDS:
        pytest.skip("torch not available")
    monkeypatch.setenv("ML_STACK_BACKEND", "torch")
    assert detect_backend() == "torch"


def test_a_bogus_override_raises_rather_than_falling_back(monkeypatch):
    """Falling back would silently run on a backend the operator did not ask for."""
    monkeypatch.setenv("ML_STACK_BACKEND", "jax")
    with pytest.raises(BackendUnavailable, match="jax"):
        detect_backend()


def test_unknown_backend_name_raises():
    with pytest.raises(BackendUnavailable, match="unknown backend"):
        get_backend("tensorflow")


@each_backend
def test_the_singleton_is_actually_one_object(name):
    """Two instances means modules bind devices on different objects and factories then
    create tensors on the wrong device."""
    assert get_backend(name) is get_backend(name)


@each_backend
def test_reset_gives_a_fresh_instance(name):
    first = get_backend(name)
    reset()
    assert get_backend(name) is not first


@each_backend
def test_backend_is_fully_populated(name):
    backend = get_backend(name)
    assert isinstance(backend, ArrayBackend)
    assert backend.name == name
    for field in ("scatter_add", "segment_sum", "make_linear", "cumsum", "cumprod", "rfft_abs"):
        assert callable(getattr(backend, field)), f"{name}.{field} is not callable"


# --------------------------------------------------------------------------- ops


@each_backend
def test_every_protocol_operation_exists(name):
    """The protocol is the contract that stops the backends diverging. A missing method
    should fail here, not deep inside somebody's forward pass."""
    from ml_stack.backend.ops import ArrayOps

    ops = get_backend(name).ops
    required = [
        attr for attr in dir(ArrayOps)
        if not attr.startswith("_") and callable(getattr(ArrayOps, attr, None))
    ]
    missing = [attr for attr in required if not hasattr(ops, attr)]
    assert not missing, f"{name} is missing {missing}"


@each_backend
def test_reductions_return_arrays_not_tuples(name):
    """torch.max(x, dim=) returns (values, indices); the protocol promises just values."""
    ops = get_backend(name).ops
    x = ops.array(np.array([[1.0, 5.0], [3.0, 2.0]], dtype=np.float32))
    assert np.asarray(ops.max(x, axis=-1)).tolist() == [5.0, 3.0]
    assert np.asarray(ops.min(x, axis=-1)).tolist() == [1.0, 2.0]


@each_backend
def test_scatter_add_accumulates_duplicate_indices(name):
    """Repeated indices must sum, not overwrite. This is what makes message passing work."""
    backend = get_backend(name)
    ops = backend.ops
    target = ops.zeros((3, 2), dtype=ops.float32)
    src = ops.array(np.ones((4, 2), dtype=np.float32))
    index = ops.array(np.array([0, 0, 1, 0]), dtype=ops.int32)

    out = np.asarray(backend.scatter_add(target, index, src))
    assert out.tolist() == [[3.0, 3.0], [1.0, 1.0], [0.0, 0.0]]


@each_backend
def test_scatter_add_does_not_mutate_its_target(name):
    """An in-place write into a tensor that is also an input corrupts the autograd graph."""
    backend = get_backend(name)
    ops = backend.ops
    target = ops.zeros((2, 2), dtype=ops.float32)
    src = ops.array(np.ones((2, 2), dtype=np.float32))
    index = ops.array(np.array([0, 1]), dtype=ops.int32)

    backend.scatter_add(target, index, src)
    assert np.asarray(target).sum() == 0.0, "scatter_add mutated its target"


@each_backend
def test_segment_sum_groups_correctly(name):
    backend = get_backend(name)
    ops = backend.ops
    values = ops.array(np.array([[1.0], [2.0], [3.0], [4.0]], dtype=np.float32))
    ids = ops.array(np.array([0, 1, 0, 2]), dtype=ops.int32)

    out = np.asarray(backend.segment_sum(values, ids, 3, 0))
    assert out.reshape(-1).tolist() == [4.0, 2.0, 4.0]


@each_backend
def test_scan_and_fft_primitives_work(name):
    backend = get_backend(name)
    ops = backend.ops
    x = ops.array(np.array([1.0, 2.0, 3.0], dtype=np.float32))

    assert np.asarray(backend.cumsum(x, -1)).tolist() == [1.0, 3.0, 6.0]
    assert np.asarray(backend.cumprod(x, -1)).tolist() == [1.0, 2.0, 6.0]
    # |rfft| of a constant signal is all energy in bin 0.
    spectrum = np.asarray(backend.rfft_abs(ops.array(np.ones(8, dtype=np.float32)), -1))
    assert spectrum[0] == pytest.approx(8.0, abs=1e-4)
    assert np.allclose(spectrum[1:], 0.0, atol=1e-4)


# --------------------------------------------------------------------------- determinism


def test_set_seeds_reports_what_it_actually_seeded():
    """'I set the seed' and 'the seed reached the library that shuffled my data' are
    different claims. Only the second makes a run reproducible."""
    report = set_seeds(1234)
    assert report.seed == 1234
    assert report.python
    assert report.numpy, "numpy is installed but was not seeded"


def test_seeding_makes_python_random_reproducible():
    import random

    set_seeds(7)
    first = [random.random() for _ in range(5)]
    set_seeds(7)
    assert [random.random() for _ in range(5)] == first


@needs_torch
def test_seeding_makes_torch_reproducible():
    import torch

    set_seeds(7)
    first = torch.randn(4)
    set_seeds(7)
    assert torch.equal(first, torch.randn(4))


@needs_mlx
def test_seeding_makes_mlx_reproducible():
    import mlx.core as mx

    set_seeds(7)
    first = np.asarray(mx.random.normal((4,)))
    set_seeds(7)
    assert np.array_equal(first, np.asarray(mx.random.normal((4,))))


# --------------------------------------------------------------------------- parity


def rms_norm(backend, x, weight, eps: float = 1e-6):
    """Shared math: written once, against the protocol, with no framework name in sight."""
    ops = backend.ops
    scale = ops.rsqrt(ops.mean(x * x, axis=-1, keepdims=True) + eps)
    return x * scale * weight


def gated_mix(backend, x, w_gate):
    """Something with a nonlinearity and a matmul, to exercise more of the surface."""
    ops = backend.ops
    gate = ops.sigmoid(ops.matmul(x, w_gate))
    return ops.tanh(x) * gate + x * (1.0 - gate)


@needs_both
@pytest.mark.parametrize("fn,extra_shape", [(rms_norm, (8,)), (gated_mix, (8, 8))])
def test_shared_math_agrees_across_backends(fn, extra_shape):
    """The whole point of the seam: one implementation, two frameworks, same numbers."""
    x = np.random.default_rng(0).standard_normal((6, 8)).astype(np.float32)
    extra = np.random.default_rng(1).standard_normal(extra_shape).astype(np.float32)

    torch_backend = get_backend("torch")
    mlx_backend = get_backend("mlx")

    torch_out = fn(torch_backend, torch_backend.ops.array(x), torch_backend.ops.array(extra))
    mlx_out = fn(mlx_backend, mlx_backend.ops.array(x), mlx_backend.ops.array(extra))

    assert_forward_parity(
        torch_out.detach().cpu().numpy(), np.asarray(mlx_out), label=fn.__name__
    )


@needs_both
def test_gradients_agree_across_backends():
    """Forward agreement is the easy half. A backend adapter can be right forward and
    wrong backward -- a detach in the wrong place, a reduction over the wrong axis."""
    import mlx.core as mx
    import torch

    x = np.random.default_rng(0).standard_normal((6, 8)).astype(np.float32)
    w = np.random.default_rng(1).standard_normal((8,)).astype(np.float32)

    tb, mb = get_backend("torch"), get_backend("mlx")

    tw = torch.tensor(w, requires_grad=True)
    (rms_norm(tb, torch.tensor(x), tw) ** 2).mean().backward()

    def loss(weight):
        return mx.mean(rms_norm(mb, mx.array(x), weight) ** 2)

    mlx_grad = mx.grad(loss)(mx.array(w))
    mx.eval(mlx_grad)

    assert_forward_parity(
        tw.grad.numpy(), np.asarray(mlx_grad), atol=1e-4, rtol=1e-3, label="d/dweight"
    )


@needs_both
def test_scatter_add_agrees_across_backends():
    """Message passing is built on this, so a disagreement here corrupts every graph model."""
    index = np.array([0, 2, 0, 1, 2, 2], dtype=np.int64)
    src = np.random.default_rng(3).standard_normal((6, 4)).astype(np.float32)

    results = []
    for name in ("torch", "mlx"):
        backend = get_backend(name)
        ops = backend.ops
        out = backend.scatter_add(
            ops.zeros((3, 4), dtype=ops.float32),
            ops.array(index, dtype=ops.int32),
            ops.array(src),
        )
        results.append(np.asarray(out))

    assert_forward_parity(results[0], results[1], label="scatter_add")
