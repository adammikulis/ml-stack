"""Cross-backend parity checking, the pytest markers that go with it, and the fakes the
suite shares -- each with the real signature (`ml_stack.testing.fakes`)."""

from __future__ import annotations

from ml_stack.testing.fakes import (
    MIRRORED,
    FakeClient,
    FakePreflight,
    FakeReport,
    FakeServe,
    ScriptedModel,
    drift,
    fake_serve,
    mirrors,
    reply_from,
)
from ml_stack.testing.markers import (
    HAVE_MLX,
    HAVE_TORCH,
    needs_both,
    needs_mlx,
    needs_torch,
)
from ml_stack.testing.parity import (
    FORWARD_ATOL,
    FORWARD_RTOL,
    GRAD_NORM_RTOL,
    ZERO_GRAD,
    ParityError,
    ParityReport,
    assert_forward_parity,
    assert_grad_parity,
    copy_torch_weights_to_mlx,
    inputs,
    mlx_grad_norms,
    run_pair,
    torch_grad_norms,
)

__all__ = [
    "FORWARD_ATOL",
    "FORWARD_RTOL",
    "GRAD_NORM_RTOL",
    "HAVE_MLX",
    "HAVE_TORCH",
    "MIRRORED",
    "FakeClient",
    "FakePreflight",
    "FakeReport",
    "FakeServe",
    "ParityError",
    "ParityReport",
    "ScriptedModel",
    "ZERO_GRAD",
    "assert_forward_parity",
    "assert_grad_parity",
    "copy_torch_weights_to_mlx",
    "drift",
    "fake_serve",
    "inputs",
    "mirrors",
    "mlx_grad_norms",
    "needs_both",
    "needs_mlx",
    "needs_torch",
    "reply_from",
    "run_pair",
    "torch_grad_norms",
]
