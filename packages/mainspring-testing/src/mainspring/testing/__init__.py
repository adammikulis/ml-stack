"""Cross-backend parity checking, plus the pytest markers that go with it.

Lab tier. Needs MLX and PyTorch to actually run a comparison; the markers let a suite skip
gracefully on a platform where one of them has no wheel.

    from mainspring.testing import needs_both, run_pair

    @needs_both
    def test_my_layer_matches():
        report = run_pair(build_torch, build_mlx, fwd_torch, fwd_mlx, (6, 8))
        assert report.checked_parameters == 4
"""

from __future__ import annotations

from mainspring.testing.markers import (
    HAVE_MLX,
    HAVE_TORCH,
    needs_both,
    needs_mlx,
    needs_torch,
)
from mainspring.testing.parity import (
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
    "ParityError",
    "ParityReport",
    "ZERO_GRAD",
    "assert_forward_parity",
    "assert_grad_parity",
    "copy_torch_weights_to_mlx",
    "inputs",
    "mlx_grad_norms",
    "needs_both",
    "needs_mlx",
    "needs_torch",
    "run_pair",
    "torch_grad_norms",
]
