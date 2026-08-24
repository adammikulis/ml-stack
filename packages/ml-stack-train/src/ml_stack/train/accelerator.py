"""What accelerator this box has, for the fleet beacon to advertise.

``ml-stack-fleet`` is device tier and cannot import a framework to ask whether there is a
CUDA card here. So the dependency is inverted: this module is registered under the
``ml_stack.device_report`` entry-point group in this package's ``pyproject.toml``, and the
daemon loads it at runtime on any box where ``ml-stack-train`` happens to be installed.
A box without the lab tier installed simply advertises less.

Every probe is best effort and each is guarded separately. A partial answer -- "MLX is
here, CUDA could not be determined" -- is more useful to placement than nothing, and a
probe that raises must not stop the daemon from booting: a box that fails to start is out
of the fleet entirely, which is strictly worse than one that undersells itself.
"""

from __future__ import annotations

from typing import Any


def report() -> dict[str, Any]:
    """Accelerator facts, as far as they can be established on this machine."""
    out: dict[str, Any] = {}
    try:
        from ml_stack.backend import available
        out["backends"] = list(available())
    except Exception:                                 # noqa: BLE001
        pass
    try:
        import torch
        out["cuda"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            out["gpu"] = torch.cuda.get_device_name(0)
            out["vram_free_gb"] = round(free / 2**30, 2)
            out["vram_total_gb"] = round(total / 2**30, 2)
    except Exception:                                 # noqa: BLE001
        pass
    return out
