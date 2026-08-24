"""What accelerator this box has, for the fleet beacon to advertise.

``ml-stack-fleet`` is device tier and cannot import a framework to ask whether there is a
card here. So the dependency is inverted: the daemon is given this probe with
``--report ml_stack.train.accelerator:report`` on any box where the lab tier is
installed, and a box without it advertises less rather than guessing.

**ROCm is the case that makes the naming matter.** PyTorch's HIP build masquerades as
CUDA -- ``torch.cuda.is_available()`` is True on an AMD card, ``torch.cuda.get_device_name``
returns the Radeon's name, and ``torch.version.cuda`` is set. Reporting that as
``cuda: True`` and stopping there is technically what torch says and practically a lie:
someone filtering for ``cuda`` to place a run that needs an NVIDIA-only kernel would land
it on a Radeon and find out inside the job. So the vendor is established separately,
``rocm`` and ``cuda`` are distinct keys, and ``accelerator`` is the one to filter on when
what you need is "a GPU, any GPU".

Every probe is guarded on its own. A partial answer -- "MLX is here, the memory reading
failed" -- is more useful to placement than nothing, and a probe that raises must not
stop the daemon booting: a box that fails to start is out of the fleet entirely, which is
strictly worse than one that undersells itself.
"""

from __future__ import annotations

from typing import Any

__all__ = ["report", "torch_report", "vendor_of"]


def vendor_of(torch: Any) -> str:
    """``nvidia`` | ``amd`` | ``cpu``, for a torch that thinks everything is CUDA.

    ``torch.version.hip`` is the honest signal: it is set only on a ROCm build, and it
    is set whether or not a card is actually present.
    """
    if getattr(torch.version, "hip", None):
        return "amd"
    if getattr(torch.version, "cuda", None):
        return "nvidia"
    return "cpu"


def torch_report() -> dict[str, Any]:
    """Card, vendor and memory, as far as torch will say."""
    import torch

    out: dict[str, Any] = {}
    vendor = vendor_of(torch)
    live = torch.cuda.is_available()
    # Both keys always present and always honest: an AMD box reports cuda False even
    # though torch's own API says otherwise, because the question "can I run a CUDA
    # kernel here" has the answer no.
    out["cuda"] = vendor == "nvidia" and live
    out["rocm"] = vendor == "amd" and live
    if not live:
        # A CPU-only torch knows nothing about this machine's accelerator, and must not
        # say so. Claiming vendor "cpu" here is how a Mac with a working Metal GPU ends
        # up advertised as having none -- a probe that found nothing overwriting one
        # that did.
        return out
    out["vendor"] = vendor
    out["accelerator"] = True

    out["gpu"] = torch.cuda.get_device_name(0)
    out["gpu_count"] = torch.cuda.device_count()
    try:
        free, total = torch.cuda.mem_get_info()
        out["vram_free_gb"] = round(free / 2**30, 2)
        out["vram_total_gb"] = round(total / 2**30, 2)
    except Exception:                                 # noqa: BLE001
        # mem_get_info is unavailable on some ROCm builds. Reporting no reading is
        # right; reporting zero would read as "no memory free" and park every run.
        pass
    return out


def mlx_report() -> dict[str, Any]:
    """Apple silicon, where the GPU has no separate memory pool."""
    import mlx.core as mx

    out: dict[str, Any] = {}
    try:
        device = mx.default_device()
    except Exception:                                 # noqa: BLE001
        return out
    if "gpu" not in str(device).lower():
        return out
    out.update({"vendor": "apple", "accelerator": True, "unified_memory": True,
                "gpu": str(device)})
    try:
        limit = mx.metal.device_info().get("max_recommended_working_set_size")
        if limit:
            out["vram_free_gb"] = out["vram_total_gb"] = round(limit / 2**30, 2)
    except Exception:                                 # noqa: BLE001
        pass
    return out


def report() -> dict[str, Any]:
    """Accelerator facts, as far as they can be established on this machine."""
    out: dict[str, Any] = {}
    try:
        from ml_stack.backend import available
        out["backends"] = list(available())
    except Exception:                                 # noqa: BLE001
        pass
    try:
        from ml_stack.backend import detect_device
        profile = detect_device()
        out["vendor"] = str(profile.vendor)
        out["device"] = profile.name
        if profile.total_memory_gb:
            out["memory_gb"] = round(profile.total_memory_gb, 2)
        out["unified_memory"] = profile.unified_memory
    except Exception:                                 # noqa: BLE001
        pass
    # Framework probes last and merged over the top: they know the live free-memory
    # reading, which is the number placement actually needs. A probe that found no
    # accelerator returns only its own negative flags and leaves the rest alone.
    for probe in (torch_report, mlx_report):
        try:
            found = probe()
        except Exception:                             # noqa: BLE001
            continue
        # `accelerator` is a claim any probe may make and none may retract: two
        # frameworks are installed on plenty of boxes and only one of them may see
        # the card.
        if not found.get("accelerator"):
            found.pop("accelerator", None)
        out.update(found)
    out["accelerator"] = bool(out.get("accelerator"))
    return out
