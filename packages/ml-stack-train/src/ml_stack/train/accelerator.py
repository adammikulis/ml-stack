"""What accelerator this box has, for the fleet beacon to advertise."""

from __future__ import annotations

from typing import Any

from ml_stack.fleet.telemetry import gpu_telemetry

__all__ = ["apple_telemetry", "report", "torch_report", "vendor_of"]


def vendor_of(torch: Any) -> str:
    """``nvidia`` | ``amd`` | ``cpu``, for a torch that thinks everything is CUDA."""
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
    out["cuda"] = vendor == "nvidia" and live
    out["rocm"] = vendor == "amd" and live
    if not live:
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
        pass
    return out


def apple_telemetry() -> dict[str, Any]:
    """Temperature, clock, power and throttle state on Apple silicon."""
    out: dict[str, Any] = {}
    try:
        import darwin_perf
    except ImportError:
        return out
    try:
        temps = darwin_perf.temperatures()
        for key, name in (("temp_c", "gpu_avg"), ("cpu_temp_c", "cpu_avg")):
            value = temps.get(name)
            if isinstance(value, (int, float)):
                out[key] = round(float(value), 1)
    except Exception:                                 # noqa: BLE001
        pass
    try:
        gpu = darwin_perf.gpu_power()
        for key, name in (("power_w", "gpu_power_w"), ("clock_mhz", "gpu_freq_mhz"),
                          ("power_limit_pct", "power_limit_pct")):
            value = gpu.get(name)
            if isinstance(value, (int, float)):
                out[key] = round(float(value), 2)
        if gpu.get("throttled"):
            out["throttled"] = True
    except Exception:                                 # noqa: BLE001
        pass
    try:
        stats = darwin_perf.system_gpu_stats()
        if stats.get("model"):
            out["gpu"] = str(stats["model"])
        if isinstance(stats.get("device_utilization"), (int, float)):
            out["gpu_util_pct"] = round(float(stats["device_utilization"]), 1)
        if stats.get("gpu_core_count"):
            out["gpu_cores"] = int(stats["gpu_core_count"])
    except Exception:                                 # noqa: BLE001
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
    out.update({"vendor": "apple", "accelerator": True, "unified_memory": True})
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
        out["device"] = out["gpu"] = profile.name
        if profile.total_memory_gb:
            out["memory_gb"] = round(profile.total_memory_gb, 2)
        out["unified_memory"] = profile.unified_memory
    except Exception:                                 # noqa: BLE001
        pass
    for probe in (torch_report, mlx_report, gpu_telemetry, apple_telemetry):
        try:
            found = probe()
        except Exception:                             # noqa: BLE001
            continue
        if not found.get("accelerator"):
            found.pop("accelerator", None)
        out.update(found)
    out["accelerator"] = bool(out.get("accelerator"))
    return out
