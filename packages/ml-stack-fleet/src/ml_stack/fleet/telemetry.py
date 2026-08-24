"""What a card is actually doing: temperature, clocks, power, utilisation."""

from __future__ import annotations

import json
import shutil
import subprocess
from typing import Any

__all__ = ["nvidia", "amd", "gpu_telemetry"]

TIMEOUT_S = 4.0
"""Short. This is called every time a beacon goes out, and a driver tool that has wedged"""

_NVIDIA_FIELDS = (
    "name", "temperature.gpu", "utilization.gpu", "utilization.memory",
    "clocks.sm", "clocks.mem", "power.draw", "power.limit",
    "memory.used", "memory.total", "fan.speed", "clocks_throttle_reasons.active",
)


def _run(argv: list[str]) -> str | None:
    try:
        done = subprocess.run(argv, capture_output=True, text=True, timeout=TIMEOUT_S)
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout if done.returncode == 0 and done.stdout.strip() else None


def _number(text: str) -> float | None:
    """A figure out of a vendor tool's field, or None for its many ways of saying no."""
    text = text.strip()
    if not text or text.lower() in ("n/a", "[n/a]", "unknown", "not supported",
                                    "[not supported]", "none"):
        return None
    cleaned = "".join(c for c in text if c.isdigit() or c in ".-")
    try:
        return float(cleaned)
    except ValueError:
        return None


def nvidia() -> dict[str, Any]:
    """Live figures for the first NVIDIA card, via ``nvidia-smi``."""
    if not shutil.which("nvidia-smi"):
        return {}
    out = _run(["nvidia-smi", f"--query-gpu={','.join(_NVIDIA_FIELDS)}",
                "--format=csv,noheader,nounits"])
    if not out:
        return {}
    parts = [p.strip() for p in out.strip().splitlines()[0].split(",")]
    if len(parts) < len(_NVIDIA_FIELDS):
        return {}
    (name, temp, util, mem_util, sm_clock, mem_clock, power, power_cap,
     used, total, fan, throttle) = parts[:len(_NVIDIA_FIELDS)]

    got: dict[str, Any] = {"vendor": "nvidia", "gpu": name}
    for key, raw in (("temp_c", temp), ("gpu_util_pct", util),
                     ("mem_util_pct", mem_util), ("clock_mhz", sm_clock),
                     ("mem_clock_mhz", mem_clock), ("power_w", power),
                     ("power_limit_w", power_cap), ("fan_pct", fan)):
        value = _number(raw)
        if value is not None:
            got[key] = round(value, 1)
    used_mb, total_mb = _number(used), _number(total)
    if used_mb is not None and total_mb is not None:
        got["vram_total_gb"] = round(total_mb / 1024, 2)
        got["vram_free_gb"] = round((total_mb - used_mb) / 1024, 2)
    if throttle and throttle.strip() not in ("Not Active", "N/A", ""):
        got["throttled"] = True
        got["throttle_reason"] = throttle.strip()
    return got


def amd() -> dict[str, Any]:
    """Live figures for the first AMD card, via ``rocm-smi``."""
    if not shutil.which("rocm-smi"):
        return {}

    out = _run(["rocm-smi", "--showtemp", "--showuse", "--showpower",
                "--showclocks", "--showmeminfo", "vram", "--json"])
    if out:
        try:
            payload = json.loads(out)
        except ValueError:
            payload = None
        if isinstance(payload, dict) and payload:
            card = next((v for k, v in payload.items()
                         if isinstance(v, dict) and k.lower().startswith("card")), None)
            if card is not None:
                return _amd_from_json(card)

    out = _run(["rocm-smi", "--showtemp", "--showuse", "--showpower", "--showclocks"])
    return _amd_from_text(out) if out else {"vendor": "amd"}


def _amd_from_json(card: dict[str, Any]) -> dict[str, Any]:
    got: dict[str, Any] = {"vendor": "amd"}
    wanted = {
        "temp_c": ("Temperature (Sensor edge) (C)", "Temperature (Sensor junction) (C)"),
        "gpu_util_pct": ("GPU use (%)",),
        "power_w": ("Average Graphics Package Power (W)", "Current Socket Graphics Package Power (W)"),
        "clock_mhz": ("sclk clock speed:", "sclk clock level:"),
        "mem_clock_mhz": ("mclk clock speed:",),
    }
    for key, names in wanted.items():
        for name in names:
            if name in card:
                value = _number(str(card[name]))
                if value is not None:
                    got[key] = round(value, 1)
                    break
    used = next((_number(str(v)) for k, v in card.items()
                 if "vram total used memory" in k.lower()), None)
    total = next((_number(str(v)) for k, v in card.items()
                  if "vram total memory" in k.lower()), None)
    if used is not None and total is not None and total > 0:
        got["vram_total_gb"] = round(total / 2**30, 2)
        got["vram_free_gb"] = round((total - used) / 2**30, 2)
    return got


def _amd_from_text(out: str) -> dict[str, Any]:
    got: dict[str, Any] = {"vendor": "amd"}
    for line in out.splitlines():
        if ":" not in line:
            continue
        label, _, raw = line.partition(":")
        label = label.lower()
        value = _number(raw)
        if value is None:
            continue
        if "temperature" in label and "temp_c" not in got:
            got["temp_c"] = round(value, 1)
        elif "gpu use" in label or "gpu utilization" in label:
            got["gpu_util_pct"] = round(value, 1)
        elif "power" in label and "power_w" not in got:
            got["power_w"] = round(value, 1)
        elif "sclk" in label and "clock_mhz" not in got:
            got["clock_mhz"] = round(value, 1)
        elif "mclk" in label and "mem_clock_mhz" not in got:
            got["mem_clock_mhz"] = round(value, 1)
    return got


def gpu_telemetry() -> dict[str, Any]:
    """Whichever vendor tool is present here. Empty when neither is."""
    return nvidia() or amd() or {}
