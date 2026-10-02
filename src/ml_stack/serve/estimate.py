"""What serving a model will cost in memory, from its GGUF header and file sizes.

``estimate`` answers in bytes for one choice of context, slots, cache types and offload;
``verdict`` rates an estimate against a machine as green, yellow or red; ``max_context``
finds the longest context that stays at or under a verdict.

Weights and the KV cache are read off the header and match what llama.cpp allocates (see
`tests/fixtures/estimate_logs`). The compute buffer is an empirical fit, a few percent of
the total.

A q8_0 cache stores 34 bytes per 32 values (1.0625 bytes), about half of f16, and is
near-lossless for chat and tool use; q4_0 saves more and loses quality. llama.cpp allows a
quantised V cache only with flash attention, so without it V stays f16.
"""

from __future__ import annotations

import functools
import math
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path

from ml_stack import hub
from ml_stack.hub import header
from ml_stack.hub.probe import MachineMemory
from ml_stack.serve.preflight import _kv_estimate_bytes, _recurrent_layers

__all__ = ["DEFAULT_KV", "Estimate", "Setup", "estimate", "estimate_meta", "max_context", "verdict"]

DEFAULT_KV = "q8_0"
"""The KV cache type assumed unless one is asked for."""

MIB = 1024**2
COMPUTE_PER_TOKEN = 4800
COMPUTE_BASE = (30 * MIB, 20 * MIB, 32 * MIB, 160 * MIB)
"""Compute-buffer fit at ``-ub 512`` with flash attention: bytes per context token, and a base
of ``30 MiB + 20 MiB`` per GiB of weights, held between 32 and 160 MiB."""
MMPROJ_FACTOR = 1.5
"""A projector's worst-case memory over its file size (784 MiB file, 1165 MiB estimated)."""

GREEN_AT = 0.70
"""A placement is green while it uses at most this share of the memory it goes in."""

HIGH_ATTENTION_HEADS = (64, 80, 96, 112, 128, 192, 256, 512)


def split_kv(kind: str) -> tuple[str, str]:
    """``(K, V)`` from ``q8_0`` or ``q8_0/f16``."""
    k, _, v = str(kind or DEFAULT_KV).strip().partition("/")
    return k.strip() or DEFAULT_KV, v.strip() or k.strip() or DEFAULT_KV


@dataclass(frozen=True, slots=True)
class Setup:
    """One way of serving a model.

    ``context`` is per slot and ``parallel`` the number of slots, so the server is asked for
    their product. ``kv_cache_type`` is ``q8_0`` or ``K/V`` when the halves differ.
    ``flash_attn`` of ``None`` lets the header decide. ``mmproj_bytes`` and ``draft_bytes``
    are for a caller that knows the sizes; `estimate` fills them from ``mmproj`` and
    ``draft`` files.
    """

    context: int = 4096
    parallel: int = 1
    n_gpu_layers: int | str = "auto"
    kv_cache_type: str = DEFAULT_KV
    flash_attn: bool | None = None
    batch: int = 512
    mmproj: str | Path | None = None
    draft: str | Path | None = None
    mmproj_bytes: int = 0
    draft_bytes: int = 0


@dataclass(frozen=True, slots=True)
class Estimate:
    """The memory one way of serving a model needs, in bytes."""

    weights_bytes: int
    kv_cache_bytes: int
    compute_buffer_bytes: int
    mmproj_bytes: int
    draft_bytes: int
    state_bytes: int
    total_bytes: int
    gpu_bytes: int
    cpu_bytes: int
    context: int
    parallel: int
    kv_cache_type: str
    flash_attn: bool
    n_gpu_layers: int | str
    trained_context: int = 0
    breakdown: Mapping[str, int] = field(default_factory=dict)
    confidence: str = "approx"
    notes: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, object]:
        """JSON-ready; ``breakdown`` is a list of ``{name, bytes}`` in drawing order."""
        row = {f: getattr(self, f) for f in self.__slots__ if f != "breakdown"}
        row["breakdown"] = [{"name": k, "bytes": v} for k, v in self.breakdown.items()]
        row["notes"] = list(self.notes)
        return row


def _layers(found: Mapping[str, object]) -> tuple[str, int]:
    arch = str(found.get("general.architecture") or "")
    return arch, int(found.get(f"{arch}.block_count") or 0)  # type: ignore[call-overload]


def recurrent_state_bytes(found: Mapping[str, object]) -> int:
    """Bytes one sequence keeps in its recurrent layers (state-space and delta-rule layers)."""
    arch, n_layer = _layers(found)
    if not n_layer:
        return 0

    def key(suffix: str) -> object:
        return found.get(f"{arch}.{suffix}")

    steps = sum(_recurrent_layers(key, n_layer))
    if not steps:
        return 0
    conv, inner = int(key("ssm.conv_kernel") or 0), int(key("ssm.inner_size") or 0)  # type: ignore[call-overload]
    state, groups = int(key("ssm.state_size") or 0), int(key("ssm.group_count") or 0)  # type: ignore[call-overload]
    if not (conv and inner and state):
        return 0
    per_layer = (max(conv - 1, 0) * (inner + 2 * max(groups, 1) * state) + state * inner) * 4
    return steps * per_layer


def _heads(found: Mapping[str, object]) -> int:
    arch, _ = _layers(found)
    return int(found.get(f"{arch}.attention.head_count") or 0)  # type: ignore[call-overload]


def flash_attention_likely(found: Mapping[str, object]) -> bool:
    """Whether the header's attention head sizes are ones llama.cpp's flash kernels cover."""
    arch, _ = _layers(found)
    dim = found.get(f"{arch}.attention.key_length")
    if not dim:
        embed, heads = found.get(f"{arch}.embedding_length"), _heads(found)
        dim = int(embed) // heads if embed and heads else 0  # type: ignore[call-overload]
    return int(dim or 0) in HIGH_ATTENTION_HEADS  # type: ignore[call-overload]


def compute_bytes(found: Mapping[str, object], weights: int, context: int, batch: int,
                  flash_attn: bool) -> int:
    """The compute buffer: a fit to measured loads, plus the score matrix without flash."""
    scale = max(batch, 1) / 512
    start, per_gib, low, high = COMPUTE_BASE
    base = min(max(start + per_gib * weights / 1024**3, low), high)
    flat = (base + COMPUTE_PER_TOKEN * context) * scale
    scores = 0 if flash_attn else _heads(found) * max(batch, 1) * context * 4
    return int(flat + scores)


def _offload(n_layer: int, ngl: int | str) -> float:
    if ngl == "auto" or n_layer <= 0:
        return 1.0
    return max(0.0, min(1.0, int(ngl) / (n_layer + 1)))


def _effective_kv(kind: str, flash: bool, notes: list[str]) -> tuple[str, str]:
    k, v = split_kv(kind)
    if v != "f16" and not flash:
        notes.append(f"a {v} V cache needs flash attention; V is counted as f16")
        v = "f16"
    return k, v


def _confidence(found: Mapping[str, object], kv: int, parts: Mapping[str, int],
                partial: bool, flash: bool) -> str:
    solid = kv > 0 and not partial and flash and not parts.get("Draft model")
    return "exact-from-header" if solid else "approx"


def estimate_meta(found: Mapping[str, object], weights_bytes: int,
                  setup: Setup | None = None, **changes: object) -> Estimate:
    """The estimate for a header already read, with ``setup`` changed by ``changes``."""
    use = replace(setup or Setup(), **changes)  # type: ignore[arg-type]
    notes: list[str] = []
    flash = flash_attention_likely(found) if use.flash_attn is None else use.flash_attn
    if use.flash_attn is None and not flash:
        notes.append("flash attention is uncertain for this head size; counted off")
    k, v = _effective_kv(use.kv_cache_type, flash, notes)
    slots = max(use.parallel, 1)
    total_ctx = use.context * slots
    kv = _kv_estimate_bytes(dict(found), total_ctx, k, v, use.batch)
    state = recurrent_state_bytes(found) * slots
    arch, n_layer = _layers(found)
    if not kv and not state:
        notes.append("the header lacks the keys for a cache size; the cache is not counted")
    if found.get(f"{arch}.expert_count"):
        used = int(found.get(f"{arch}.expert_used_count") or 0)  # type: ignore[call-overload]
        count = int(found[f"{arch}.expert_count"])  # type: ignore[call-overload]
        notes.append(f"mixture of experts: {used} of {count} experts run per token, "
                     "all of them stay in memory")
    compute = compute_bytes(found, int(weights_bytes), total_ctx, use.batch, flash)
    projector = int(use.mmproj_bytes * MMPROJ_FACTOR)
    parts = {"Weights": int(weights_bytes), "KV cache": kv, "Recurrent state": state,
             "Compute buffers": compute, "Vision projector": projector,
             "Draft model": int(use.draft_bytes)}
    parts = {name: size for name, size in parts.items()
             if size or name in ("Weights", "KV cache")}
    total = sum(parts.values())
    share = _offload(n_layer, use.n_gpu_layers)
    gpu = int(share * (weights_bytes + kv + state) + (compute if share else 0) + projector
              + use.draft_bytes)
    partial = share < 1.0
    if partial:
        notes.append("part of the model runs on the CPU; compute buffers are counted twice")
        total += compute
    gpu = min(gpu, total)
    return Estimate(int(weights_bytes), kv, compute, projector, int(use.draft_bytes), state,
                    total, gpu, total - gpu, use.context, slots,
                    f"{k}/{v}" if k != v else k, flash, use.n_gpu_layers,
                    int(found.get(f"{arch}.context_length") or 0),  # type: ignore[call-overload]
                    parts, _confidence(found, kv, parts, partial, flash), tuple(notes))


@functools.lru_cache(maxsize=64)
def _meta_cached(path: str, size: int, mtime_ns: int) -> tuple[tuple[str, object], ...]:
    got = header.meta(path)
    return tuple((got or {}).items())


def read_meta(path: Path | str) -> dict[str, object]:
    """The header of a GGUF, kept per file until the file changes; empty when unreadable."""
    try:
        st = Path(path).stat()
    except OSError:
        return {}
    return dict(_meta_cached(str(path), st.st_size, st.st_mtime_ns))


def _bytes_of(path: Path) -> int:
    total = 0
    for part in hub.shards_beside(path):
        try:
            total += part.stat().st_size
        except OSError:
            continue
    return total


def _locate(model: str | Path) -> Path:
    path = hub.located(model, loose=True)
    if path is None and str(model).startswith("hf:"):
        found = hub.installed_for(str(model))
        path = found.path if found else None
    if path is None:
        raise FileNotFoundError(f"{model} is not installed")
    return path


def _sizes(path: Path, use: Setup) -> Setup:
    """``use`` with the projector and draft sizes filled in from their files."""
    projector = Path(use.mmproj) if use.mmproj else None
    if projector is None and not use.mmproj_bytes:
        beside = [m for m in hub.discover(formats=("gguf",)) if m.path == path]
        projector = beside[0].mmproj if beside else None
    return replace(
        use, mmproj_bytes=use.mmproj_bytes or (_bytes_of(projector) if projector else 0),
        draft_bytes=use.draft_bytes or (_bytes_of(Path(use.draft)) if use.draft else 0))


def estimate(model: str | Path, setup: Setup | None = None, **changes: object) -> Estimate:
    """The estimate for an installed model: a path, a file name, or an ``hf:`` reference.

    ``setup`` is changed by ``changes``: ``estimate(model, context=16384)``. Raises
    ``FileNotFoundError`` when the model is not on this machine.
    """
    path = _locate(model)
    use = _sizes(path, replace(setup or Setup(), **changes))  # type: ignore[arg-type]
    return estimate_meta(read_meta(path), _bytes_of(path), use)


def reserve_default(machine: MachineMemory) -> int:
    """Memory kept back for the operating system and the application: the larger of 2 GiB
    and a tenth of installed RAM."""
    return max(2 * 1024**3, machine.total_ram // 10)


def _pools(est: Estimate, machine: MachineMemory, reserve: int) -> list[tuple[str, int, int]]:
    """``(name, need, budget)`` for each place the estimate lands on this machine."""
    ram = max(0, machine.available_ram - reserve)
    if machine.unified:
        limit = machine.gpu_limit_bytes or machine.total_ram
        return [("unified memory", est.total_bytes, min(ram, limit))]
    pools = []
    if machine.gpus and est.gpu_bytes:
        pools.append(("GPU memory", est.gpu_bytes, machine.vram_free))
        pools.append(("RAM", est.cpu_bytes, ram))
    else:
        pools.append(("RAM", est.total_bytes, ram))
    return pools


def headroom(est: Estimate, machine: MachineMemory, reserve_bytes: int | None = None
             ) -> tuple[str, int, float]:
    """``(pool, bytes free after the estimate, share used)`` for the pool that is fullest."""
    reserve = reserve_default(machine) if reserve_bytes is None else reserve_bytes
    worst = ("none", 0, 0.0)
    for name, need, budget in _pools(est, machine, reserve):
        used = need / budget if budget > 0 else math.inf
        if need and used >= worst[2]:
            worst = (name, budget - need, used)
    return worst


def verdict(est: Estimate, machine: MachineMemory, reserve_bytes: int | None = None) -> str:
    """``green`` up to 70% of the memory it goes in, ``yellow`` while it still fits,
    ``red`` when it does not, ``none`` when the machine's memory is unknown.

    The memory is the unified working-set limit on Apple silicon, a card's free VRAM for
    the GPU part, and available RAM less ``reserve_bytes`` (default `reserve_default`) for
    the CPU part. The fullest part decides.
    """
    if not (machine.available_ram or machine.vram_free):
        return "none"
    _, _, used = headroom(est, machine, reserve_bytes)
    if used <= GREEN_AT:
        return "green"
    return "yellow" if used <= 1.0 else "red"


ORDER = {"green": 0, "yellow": 1, "red": 2, "none": 3}


def max_context(model: str | Path, machine: MachineMemory, setup: Setup | None = None, *,
                max_verdict: str = "yellow", reserve_bytes: int | None = None) -> int:
    """The longest context per slot, in steps of 256, whose estimate is at or under
    ``max_verdict`` on ``machine``; 0 when none is, and never above the trained context."""
    path = _locate(model)
    use = _sizes(path, setup or Setup())
    found, weights = read_meta(path), _bytes_of(path)
    arch, _ = _layers(found)
    cap = int(found.get(f"{arch}.context_length") or 0) or 1_048_576  # type: ignore[call-overload]
    best, low, high = 0, 1, max(1, cap // 256)
    while low <= high:
        mid = (low + high) // 2
        est = estimate_meta(found, weights, use, context=mid * 256)
        if ORDER[verdict(est, machine, reserve_bytes)] <= ORDER[max_verdict]:
            best, low = mid * 256, mid + 1
        else:
            high = mid - 1
    return best
