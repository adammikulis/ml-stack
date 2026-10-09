"""The wiring limit a model and its caches need on unified memory: what is counted, what is
proposed, and what is left for the rest of the machine."""

from __future__ import annotations

import math
import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from poolhouse import hub
from poolhouse.serve import estimate as est_mod, mtp as mtp_mod
from poolhouse.serve.preflight import _recurrent_layers
from poolhouse.serve.process import machine_memory

__all__ = ["CONTEXTS", "KV_TYPES", "MIN_MB", "Ask", "Hooks", "Plan", "current_mb", "max_mb",
           "plan", "reserve_bytes", "warn_below_bytes"]

GIB = 1024**3
MIB = 1024**2
KEY = "iogpu.wired_limit_mb"
SYSCTL = "/usr/sbin/sysctl"
KV_TYPES = ("q8_0", "f16", "q4_0")
CONTEXTS = (8192, 16384, 32768, 65536, 131072, 196608, 262144, 393216, 524288, 1048576)
MIN_MB = 4096
RESERVE_FLOOR = 8 * GIB
"""The least the rest of the machine is left; a plan never proposes a limit under it."""
WARN_BELOW = 12 * GIB
"""A limit that leaves less than this is proposed with a swap warning."""
MARGIN_MIN = 2 * GIB
MARGIN_SHARE = 0.03
HEAD_COMPUTE = 32 * MIB
CHECKPOINTS = 32
NGRAM_MOD = 16 * MIB
CACHE_RAM_MB = 8192

Runner = Callable[[Sequence[str], bool], tuple[int, str, str]]


@dataclass(frozen=True, slots=True)
class Ask:
    """What to size: the context, the cache type, whether the MTP head is shared, and the
    prompt cache cap."""

    context: int = 131072
    kv: str = "q8_0"
    mtp: bool = True
    cache_ram_mb: int = CACHE_RAM_MB


@dataclass(frozen=True, slots=True)
class Hooks:
    """What a caller may replace: the machine's facts and the privileged call. Everything
    left as None is read from this machine."""

    total: int | None = None
    current: int | None = None
    others: int | None = None
    env: Mapping[str, str] | None = None
    runner: Runner | None = None
    system: str | None = None
    terminal: tuple[bool, bool] | None = None
    daemon: Path | None = None
    read: Callable[[], int] | None = None
    contexts: Sequence[int] = CONTEXTS

    def total_bytes(self) -> int:
        return hub.total_memory() if self.total is None else self.total

    def live_mb(self) -> int:
        if self.read is not None:
            return self.read()
        return current_mb() if self.current is None else self.current


def reserve_bytes(total: int) -> int:
    """What the rest of the machine keeps at the least: 8 GiB."""
    return min(RESERVE_FLOOR, max(total, 0))


def warn_below_bytes(total: int) -> int:
    """Leaving the rest of the machine less than this is allowed and warned about."""
    return min(WARN_BELOW, max(total, 0))


def max_mb(total: int) -> int:
    """The highest limit a person may set: installed memory less the 8 GiB floor."""
    return max(0, (total - reserve_bytes(total)) // MIB)


def current_mb() -> int:
    """The raw `iogpu.wired_limit_mb` (0 means the default share), or 0 when unreadable."""
    try:
        got = subprocess.run([SYSCTL, "-n", KEY], capture_output=True, text=True, timeout=5)
        said = got.stdout.strip()
        return int(said) if got.returncode == 0 and said.isdigit() else 0
    except (OSError, subprocess.SubprocessError):
        return 0


@dataclass(frozen=True, slots=True)
class Row:
    """One context length: what it needs, what is left, and whether it fits."""

    context: int
    need_bytes: int
    limit_mb: int
    left_bytes: int
    fits: bool
    enough_now: bool


@dataclass(frozen=True, slots=True)
class Grower:
    """Memory that grows as the server is used: its cap, where it lives, and the bytes counted
    at that cap (0 when there is no cap)."""

    label: str
    bytes: int
    cap: str
    where: str


def _growers(est: Any, cache_ram_mb: int) -> tuple[Grower, ...]:
    checkpoints = int(est.state_bytes) * CHECKPOINTS
    cache = (Grower("prompt cache (--cache-ram)", cache_ram_mb * MIB,
                    f"capped at {cache_ram_mb} MiB", "host RAM") if cache_ram_mb >= 0 else
             Grower("prompt cache (--cache-ram -1)", 0, "no cap", "host RAM"))
    return (cache,
            Grower(f"context checkpoints (up to {CHECKPOINTS} per slot, about)", checkpoints,
                   f"capped at {CHECKPOINTS} per slot (--ctx-checkpoints)", "host RAM"),
            Grower("n-gram table, ngram-mod (only with --spec-type ngram-mod)", NGRAM_MOD,
                   "fixed size, 4M entries", "host RAM"),
            Grower("n-gram table, ngram-cache dynamic file (only with --lookup-cache-dynamic)",
                   0, "no cap: llama.cpp has no size option for it", "host RAM"))


@dataclass(frozen=True, slots=True)
class Plan:
    """The limit a model at one context and cache type needs, against this machine."""

    model: str
    context: int
    kv: str
    mtp: bool
    total: int
    current_mb: int
    default_mb: int
    held_others: int
    counted: tuple[tuple[str, int], ...]
    grows: tuple[Grower, ...]
    notes: tuple[str, ...]
    need_bytes: int
    margin_bytes: int
    needed_mb: int
    max_mb: int
    fits: bool
    enough_now: bool
    proposed_mb: int
    left_bytes: int
    warn: str
    largest_context: int
    table: tuple[Row, ...] = field(default_factory=tuple)

    def as_dict(self) -> dict[str, Any]:
        """JSON-ready."""
        out = {name: getattr(self, name) for name in self.__slots__ if name != "table"}
        out["counted"] = [{"label": k, "bytes": v} for k, v in self.counted]
        out["grows"] = [{name: getattr(g, name) for name in g.__slots__} for g in self.grows]
        out["notes"] = list(self.notes)
        out["table"] = [{name: getattr(r, name) for name in r.__slots__} for r in self.table]
        return out


def _mtp_part(path: Path, found: Mapping[str, object], est: Any, mtp: bool
              ) -> tuple[list[tuple[str, int]], list[str]]:
    """What the MTP head adds on top of the main model, and a line saying what was counted."""
    if not mtp:
        return [], ["MTP head: not counted (off)"]
    arch, n_layer = est_mod._layers(found)
    nextn = int(found.get(f"{arch}.nextn_predict_layers") or 0)  # type: ignore[call-overload]
    recurrent = (sum(_recurrent_layers(lambda s: found.get(f"{arch}.{s}"), n_layer))
                 if n_layer else 0)
    attention = max(n_layer - recurrent, 1)
    depth = mtp_mod.DEPTH.get(arch, 0)
    shape = f"draft depth {depth}" if depth else "the server's default draft depth"
    if mtp_mod.embeds_head(path):
        layers = nextn or 1
        kind, size = "embedded in the weights file (blk.N.nextn tensors), weights shared, 0 extra", 0
    else:
        heads = [h for h in hub.heads_for(path) if h.spec_type == mtp_mod.KIND and not h.build]
        if not heads:
            return [], ["MTP head: none ships beside the weights; nothing counted"]
        head = sorted(heads, key=lambda h: ("q8_0" not in h.path.lower(), h.bytes))[0]
        layers, size = 1, int(head.bytes)
        kind = f"separate file {Path(head.path).name}, its own weights counted"
    kv_extra = int(est.kv_cache_bytes / attention * layers)
    parts = [("MTP head weights", size), ("MTP head KV cache", kv_extra),
             ("MTP head compute buffer", HEAD_COMPUTE)]
    line = (f"MTP head: {kind}; shares the main model's weights and cache, so only its "
            f"{layers} extra layer(s) of KV cache and a compute buffer are added ({shape})")
    return [p for p in parts if p[1]], [line]


def _limit_for(need: int) -> int:
    margin = max(MARGIN_MIN, int(need * MARGIN_SHARE))
    return math.ceil((need + margin) / MIB / 256) * 256


def _largest(row: Callable[[int], Row], cap: int) -> int:
    low, high, best = 0, 4096, 0
    while low <= high:
        mid = (low + high) // 2
        ctx = mid * 256
        if ctx and ctx <= cap and row(ctx).fits:
            best, low = ctx, mid + 1
        else:
            high = mid - 1
    return best


def plan(model: str, ask: Ask | None = None, hooks: Hooks | None = None) -> Plan:
    """The wiring limit ``model`` needs for ``ask``, MTP head shared.

    Raises ``FileNotFoundError`` when the model is not installed and ``ValueError`` on a
    cache type or context that is not a known one.
    """
    ask, hooks = ask or Ask(), hooks or Hooks()
    if ask.kv not in KV_TYPES:
        raise ValueError(f"cache type {ask.kv!r} is not one of {', '.join(KV_TYPES)}")
    if isinstance(ask.context, bool) or not isinstance(ask.context, int) or ask.context < 256:
        raise ValueError("context must be a whole number of tokens, at least 256")
    total, now = hooks.total_bytes(), hooks.live_mb()
    held = (int((machine_memory() or {}).get("others") or 0) if hooks.others is None
            else hooks.others)
    default_mb = int(total * 0.75) // MIB
    effective = now or default_mb
    path = est_mod._locate(model)
    found, weights = est_mod.read_meta(path), est_mod._bytes_of(path)
    setup = est_mod.Setup(context=ask.context, kv_cache_type=ask.kv)

    def need_at(ctx: int) -> tuple[Any, list[tuple[str, int]], list[str]]:
        est = est_mod.estimate_meta(found, weights, setup, context=ctx)
        extra, said = _mtp_part(path, found, est, ask.mtp)
        return est, extra, said

    est, extra, said = need_at(ask.context)
    counted = [(k, int(v)) for k, v in est.breakdown.items() if v] + extra
    need = sum(v for _, v in counted)
    margin = max(MARGIN_MIN, int(need * MARGIN_SHARE))
    limit_cap = max_mb(total)
    needed_mb = _limit_for(need)

    def row(ctx: int) -> Row:
        e, x, _ = need_at(ctx)
        n = e.total_bytes + sum(v for _, v in x)
        mb = _limit_for(n)
        return Row(ctx, n, mb, max(total - mb * MIB, 0), mb <= limit_cap, mb <= effective)

    arch = est_mod._layers(found)[0]
    trained = int(found.get(f"{arch}.context_length") or 0)  # type: ignore[call-overload]
    shown = sorted({c for c in hooks.contexts if not trained or c <= trained} | {ask.context})
    fits = needed_mb <= limit_cap
    left = max(total - needed_mb * MIB, 0) if fits else 0
    warn = ""
    if fits and left < warn_below_bytes(total):
        warn = (f"this leaves {left / GIB:.1f} GiB for the rest of the machine, under "
                f"{WARN_BELOW // GIB} GiB; it may swap")
    return Plan(model=Path(path).name, context=ask.context, kv=ask.kv, mtp=ask.mtp, total=total,
                current_mb=now, default_mb=default_mb, held_others=held,
                counted=tuple(counted), grows=_growers(est, ask.cache_ram_mb),
                notes=tuple(list(est.notes) + said), need_bytes=need, margin_bytes=margin,
                needed_mb=needed_mb, max_mb=limit_cap, fits=fits,
                enough_now=needed_mb <= effective, proposed_mb=needed_mb if fits else 0,
                left_bytes=left, warn=warn, largest_context=_largest(row, trained or 1_048_576),
                table=tuple(row(c) for c in shown))
