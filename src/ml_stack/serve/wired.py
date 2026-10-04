"""The wiring limit a model and its caches need on unified memory: the plan, and the one
privileged change (`iogpu.wired_limit_mb`) that is only ever made by a person."""

from __future__ import annotations

import math
import os
import platform
import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ml_stack import home
from ml_stack.files import read_json, write_json
from ml_stack.sentinel.human import AGENT_MARKERS, HumanRequired, require_person

__all__ = ["CONTEXTS", "KV_TYPES", "MIN_MB", "Applied", "Plan", "apply", "current_mb",
           "original_mb", "plan", "reset", "reserve_bytes", "warn_below_bytes"]

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


def reserve_bytes(total: int) -> int:
    """What the rest of the machine keeps at the least: 8 GiB."""
    return min(RESERVE_FLOOR, max(total, 0))


def warn_below_bytes(total: int) -> int:
    """Leaving the rest of the machine less than this is allowed and warned about."""
    return min(WARN_BELOW, max(total, 0))


def max_mb(total: int) -> int:
    """The highest limit a person may set: installed memory less the 8 GiB floor."""
    return max(0, (total - reserve_bytes(total)) // MIB)


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
    from ml_stack import hub
    from ml_stack.serve import estimate as est_mod
    from ml_stack.serve import mtp as mtp_mod
    from ml_stack.serve.preflight import _recurrent_layers

    if not mtp:
        return [], ["MTP head: not counted (off)"]
    arch, n_layer = est_mod._layers(found)
    nextn = int(found.get(f"{arch}.nextn_predict_layers") or 0)  # type: ignore[call-overload]
    recurrent = sum(_recurrent_layers(lambda s: found.get(f"{arch}.{s}"), n_layer)) if n_layer else 0
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


def _facts(total: int | None, current: int | None, held: int | None
           ) -> tuple[int, int, int]:
    from ml_stack import hub

    total = hub.total_memory() if total is None else total
    now = current_mb() if current is None else current
    if held is None:
        from ml_stack.serve import ops

        held = int((ops.memory().held or {}).get("others") or 0)
    return total, now, held


def plan(model: str, *, context: int = 131072, kv: str = "q8_0", mtp: bool = True,
         cache_ram_mb: int = CACHE_RAM_MB, total: int | None = None,
         current: int | None = None, others: int | None = None, contexts: Sequence[int] = CONTEXTS) -> Plan:
    """The wiring limit ``model`` needs at ``context`` with a ``kv`` cache, MTP head shared.

    Raises ``FileNotFoundError`` when the model is not installed and ``ValueError`` on a
    cache type or context that is not a known one.
    """
    from ml_stack.serve import estimate as est_mod

    if kv not in KV_TYPES:
        raise ValueError(f"cache type {kv!r} is not one of {', '.join(KV_TYPES)}")
    if isinstance(context, bool) or not isinstance(context, int) or context < 256:
        raise ValueError("context must be a whole number of tokens, at least 256")
    total, now, held = _facts(total, current, others)
    path = est_mod._locate(model)
    found, weights = est_mod.read_meta(path), est_mod._bytes_of(path)
    setup = est_mod.Setup(context=context, kv_cache_type=kv)

    def need_at(ctx: int) -> tuple[Any, list[tuple[str, int]], list[str]]:
        est = est_mod.estimate_meta(found, weights, setup, context=ctx)
        extra, said = _mtp_part(path, found, est, mtp)
        return est, extra, said

    est, extra, said = need_at(context)
    counted = [(k, int(v)) for k, v in est.breakdown.items() if v] + extra
    need = sum(v for _, v in counted)
    margin = max(MARGIN_MIN, int(need * MARGIN_SHARE))
    limit_cap = max_mb(total)
    needed_mb = math.ceil((need + margin) / MIB / 256) * 256
    default_mb = int(total * 0.75) // MIB

    def row(ctx: int) -> Row:
        e, x, _ = need_at(ctx)
        n = e.total_bytes + sum(v for _, v in x)
        mb = math.ceil((n + max(MARGIN_MIN, int(n * MARGIN_SHARE))) / MIB / 256) * 256
        return Row(ctx, n, mb, max(total - mb * MIB, 0), mb <= limit_cap, mb <= now)

    arch = est_mod._layers(found)[0]
    trained = int(found.get(f"{arch}.context_length") or 0)  # type: ignore[call-overload]
    shown = sorted({c for c in contexts if not trained or c <= trained} | {context})
    table = tuple(row(c) for c in shown)
    fits = needed_mb <= limit_cap
    proposed = needed_mb if fits else 0
    left = max(total - needed_mb * MIB, 0) if fits else 0
    warn = ""
    if fits and left < warn_below_bytes(total):
        warn = (f"this leaves {left / GIB:.1f} GiB for the rest of the machine, under "
                f"{WARN_BELOW // GIB} GiB; it may swap")
    low, high, best = 0, 4096, 0
    cap = trained or 1_048_576
    while low <= high:
        mid = (low + high) // 2
        ctx = mid * 256
        if ctx and ctx <= cap and row(ctx).fits:
            best, low = ctx, mid + 1
        else:
            high = mid - 1
    return Plan(model=Path(path).name, context=context, kv=kv, mtp=mtp, total=total,
                current_mb=now, default_mb=default_mb, held_others=held,
                counted=tuple(counted), grows=_growers(est, cache_ram_mb), notes=tuple(list(est.notes) + said),
                need_bytes=need, margin_bytes=margin, needed_mb=needed_mb, max_mb=limit_cap,
                fits=fits, enough_now=needed_mb <= now, proposed_mb=proposed, left_bytes=left,
                warn=warn, largest_context=best, table=table)


# -- the privileged change -------------------------------------------------------------------
def current_mb() -> int:
    """The raw `iogpu.wired_limit_mb` (0 means the default share), or 0 when unreadable."""
    try:
        got = subprocess.run([SYSCTL, "-n", KEY], capture_output=True, text=True, timeout=5)
        said = got.stdout.strip()
        return int(said) if got.returncode == 0 and said.isdigit() else 0
    except (OSError, subprocess.SubprocessError):
        return 0


def _record() -> Path:
    return home.state("wired-limit.json")


def original_mb() -> int | None:
    """The limit this machine reported before the first change, or None when none was made."""
    held = read_json(_record(), {})
    value = held.get("original_mb") if isinstance(held, dict) else None
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _remember(before: int) -> None:
    if original_mb() is None:
        write_json(_record(), {"schema": 1, "original_mb": int(before)})


def checked(value: object, total: int, *, lowest: int = MIN_MB) -> int:
    """``value`` as a whole number of MB within ``[lowest, installed - 8 GiB]``, else
    `ValueError`. A bool, a float, a string or anything else that is not an int is refused."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("the limit must be a whole number of MB")
    if value < lowest or value > max_mb(total):
        raise ValueError(f"the limit must be between {lowest} and {max_mb(total)} MB")
    return value


@dataclass(frozen=True, slots=True)
class Applied:
    """What a change did: the command, whether it ran, and the limit before and after."""

    argv: tuple[str, ...]
    ok: bool
    message: str
    before_mb: int
    after_mb: int


def _run(argv: Sequence[str], capture: bool) -> tuple[int, str, str]:
    try:
        done = subprocess.run(list(argv), capture_output=capture, text=True, timeout=300,
                              check=False)
    except (OSError, subprocess.SubprocessError) as exc:
        return 1, "", type(exc).__name__
    return done.returncode, done.stdout or "", done.stderr or ""


def argv_for(mb: int, via: str) -> list[str]:
    """The command that sets the limit. ``mb`` is an int and is the only variable part."""
    mb = int(mb)
    if via == "sudo":
        return ["sudo", SYSCTL, "-w", f"{KEY}={mb}"]
    if via == "osascript":
        script = f'do shell script "{SYSCTL} -w {KEY}={mb}" with administrator privileges'
        return ["osascript", "-e", script]
    raise ValueError(f"unknown way to ask for administrator rights: {via!r}")


def _refuse_agents(env: Mapping[str, str] | None) -> None:
    env = os.environ if env is None else env
    marked = [name for name in AGENT_MARKERS if env.get(name)]
    if marked:
        raise HumanRequired(f"changing the wiring limit is for a person; this process was "
                            f"started by an agent ({marked[0]} is set)")


def _change(mb: int, via: str, *, env: Mapping[str, str] | None, runner: Runner | None,
            system: str | None, read: Callable[[], int]) -> Applied:
    _refuse_agents(env)
    if (system or platform.system()) != "Darwin":
        raise ValueError("the wiring limit is a macOS setting")
    before = read()
    _remember(before)
    argv = argv_for(mb, via)
    code, _, err = (runner or _run)(argv, via != "sudo")
    after = read()
    if code != 0:
        why = "cancelled" if "-128" in err else (err.strip().splitlines() or ["failed"])[-1]
        return Applied(tuple(argv), False, why[:200], before, after)
    return Applied(tuple(argv), True, "set", before, after)


def apply(mb: object, *, via: str, total: int | None = None, env: Mapping[str, str] | None = None,
          runner: Runner | None = None, system: str | None = None,
          read: Callable[[], int] = current_mb, terminal: tuple[bool, bool] | None = None
          ) -> Applied:
    """Set the limit for this boot to ``mb`` MB. ``via`` is ``sudo`` (a terminal, sudo asks)
    or ``osascript`` (macOS's own administrator dialog). Raises `HumanRequired` for a process
    an agent started, and for ``sudo`` without a terminal; `ValueError` for a bad ``mb``."""
    from ml_stack import hub

    _refuse_agents(env)
    if via == "sudo":
        require_person("raise the wiring limit", terminal, env)
    value = checked(mb, hub.total_memory() if total is None else total)
    return _change(value, via, env=env, runner=runner, system=system, read=read)


def reset(*, via: str, total: int | None = None, env: Mapping[str, str] | None = None,
          runner: Runner | None = None, system: str | None = None,
          read: Callable[[], int] = current_mb, terminal: tuple[bool, bool] | None = None
          ) -> Applied:
    """Put the limit back to what it was before the first change (0, the default share, when
    none was recorded)."""
    from ml_stack import hub

    _refuse_agents(env)
    if via == "sudo":
        require_person("reset the wiring limit", terminal, env)
    target = original_mb() or 0
    total = hub.total_memory() if total is None else total
    if target and target > total // MIB:
        raise ValueError("the recorded original limit is larger than installed memory")
    return _change(target, via, env=env, runner=runner, system=system, read=read)
