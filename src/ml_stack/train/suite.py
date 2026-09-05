"""A measurement run over several seeds, as a `bench.record.Measured`.

A `Suite` is a name, a line saying what it measures, and a function that runs one seed on
one backend and returns a flat mapping of numbers. `run` seeds each one, takes the lock,
records the commit, the host, the wall clock, the peak memory and how busy the card was
when it started, and folds the seeds into a `Spread` per metric. `said` reads one back
out; `register` adds one; `ml-stack-suite` is the command.
"""

from __future__ import annotations

import hashlib
import json
import socket
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from ml_stack.bench.keep import _commit
from ml_stack.bench.record import Measured, Spread
from ml_stack.serve.binary import CACHE_ROOT
from ml_stack.train.backend import detect_backend, set_seeds

__all__ = ["KIND", "Suite", "busy_pct", "file_name", "known", "peak_memory_bytes",
           "register", "registered", "run", "said", "suite_lock"]

#: What one seed of a suite hands back: a flat mapping of numbers.
Metrics = Mapping[str, float]
Measure = Callable[..., Metrics]

_SUITES: dict[str, "Suite"] = {}

#: What a suite run is marked with, so it is never read as an answering run.
KIND = "suite"

#: Where two runs wait for each other, so neither times the other's work.
LOCK = CACHE_ROOT / "suite.lock"

# Above this much of the card in use when a run starts, its wall clock is somebody else's
# as much as its own, and the result says so.
BUSY_PCT = 40.0


@dataclass(frozen=True)
class Suite:
    """One measurable thing, and the backends it can be measured on."""

    name: str
    about: str
    measure: Measure
    backends: tuple[str, ...] = ()
    """Backend names this suite runs on; empty means any."""


def register(name: str, about: str,
             *, backends: Sequence[str] = ()) -> Callable[[Measure], Measure]:
    """Register the decorated function as the suite ``name``."""
    def take(fn: Measure) -> Measure:
        if name in _SUITES:
            raise ValueError(f"a suite called {name!r} is already registered")
        _SUITES[name] = Suite(name, about, fn, tuple(backends))
        return fn
    return take


def known() -> dict[str, str]:
    """Every registered suite's name and what it measures."""
    return {name: suite.about for name, suite in sorted(_SUITES.items())}


def registered(name: str) -> Suite:
    """The suite called ``name``. `KeyError` names what there is instead."""
    if name not in _SUITES:
        raise KeyError(f"no suite called {name!r}; there is "
                       f"{', '.join(sorted(_SUITES)) or 'none'}")
    return _SUITES[name]


def peak_memory_bytes(backend: str) -> int:
    """The most memory the backend says it allocated, or 0 when it does not say."""
    try:
        if backend == "mlx":
            import mlx.core as mx

            return int(mx.get_peak_memory())
        if backend == "torch":
            import torch

            if torch.cuda.is_available():
                return int(torch.cuda.max_memory_allocated())
            mps = getattr(torch.backends, "mps", None)
            if mps is not None and mps.is_available():
                return int(torch.mps.driver_allocated_memory())
    except Exception:  # noqa: BLE001 - a peak nobody reports is 0, never a failed run
        return 0
    return 0


def busy_pct() -> float:
    """How much of the card is in use right now, or 0.0 when nothing here says."""
    try:
        from ml_stack.fleet.telemetry import gpu_telemetry

        found = gpu_telemetry()
    except Exception:  # noqa: BLE001 - no vendor tool is not a failed run
        return 0.0
    for key in ("gpu_util_pct", "utilization", "gpu_use_pct"):
        value = found.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return 0.0


def suite_lock(path: Path | str | None = None, *, wait: bool = True):
    """The lock two runs take turns on -- `ml_stack.lock.only_one` over `LOCK`."""
    from ml_stack.lock import only_one

    return only_one(path or LOCK, wait=wait)


def file_name(name: str, backend: str, commit: str, config: Mapping[str, Any],
              seeds: Sequence[int]) -> str:
    """What this measurement is written as: the suite, the backend, the commit, the
    readable part of its arguments and a digest of the rest."""
    stamp = json.dumps({"config": dict(config), "seeds": list(seeds)},
                       sort_keys=True, default=str)
    digest = hashlib.sha256(stamp.encode()).hexdigest()[:8]
    plain = "-".join(f"{k}_{v}" for k, v in sorted(config.items())
                     if isinstance(v, (str, int, bool)) and len(str(v)) <= 16)[:60]
    told = f"{plain}-seeds_{len(seeds)}" if plain else f"seeds_{len(seeds)}"
    return f"{name}.{backend}.{commit or 'unknown'}.{told}.{digest}.json"


def said(one: Measured) -> str:
    """One suite run as a person reads it: every metric with its spread, one screen."""
    head = f"{one.label} [{one.server.get('backend') or ''}] @ {one.sha or 'unknown'}"
    if one.dirty:
        head += " (a dirty tree -- not reproducible)"
    peak = int(one.server.get("peak_memory_bytes") or 0)
    out = [head, f"  seeds={list(one.seeds)} {one.seconds:.1f}s"
                 + (f" peak {peak / 1e9:.2f}G" if peak else "")]
    busy = float(one.server.get("busy_pct_at_start") or 0.0)
    if busy > BUSY_PCT:
        out.append(f"  the card was {busy:.0f}% busy when this started; the clock is not "
                   "this run's alone")
    for key in sorted(one.metrics):
        got = one.metrics[key]
        out.append(f"  {key:32s} {got.mean:12.5f} +/- {got.std:.5f}  (n={got.n})")
    out += [f"  FAILED {why}" for why in one.failures]
    return "\n".join(out)


@dataclass(frozen=True)
class _Under:
    """What every fold of one run shares."""

    name: str
    backend: str
    commit: str
    seeds: tuple[int, ...]
    config: Mapping[str, Any]
    busy: float = 0.0


def _folded(under: _Under, each: Sequence[Mapping[str, float]], *, seconds: float = 0.0,
            failures: Sequence[str] = (), peak: int = 0) -> Measured:
    """Mean and spread over the seeds that finished, not the seeds that were asked for."""
    metrics = {key: Spread.over([one[key] for one in each if key in one])
               for key in sorted({k for one in each for k in one})}
    host = socket.gethostname()
    return Measured(
        label=under.name, kind=KIND, at=time.strftime("%FT%T"), host=host,
        commit=under.commit, seconds=seconds, seeds=under.seeds, metrics=metrics,
        failures=tuple(failures),
        server={"backend": under.backend, "host": host, "commit": under.commit,
                "peak_memory_bytes": peak, "busy_pct_at_start": under.busy},
        extra={"config": dict(under.config)})


def run(name: str, *, backend: str = "", seeds: Sequence[int] = (0, 1, 2),
        out_dir: Path | str | None = None, where: Path | str | None = None,
        wait: bool = True, lock: Path | str | None = None, **arguments: Any) -> Measured:
    """Run one suite on one backend over ``seeds``, and write the result after each one.

    ``backend`` is a name `ml_stack.train.backend` knows; left out, the detected default.
    ``arguments`` reach the suite's own function. ``lock`` is the file runs take turns on,
    `LOCK` unless said. Returns the record; a seed that raised is recorded in ``failures``
    and left out of every mean rather than averaged away.
    """
    suite = registered(name)
    if not seeds:
        raise ValueError(f"{name} needs at least one seed: one run measures no spread")
    if not backend:
        backend = detect_backend()
    if suite.backends and backend not in suite.backends:
        raise ValueError(f"{name} runs on {', '.join(suite.backends)}, not {backend!r}")

    under = _Under(name, backend, _commit(Path(where) if where is not None else None),
                   tuple(int(s) for s in seeds), dict(arguments))
    written = None
    if out_dir is not None:
        written = Path(out_dir)
        written.mkdir(parents=True, exist_ok=True)
    into = written / file_name(name, backend, under.commit.split(" ")[0], arguments,
                               under.seeds) if written is not None else None

    each: list[dict[str, float]] = []
    failures: list[str] = []
    out = _folded(under, each)
    with suite_lock(lock, wait=wait):
        under = replace(under, busy=busy_pct())
        began = time.perf_counter()
        for seed in under.seeds:
            set_seeds(seed)
            try:
                each.append({str(k): float(v) for k, v in suite.measure(
                    backend=backend, seed=seed, **arguments).items()})
            except Exception as exc:  # noqa: BLE001 - a seed that failed is reported, not averaged
                failures.append(f"seed {seed}: {type(exc).__name__}: {exc}")
            out = _folded(under, each, seconds=time.perf_counter() - began,
                          failures=failures, peak=peak_memory_bytes(backend))
            if into is not None:
                into.write_text(
                    json.dumps(out.to_dict(), indent=2, sort_keys=True, default=str),
                    encoding="utf-8")
    return out


# ---------------------------------------------------------------------------- the command

def _value(text: str) -> Any:
    """``--set width=512`` as the number, boolean or string it reads as."""
    low = text.strip().lower()
    if low in ("true", "false"):
        return low == "true"
    for kind in (int, float):
        try:
            return kind(text)
        except ValueError:
            continue
    return text


def _imported(modules: Sequence[str]) -> None:
    """Import each module, so the suites it registers are registered."""
    import importlib

    for name in modules:
        importlib.import_module(name)


def parser():
    import argparse

    top = argparse.ArgumentParser(
        prog="ml-stack-suite",
        description="A measurement run over several seeds, written down with the commit, "
                    "the spread and what else was running.")
    top.add_argument("--import", dest="imports", action="append", default=[],
                     metavar="MODULE",
                     help="a module to import so its @register runs; repeatable")
    subs = top.add_subparsers(dest="command", required=True)
    subs.add_parser("list", help="every suite that is registered, and what it measures")
    one = subs.add_parser("run", help="run one suite over its seeds and write the result")
    one.add_argument("name")
    one.add_argument("--backend", default="", help="which backend to measure on "
                                                   "(default: the detected one)")
    one.add_argument("--seed", type=int, action="append", default=[], metavar="N",
                     help="a seed to run; repeatable (default: 0 1 2)")
    one.add_argument("--out", default="", metavar="DIR",
                     help="write the result here, after every seed")
    one.add_argument("--set", dest="settings", action="append", default=[], metavar="K=V",
                     help="an argument for the suite's own function; repeatable")
    one.add_argument("--no-wait", action="store_true",
                     help="refuse at once when another run holds the lock, rather than "
                          "waiting for it")
    return top


def main(argv: Sequence[str] | None = None) -> int:
    import sys

    args = parser().parse_args(argv)
    _imported(args.imports)
    if args.command == "list":
        found = known()
        if not found:
            print("no suite is registered; name a module with --import", file=sys.stderr)
            return 1
        width = max(len(name) for name in found)
        for name, about in found.items():
            print(f"{name:{width}}  {about}")
        return 0

    settings: dict[str, Any] = {}
    for pair in args.settings:
        if "=" not in pair:
            print(f"--set takes k=v, not {pair!r}", file=sys.stderr)
            return 2
        key, told = pair.split("=", 1)
        settings[key.strip()] = _value(told)
    try:
        out = run(args.name, backend=args.backend, seeds=tuple(args.seed) or (0, 1, 2),
                  out_dir=args.out or None, wait=not args.no_wait, **settings)
    except (KeyError, ValueError) as why:
        print(f"error: {why}", file=sys.stderr)
        return 2
    print(said(out))
    return 1 if out.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
