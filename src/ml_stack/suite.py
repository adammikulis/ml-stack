"""A measurement run more than once, written down with everything that makes it comparable.

A `Suite` is a name, a line saying what it measures, and a function that runs one seed on
one backend and returns a flat mapping of numbers. Everything around that -- seeding, the
commit it ran on, the wall clock, the peak memory, the machine's own load at the start,
the spread across seeds and the file it lands in -- is here, so a suite holds the
measurement and nothing else.

What a result carries, and why each part is not optional:

* **The seed, and more than one of them.** A run that does not seed cannot be repeated, and
  one run reports no spread, so a difference between two of them cannot be told from noise.
  `run` takes several seeds and reports each metric's mean, its standard deviation and how
  many seeds actually produced it.
* **The commit, and whether the tree was dirty.** A number from an edited checkout is not
  reproducible; the record says so rather than implying otherwise.
* **What else was running.** A wall clock taken while something else held the card
  describes the contention as much as the work, so the load at the start is recorded
  (`ml_stack.fleet.telemetry`) and a run that started on a busy machine says so.
* **Every seed as it finishes.** A seed can run for hours, so the file is written after
  each one: a run killed in its third seed leaves the two it has already paid for.
  ``seeds`` is what was asked for and each metric's ``n`` is what arrived.

Two runs of one suite differ by their arguments and their seeds, so the file name carries
both -- otherwise scanning three configurations leaves one file, and the survivor is
whichever ran last.

    from ml_stack.suite import Suite, register, run

    @register("attention", "one forward pass, per backend")
    def attention(*, backend, seed, width=512):
        return {"seconds": timed(width), "loss": measured(width)}

    print(run("attention", backend="torch", seeds=(0, 1, 2), out_dir=here).said())
"""

from __future__ import annotations

import hashlib
import json
import statistics
import subprocess
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ml_stack.serve.binary import CACHE_ROOT

__all__ = ["Result", "Suite", "known", "peak_memory_bytes", "provenance", "register",
           "registered", "run", "suite_lock"]

#: What one seed of a suite hands back: a flat mapping of numbers.
Metrics = Mapping[str, float]
Measure = Callable[..., Metrics]

_SUITES: dict[str, "Suite"] = {}

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


@dataclass
class Result:
    """One suite, one backend, every seed: the numbers and what they were measured under."""

    suite: str
    backend: str
    commit: str
    dirty: bool
    seeds: list[int]
    #: metric -> ``{"mean", "std", "n", "values"}``
    metrics: dict[str, dict[str, Any]]
    config: dict[str, Any] = field(default_factory=dict)
    seconds: float = 0.0
    peak_memory_bytes: int = 0
    busy_pct_at_start: float = 0.0
    failures: list[str] = field(default_factory=list)

    def said(self) -> str:
        """The result as a person reads it: every metric with its spread, one screen."""
        head = f"{self.suite} [{self.backend}] @ {self.commit or 'unknown'}"
        if self.dirty:
            head += " (a dirty tree -- not reproducible)"
        out = [head, f"  seeds={self.seeds} {self.seconds:.1f}s"
                     + (f" peak {self.peak_memory_bytes / 1e9:.2f}G"
                        if self.peak_memory_bytes else "")]
        if self.busy_pct_at_start > BUSY_PCT:
            out.append(f"  the card was {self.busy_pct_at_start:.0f}% busy when this "
                       "started; the clock is not this run's alone")
        for key in sorted(self.metrics):
            m = self.metrics[key]
            out.append(f"  {key:32s} {m['mean']:12.5f} +/- {m['std']:.5f}  (n={m['n']})")
        out += [f"  FAILED {why}" for why in self.failures]
        return "\n".join(out)


def register(name: str, about: str, *, backends: Sequence[str] = ()) -> Callable[[Measure], Measure]:
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
        raise KeyError(f"no suite called {name!r}; there is {', '.join(sorted(_SUITES)) or 'none'}")
    return _SUITES[name]


def provenance(where: Path | str | None = None) -> tuple[str, bool]:
    """``(short commit, the tree was dirty)`` for the checkout at ``where``, or ``("", False)``."""
    def git(*words: str) -> str:
        return subprocess.run(["git", *(["-C", str(where)] if where else []), *words],
                              capture_output=True, text=True, timeout=15,
                              check=False).stdout.strip()
    sha = git("rev-parse", "--short", "HEAD")
    return (sha, bool(sha and git("status", "--porcelain")))


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
            if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
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
    said = f"{plain}-seeds_{len(seeds)}" if plain else f"seeds_{len(seeds)}"
    return f"{name}.{backend}.{commit or 'unknown'}.{said}.{digest}.json"


def run(name: str, *, backend: str = "", seeds: Sequence[int] = (0, 1, 2),
        out_dir: Path | str | None = None, where: Path | str | None = None,
        wait: bool = True, lock: Path | str | None = None, **arguments: Any) -> Result:
    """Run one suite on one backend over ``seeds``, and write the result after each one.

    ``backend`` is a name `ml_stack.backend` knows; left out, the detected default.
    ``arguments`` reach the suite's own function. ``lock`` is the file runs take turns on,
    `LOCK` unless said. Returns the aggregate; a seed that raised is recorded in
    ``failures`` and left out of every mean rather than averaged away.
    """
    suite = registered(name)
    if not seeds:
        raise ValueError(f"{name} needs at least one seed: one run measures no spread")
    if not backend:
        from ml_stack.backend import detect_backend

        backend = detect_backend()
    if suite.backends and backend not in suite.backends:
        raise ValueError(f"{name} runs on {', '.join(suite.backends)}, not {backend!r}")

    from ml_stack.backend import set_seeds

    commit, dirty = provenance(where)
    written = None
    if out_dir is not None:
        written = Path(out_dir)
        written.mkdir(parents=True, exist_ok=True)

    each: list[dict[str, float]] = []
    failures: list[str] = []
    result = _folded(name, backend, commit, dirty, seeds, each, arguments, 0.0, failures, 0.0)
    with suite_lock(lock, wait=wait):
        busy = busy_pct()
        began = time.perf_counter()
        for seed in seeds:
            set_seeds(seed)
            try:
                each.append({str(k): float(v) for k, v in suite.measure(
                    backend=backend, seed=seed, **arguments).items()})
            except Exception as exc:  # noqa: BLE001 - a seed that failed is reported, not averaged
                failures.append(f"seed {seed}: {type(exc).__name__}: {exc}")
            result = _folded(name, backend, commit, dirty, seeds, each, arguments,
                             time.perf_counter() - began, failures, busy,
                             peak=peak_memory_bytes(backend))
            if written is not None:
                (written / file_name(name, backend, commit, arguments, seeds)).write_text(
                    json.dumps(asdict(result), indent=2, sort_keys=True), encoding="utf-8")
    return result


def _folded(name: str, backend: str, commit: str, dirty: bool, seeds: Sequence[int],
            each: list[dict[str, float]], arguments: Mapping[str, Any], seconds: float,
            failures: list[str], busy: float, *, peak: int = 0) -> Result:
    """Mean and spread over the seeds that finished, not the seeds that were asked for."""
    metrics: dict[str, dict[str, Any]] = {}
    for key in sorted({k for one in each for k in one}):
        values = [one[key] for one in each if key in one]
        metrics[key] = {"mean": statistics.fmean(values),
                        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
                        "n": len(values), "values": values}
    return Result(suite=name, backend=backend, commit=commit, dirty=dirty,
                  seeds=[int(s) for s in seeds], metrics=metrics, config=dict(arguments),
                  seconds=seconds, peak_memory_bytes=peak, busy_pct_at_start=busy,
                  failures=list(failures))


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
    top.add_argument("--import", dest="imports", action="append", default=[], metavar="MODULE",
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
        key, said = pair.split("=", 1)
        settings[key.strip()] = _value(said)
    try:
        out = run(args.name, backend=args.backend, seeds=tuple(args.seed) or (0, 1, 2),
                  out_dir=args.out or None, wait=not args.no_wait, **settings)
    except (KeyError, ValueError) as why:
        print(f"error: {why}", file=sys.stderr)
        return 2
    print(out.said())
    return 1 if out.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
