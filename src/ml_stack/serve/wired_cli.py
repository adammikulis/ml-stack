"""``ml-stack-serve memory --for MODEL``: the wiring limit a model needs, and changing it."""

from __future__ import annotations

import argparse
import platform
import sys

from ml_stack.log import say, warn
from ml_stack.sentinel.human import HumanRequired
from ml_stack.serve import wired

__all__ = ["run"]

GIB = wired.GIB


def _gib(n: int) -> str:
    if 0 < n < GIB // 2:
        return f"{n / wired.MIB:.0f} MiB"
    return f"{n / GIB:.1f} GiB"


def _ctx(n: int) -> str:
    return f"{n // 1024}K" if n % 1024 == 0 else str(n)


def show(plan: wired.Plan) -> None:
    """Print a plan: what was counted, the verdict, before and after, and the context table."""
    say(f"{plan.model} at {_ctx(plan.context)} context, {plan.kv} cache, "
        f"MTP {'shared' if plan.mtp else 'off'}\n")
    say("fixed, in wired memory:")
    for label, size in plan.counted:
        say(f"  {label:<26}{_gib(size):>10}")
    say(f"  {'safety margin':<26}{_gib(plan.margin_bytes):>10}")
    say(f"  {'needs':<26}{_gib(plan.need_bytes + plan.margin_bytes):>10}"
        f"   -> a limit of {plan.needed_mb} MB")
    host = sum(g.bytes for g in plan.grows)
    say("\ngrows with use, in ordinary host RAM (counted against the rest of the machine, "
        "not the wired limit):")
    for g in plan.grows:
        say(f"  {g.label:<62}{_gib(g.bytes) if g.bytes else '-':>10}  {g.cap}")
    for line in plan.notes:
        say(f"note: {line}")
    say(f"\ncurrent limit   {plan.current_mb} MB ({_gib(plan.current_mb * 1024**2)})"
        f"{' (0 = default share)' if not plan.current_mb else ''}; default share about "
        f"{plan.default_mb} MB; installed {_gib(plan.total)}")
    if plan.held_others:
        say(f"the rest of the machine holds {_gib(plan.held_others)} now")
    if not plan.fits:
        say(f"\nDOES NOT FIT: needs {plan.needed_mb} MB, the most that leaves "
            f"{_gib(wired.reserve_bytes(plan.total))} for the rest of the machine is "
            f"{plan.max_mb} MB.")
        say(f"the longest context that does fit: {_ctx(plan.largest_context)}"
            if plan.largest_context else "no context length fits")
    elif plan.enough_now:
        say(f"\nFITS under the current limit; nothing to change (needs {plan.needed_mb} MB)")
    else:
        say(f"\nproposed limit  {plan.proposed_mb} MB ({_gib(plan.proposed_mb * 1024**2)}); "
            f"that leaves {_gib(plan.left_bytes)} for everything else"
            f" (and {_gib(max(plan.left_bytes - host, 0))} once the server's host RAM is at "
            f"its caps)")
    if plan.warn:
        say(f"WARNING: {plan.warn}")
    if plan.fits and plan.left_bytes - host < wired.reserve_bytes(plan.total):
        say("WARNING: with the server's host RAM at its caps less than "
            f"{_gib(wired.reserve_bytes(plan.total))} is left; lower the prompt cache "
            "(--cache-ram here, cache_ram_mb on a lease)")
    say("\ncontext  counted    limit MB  leaves     fits")
    for r in plan.table:
        mark = "yes" if r.fits else "no"
        mark += ", already" if r.fits and r.enough_now else ""
        say(f"{_ctx(r.context):<8} {_gib(r.need_bytes):<10} {r.limit_mb:<9} "
            f"{_gib(r.left_bytes):<10} {mark}")


def _how(args: argparse.Namespace) -> str:
    interactive = sys.stdin.isatty() and sys.stdout.isatty()
    return "sudo" if interactive else "osascript"


def _report(done: wired.Applied) -> int:
    if not done.ok:
        warn(f"not changed: {done.message}")
        return 1
    say(f"limit was {done.before_mb} MB, now {done.after_mb} MB. This resets on reboot "
        "unless it is persisted (--persist, which needs root).")
    say("  undo:  ml-stack-serve memory --reset")
    return 0


def run(args: argparse.Namespace) -> int:
    """Handle ``--for``, ``--apply`` and ``--reset``; returns an exit code."""
    try:
        if args.reset:
            say("putting the wiring limit back")
            return _report(wired.reset(via=_how(args)))
        plan = wired.plan(args.for_model, context=args.ctx, kv=args.kv, mtp=args.mtp,
                          cache_ram_mb=args.cache_ram)
    except (FileNotFoundError, ValueError) as no:
        warn(f"error: {no}")
        return 2
    except HumanRequired as no:
        warn(f"refused: {no}")
        return 3
    show(plan)
    if not args.apply:
        say("\n  ml-stack-serve memory --for MODEL --apply   to raise it for this boot")
        return 0 if plan.fits else 1
    if not plan.fits or not plan.proposed_mb:
        warn("not changed: the model does not fit at this context")
        return 1
    if platform.system() != "Darwin":
        warn("not changed: the wiring limit is a macOS setting")
        return 2
    how = _how(args)
    say(f"\nraising it to {plan.proposed_mb} MB for this boot"
        + ("; sudo will ask for your password" if how == "sudo"
           else "; macOS will ask for your password"))
    try:
        return _report(wired.apply(plan.proposed_mb, via=how))
    except HumanRequired as no:
        warn(f"refused: {no}")
        return 3
    except ValueError as no:
        warn(f"error: {no}")
        return 2


