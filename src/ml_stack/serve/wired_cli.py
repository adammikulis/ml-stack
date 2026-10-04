"""``ml-stack-serve memory --for MODEL``: the wiring limit a model needs, and changing it."""

from __future__ import annotations

import argparse
import platform
import sys

from ml_stack.log import say, warn
from ml_stack.sentinel.human import HumanRequired
from ml_stack.serve import wired, wired_apply

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


def _how() -> str:
    return "sudo" if sys.stdin.isatty() and sys.stdout.isatty() else "osascript"


def _report(done: wired_apply.Applied, kept: str) -> int:
    if not done.ok:
        warn(f"not changed: {done.message}")
        return 1
    say(f"limit was {done.before_mb} MB, now {done.after_mb} MB. {kept}")
    say("  undo:  ml-stack-serve memory --reset")
    return 0


def _kept_line(keep: bool | None) -> str:
    if keep:
        return "It is kept across restarts by a boot-time daemon (remove it: --unpersist)."
    if keep is False:
        return "The boot-time daemon is removed."
    return "This resets on reboot unless it is persisted (--persist N)."


def _number(args: argparse.Namespace, plan: wired.Plan | None) -> int | None:
    """The MB asked for: a number after --apply/--persist, else the plan's proposal."""
    for given in (args.apply, args.persist):
        if given not in (None, "", True, False) and str(given).lstrip("-").isdigit():
            return int(given)
    return plan.proposed_mb if plan and plan.proposed_mb else None


def _change(args: argparse.Namespace, plan: wired.Plan | None) -> int:
    if platform.system() != "Darwin":
        warn("not changed: the wiring limit is a macOS setting")
        return 2
    keep = True if args.persist is not None else (False if args.unpersist else None)
    mb = None if args.unpersist and args.apply is None else _number(args, plan)
    if mb is None and not args.unpersist:
        warn("not changed: say how many MB, or name a model with --for that fits")
        return 1
    how = _how()
    say(f"\nchanging it{f' to {mb} MB' if mb is not None else ''}; "
        + ("sudo will ask for your password" if how == "sudo"
           else "macOS will ask for your password"))
    try:
        done = wired_apply.set_limit(mb, keep=keep, via=how)
    except HumanRequired as no:
        warn(f"refused: {no}")
        return 3
    except ValueError as no:
        warn(f"error: {no}")
        return 2
    return _report(done, _kept_line(keep))


def run(args: argparse.Namespace) -> int:
    """Handle ``--for``, ``--apply``, ``--persist``, ``--unpersist`` and ``--reset``."""
    plan = None
    try:
        if args.reset:
            say("putting the wiring limit back")
            return _report(wired_apply.reset(via=_how()), _kept_line(False))
        if args.for_model:
            plan = wired.plan(args.for_model, wired.Ask(args.ctx, args.kv, args.mtp,
                                                        args.cache_ram))
    except (FileNotFoundError, ValueError) as no:
        warn(f"error: {no}")
        return 2
    except HumanRequired as no:
        warn(f"refused: {no}")
        return 3
    if plan is not None:
        show(plan)
    wants = args.apply is not None or args.persist is not None or args.unpersist
    if not wants:
        say("\n  ml-stack-serve memory --for MODEL --apply   to raise it for this boot")
        return 0 if plan is None or plan.fits else 1
    if plan is not None and not plan.fits and _number(args, None) is None:
        warn("not changed: the model does not fit at this context")
        return 1
    return _change(args, plan)
