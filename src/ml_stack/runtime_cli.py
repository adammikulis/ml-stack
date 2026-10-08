"""ml-stack-runtime: ensure, status and rollback for the installed runtime."""

from __future__ import annotations

import argparse
import json
import os
import time
from collections.abc import Callable
from pathlib import Path

from ml_stack import jobs, runtime, runtime_board, runtime_deploy, runtime_store
from ml_stack.command import Group, flag
from ml_stack.fleet import runtime_wheel
from ml_stack.home import expand
from ml_stack.lock import held_by
from ml_stack.log import say, warn


def _source(path: Path) -> Path:
    if not (path / "src" / "ml_stack" / "__init__.py").is_file():
        raise runtime_deploy.DeployError(f"{path} is not an ml-stack source checkout")
    return path


def _checkout(named: str) -> Path:
    if named:
        return _source(expand(named).resolve())
    for candidate in (runtime_store.read_state().get("checkout"), runtime_wheel.source_checkout()):
        if candidate:
            return _source(expand(str(candidate)).resolve())
    raise runtime_deploy.DeployError("no source checkout recorded: pass --checkout")


def _launchers(named: str) -> Path | None:
    recorded = runtime_store.read_state().get("launchers")
    chosen = named or (str(recorded) if recorded else "")
    if not chosen:
        return None
    path = expand(chosen)
    if not path.is_absolute() and not named:
        raise runtime_deploy.DeployError(f"the recorded launcher directory {chosen!r} must be absolute")
    return path.resolve()


def plan_from(args: argparse.Namespace) -> runtime_deploy.Plan:
    """The deploy plan an ensure, status or rollback command names."""
    checkout = _checkout(args.checkout)
    commit = runtime_deploy.resolve_commit(checkout, getattr(args, "ref", "HEAD"))
    return runtime_deploy.Plan(checkout, commit, _launchers(args.launchers),
                               runtime_deploy.floor_of(checkout, commit), args.timeout, args.wait, args.agent)


def _start_background(argv: list[str]) -> None:
    root = runtime_deploy.prepare_root()
    log = root / "ensure.log"
    os.close(os.open(log, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600))
    jobs.detach("ml_stack.runtime_cli", argv, log=log)


def _building() -> str:
    lock = runtime.directory() / "deploy.lock"
    return held_by(lock) if lock.exists() else ""


def _again(args: argparse.Namespace) -> list[str]:
    """The ensure command line that repeats this one without --background."""
    words = ["ensure", "--ref", args.ref, "--timeout", str(args.timeout), "--wait", str(args.wait)]
    for name in ("checkout", "launchers", "agent", "label"):
        if getattr(args, name):
            words += [f"--{name}", getattr(args, name)]
    return words + [f"--{name.replace('_', '-')}" for name in ("force", "force_build") if getattr(args, name)]


def _ensure(args: argparse.Namespace) -> int:
    plan = plan_from(args)
    if args.background:
        settled = runtime_deploy.settled(plan)
        if settled:
            say(f"runtime {plan.commit[:7]} is {settled}")
        elif _building():
            say("runtime build already running")
        else:
            _start_background(_again(args))
            say(f"runtime {plan.commit[:7]} is building in the background; log: {runtime.directory() / 'ensure.log'}")
        return 0
    previous = str(runtime_store.selection().get("commit", ""))
    outcome = runtime_deploy.ensure(plan, force=args.force, force_build=args.force_build)
    runtime_board.announce(outcome, previous, verb="ensure", agent=args.agent, label=args.label)
    say(f"{outcome.action} {outcome.commit[:7]} {outcome.detail}".strip())
    return 0 if outcome.ok else 1


def _rollback(args: argparse.Namespace) -> int:
    previous = str(runtime_store.selection().get("commit", ""))
    outcome = runtime_deploy.rollback(plan_from(args), args.to)
    runtime_board.announce(outcome, previous, verb="rollback to", agent=args.agent, label=args.label)
    say(f"{outcome.action} {outcome.commit[:7]} {outcome.detail}".strip())
    return 0 if outcome.ok else 1


def _status(args: argparse.Namespace) -> int:
    plan = plan_from(args)
    say(json.dumps(status_record(plan), indent=2) if args.json else "\n".join(status_lines(plan)))
    return 0


def _reported(run: Callable[[argparse.Namespace], int]) -> Callable[[argparse.Namespace], int]:
    def wrapped(args: argparse.Namespace) -> int:
        try:
            return run(args)
        except (runtime_deploy.DeployError, OSError, ValueError) as exc:
            warn(f"ml-stack runtime: {exc}")
            return 1
    return wrapped


def _verdict(plan: runtime_deploy.Plan) -> str:
    settled = runtime_deploy.settled(plan)
    if settled == "held":
        return f"held, ensure will not build {plan.commit[:7]} until the branch moves"
    return "yes" if settled else f"no, ensure would build {plan.commit[:7]}"


def status_lines(plan: runtime_deploy.Plan) -> list[str]:
    """The installed runtime, the commit it should be, the fallbacks kept and the last outcome."""
    root = runtime.directory()
    row, record = runtime_store.selection(root), runtime_store.read_state(root)
    chosen = runtime_deploy.healthy(plan)
    lines = [f"runtimes     {root}",
             f"source       {plan.checkout} at {plan.commit[:7]}" + (f" (epoch floor {plan.floor})" if plan.floor else ""),
             f"selected     {str(row.get('commit', 'none'))[:7]} {'healthy' if chosen else 'NOT HEALTHY'}  {row.get('prefix', '')}",
             f"current      {_verdict(plan)}",
             f"launchers    {plan.launchers or 'none recorded'}"]
    for candidate in runtime_store.candidates(root):
        mark = "*" if str(candidate.prefix) == row.get("prefix") else " "
        when = time.strftime("%F %T", time.localtime(runtime_store.verified_at(candidate.prefix)))
        lines.append(f"  {mark} {candidate.commit[:7]}  verified {when}  {candidate.prefix}")
    if record.get("held"):
        lines.append(f"held         {record['held'].get('commit', '')[:7]}: {record['held'].get('reason', '')}")
    if record.get("last_failure"):
        failure = record["last_failure"]
        lines.append(f"last failure {failure.get('commit', '')[:7]}: {failure.get('detail', '')[-300:]}")
    if _building():
        lines.append(f"building     {_building()}")
    for tree in runtime_store.unmanaged(root):
        lines.append(f"unmanaged    {tree['path']}  {tree['bytes']} bytes  {'process inside' if tree['in_use'] else 'idle'}")
    return lines


def status_record(plan: runtime_deploy.Plan) -> dict:
    """The status as data: the selected commit, whether it is current and healthy, and the kept runtimes."""
    row = runtime_store.selection()
    return {"selected": row.get("commit", ""), "wanted": plan.commit, "current": runtime_deploy.current(plan), "settled": runtime_deploy.settled(plan),
            "healthy": runtime_deploy.healthy(plan) is not None, "floor": plan.floor,
            "kept": [{"commit": c.commit, "prefix": str(c.prefix), "verified_at": runtime_store.verified_at(c.prefix)}
                     for c in runtime_store.candidates()],
            "state": runtime_store.read_state(), "building": _building(),
            "unmanaged": runtime_store.unmanaged()}


COMMON = [
    flag("--checkout", default="", help="source checkout (default: the one recorded)"),
    flag("--launchers", default="", help="absolute directory holding the console launchers (default: the one recorded)"),
    flag("--timeout", type=float, default=runtime_deploy.BUILD_TIMEOUT),
    flag("--wait", type=float, default=runtime_deploy.CLAIM_WAIT_S, help="seconds to wait for another owner's install claim"),
    flag("--agent", default="", help="workspace agent that holds the claims and posts the outcome (default: the environment's)"),
    flag("--label", default="", help="helper label shown beside the agent"),
]

GROUP = Group("ml-stack runtime", "Build, verify and select the installed runtime from the source checkout; status; rollback.")
GROUP.add("ensure", _reported(_ensure), help="make the checkout's commit the selected runtime", options=[
    *COMMON, flag("--ref", default="HEAD"),
    flag("--force", action="store_true", help="rebuild even when held or current"),
    flag("--force-build", action="store_true", help="build again at once after a failed build (needs an agent)"),
    flag("--background", action="store_true", help="return at once; build in a detached process")])
GROUP.add("status", _reported(_status), help="show the selected runtime, the fallbacks kept and unmanaged trees",
          options=[*COMMON, flag("--json", action="store_true")])
GROUP.add("rollback", _reported(_rollback), help="select the newest earlier verified runtime (needs an agent)",
          options=[*COMMON, flag("--to", default="", help="commit prefix to select")])

main = GROUP.run


if __name__ == "__main__":
    raise SystemExit(main())
