"""ml-stack-runtime: ensure, status and rollback for the installed runtime."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from ml_stack import runtime, runtime_board, runtime_deploy, runtime_store
from ml_stack.fleet import runtime_wheel
from ml_stack.home import expand
from ml_stack.lock import held_by
from ml_stack.platform import detached_kwargs


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
    root = runtime.directory()
    root.mkdir(parents=True, exist_ok=True)
    with (root / "ensure.log").open("ab") as log:
        log.write(f"started: {time.strftime('%FT%T')} argv: {' '.join(argv)}\n".encode())
        log.flush()
        subprocess.Popen([sys.executable, "-m", "ml_stack.runtime_cli", *argv], stdin=subprocess.DEVNULL,
                         stdout=log, stderr=subprocess.STDOUT, **detached_kwargs())


def _building() -> str:
    lock = runtime.directory() / "deploy.lock"
    return held_by(lock) if lock.exists() else ""


def _ensure(args: argparse.Namespace, argv: list[str]) -> int:
    plan = plan_from(args)
    if args.background:
        settled = runtime_deploy.settled(plan)
        if settled:
            print(f"runtime {plan.commit[:7]} is {settled}")
        elif _building():
            print("runtime build already running")
        else:
            _start_background([word for word in argv if word != "--background"])
            print(f"runtime {plan.commit[:7]} is building in the background; log: {runtime.directory() / 'ensure.log'}")
        return 0
    previous = str(runtime_store.selection().get("commit", ""))
    outcome = runtime_deploy.ensure(plan, force=args.force)
    runtime_board.announce(outcome, previous, verb="ensure", agent=args.agent, label=args.label)
    print(f"{outcome.action} {outcome.commit[:7]} {outcome.detail}".strip())
    return 0 if outcome.ok else 1


def _rollback(args: argparse.Namespace) -> int:
    previous = str(runtime_store.selection().get("commit", ""))
    outcome = runtime_deploy.rollback(plan_from(args), args.to)
    runtime_board.announce(outcome, previous, verb="rollback to", agent=args.agent, label=args.label)
    print(f"{outcome.action} {outcome.commit[:7]} {outcome.detail}".strip())
    return 0 if outcome.ok else 1


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
    return lines


def status_record(plan: runtime_deploy.Plan) -> dict:
    """The status as data: the selected commit, whether it is current and healthy, and the kept runtimes."""
    row = runtime_store.selection()
    return {"selected": row.get("commit", ""), "wanted": plan.commit, "current": runtime_deploy.current(plan), "settled": runtime_deploy.settled(plan),
            "healthy": runtime_deploy.healthy(plan) is not None, "floor": plan.floor,
            "kept": [{"commit": c.commit, "prefix": str(c.prefix), "verified_at": runtime_store.verified_at(c.prefix)}
                     for c in runtime_store.candidates()],
            "state": runtime_store.read_state(), "building": _building()}


def main(argv: list[str] | None = None) -> int:
    """Entry point of ml-stack-runtime."""
    parser = argparse.ArgumentParser(prog="ml-stack-runtime", description=__doc__)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--checkout", default="", help="source checkout (default: the one recorded)")
    common.add_argument("--launchers", default="", help="directory holding the console launchers")
    common.add_argument("--timeout", type=float, default=runtime_deploy.BUILD_TIMEOUT)
    common.add_argument("--agent", default="", help="workspace agent that posts the outcome (default: the environment's)")
    common.add_argument("--label", default="", help="helper label shown beside the agent")
    common.add_argument("--wait", type=float, default=runtime_deploy.CLAIM_WAIT_S,
                        help="seconds to wait for another owner's install claim")
    sub = parser.add_subparsers(dest="command", required=True)
    ensure = sub.add_parser("ensure", parents=[common], help="make the checkout's commit the selected runtime")
    ensure.add_argument("--ref", default="HEAD")
    ensure.add_argument("--force", action="store_true", help="rebuild even when held or current")
    ensure.add_argument("--background", action="store_true", help="return at once; build in a detached process")
    state = sub.add_parser("status", parents=[common], help="show the selected runtime and the fallbacks kept")
    state.add_argument("--json", action="store_true")
    back = sub.add_parser("rollback", parents=[common], help="select the newest earlier verified runtime")
    back.add_argument("--to", default="", help="commit prefix to select")
    words = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(words)
    try:
        if args.command == "ensure":
            return _ensure(args, words)
        if args.command == "rollback":
            return _rollback(args)
        plan = plan_from(args)
        print(json.dumps(status_record(plan), indent=2) if args.json else "\n".join(status_lines(plan)))
        return 0
    except (runtime_deploy.DeployError, OSError, ValueError) as exc:
        print(f"ml-stack-runtime: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
