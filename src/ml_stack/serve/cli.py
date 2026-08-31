"""``ml-serve`` -- see what is serving, put a model up, take one down."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from ml_stack.client import is_healthy, reported_models
from ml_stack.client.health import serving_params
from ml_stack.serve.backend import ServerFailed, ServerInfo, ServerSpec
from ml_stack.serve.binary import BinaryNotFound
from ml_stack.serve.manager import STATE_FILE, ServerManager, recorded_servers
from ml_stack.serve.ports import DEFAULT_HOST, server_pids_on_port
from ml_stack.serve.process import pid_exists

_SPEC = ServerSpec(model="")
DEFAULT_PORT = _SPEC.port
DEFAULT_CONTEXT = _SPEC.context
DEFAULT_PARALLEL = _SPEC.parallel
DEFAULT_TIMEOUT = 300.0
PROBE_TIMEOUT = 2.0


@dataclass(frozen=True, slots=True)
class Snapshot:
    """One server that is answering, and what a lease for it would do."""

    port: int
    base_url: str
    model: str | None = None
    quant: str | None = None
    context: int | None = None
    slots: int | None = None
    pid: int | None = None
    owner_pid: int | None = None
    holder_running: bool = False
    recorded: bool = False
    verdict: str = ""
    reason: str = ""


def base_url_for(port: int) -> str:
    return f"http://{DEFAULT_HOST}:{port}"


def _int_or_none(value: object) -> int | None:
    return value if isinstance(value, int) else None


def look(port: int, records: dict[int, dict]) -> Snapshot | None:
    """What is serving on ``port``, or ``None`` when nothing answers there."""
    url = base_url_for(port)
    if not is_healthy(url, timeout=PROBE_TIMEOUT):
        return None

    params = serving_params(url)
    models = reported_models(url)
    entry = records.get(port) or {}

    reported = (params.model if params else None) or (models[0] if models else None)
    reported = reported or entry.get("model")
    owner = _int_or_none(entry.get("owner_pid"))

    return Snapshot(
        port=port,
        base_url=url,
        model=Path(str(reported)).name if reported else None,
        quant=params.quant if params else None,
        context=params.n_ctx if params else None,
        slots=params.total_slots if params else None,
        pid=_int_or_none(entry.get("pid")),
        owner_pid=owner,
        holder_running=pid_exists(owner),
        recorded=bool(entry),
    )


def judge(manager: ServerManager, snapshot: Snapshot, spec: ServerSpec) -> Snapshot:
    """``snapshot`` with the verdict a lease for ``spec`` would reach."""
    try:
        info = manager.adopt(spec)
    except ServerFailed as exc:
        return replace(snapshot, verdict="refuse", reason=str(exc))
    return replace(snapshot, verdict="adopt" if info is not None else "start")


def _lease_line(snapshot: Snapshot) -> str:
    if not snapshot.recorded:
        return "none on record -- 'ml-serve down' will not stop this server"
    pid = f"server pid {snapshot.pid}" if snapshot.pid else "server pid not recorded"
    if snapshot.owner_pid is None:
        return pid
    if snapshot.owner_pid == snapshot.pid:
        return f"{pid}, started by 'ml-serve up'"
    if snapshot.holder_running:
        return f"{pid}, held by process {snapshot.owner_pid}"
    return f"{pid}, the process that started it (pid {snapshot.owner_pid}) has gone"


def _verdict_line(snapshot: Snapshot, model: str, parallel: int) -> str:
    ask = f"'ml-serve up {model} --parallel {parallel}'"
    if snapshot.verdict == "adopt":
        return f"{ask} would adopt this server"
    if snapshot.verdict == "refuse":
        return f"{ask} would be refused -- {snapshot.reason}"
    return f"{ask} would start its own server"


def cmd_status(args: argparse.Namespace) -> int:
    records = recorded_servers(STATE_FILE)
    ports = sorted({*records, args.port})
    manager = ServerManager(state_file=STATE_FILE)

    found: list[Snapshot] = []
    for port in ports:
        snapshot = look(port, records)
        if snapshot is None:
            continue
        model = args.model or snapshot.model
        if model:
            snapshot = judge(
                manager,
                snapshot,
                ServerSpec(model=model, port=port, context=args.context,
                           parallel=args.parallel),
            )
        found.append(snapshot)

    if args.json:
        print(json.dumps(
            {"serving": bool(found), "ports_checked": ports,
             "servers": [asdict(s) for s in found]},
            indent=2))
        return 0 if found else 1

    if not found:
        print("nothing is serving on port " + ", ".join(str(p) for p in ports) + ".")
        print(f"  'ml-serve up <model>' would start one on port {args.port}.")
        return 1

    for snapshot in found:
        quant = f"  ({snapshot.quant})" if snapshot.quant else ""
        print(snapshot.base_url)
        print(f"  model    {snapshot.model or 'not reported'}{quant}")
        print(f"  context  {snapshot.context if snapshot.context is not None else 'not reported'}"
              " per slot")
        print(f"  slots    {snapshot.slots if snapshot.slots is not None else 'not reported'}")
        print(f"  lease    {_lease_line(snapshot)}")
        if snapshot.verdict:
            print("  " + _verdict_line(snapshot, args.model or snapshot.model or "<model>",
                                       args.parallel))
    return 0


def cmd_up(args: argparse.Namespace) -> int:
    manager = ServerManager(state_file=STATE_FILE)
    spec = ServerSpec(model=args.model, port=args.port, context=args.context,
                      parallel=args.parallel)
    try:
        info = manager.lease(spec, timeout=args.timeout)
    except (ServerFailed, BinaryNotFound, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if not info.adopted:
        manager.detach(info)

    if args.json:
        print(json.dumps({"base_url": info.base_url, "port": info.port, "pid": info.pid,
                          "adopted": info.adopted, "model": str(spec.model),
                          "context": spec.context, "parallel": spec.parallel}, indent=2))
        return 0

    where = f" (pid {info.pid})" if info.pid else ""
    print(f"{'adopted' if info.adopted else 'started'} {info.base_url}{where}")
    return 0


def cmd_down(args: argparse.Namespace) -> int:
    records = recorded_servers(STATE_FILE)
    entry = records.get(args.port)
    url = base_url_for(args.port)

    if entry is None:
        if not is_healthy(url, timeout=PROBE_TIMEOUT):
            print(f"nothing is serving on port {args.port}.")
            return 1
        held = server_pids_on_port(args.port)
        where = f" (pid {held[0]})" if held else ""
        print(f"error: something is serving on {url}{where}, and this machine has no "
              "record of starting it.", file=sys.stderr)
        print("  stop it the way it was started.", file=sys.stderr)
        return 2

    owner = _int_or_none(entry.get("owner_pid"))
    pid = _int_or_none(entry.get("pid"))
    if owner is not None and owner != pid and pid_exists(owner):
        print(f"error: {url} is held by process {owner}, which is still running.",
              file=sys.stderr)
        print("  that process started it and will stop it.", file=sys.stderr)
        return 2

    running = pid_exists(pid)
    ServerManager(state_file=STATE_FILE).release(
        ServerInfo(base_url=str(entry.get("base_url") or url), port=args.port, pid=pid,
                   backend=str(entry.get("backend") or "")))

    if running:
        print(f"stopped {url} (pid {pid})")
    else:
        print(f"nothing was running on port {args.port}; removed the record")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="ml-serve",
        description="See which model is being served on this machine, put one up, take it down.")
    sub = ap.add_subparsers(dest="cmd", required=True, metavar="{status,up,down}")

    status = sub.add_parser("status", help="what is serving, and what a lease would do")
    status.add_argument("--port", type=int, default=DEFAULT_PORT,
                        help=f"port to check besides the recorded ones (default: {DEFAULT_PORT})")
    status.add_argument("--model", default="",
                        help="ask what leasing this model would do (default: whatever is "
                             "already serving)")
    status.add_argument("--context", type=int, default=DEFAULT_CONTEXT,
                        help=f"the context that lease would ask for (default: "
                             f"{DEFAULT_CONTEXT})")
    status.add_argument("--parallel", type=int, default=DEFAULT_PARALLEL,
                        help=f"the slots that lease would ask for (default: "
                             f"{DEFAULT_PARALLEL})")
    status.add_argument("--json", action="store_true",
                        help="print one JSON object instead of the human listing")

    up = sub.add_parser("up", help="serve a model, or adopt the one already serving it")
    up.add_argument("model", help="path to a .gguf file, or hf:owner/repo/file.gguf")
    up.add_argument("--port", type=int, default=DEFAULT_PORT,
                    help=f"port to serve on (default: {DEFAULT_PORT})")
    up.add_argument("--context", type=int, default=DEFAULT_CONTEXT,
                    help=f"tokens across all slots (default: {DEFAULT_CONTEXT})")
    up.add_argument("--parallel", type=int, default=DEFAULT_PARALLEL,
                    help=f"slots to serve at once (default: {DEFAULT_PARALLEL})")
    up.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT,
                    help=f"seconds to wait for it to load (default: {DEFAULT_TIMEOUT:.0f})")
    up.add_argument("--json", action="store_true",
                    help="print one JSON object instead of the human line")

    down = sub.add_parser("down", help="stop a server started on this machine")
    down.add_argument("--port", type=int, default=DEFAULT_PORT,
                      help=f"port of the server to stop (default: {DEFAULT_PORT})")

    args = ap.parse_args(argv)
    return {"status": cmd_status, "up": cmd_up, "down": cmd_down}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
