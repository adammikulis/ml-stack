"""``ml-stack-serve status``: what is serving, what its draft head is keeping, and what a
lease would do."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

from ml_stack.command import flag, option
from ml_stack.hub import pretty_name
from ml_stack.log import say
from ml_stack.serve import holding, lease_cli, ops, provenance
from ml_stack.serve.backend import ServerSpec, logs_of, parse_context
from ml_stack.serve.broker import BrokerError
from ml_stack.serve.ops import base_url_for
from ml_stack.serve.quant_guard import status_note
from ml_stack.units import human_bytes

__all__ = ["OPTIONS", "cmd_status"]

_SPEC = ServerSpec(model="")
DEFAULT_PORT = _SPEC.port
DEFAULT_CONTEXT = _SPEC.context
DEFAULT_PARALLEL = _SPEC.parallel

OPTIONS = [
    option("port", default=DEFAULT_PORT,
           help=f"port to check besides the recorded ones (default: {DEFAULT_PORT})"),
    option("model", help="ask what leasing this model would do (default: whatever is "
                         "already serving)"),
    option("context", type=parse_context, default=DEFAULT_CONTEXT,
           help=f"the context that lease would ask for -- 32768, 256k, 1m "
                f"(default: {DEFAULT_CONTEXT})"),
    option("parallel", default=DEFAULT_PARALLEL,
           help=f"the slots that lease would ask for (default: {DEFAULT_PARALLEL})"),
    flag("--every", action="store_true",
         help="every llama-server process on this machine, leased or not -- a stray "
              "one holds memory a lease cannot see"),
    option("json", help="print one JSON object instead of the human listing"),
]


def _lease_line(snapshot: ops.Snapshot) -> str:
    if not snapshot.recorded:
        return "none on record -- 'ml-stack-serve down' will not stop this server"
    pid = f"server pid {snapshot.pid}" if snapshot.pid else "server pid not recorded"
    if snapshot.owner_pid is None:
        return pid
    if snapshot.owner_pid == snapshot.pid:
        return f"{pid}, started by 'ml-stack-serve up'"
    if snapshot.holder_running:
        return f"{pid}, held by process {snapshot.owner_pid}"
    return (f"{pid}, orphaned -- the process that started it (pid {snapshot.owner_pid}) "
            "has gone; 'ml-stack-serve down --orphans' stops it")


def _verdict_line(snapshot: ops.Snapshot, model: str, parallel: int) -> str:
    slots = f" --parallel {parallel}" if int(parallel or 1) != 1 else ""
    ask = f"'ml-stack-serve up {model}{slots}'"
    if snapshot.verdict == "adopt":
        return f"{ask} would adopt this server"
    if snapshot.verdict == "refuse":
        return f"{ask} would be refused -- {snapshot.reason}"
    return f"{ask} would start its own server"


def _say_processes(as_json: bool) -> int:
    """``status --every``: every llama-server on this machine, leased or not."""
    from ml_stack.fleet.models import sized

    got = ops.processes()
    if as_json:
        say(json.dumps({"serving": bool(got.found), "servers": list(got.found),
                        "foreign": list(got.foreign)}, indent=2))
        return 0 if got.found else 1
    if not got.found:
        say("no llama-server is running on this machine.")
        return 1
    for one in got.found:
        if one.get("defunct"):
            # a zombie holds no memory and answers no port; it is waiting to be reaped
            say(f"  pid {one['pid']}  defunct -- exited, not yet reaped; holds nothing")
            continue
        leased = "leased" if one["port"] in got.leased else "NOT leased -- nobody records it"
        rss = f"{one['rss'] / 2**30:.1f}G" if one["rss"] else "?"
        say(f"  :{one['port']}  pid {one['pid']}  {pretty_name(one['model']) or '?'}  "
            f"{rss} resident  {leased}  ({one['binary']})")
        cache = ops.cache_of(one["model"])
        if cache is not None:
            say(f"      cache  {cache[0]}  ({sized(cache[1])})")
        if one["port"] not in got.leased:
            say(f"    unmanaged -- pid {one['pid']}, not started by ml-stack; reported and left alone")
    if got.strays:
        say(f"  {len(got.strays)} not leased: 'ml-stack-serve down --port N' stops one")
    return 0


def _drafting_lines(drafting: ops.Drafting | None) -> list[str]:
    """What `status` says about a server's draft head: what it is, and how much is kept."""
    if drafting is None:
        return []
    if not drafting.loaded:
        return ["  drafting no draft head -- " + (drafting.note
                or "every token is written by the model itself")]
    named = [drafting.head or ("the weights' own MTP layer" if drafting.spec_type == "draft-mtp"
                               else "a head the command line does not name")]
    if drafting.spec_type:
        named.append(drafting.spec_type)
    if drafting.ahead is not None:
        named.append(f"{drafting.ahead} token(s) ahead per pass")
    return [f"  drafting {', '.join(named)}", "           " + _kept_line(drafting)]


def _kept_line(drafting: ops.Drafting) -> str:
    """How much of the draft the model has kept since this server came up."""
    if not drafting.metrics:
        return ("acceptance unknown: this server was started without --metrics, so it "
                "reports no counters")
    counted = drafting.counted
    if counted is None:
        return "acceptance unknown: this server answered no counters"
    if not counted.drafts:
        return "nothing drafted yet since the server came up"
    return (f"{(counted.acceptance or 0) * 100:.1f}% of drafted tokens kept since the server "
            f"came up (higher is better), "
            f"{counted.tokens_per_draft or 0:.1f} tokens per verification pass")


def _broker_servers() -> list[dict]:
    """The servers and holders the running broker reports; none when no broker is running."""
    try:
        return list(lease_cli.snapshot()["servers"])
    except (BrokerError, OSError, KeyError):
        return []


def _say_leases(held: list[holding.Hold]) -> None:
    """Each lease `ml-stack-serve up` holds: who, what, how much memory, since when, why, and
    everyone else using its server."""
    servers = _broker_servers()
    for one in held:
        since = time.strftime("%F %T", time.localtime(one.since)) if one.since else "-"
        where = f"port {one.port}" if one.port else one.status
        say(f"lease {one.id}  {Path(one.model).name}  context {one.context:,}  {where}  "
            f"held by pid {one.pid}  {human_bytes(one.memory) if one.memory else 'memory unmeasured'}"
            f"  since {since}" + (f"  idle {one.idle_s:.0f}s" if one.idle_s else ""))
        for line in provenance.lines(one.why):
            say(line)
        users = [*holding.joined(one.id), *lease_cli.other_holders(servers, one.port, besides=one.lease)]
        for other in users:
            say(f"  also used by {other.get('requester')} (pid {other.get('pid')}): {other.get('reason')}")


def cmd_status(args: argparse.Namespace) -> int:
    if getattr(args, "every", False):
        return _say_processes(bool(args.json))

    found = ops.status(port=args.port, model=args.model, context=args.context,
                       parallel=args.parallel)
    held = holding.holds()
    if args.json:
        say(json.dumps(
            {"serving": bool(found.servers) or bool(found.foreign),
             "leases": [asdict(h) for h in held],
             "ports_checked": list(found.ports),
             "servers": [asdict(s) for s in found.servers],
             "foreign": list(found.foreign)},
            indent=2))
        return 0 if (found.servers or found.foreign) else 1

    _say_leases(held)
    if not found.servers and not found.foreign:
        say("nothing is serving on port " + ", ".join(str(p) for p in found.ports) + ".")
        say(f"  'ml-stack-serve up <model>' would start one on port {args.port}.")
        for port in found.ports:
            written = logs_of("llama-server", port)
            if written:
                say(f"  the last server on port {port} wrote {written[-1]}")
        return 1

    for snapshot in found.servers:
        quant = f"  ({snapshot.quant})" if snapshot.quant else ""
        say(snapshot.base_url)
        say(f"  model    {snapshot.model or 'not reported'}{quant}")
        say(f"  context  {snapshot.context if snapshot.context is not None else 'not reported'}"
            " per slot")
        say(f"  slots    {snapshot.slots if snapshot.slots is not None else 'not reported'}")
        say(f"  lease    {_lease_line(snapshot)}")
        if snapshot.iq_warning:
            say(f"  WARNING  {status_note(snapshot.iq_warning)}")
        if snapshot.log:
            say(f"  log      {snapshot.log}")
        if snapshot.load_s is not None:
            warm = f", warm-up {snapshot.warmup_s:.1f}s" if snapshot.warmup_s is not None else ""
            say(f"  loaded   in {snapshot.load_s:.1f}s{warm}")
        for line in _drafting_lines(snapshot.drafting):
            say(line)
        if snapshot.verdict:
            say("  " + _verdict_line(snapshot, args.model or snapshot.model or "<model>",
                                     args.parallel))
    for held in found.foreign:
        say(base_url_for(held["port"]))
        say(f"  unmanaged -- pid {held['pid']}, not started by ml-stack; reported and left alone")
        say("  drafting " + (f"{held['draft']}, from its command line"
                             if held.get("draft") else
                             "no draft head -- every token is written by the model itself"))
    return 0
