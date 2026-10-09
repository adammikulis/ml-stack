"""``poolhouse-models``: what is installed, what fits and what to pull -- list, where, info,
which, fit, suggest, pull, search, recommend and machine -- beside the Hub commands."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable

from poolhouse import hub
from poolhouse.hub import cli as hub_cli, hf_cache, origins
from poolhouse.log import say, warn
from poolhouse.serve import estimate as est, suggest as pick
from poolhouse.units import human_bytes

COMMANDS = ("list", "where", "info", "which", "fit", "suggest", "pull", "snapshot", "search",
            "recommend", "machine")


def _emit(args: argparse.Namespace, payload: object, text: Callable[[], None]) -> int:
    if args.json:
        say(json.dumps(payload, indent=2, default=str))
    else:
        text()
    return 0


def add(sub: argparse._SubParsersAction) -> None:
    """Register the commands on ``poolhouse-models``' subparsers."""
    def make(name: str, help_: str, *, arg: str = "", optional: bool = False
             ) -> argparse.ArgumentParser:
        one = sub.add_parser(name, help=help_)
        if arg:
            one.add_argument(arg, nargs="?" if optional else None)
        one.add_argument("--json", action="store_true", help="print JSON")
        return one

    lst = make("list", "every model installed on this machine")
    lst.add_argument("--format", action="append", choices=hub.FORMATS)
    lst.add_argument("--source", action="append", help="only this tool's folders")
    lst.add_argument("--kind", choices=hub.KINDS, help="only models of this kind")
    lst.add_argument("--all", action="store_true", help="include projectors and draft heads")
    make("where", "the folders searched and what each holds")
    make("info", "one installed model in full", arg="model")
    make("which", "the file a name would be served from, or the best model that fits",
         arg="model", optional=True)
    fit = make("fit", "what serving a model costs in memory, and whether it fits",
               arg="model")
    suggest = make("suggest", "the settings that suit a model on this machine", arg="model")
    for one in (fit, suggest):
        one.add_argument("--goal", choices=("agent", "chat", "long-context", "fast"),
                         default="agent")
    fit.add_argument("--context", type=int, default=4096, help="tokens per slot")
    fit.add_argument("--parallel", type=int, default=1)
    fit.add_argument("--kv", default=est.DEFAULT_KV, help="cache type, e.g. q8_0 or q8_0/f16")
    fit.add_argument("--ngl", default="auto", help="layers on the GPU, or auto")
    fit.add_argument("--batch", type=int, default=512)
    suggest.add_argument("--max-verdict", choices=("green", "yellow", "red"), default="green")
    pull = make("pull", "download hf:owner/repo[/file.gguf][:QUANT]", arg="ref")
    pull.add_argument("--dest", help="folder to download into (default: the store)")
    pull.add_argument("--no-peers", action="store_true",
                      help="do not ask paired devices first; download from the Hub (also POOLHOUSE_NO_PEERS=1)")
    snapshot = make("snapshot", "download a public Hub model or dataset into the persistent Hugging Face cache",
                    arg="repo")
    snapshot.add_argument("--repo-type", choices=("model", "dataset"), default="model")
    snapshot.add_argument("--revision", default="main")
    find = make("search", "GGUF repositories on the Hub", arg="query")
    find.add_argument("--max-size", type=float, default=0, help="largest build, in GB")
    find.add_argument("--quant", default="")
    find.add_argument("--limit", type=int, default=10)
    rec = make("recommend", "models ranked for this machine")
    rec.add_argument("--goal", choices=("agent", "chat", "long-context", "fast"),
                     default="agent")
    rec.add_argument("--query", default="", help="also rank what the Hub offers for this")
    make("machine", "the memory this machine has for a model")


def _table(rows: list[list[str]], head: list[str]) -> None:
    widths = [max(len(r[i]) for r in [head, *rows]) for i in range(len(head))]
    for row in [head, *rows]:
        say("  ".join(c.ljust(w) for c, w in zip(row, widths, strict=True)).rstrip())


def _list(args: argparse.Namespace) -> int:
    found = hub.discover(formats=args.format or hub.FORMATS, include=args.source,
                         companions=args.all, kind=args.kind)
    came = {m.id: origins.latest(m.path.name, m.size_bytes) for m in found}
    rows = [[m.name[:40], m.kind, human_bytes(m.size_bytes), m.quantization, m.format, m.source,
             _from(came[m.id]), "" if m.is_complete else "incomplete", m.id[:60]] for m in found]
    return _emit(args, [m.as_dict() | {"from_peer": came[m.id]} for m in found], lambda: (
        _table(rows, ["NAME", "KIND", "SIZE", "QUANT", "FORMAT", "SOURCE", "FROM", "", "ID"]),
        say(f"\n{len(found)} models, {human_bytes(sum(m.size_bytes for m in found))}")))


def _from(line: dict | None) -> str:
    """A model that came from a paired device says which, and whose licence acceptance it
    relied on (`hub.origins`)."""
    if not line:
        return ""
    relied = line.get("relies_on") or {}
    return f"peer {line.get('peer', '?')}" + (
        f", licence accepted by {relied.get('accepted_by') or '?'} on {relied.get('device', '?')}"
        if relied else "")


def _where(args: argparse.Namespace) -> int:
    counts: dict[str, int] = {}
    for m in hub.discover(companions=True):
        counts[m.source] = counts.get(m.source, 0) + 1
    shown = [{"source": p.label, "path": str(p.path), "present": p.path.is_dir(),
              "verified": p.verified, "models": counts.get(p.label, 0)}
             for p in hub.standard()]
    return _emit(args, shown, lambda: _table(
        [[r["source"], "yes" if r["present"] else "no", "yes" if r["verified"] else "no",
          str(r["path"])] for r in shown], ["SOURCE", "PRESENT", "VERIFIED", "PATH"]))


def _one(name: str) -> hub.ModelInfo | None:
    pool = hub.discover(companions=True)
    got = hub.installed_find(name, pool) or ([m for m in pool if m.path == hub.located(name)])
    return got[0] if got else None


def _info(args: argparse.Namespace) -> int:
    one = _one(args.model)
    if one is None:
        warn(f"{args.model} is not installed; `poolhouse-models list` shows what is")
        return 1
    row = one.as_dict()
    if one.format == "gguf":
        row["estimate_at_4k"] = est.estimate(one.path, context=4096).as_dict()
    return _emit(args, row, lambda: [say(f"{k:16} {v}") for k, v in row.items()
                                     if k != "estimate_at_4k" and v not in (None, "", [], ())])


def _which(args: argparse.Namespace) -> int:
    if not args.model:
        ranked = pick.recommend(goal="agent", limit=1)
        got = ranked[0].choice.candidate if ranked else None
        out = {"model": got.name if got else "", "path": str(got.path) if got and got.path
               else "", "verdict": ranked[0].choice.verdict if ranked else "none"}
        return _emit(args, out, lambda: say(
            f"best installed model for this machine: {out['model']} ({out['verdict']})\n"
            f"  {out['path']}" if got else "no installed model fits"))
    ref = args.model
    local = hub.installed_for(ref) if ref.startswith("hf:") else None
    one = local or _one(ref)
    out = {"asked": ref, "path": str(one.path) if one else "", "reused": bool(one),
           "download": not one and ref.startswith("hf:")}
    return _emit(args, out, lambda: say(
        f"{ref} -> {one.path}  (installed, from {one.source}; nothing to download)" if one
        else f"{ref} is not installed" + ("; serving would download it" if out["download"]
                                          else "")))


def _fit(args: argparse.Namespace) -> int:
    ngl = args.ngl if args.ngl == "auto" else int(args.ngl)
    got = est.estimate(args.model, context=args.context, parallel=args.parallel,
                       kv_cache_type=args.kv, n_gpu_layers=ngl, batch=args.batch)
    machine = hub.machine_memory()
    rated = est.verdict(got, machine)
    pool, left, used = est.headroom(got, machine)
    out = {**got.as_dict(), "verdict": rated, "pool": pool, "headroom_bytes": left,
           "meters": [m.as_dict() for m in est.meters(got, machine)],
           "share_used": round(used, 3),
           "max_context": est.max_context(args.model, machine, est.Setup(
               parallel=args.parallel, kv_cache_type=args.kv, n_gpu_layers=ngl,
               batch=args.batch))}

    def text() -> None:
        say(f"{args.model}  context {got.context} x {got.parallel} slot(s), KV {got.kv_cache_type}")
        for name, size in got.breakdown.items():
            say(f"  {name:18}{human_bytes(size):>9}")
        say(f"  {'total':18}{human_bytes(got.total_bytes):>9}   [{got.confidence}]")
        say(f"verdict {rated}: {used:.0%} of {pool}; the longest context that is at most "
            f"yellow is {out['max_context']}")
        for note in got.notes:
            say(f"note: {note}")
    return _emit(args, out, text)


def _suggest(args: argparse.Namespace) -> int:
    got = pick.suggest(args.model, goal=args.goal, max_verdict=args.max_verdict)
    return _emit(args, got.as_dict(), lambda: [say(
        f"context {got.context}  slots {got.parallel}  gpu layers {got.n_gpu_layers}  "
        f"kv {got.kv_cache_type}  batch {got.batch}  flash {got.flash_attn}  -> {got.verdict}"),
        *[say(f"  {r}") for r in got.reasons],
        *[say(f"  alternative, {o.label}: {o.verdict}") for o in got.alternatives]])


def _pull(args: argparse.Namespace) -> int:
    def show(p: hub.Progress) -> None:
        if p.phase != "downloading" or p.file_total:
            warn(f"\r{p.phase:11} {p.fraction:6.1%}  {human_bytes(p.bytes_per_second)}/s  "
                 f"{p.file}", end="", flush=True)

    got = hub.pull(args.ref, args.dest, show, peers=False if args.no_peers else None)
    warn("")
    return _emit(args, {"path": str(got)}, lambda: say(str(got)))


def _search(args: argparse.Namespace) -> int:
    found = hub.search(args.query, hub.Filters(max_bytes=int(args.max_size * 1e9),
                                               quant=args.quant, limit=args.limit))
    payload = [{"id": r.id, "downloads": r.downloads, "gated": r.gated,
                "builds": [{"build": b, "bytes": s, "shards": n, "quantization": q}
                           for b, s, n, q in r.builds()]} for r in found]

    def text() -> None:
        for r in found:
            say(f"{r.downloads:>10}  {r.id}" + ("  (gated)" if r.gated else ""))
            for build, size, shards, quant in r.builds()[:6]:
                say(f"{'':>12}{human_bytes(size):>8}  {quant or '-':8} {build}"
                    + (f"  {shards} shards" if shards > 1 else ""))
    return _emit(args, payload, text)


def _recommend(args: argparse.Namespace) -> int:
    got = pick.recommend(goal=args.goal, query=args.query)
    payload = [{"name": r.choice.candidate.name, "ref": r.ref, "installed": r.installed,
                "verdict": r.choice.verdict, "size_bytes": r.choice.candidate.size_bytes}
               for r in got]
    return _emit(args, payload, lambda: _table(
        [[r["verdict"], str(r["name"])[:50], human_bytes(float(r["size_bytes"])),
          "installed" if r["installed"] else "download", str(r["ref"])[:60]] for r in payload],
        ["FIT", "NAME", "SIZE", "", "REF"]))


def _machine(args: argparse.Namespace) -> int:
    got = hub.machine_memory()
    return _emit(args, got.as_dict(), lambda: [say(f"{k:16} {v}") for k, v in
                                               got.as_dict().items()])


def _snapshot(args: argparse.Namespace) -> int:
    path = hf_cache.fetch(args.repo, revision=args.revision, repo_type=args.repo_type)
    return _emit(args, {"repo": args.repo, "repo_type": args.repo_type,
                        "requested_revision": args.revision,
                        "resolved_revision": path.name, "path": str(path)},
                 lambda: say(str(path)))


RUN: dict[str, Callable[[argparse.Namespace], int]] = {
    "list": _list, "where": _where, "info": _info, "which": _which, "fit": _fit,
    "suggest": _suggest, "pull": _pull, "search": _search, "recommend": _recommend,
    "machine": _machine, "snapshot": _snapshot,
}


def run(argv: list[str] | None = None) -> int:
    """``poolhouse-models``: the Hub commands and these."""
    return hub_cli.main(argv, add, RUN)
