"""``ml-stack-models``: find a model on the Hub, read its card, fetch it, print its layout."""

from __future__ import annotations

import argparse
from pathlib import Path

from ml_stack import hub
from ml_stack.hub.naming import _SHARD
from ml_stack.log import say, warn
from ml_stack.units import human_bytes


def _head_lines(repo: str) -> list[str]:
    """What `ml-stack-models files` says about a draft head: that one exists, what its
    README warns, and what `choose_head` would serve through the default build and through
    each named build on this machine.

    This listing never serves anything, so the head is named even when every build here
    would withhold it -- ``borrows=True`` -- and the per-build lines say who would.
    """
    from ml_stack.serve.binary import find_binary, named_builds

    shown = hub.choose_head(hub.ref(repo), binary=None, borrows=True)
    if not shown.path:
        return []
    out = [f"draft head shipped with it: {shown.path}"]
    if shown.note:
        out.append(f"  {shown.note}")
    builds: list[tuple[str, Path | None]] = [("this build", find_binary())]
    builds += [(f"--build {name}", binary) for name, binary in named_builds()]
    for label, binary in builds:
        one = hub.choose_head(hub.ref(repo), binary=binary)
        kind = "fork" if one.borrows else "mainline"
        said = f"{one.path} ({one.why})" if one.path else one.why
        out.append(f"  {label} ({kind}): {said}")
    return out


def _parser() -> argparse.ArgumentParser:
    """``ml-stack-models``' parser: find, files, card, fetch, layout."""
    ap = argparse.ArgumentParser(
        prog="ml-stack-models",
        description="Find a model that is newer than anything you remember, and serve it.")
    sub = ap.add_subparsers(dest="cmd", required=True,
                            metavar="{find,files,card,fetch,layout}")

    look = sub.add_parser("find", help="repositories matching some words")
    look.add_argument("words", nargs="+", help="e.g. gemma-4 E4B")
    look.add_argument("--prefer", default=",".join(hub.PREFER),
                      help="publishers to rank first, comma separated (default: %(default)s)")
    look.add_argument("--all", action="store_true", help="not only GGUF repositories")
    look.add_argument("--limit", type=int, default=12)

    what = sub.add_parser("files", help="what is in one repository, and how to serve each")
    what.add_argument("repo", help="owner/name")
    what.add_argument("--ending", default=".gguf")
    what.add_argument("--every", action="store_true",
                      help="one line per file rather than per build; a sharded model is "
                           "forty lines this way and its totals are what you wanted")

    said = sub.add_parser("card", help="what a model's own card asks for -- sampler settings "
                                       "first, because those are what get guessed at")
    said.add_argument("repo", help="owner/name")
    said.add_argument("--full", action="store_true", help="print the whole card as well")

    got = sub.add_parser("fetch", help="download hf: references into the cache, without "
                                       "serving them -- every shard of a sharded model")
    got.add_argument("refs", nargs="+", metavar="REF",
                     help="hf:owner/repo/file.gguf, one or more")

    shape = sub.add_parser("layout", help="the attention layout off a GGUF header: which "
                                          "layers hold a full cache, slide, recur or share "
                                          "it, plus experts, indexers and lookup tables")
    shape.add_argument("model", help="a path, an hf: reference already fetched, or a file "
                                     "name copied from `files`")
    shape.add_argument("--json", action="store_true", help="the same as JSON")
    return ap


def _layout(args) -> int:
    """``ml-stack-models layout``: the attention layout off a GGUF header."""
    import struct

    from ml_stack.serve.layout import layout, render

    named = str(hub.located(args.model) or args.model)
    if named.startswith("hf:"):
        named = str(hub.fetch(named))
    try:
        shape = layout(named)
    except (OSError, ValueError, struct.error) as exc:
        warn(f"error: cannot read {args.model}: {exc}")
        return 2
    say(shape.to_json() if args.json else render(shape))
    return 0


def _find(args) -> int:
    """``ml-stack-models find``: repositories matching some words."""
    prefer = tuple(p.strip().lower() for p in args.prefer.split(",") if p.strip())
    found = hub.find(" ".join(args.words), prefer=prefer, gguf=not args.all,
                     limit=args.limit)
    if not found:
        warn("nothing matched")
        return 1
    for one in found:
        say(f"{one.downloads:>10}  {one.repo}")
    say(f"\nml-stack-models files {found[0].repo}")
    return 0


def _card(args) -> int:
    """``ml-stack-models card``: the sampler settings a model's own card asks for."""
    text = hub.card(args.repo)
    asked = hub.advice(text)
    if asked:
        say(f"{args.repo} asks for:")
        for name, value in asked.items():
            say(f"  {name:16} {value:g}")
        flags = " ".join(f"--{n.replace('_', '-')} {v:g}" for n, v in asked.items()
                         if n in ("temperature", "top_p", "top_k", "min_p"))
        say(f"\nml-stack-bench run {flags}")
    else:
        say(f"{args.repo}'s card names no sampler settings. That is an answer: "
            f"nobody has chosen one, so the caller's default stands.")
    if args.full:
        say("\n" + text)
    return 0


def _fetch(args) -> int:
    """``ml-stack-models fetch``: every shard of each reference, with what came down."""
    for one in args.refs:
        path = hub.fetch(one)
        total = 0
        for shard in hub.shards_beside(path):
            size = shard.stat().st_size if shard.exists() else 0
            total += size
            say(f"{human_bytes(size):>8}  {shard}")
        if _SHARD.search(path.name):
            say(f"{human_bytes(total):>8}  in all")
    return 0


def _builds(repo: str, ending: str, listing: list[tuple[str, int]]) -> None:
    """One line per build: its total, whether it fits here and whether it is downloaded."""
    fits = hub.room()
    mine = hub.on_disk()
    if fits:
        say(f"this machine can serve about {human_bytes(fits)}\n")
    for name, size, shards in hub.builds(repo, ending=ending):
        downloaded = sum(1 for f, _s in hub.build_files(repo, name, ending) if f in mine)
        mark = "" if not fits else ("  fits" if size < fits * 0.95 else "  TOO BIG")
        if downloaded:
            mark = ("  ON THIS MACHINE" if downloaded >= shards
                    else f"  {downloaded}/{shards} downloaded") + mark
        many = f"  {shards} shards" if shards > 1 else ""
        # IQ builds decode through lookup tables Metal runs slowly: on a Mac the
        # smaller IQ file was the slower model (README, "What this measured")
        slow = "  IQ: slower on Metal, take a K-quant" if hub.iq_on_metal(name) else ""
        say(f"{human_bytes(size):>8}  {name}{many}{mark}{slow}")
    for name, size in listing:
        if hub.aside(name):
            say(f"{human_bytes(size):>8}  {hub.ref(repo, name)}  (alongside)")
    say(f"\nml-stack-models files {repo} --every  for individual files")


def _files(args) -> int:
    """``ml-stack-models files``: what a repository holds, by build or file by file."""
    listing = hub.files(args.repo, ending=args.ending)
    if not listing:
        warn(f"no {args.ending} in {args.repo}")
        return 1
    if not args.every:
        _builds(args.repo, args.ending, listing)
        for line in _head_lines(args.repo):
            say(line)
        return 0
    for name, size in listing:
        note = "  (alongside)" if hub.aside(name) else ""
        say(f"{human_bytes(size):>8}  {hub.ref(args.repo, name)}{note}")
    lines = _head_lines(args.repo)
    if lines:
        say("")
    for line in lines:
        say(line)
    return 0


def main(argv: list[str] | None = None) -> int:
    """``ml-stack-models`` -- find a model on the Hub and print how to serve it."""
    args = _parser().parse_args(argv)
    ran = {"layout": _layout, "find": _find, "card": _card, "fetch": _fetch,
           "files": _files}[args.cmd]
    try:
        return ran(args)
    except Exception as exc:  # noqa: BLE001 - the Hub is somebody else's machine
        warn(f"error: {type(exc).__name__}: {exc}")
        return 2
