"""``ml-stack-store``: what a store holds, read two ways, and whether they agree.

    ml-stack-store check PATH          # every doc, node and edge by key and by scan
    ml-stack-store check PATH --fix    # and rewrite the docs a scan reads empty
    ml-stack-store docs PATH           # the documents, with their sizes
    ml-stack-store tidy PATH --apply   # the hygiene pass over a whole store
    ml-stack-store tidy --gold --base-url URL --fail-under 0.8   # score the judge

Measured 2026-09-01: twelve bench runs read back empty through a full scan of ``Doc.value``
while a lookup by key returned them whole, and nothing in the store said so. ``check``
is the question to ask before believing a document is gone; it exits 1 on any finding.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ml_stack.graph.store import GraphStore, StoreMismatch, StoreNeedsUpgrade


def _check(path: Path, fix: bool) -> int:
    with GraphStore(path, read_only=not fix) as store:
        findings = store.check()
        if not fix or not findings:
            for line in findings:
                print(line)
            print(f"{path}: {'clean' if not findings else f'{len(findings)} findings'}")
            return 1 if findings else 0
        for line in findings:
            print(line)
        try:
            rewrote = store.repair()
        except StoreMismatch as why:
            print(f"rewrite refused: {why}")
            rewrote = []
        for line in rewrote:
            print(line)
        left = store.check()
        for line in left:
            print(f"still: {line}")
        print(f"{path}: rewrote {len(rewrote)}, "
              f"{'clean' if not left else f'{len(left)} findings remain'}")
        return 1 if left else 0


def _docs(path: Path) -> int:
    with GraphStore(path, read_only=True) as store:
        rows = store.query("MATCH (d:Doc) RETURN d.key AS key, d.value AS value ORDER BY d.key")
        for row in rows:
            keyed = store.query("MATCH (d:Doc {key:$key}) RETURN d.value AS value",
                                {"key": row["key"]})
            by_key = len((keyed[0]["value"] if keyed else "") or "")
            by_scan = len(row["value"] or "")
            note = "" if by_scan == by_key else f"  (scan reads {by_scan})"
            print(f"{row['key']}\t{by_key} chars{note}")
        print(f"{len(rows)} docs")
    return 0


def _gold(gold: str, base_url: str, fail_under: float) -> int:
    from ml_stack.graph.tidy import judge_gold, load_gold

    if not base_url:
        print("--gold needs --base-url: a model already serving to answer the pairs",
              file=sys.stderr)
        return 2
    from ml_stack.client import Client

    scored = judge_gold(Client(base_url, n_predict=1024), load_gold(gold or None), log=print)
    if fail_under and scored.accuracy < fail_under:
        print(f"below the bar: {scored.accuracy:.0%} of {scored.total} against "
              f"{fail_under:.0%}", file=sys.stderr)
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="ml-stack-store",
        description="What a graph store holds, read by key and by scan, and whether they agree.")
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser(
        "check", help="read every doc, node and edge two ways and print each disagreement")
    check.add_argument("path", type=Path, help="the store directory")
    check.add_argument("--fix", action="store_true",
                       help="rewrite a doc a scan reads empty while a lookup by key reads it whole")
    docs = sub.add_parser("docs", help="list the documents with their sizes")
    docs.add_argument("path", type=Path, help="the store directory")
    hygiene = sub.add_parser(
        "tidy", help="the hygiene pass: merge duplicate nodes and edges, fold inverse pairs, "
                     "flag doubtful labels, report conflicts and orphans -- dry unless --apply")
    hygiene.add_argument("path", type=Path, nargs="?", help="the store directory")
    hygiene.add_argument("--apply", action="store_true",
                         help="write the merges, folds and flags; without it, say what would be done")
    hygiene.add_argument("--written", type=Path, default=None, metavar="FILE",
                         help="a JSON object {name: the name it is} -- the possible duplicates "
                              "a person settled; applied whatever the weights")
    hygiene.add_argument("--base-url", default="", metavar="URL",
                         help="a model already serving, to judge the names a spelling apart "
                              "from what it knows; with one the pass is automated and applies "
                              "what it decides (ml-stack-ingest tidy --model also re-reads the "
                              "source passages)")
    hygiene.add_argument("--gold", nargs="?", const="", default=None, metavar="FILE",
                         help="score the judge instead of tidying a store: pairs whose right "
                              "verdicts are known, accuracy overall and per class, and how many "
                              "needed the source passages (the shipped set when no file is named)")
    hygiene.add_argument("--fail-under", type=float, default=0.0, metavar="F",
                         help="with --gold: exit 1 when accuracy is under F (0-1)")
    args = parser.parse_args(argv)

    if args.command == "tidy" and args.gold is not None:
        return _gold(args.gold, args.base_url, args.fail_under)
    if args.path is None:
        parser.error("the store directory is needed")
    path = args.path.expanduser()
    if not path.exists():
        print(f"{path}: no store there", file=sys.stderr)
        return 2
    try:
        if args.command == "check":
            return _check(path, args.fix)
        if args.command == "tidy":
            from ml_stack.graph.tidy import ModelJudge, tidy, written_from

            judge = None
            if args.base_url:
                from ml_stack.client import Client

                judge = ModelJudge(Client(args.base_url, n_predict=1024))
            tidy(path, dry_run=not args.apply, written=written_from(args.written),
                 judge=judge, log=print)
            return 0
        return _docs(path)
    except StoreNeedsUpgrade as why:
        print(str(why), file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
