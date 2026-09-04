"""``ml-stack-ingest``: the parser, the words it takes instead of a document, and the
detached run's record."""

from __future__ import annotations

import argparse
import os
import platform
import subprocess
import sys
import time
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any

from ml_stack import jobs
from ml_stack.ingest.ask import asked_f1, asked_lines, graph_of, read_asked, score_asked
from ml_stack.ingest.extract import PER_SECTION, schema
from ml_stack.ingest.fold import fold
from ml_stack.ingest.gold import gold_lines, gold_score, read_gold
from ml_stack.ingest.migrate import migrate
from ml_stack.ingest.progress import GIVE_UP, Progress, _folded_at, status
from ml_stack.ingest.reads import _read_json
from ml_stack.ingest.run import Stopped, _read_run, _stopping
from ml_stack.ingest.sources import show, sources

__all__ = ["HOME", "STOP_WAIT", "detach", "main", "parser", "retry", "stop", "wait"]


HOME = Path(os.environ.get("MLSTACK_INGEST_HOME") or "~/.ml-stack/ingest").expanduser()
"""Where a detached run's log and its record of itself live. Not the store: the store is
the caller's, named by ``--out``."""


_WINDOWS_DETACHED = 0x00000200 | 0x00000008     # CREATE_NEW_PROCESS_GROUP | DETACHED_PROCESS


KIND = "ingest"
"""The kind of job a detached run is recorded as, in `ml_stack.jobs`."""

STOP_WAIT = 900.0       # a fold over a 7,000-node source took minutes on the way out


def _home() -> Path:
    """Where a detached run's log and its record of itself live."""
    from ml_stack import ingest

    return Path(ingest.HOME)


def _jobs_home(home: Path | None = None) -> Path:
    """The `ml_stack.jobs` record directory under an ingest home."""
    return Path(home) / "jobs" if home is not None else _home() / "jobs"


def _adopt(home: Path | None = None) -> None:
    """Take over an ``ingesting.json`` -- a run started before the record moved into
    `ml_stack.jobs` -- as this machine's ``ingest`` job, so `stop` and `wait` still find it."""
    old = Path(home) / "ingesting.json" if home is not None else _home() / "ingesting.json"
    held = _read_json(old)
    if not isinstance(held, dict) or not int(held.get("pid") or 0):
        return
    if not jobs.alive(KIND, home=_jobs_home(home)):
        jobs.record(KIND, pid=int(held["pid"]), argv=held.get("argv") or (),
                    log=str(held.get("log") or ""), started=str(held.get("started") or ""),
                    home=_jobs_home(home), refuse_if_alive=False)
    old.unlink(missing_ok=True)


def detach(argv: Sequence[str]) -> Path:
    """Run ``ml-stack-ingest argv`` in the background, owned by no terminal; return its log.

    A run over a directory of documents is hours. A child of a shell -- `nohup`, `&`, a
    redirect into a
    scratch directory -- dies with the shell, or with the agent that opened it, so the
    command re-runs itself in a new session with its output in a log under ``HOME/logs``
    and gives the shell back at once. The pid is recorded through `ml_stack.jobs`, which
    refuses a second one beside a run still going; ``status`` reads the progress file the
    run writes.
    """
    rest = [a for a in argv if a != "--detach"]
    logs = _home() / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    log = logs / f"ingest-{time.strftime('%Y%m%dT%H%M%S')}.log"
    command = [sys.executable, "-m", "ml_stack.ingest", *rest]
    extra: dict[str, Any] = ({"creationflags": _WINDOWS_DETACHED}
                             if platform.system() == "Windows" else {"start_new_session": True})
    with log.open("ab") as out:
        out.write((f"argv: {' '.join(rest)}\nstarted: {time.strftime('%FT%T')}\n")
                  .encode("utf-8"))
        out.flush()
        child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=out,
                                 stderr=subprocess.STDOUT,
                                 env={**os.environ, "PYTHONUNBUFFERED": "1"}, **extra)
    jobs.record(KIND, pid=child.pid, argv=rest, log=str(log), home=_jobs_home())
    return log


def retry(out: str | Path, *, say: Callable[[str], None] = print) -> int:
    """``ml-stack-ingest retry --out STORE``: the units given up on are read again by the
    next ``--resume`` -- for after the fix that made them fail is in."""
    where = Progress.beside(out)
    if not where.is_file():
        say(f"nothing ingested into {out}: no {where.name}")
        return 1
    progress = Progress(where)
    freed = 0
    for one in progress.state["sources"].values():
        for entry in (one.get("done") or {}).values():
            if entry.get("error") and int(entry.get("attempts") or 1) >= GIVE_UP:
                entry["attempts"] = 0
                freed += 1
    progress.save()
    say(f"{freed} unit(s) will be read again on the next --resume")
    return 0


def stop(*, say: Callable[[str], None] = print, home: Path | None = None,
         wait: float = STOP_WAIT) -> int:
    """``ml-stack-ingest stop``: end the detached run and wait for its last fold to land.

    The run folds the source it is on before it exits -- minutes, for a source of thousands
    of nodes -- so this waits up to ``wait`` seconds for the process to go, saying so every
    half minute, and then says whether the store moved. The record is kept while the run
    is still ending, so `detach` refuses to start another beside it. Whatever was read is
    kept either way, and the same command with ``--resume`` reads on.
    """
    where = _jobs_home(home)
    _adopt(home)
    held = jobs.held(KIND, home=where)
    pid = int(held.get("pid") or 0)
    if not pid:
        say("no detached ingest is recorded on this machine")
        return 1
    out = _out_of(held.get("argv") or ())
    before = _folded_at(out)

    def waiting(line: str) -> None:
        # jobs indents what it says while it is still waiting; its conclusions are reworded
        if line.startswith("  "):
            say(line.replace("still ending", "still folding"))

    if jobs.stop(KIND, say=waiting, wait=wait, home=where):
        if jobs.held(KIND, home=where):
            say(f"asked the detached ingest (pid {pid}) to stop; it had not ended after "
                f"{wait:.0f}s, so its last fold is still being written -- its record stays, "
                f"and no new run starts beside it until it has")
        else:
            say(f"the recorded ingest (pid {pid}) had already ended")
        return 1
    after = _folded_at(out)
    moved = [f"{slug} at unit {units}" for slug, units in sorted(after.items())
             if units != before.get(slug)]
    say(f"stopped the detached ingest (pid {pid}); "
        + (f"folded {', '.join(moved)} into {out}" if moved
           else f"nothing new was folded into {out}" if out
           else "its units so far are kept")
        + "; the same command with --resume reads on")
    return 0


def wait(*, say: Callable[[str], None] = print, home: Path | None = None,
         every: float = 60.0) -> int:
    """``ml-stack-ingest wait``: block until the detached run this machine records has
    ended, saying so every minute -- so the next command can follow it without a loop
    written by hand (`ml-stack-ingest wait && ml-stack-ingest tidy --out ... --model ...`)."""
    where = _jobs_home(home)
    _adopt(home)
    if not jobs.alive(KIND, home=where):
        say("no detached ingest is running")
        return 0
    return jobs.wait(KIND, say=say, every=every, home=where)


def _recorded_alive(home: Path | None = None) -> int:
    """The pid of the detached run this machine records, when it is still alive; else 0."""
    _adopt(home)
    return jobs.alive(KIND, home=_jobs_home(home))


def _out_of(argv: Iterable[str]) -> str:
    """The ``--out`` a recorded run was started with."""
    argv = list(argv)
    for index, word in enumerate(argv):
        if word == "--out" and index + 1 < len(argv):
            return argv[index + 1]
        if word.startswith("--out="):
            return word[len("--out="):]
    return ""


_WORDS = ("status", "show", "sources", "ask", "fold", "import", "retry", "tidy", "migrate")
"""What a run does instead of reading a document, when one is named where a PDF would be."""


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="ml-stack-ingest", allow_abbrev=False,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Read documents into a knowledge graph, section by section: "
                    "`ml-stack-ingest DOC.pdf ... --out STORE`.",
        epilog="Instead of documents, one of these words:\n"
               "  status   how far the run into --out has got, what failed, what is in the\n"
               "           store, what it cost per unit, and how long the rest will take\n"
               "  show     what was read: concepts, relations and the folds each source made\n"
               "  sources  every source in the store: what the store holds for each, the\n"
               "           concepts more than one source names, the names tidy joined across\n"
               "           sources, and the relations between their vocabularies\n"
               "  ask      ask the store a question with a model -- `ask --out STORE \"...\"` --\n"
               "           or score a set of questions with --gold FILE\n"
               "  fold     fold every source that has reads -- part-read ones too -- into the\n"
               "           store, replacing what the store held for it\n"
               "  import   a nodes/edges CSV pair another extractor wrote, into this store\n"
               "           as one source -- `import DIR --out STORE`, or the two files\n"
               "  retry    let the units given up on be read again by the next --resume\n"
               "  migrate  bring a store written before sources were called sources up to\n"
               "           date: `book:` node ids, their edges, the unit documents, the\n"
               "           progress file and the reads files\n"
               "  stop     end the detached run, after it has folded what it has read\n"
               "  wait     block until the detached run has ended\n")
    ap.add_argument("docs", nargs="*", metavar="DOC",
                    help="the PDFs to read; or one of `status`, `show`, `sources`, `ask`, "
                         "`fold`, `import`, `retry`, `migrate`, `stop` (see below), which "
                         "does that and stops. `ask` takes the question after it, `import` "
                         "the CSV pair")
    ap.add_argument("--out", default="", metavar="STORE",
                    help="the GraphStore to write into; one store holds every source. "
                         "Required to read anything; --gold writes nothing and needs none")
    ap.add_argument("--model", default="", metavar="M",
                    help="a model to put up, read with and take down: a name, a path or an "
                         "hf: reference")
    ap.add_argument("--base-url", default="http://127.0.0.1:8080",
                    help="the model reading, when nothing is served (default: %(default)s)")
    ap.add_argument("--profile", action=argparse.BooleanOptionalAction, default=True,
                    help="serve --model in its measured shape from ml-stack's profiles "
                         "(build, head, cache type, thinking budget, raw flags); "
                         "--no-profile serves it bare")
    ap.add_argument("--images", action="store_true",
                    help="show the model each section's figures as pictures, not only their "
                         "captions; needs a served projector, and without one the captions "
                         "are all it gets")
    ap.add_argument("--sample", type=int, default=0, metavar="N",
                    help="read only the first N sections of each document -- a smoke of "
                         "the whole path before a night is spent on it; with `show`, how "
                         "many concepts and relations to print per source (default 5); with "
                         "`sources`, how many shared concepts and cross-source relations "
                         "(default 10)")
    ap.add_argument("--apply", action="store_true",
                    help="with tidy: write the merges, folds and flags; without it, say what "
                         "would be done")
    ap.add_argument("--no-tidy", action="store_true",
                    help="do not run the hygiene pass over the store at the end of each "
                         "source (it runs by default, with this run's model judging the "
                         "names a spelling apart and re-reading the source where it must)")
    ap.add_argument("--written", default="", metavar="FILE",
                    help="with tidy: a JSON object {name: the name it is} -- the possible "
                         "duplicates a person settled")
    ap.add_argument("--rebuild", action="store_true",
                    help="with fold: drop each source's own nodes and edges first and write "
                         "the full fold from its reads -- the only way anything leaves the "
                         "store, for after a fix that changed what a read means")
    ap.add_argument("--dry-run", action="store_true",
                    help="with fold or import: say what would be written, and write nothing")
    ap.add_argument("--slug", default="", metavar="SLUG",
                    help="with import: name the source this; by default the file it was read "
                         "out of names it")
    ap.add_argument("--confidence", default="medium", choices=("low", "medium", "high"),
                    metavar="LEVEL",
                    help="with import: take rows at this confidence and above -- low, "
                         "medium or high (default: %(default)s)")
    ap.add_argument("--provisional", action=argparse.BooleanOptionalAction, default=True,
                    help="with import: take rows their extractor left provisional "
                         "(default); --no-provisional leaves them")
    ap.add_argument("--core-only", action="store_true",
                    help="keep to the core verbs and kinds. Reading a source, the schema "
                         "is fenced to them, so a section is read with the shared "
                         "vocabulary and nothing else; importing, only the predicates that "
                         "map onto them are written and the rest are left, where without "
                         "it every predicate comes in and the ones outside them are marked "
                         "as extensions")
    ap.add_argument("--source", default="", metavar="SLUG",
                    help="with `show` or `fold`, only this source")
    ap.add_argument("--chapter", default="", metavar="N",
                    help="read only this chapter of each document")
    ap.add_argument("--resume", action="store_true",
                    help="skip the sections the progress file beside --out already records "
                         "as done")
    ap.add_argument("--detach", action="store_true",
                    help=f"run this in the background, owned by nobody's terminal, with its "
                         f"output in a log under {_home() / 'logs'}")
    ap.add_argument("--gold", default="", metavar="FILE",
                    help="score the extraction against a gold set of passages with known "
                         "triples -- recall, precision and the misses -- instead of reading "
                         "anything. With `ask`, a set of questions with the entries each "
                         "answer should select: {\"question\", \"expected\": [ids or labels]}")
    ap.add_argument("--fail-under", type=float, default=None, metavar="F1",
                    help="exit 1 when --gold scores below this F1 (0-1), reading a gold set "
                         "or asking one")
    ap.add_argument("--n-max", type=int, default=None, metavar="N",
                    help="tokens the draft head guesses ahead each step, over the profile's "
                         "measured length -- extraction accepts far more of them than "
                         "answering does, so measure it here (default: the profile's)")
    ap.add_argument("--per-section", type=float, default=PER_SECTION, metavar="SECONDS",
                    help="the most one section may take (default: %(default)s)")
    ap.add_argument("--max-tokens", type=int, default=0, metavar="N",
                    help="where a long section is split, in tokens (default: the reader's "
                         "own 2500)")
    ap.add_argument("--n-predict", type=int, default=16384, metavar="N",
                    help="the answer's ceiling; a ceiling is not a budget, and a low one "
                         "truncates the extraction (default: %(default)s)")
    ap.add_argument("--context", type=int, default=32768, metavar="N",
                    help="context of the one slot a --model is served with -- extraction "
                         "reads one unit at a time and never splits the GPU (default: "
                         "%(default)s)")
    ap.add_argument("--serve-port", type=int, default=8099)
    ap.add_argument("--no-queue", action="store_true",
                    help="refuse at once when the bench is measuring, instead of waiting "
                         "for it (the ingest and the bench take one lock, so one job is on "
                         "the GPU at a time)")
    ap.add_argument("--cache", default="", metavar="DIR",
                    help="keep each extraction under this directory and do not ask twice "
                         "for the same section and schema")
    ap.add_argument("--temperature", type=float, default=None,
                    help="override the sampling temperature")
    ap.add_argument("--top-p", type=float, default=None, help="override top_p")
    ap.add_argument("--top-k", type=int, default=None, help="override top_k")
    ap.add_argument("--min-p", type=float, default=None, help="override min_p")
    return ap


def _parsed(rest: Sequence[str]) -> Any:
    """The arguments, with a word written after an option still a word.

    ``ask --out STORE "the question"`` puts a positional after an option, and argparse
    cannot gather a ``nargs="*"`` positional across one -- it reads the question as an
    argument it does not recognise. Words it could not place join ``docs``; an option it
    could not place is still an error, so an abbreviated or misspelled flag is refused
    rather than read as a document.
    """
    ap = parser()
    args, extra = ap.parse_known_args(list(rest))
    misplaced = [word for word in extra if word.startswith("-")]
    if misplaced:
        ap.error("unrecognized arguments: " + " ".join(misplaced))
    args.docs = [*args.docs, *[word for word in extra if not word.startswith("-")]]
    return args


def main(argv: Sequence[str] | None = None) -> int:
    """``ml-stack-ingest``: documents into one graph, or a gold set scored."""
    from ml_stack.lock import Busy

    rest = list(sys.argv[1:] if argv is None else argv)
    args = _parsed(rest)
    try:
        return _dispatch(args, rest)
    except Busy as why:
        print(f"error: {why}. The bench is measuring; wait for it, or leave out --no-queue "
              f"to queue behind it.", file=sys.stderr)
        return 3


def _dispatch(args: Any, rest: list[str]) -> int:
    from ml_stack import ingest

    if args.docs[:1] == ["stop"]:
        return stop()
    if args.docs[:1] == ["wait"]:
        return wait()
    word = args.docs[0] if args.docs[:1] and args.docs[0] in _WORDS else ""
    if word:
        if not args.out:
            print(f"error: {word} needs --out STORE", file=sys.stderr)
            return 2
        if word == "fold":
            return fold(args.out, source=args.source, rebuild=args.rebuild,
                        dry_run=args.dry_run)
        if word == "import":
            return ingest.bring(args.out, args.docs[1:], slug=args.slug,
                                confidence=args.confidence, provisional=args.provisional,
                                core_only=args.core_only, dry_run=args.dry_run)
        if word == "show":
            return show(args.out, source=args.source, most=args.sample or 5)
        if word == "sources":
            return sources(args.out, most=args.sample or 10)
        if word == "ask":
            return _ask_run(args)
        if word == "retry":
            return retry(args.out)
        if word == "migrate":
            return migrate(args.out)
        if word == "tidy":
            # the hygiene pass is graph.tidy's -- a book, a Slack community, any store --
            # and lives beside the fold here only so the ingest commands are in one place
            from ml_stack.graph.tidy import tidy as hygiene
            from ml_stack.graph.tidy import written_from

            if args.model or args.base_url != parser().get_default("base_url"):
                # automated: the model judges the names a spelling apart, re-reading the
                # sources where it must, and the pass applies what it decides -- after the
                # run that is reading, never beside it (one job on the GPU)
                alive = _recorded_alive()
                if alive:
                    print(f"error: a detached ingest (pid {alive}) is still reading; "
                          f"`ml-stack-ingest wait` first, then tidy", file=sys.stderr)
                    return 2
                try:
                    with _stopping(), ingest._serving(args) as client:
                        judge = ingest._judge(client, args.out, model=args.model)
                        report = hygiene(args.out, written=written_from(args.written),
                                         judge=judge, log=print)
                except Stopped:
                    print("stopped before the tidy finished; the store is as it was")
                    return 1
                return 0 if report.sound else 1
            report = hygiene(args.out, dry_run=not args.apply,
                             written=written_from(args.written), log=print)
            return 0 if report.sound else 1
        return status(args.out)
    if not args.docs and not args.gold:
        print("error: name at least one document, or --gold FILE", file=sys.stderr)
        return 2
    if args.docs and not args.out:
        print("error: reading a document needs --out STORE to write it into", file=sys.stderr)
        return 2
    if args.detach:
        alive = _recorded_alive()
        if alive:
            # one run at a time, on one model, into one store: a second run beside one
            # that is still folding its way out adopted its server and lost it when the
            # first finished (2026-09-03). The record is the lease; it is cleared when
            # the run ends or `stop` sees it end
            print(f"error: a detached ingest (pid {alive}) is still running or still "
                  f"folding on its way out; `ml-stack-ingest stop` waits for it", file=sys.stderr)
            return 2
        log = detach(rest)
        print(f"detached; the log is {log}")
        print(f"  ml-stack-ingest status --out {args.out}")
        return 0
    if args.gold:
        return _gold_run(args)
    return _read_run(args)


def _ask_run(args: Any) -> int:
    from ml_stack import ingest

    question = " ".join(str(d) for d in args.docs[1:]).strip()
    if not question and not args.gold:
        print("error: ask needs a question, or --gold FILE", file=sys.stderr)
        return 2
    if not Path(args.out).expanduser().exists():
        print(f"error: no store at {args.out}", file=sys.stderr)
        return 2
    graph = graph_of(args.out)
    if not graph["nodes"]:
        print(f"error: nothing in {args.out} to ask about", file=sys.stderr)
        return 2
    print(f"{args.out}: {len(graph['nodes'])} node(s), {len(graph['edges'])} edge(s)")
    # the asking comes from the same profile the serving does, so a model measured with
    # one way of asking is not served in its shape and asked in somebody else's
    measured = ingest._find_model(args.model) if args.model else None
    try:
        with _stopping(), ingest._serving(args) as client:
            if not args.gold:
                ingest.ask(graph, question, client, profile=measured)
                return 0
            asked = read_asked(args.gold)
            print(f"asking {len(asked)} question(s) from {args.gold}")
            rows = score_asked(graph, client, asked, log=print, profile=measured)
    except Stopped:
        print("stopped before the answer was finished")
        return 1
    for line in asked_lines(rows):
        print(line)
    f1 = asked_f1(rows)
    if args.fail_under is not None and f1 < args.fail_under:
        print(f"error: F1 {f1:.2f} is under {args.fail_under:g}", file=sys.stderr)
        return 1
    return 0


def _gold_run(args: Any) -> int:
    from ml_stack import ingest

    passages = read_gold(args.gold)
    print(f"gold: {len(passages)} passages from {args.gold}")
    try:
        with _stopping(), ingest._serving(args) as client:
            scored = gold_score(client, passages, schema(core_only=args.core_only),
                                per_section=args.per_section, log=print)
    except Stopped:
        print("stopped before the gold set was scored")
        return 1
    for line in gold_lines(scored):
        print(line)
    if args.fail_under is not None and scored.f1 < args.fail_under:
        print(f"error: F1 {scored.f1:.2f} is under {args.fail_under:g}", file=sys.stderr)
        return 1
    return 0
