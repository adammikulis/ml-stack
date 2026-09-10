"""The read run: every unit of every source through the model, folded into the store as
it goes, tidied at the end of each source, and stoppable."""

from __future__ import annotations

import hashlib
import math
import time
import urllib.parse
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any

from ml_stack.client.counters import (
    Ledger,
    counting,
    sampling_named,
    survival_lines,
)
from ml_stack.home import expand
from ml_stack.ingest.extract import schema
from ml_stack.ingest.fold import fold_into
from ml_stack.ingest.judge import run_record, write_run
from ml_stack.ingest.progress import Progress
from ml_stack.ingest.reads import Read, _keep_reads, _read_json, reads_path
from ml_stack.ingest.serving import _sampling
from ml_stack.log import say, warn
from ml_stack.serve.profile import INGEST
from ml_stack.sources.html import Marks

__all__ = ["FOLD_EVERY", "FOLD_SECONDS", "Stopped", "read_unit", "reader_for"]


FOLD_EVERY = 25
"""Units read between folds of a source into the store.

A fold costs what `entities.fold_names` costs, which is every concept name against every
other: measured over invented units, 400 units of a 12-word vocabulary folded and wrote in
3.8 s, and 300 units of a 2,700-word one took 44 s to fold and 9 s to write. It grows with
the square of the vocabulary rather than with the units, so the interval is a real cost and
not a formality. A chapter's end folds once this many units have gone by since the last
fold; a chapter longer than twice this folds inside itself; the end of a source and a stop
always fold. What the last fold actually took widens it -- see `FOLD_SECONDS`."""

FOLD_SECONDS = 20.0
"""The most a fold may take before the run waits longer between folds.

`fold_source` folds every name against every other, so it grows with the square of the
vocabulary while the write grows with the units. Measured over invented units of a
vocabulary as wide as the source (`tests/test_ingest_sources.py`, an M-series laptop): 300
units and 300 concepts, 0.25 s to fold and 3.3 s to fold and write; 1,000 units and 1,000
concepts, 2.9 s and 11.8 s; 3,000 units and 3,000 concepts, 28.5 s and 82.2 s. A source of
thousands of nodes therefore costs a minute or more at every chapter end, so
`_fold_interval` reads as many units again between folds as the last fold ran over this."""


class Stopped(BaseException):
    """SIGTERM reached a run. Not an `Exception`: one section's extraction catches every
    `Exception` there is, and a stop is not one section's failure."""


def read_unit(client: Any, unit: Any, shape: Mapping[str, Any], **asking: Any) -> Read:
    """One unit, read a second time when the server reset inside the first read.

    `ServerUnreachable` from a server that still answers is a connection dropped mid
    request, not a server that has gone: the unit is read once more, and whatever the
    second read says is what is written down. A server that does not answer is not retried
    -- that is the dead-server path, and the run stops on it.
    """
    from ml_stack import ingest

    row = ingest.extract_unit(client, unit, shape, **asking)
    if row.error.startswith("ServerUnreachable") and ingest._alive(client):
        row = ingest.extract_unit(client, unit, shape, **asking)
        row.retried = True
    return row


def reader_for(where: str, *, images: bool = False, chapter: str | int | None = None,
              cache_dir: str | Path | None = None, marks: Any = None
              ) -> Callable[[], Any]:
    """The reader for ``where``: a PDF, an HTML or XML file, or a URL fetched first.

    ``.pdf`` reads through `ml_stack.sources.pdf`; ``.html``, ``.htm`` and ``.xml`` through
    `ml_stack.sources.html`. http(s) is checked by `ml_stack.http.check`, fetched by
    `ml_stack.media.download.fetch` into ``cache_dir`` (the state root by default), and
    dispatched the same way once it is down.

    ``marks`` says how this source marks a section -- a
    :class:`~ml_stack.sources.html.Marks`. Left out, each reader takes its own default,
    which no corpus with a spelling of its own will fit.
    """
    text = str(where)
    how = marks if marks is not None else Marks()
    if urllib.parse.urlsplit(text).scheme in ("http", "https"):
        return lambda: _read_url(text, images=images, chapter=chapter, cache_dir=cache_dir,
                                 marks=how)
    return lambda: _read_local(text, images=images, chapter=chapter, marks=how)


def _read_local(where: str, *, images: bool, chapter: str | int | None,
               marks: Any = None) -> Any:
    """A PDF, HTML or XML file, read by its suffix."""
    from ml_stack.sources import html, pdf

    marks = marks if marks is not None else Marks()
    suffix = Path(where).suffix.casefold()
    if suffix in (".html", ".htm"):
        return html.read(where, sections=marks.rule())
    if suffix == ".xml":
        return html.read_xml(where, marks=marks)
    return pdf.read(where, images=images, chapter=chapter)


def _read_url(url: str, *, images: bool, chapter: str | int | None,
             cache_dir: str | Path | None, marks: Any = None) -> Any:
    """A document fetched from the web, then dispatched by what came down."""
    from ml_stack.home import state
    from ml_stack.http import check
    from ml_stack.media.download import fetch
    from ml_stack.sources import html, pdf

    marks = marks if marks is not None else Marks()
    safe = check(url)
    root = expand(cache_dir) if cache_dir else state("ingest", "downloads")
    local = fetch(safe, root / _cache_name(safe))
    suffix = local.suffix.casefold()
    if suffix in (".html", ".htm"):
        return html.read(local, url=safe, sections=marks.rule())
    if suffix == ".xml":
        return html.read_xml(local, url=safe, marks=marks)
    return pdf.read(local, images=images, chapter=chapter)


def _cache_name(url: str) -> str:
    """A filename for a downloaded document: its own name, deduped by a hash of the url."""
    parts = urllib.parse.urlsplit(url)
    name = Path(parts.path).name or "download"
    stem, suffix = Path(name).stem or "download", Path(name).suffix
    return f"{stem}-{hashlib.sha256(url.encode()).hexdigest()[:16]}{suffix}"


def _opened(text: str, *, images: bool, chapter: str | int | None,
           marks: Any = None) -> Any:
    """One of `args.docs`, read into a `Document`; `FileNotFoundError` for a local path
    that is not one."""
    if urllib.parse.urlsplit(text).scheme not in ("http", "https") and not expand(text).is_file():
        raise FileNotFoundError(f"no such document: {expand(text)}")
    return reader_for(text, images=images, chapter=chapter, marks=marks)()


@contextmanager
def _stopping() -> Any:
    """Turn SIGTERM into `Stopped` for the length of a run, and put the old handler back."""
    import signal

    def raise_it(*_: Any) -> None:
        raise Stopped("SIGTERM")

    try:
        before = signal.signal(signal.SIGTERM, raise_it)
    except ValueError:            # not the main thread: nothing to install onto
        yield
        return
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, before)


def _read_run(args: Any) -> int:
    from ml_stack import ingest
    from ml_stack.client.spent import Spent
    from ml_stack.http import Refused
    from ml_stack.ingest.vocabulary import Vocabulary
    from ml_stack.media.download import DownloadError
    from ml_stack.sources import units as source_units

    progress = Progress(Progress.beside(args.out))
    spent = Spent()
    core_only = bool(getattr(args, "core_only", False))
    shape = schema(core_only=core_only)
    words = None if core_only else Vocabulary.read(args.out)
    started = time.time()
    code = 0
    stopped = False

    folded_seconds = 0.0

    def keep(slug: str, title: str, rows: Sequence[Mapping[str, Any]],
             units_by_id: Mapping[str, Any]) -> dict[str, Any]:
        nonlocal folded_seconds
        judge = None if args.no_tidy else ingest._judge(client, args.out, model=args.model)
        got = fold_into(args.out, slug, title=title, reads=rows, units_by_id=units_by_id,
                        progress=progress, judge=judge)
        if words is not None:
            words.write(args.out)
        folded_seconds = float(got["seconds"])
        landed = got.get("absorbed") or {}
        say(f"  folded {slug} at unit {got['units']} in {got['seconds']:.1f}s: "
            f"{got['nodes']} nodes, {got['edges']} edges"
            + (" (partial)" if got["partial"] else "")
            + (f"; {landed['same_name'] + landed['plural']} name(s) landed on existing "
                 f"nodes, judge: {landed['judged_same']} same, "
                 f"{landed['judged_different']} different, {landed['possible']} left"
                 if landed else "")
            + (f"; the next fold is {_fold_interval(folded_seconds)} unit(s) away"
                 if _fold_interval(folded_seconds) != ingest.FOLD_EVERY else ""))
        return got

    try:
        with _stopping(), ingest._serving(args) as client, _counted(args, client) as counted:
            run_id = write_run(args.out,
                               run_record(args, serving=ingest._serving_said(args)))
            say(f"  run {run_id}: units read now point at it")
            for path in args.docs:
                text = str(path)
                began = time.time()
                try:
                    marks = Marks.from_args(args)
                    document = _opened(text, images=args.images,
                                      chapter=args.chapter or None, marks=marks)
                except (FileNotFoundError, Refused, DownloadError, ValueError) as why:
                    warn(f"error: {why}")
                    code = 2
                    continue
                wanted = source_units.units(document, **({"max_tokens": args.max_tokens}
                                                         if args.max_tokens else {}))
                if args.sample:
                    wanted = wanted[:args.sample]
                slug = document.slug
                progress.source(slug, title=document.title, path=document.path,
                                sections=len(wanted), marks=asdict(marks))
                banks = source_units.question_banks(document, **({"max_tokens": args.max_tokens}
                                                                 if args.max_tokens else {}))
                say(f"{document.title}: {len(document.chapters)} chapter(s), "
                    f"{len(wanted)} unit(s) over {document.page_count} pages, headings from "
                    f"the {document.how}" + (", OpenStax" if document.openstax else "")
                    + (f", {banks} question-bank part(s) skipped" if banks else "")
                    + f" -- read in {time.time() - began:.0f}s")

                units_by_id = {unit.id: unit for unit in wanted}
                folded_seconds = float(
                    (progress.state["sources"].get(slug) or {}).get("folded_seconds") or 0.0)
                held_reads = _read_json(reads_path(args.out, slug))
                held_reads = held_reads if isinstance(held_reads, dict) else {}
                reads_by_unit: dict[str, dict[str, Any]] = {}
                to_read = []
                for unit in wanted:
                    if args.resume and progress.done(slug, unit.id) and unit.id in held_reads:
                        reads_by_unit[unit.id] = held_reads[unit.id]
                        continue
                    to_read.append(unit)

                # one at a time, on the one slot: each unit is written down the moment it
                # finishes, so a run killed mid-source loses at most the unit in flight,
                # and the source is folded into the store as it goes, so a run that will
                # take days can be asked questions today
                since = 0
                try:
                    for index, unit in enumerate(to_read):
                        row = read_unit(client, unit, shape, images=args.images,
                                        per_section=args.per_section,
                                        cache_dir=args.cache or None, vocabulary=words)
                        fresh = words.note(row.extracted, unit.id) if words is not None else []
                        row.run = run_id
                        reads_by_unit[unit.id] = asdict(row)
                        for call in row.calls:
                            spent.add(_call_of(call))
                        progress.note(slug, row)
                        _keep_reads(args.out, slug, [reads_by_unit[unit.id]])
                        if row.error.startswith("ServerUnreachable") and not ingest._alive(client):
                            # the server is gone -- killed, crashed, evicted. Every unit
                            # after this one would fail in a second and be written down as
                            # a failure (2026-09-03: 209 of them, in under a minute), so the
                            # run folds what it has and ends; the unit is written down but
                            # not counted against, and --resume reads on once something
                            # serves again
                            say(f"  the model server went away at {unit.id}; folding what "
                                f"was read and stopping -- --resume reads on")
                            raise Stopped("server gone")
                        since += 1
                        say(f"  ch {unit.chapter or '-':>3}  "
                            f"{unit.section or unit.section_title[:12]:<8}"
                            f" {row.seconds:6.1f}s  {row.concepts:>3}c {row.relations:>3}r "
                            f"{row.figures:>2}f" + (f" {row.images}img" if row.images else "")
                            + (" (read again after a reset)" if row.retried else "")
                            + (f"  coined {', '.join(fresh)}" if fresh else "")
                            + (f"  {row.error}" if row.error else ""))
                        ahead = to_read[index + 1] if index + 1 < len(to_read) else None
                        if ahead is not None and _time_to_fold(
                                since, ahead.chapter != unit.chapter,
                                seconds=folded_seconds):
                            keep(slug, document.title, _rows(wanted, reads_by_unit), units_by_id)
                            since = 0
                except Stopped:
                    stopped = True

                counts = keep(slug, document.title, _rows(wanted, reads_by_unit), units_by_id)
                if not args.no_tidy:
                    # the hygiene pass over the store, with this run's model as the judge
                    # and the units still in memory as its source -- automated, recorded,
                    # nothing deferred
                    from ml_stack.graph.tidy import tidy as hygiene
                    texts = {unit.id: unit.text for unit in wanted}
                    judged = hygiene(args.out, judge=ingest._judge(
                        client, args.out, model=args.model, texts=texts), log=None)
                    say(f"  tidied: {judged.said()}")
                say(f"  {document.title}: {counts['nodes']} nodes, {counts['edges']} edges "
                    f"into {args.out}")
                if stopped:
                    break
    except Stopped:
        stopped = True

    if words is not None:
        for line in words.lines():
            say(line)
    totals = progress.totals()
    say(f"\n{totals['sections']} section(s) of {totals['sources']} source(s) in "
        f"{(time.time() - started) / 60:.1f} min; {spent.calls} calls, "
        f"{spent.prompt_tokens} prompt and {spent.completion_tokens} completion tokens"
        + (f"; {totals['failed']} failed" if totals["failed"] else ""))
    if spent.drafted:
        say(_drafting_line(spent))
    for line in _counter_lines(counted):
        say(line)
    if stopped:
        say("stopped: what was read is folded into the store; "
            f"the same command with --resume reads on ({args.out})")
    return code


@contextmanager
def _counted(args: Any, client: Any) -> Iterator[Any]:
    """The server's speculative counters either side of the whole read, labelled."""
    base = str(getattr(client, "base_url", "") or "")
    asked = _sampling(args)
    if not base:
        yield None
        return
    with counting(base, workload=INGEST,
                  sampling=sampling_named(asked.get("temperature"), top_p=asked.get("top_p")),
                  label=str(getattr(args, "out", "") or "")) as block:
        yield block


def _counter_lines(block: Any) -> list[str]:
    """What the counters said about this read, or why they said nothing."""
    if block is None:
        return ["the server was not reachable for its speculative counters"]
    led = Ledger([block])
    out = list(led.lines())
    if (moved := block.delta) is not None and moved.drafts:
        out += ["", "  acceptance by position within the draft:", *survival_lines(moved)]
    return out


def _drafting_line(spent: Any) -> str:
    """What the draft head guessed, what was kept, and the time each half took."""
    line = (f"drafted {spent.draft_tokens}, kept {spent.draft_taken}"
            f" ({(spent.acceptance or 0) * 100:.1f}%)")
    if spent.verify_n:
        line += f" over {spent.verify_n} pass(es), {spent.tokens_per_pass:.2f} tokens each"
    if spent.draft_ms:
        line += (f"; {spent.draft_ms / 1000:.1f}s drafting"
                 f" + {(spent.verify_ms or 0) / 1000:.1f}s checking")
    return line


def _fold_interval(seconds: float, *, every: int | None = None,
                   most: float | None = None) -> int:
    """Units to read between folds, given what the last fold of this source took.

    A fold under ``most`` seconds is paid at every chapter end. One over it is paid once
    for every ``most`` seconds it ran to: a fold of a minute is worth waiting three
    chapters for, and a source of nine thousand nodes is not folded for minutes at every
    chapter end. Nothing measured makes it longer, so the first fold of a source is
    `FOLD_EVERY` as it always was.
    """
    # read off the package rather than bound as defaults: a caller that moves either moves it
    from ml_stack import ingest

    every = ingest.FOLD_EVERY if every is None else int(every)
    most = ingest.FOLD_SECONDS if most is None else float(most)
    if seconds <= most or most <= 0:
        return int(every)
    return int(every) * math.ceil(seconds / most)


def _time_to_fold(since: int, boundary: bool, *, seconds: float = 0.0) -> bool:
    """Whether the source in flight should be folded into the store now.

    ``seconds`` is what the last fold of this source took -- see `_fold_interval`.
    """
    every = _fold_interval(seconds)
    return since >= (every if boundary else 2 * every)


def _rows(wanted: Iterable[Any], reads_by_unit: Mapping[str, Any]) -> list[dict[str, Any]]:
    """One source's reads so far, in the order the source has them."""
    return [reads_by_unit[unit.id] for unit in wanted if unit.id in reads_by_unit]


def _call_of(record: Mapping[str, Any]) -> Any:
    from ml_stack.telemetry import Call

    fields = {f for f in Call.__dataclass_fields__}
    return Call(**{k: v for k, v in record.items() if k in fields})
