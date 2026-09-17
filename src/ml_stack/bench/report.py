"""Everything measured so far, as one document: how each model was asked, what a draft
head was worth, how much memory it wants, and what to serve.

`Doc` builds a document once and renders it as Markdown or as plain text; `report` fills
it -- answering per model, across models, extraction, ingest, draft heads, memory, and
what to serve -- and `main` is ``ml-stack-bench report``. Nothing here measures or serves
anything: it reads the kept runs and the fit records and arranges them, and a part that
was never measured says "not measured".
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

# The package is the namespace the tests and `selfcheck` patch -- `bench.runs`, `bench.home_dir()`
# -- so anything patchable is looked up there at call time, never bound here at import.
from ml_stack import bench
from ml_stack.bench.gathered import (
    MIN_MESSAGES,
    _pct,
    _scores,
    across,
    answering,
    asking_of,
    best_extractor,
    by_model,
    cache_of,
    extract_model_of,
    extractions,
    fit_for,
    model_of,
    read_messages,
    recommended_head,
    thinking_of,
)
from ml_stack.bench.keep import SHORT
from ml_stack.bench.profiles import write_profiles
from ml_stack.bench.record import of
from ml_stack.bench.score import NOISE, derived, host_of, hosts_of, per_question
from ml_stack.bench.show import _gb, drafted
from ml_stack.log import say, warn
from ml_stack.units import human_bytes

__all__ = ["Doc", "fits_named", "report"]


# ---------------------------------------------------------------- rendering both ways

class Doc:
    """A document built once and rendered as Markdown or as plain text.

    Two renderings of the same structure rather than two writers: a table that is right in
    one and stale in the other is the failure this exists to prevent, and every section
    here is built by the same code whichever way it comes out.
    """

    def __init__(self, md: bool = True) -> None:
        self.md = md
        self.lines: list[str] = []

    def head(self, level: int, text: str) -> None:
        if self.md:
            self._blank()
            self.lines.append(f"{'#' * level} {text}")
        else:
            self._blank()
            self.lines.append(text if level > 2 else text.upper())
            if level <= 2:
                self.lines.append("=" if level == 1 else "-")
                self.lines[-1] *= len(text)
        self.lines.append("")

    def para(self, text: str) -> None:
        self.lines.append(text)
        self.lines.append("")

    def bullet(self, text: str) -> None:
        self.lines.append(f"- {text}" if self.md else f"  {text}")

    def note(self, text: str) -> None:
        """A footnote: what was left out and why."""
        self._blank()
        self.lines.append(f"*{text}*" if self.md else f"({text})")
        self.lines.append("")

    def pre(self, text: str) -> None:
        """Something already laid out in columns -- a `drafts` summary -- kept as it is."""
        if self.md:
            self.lines += ["```", *text.splitlines(), "```", ""]
        else:
            self.lines += [f"  {line}" for line in text.splitlines()] + [""]

    def table(self, headers: Sequence[str], rows: Sequence[Sequence[str]],
              best: int = -1) -> None:
        """A table. ``best`` is the index of the row to mark -- the one to serve."""
        marked = [[(self._strong(c) if n == best and m == 0 else c)
                   for m, c in enumerate(row)] for n, row in enumerate(rows)]
        if self.md:
            self.lines.append("| " + " | ".join(headers) + " |")
            self.lines.append("| " + " | ".join("---" for _ in headers) + " |")
            for row in marked:
                self.lines.append("| " + " | ".join(row) + " |")
        else:
            width = [max(len(str(h)), *(len(str(r[n])) for r in marked)) if marked
                     else len(str(h)) for n, h in enumerate(headers)]
            def line(cells: Sequence[str]) -> str:
                return "  " + "  ".join(str(c).ljust(width[n]) if n == 0
                                        else str(c).rjust(width[n])
                                        for n, c in enumerate(cells))
            self.lines.append(line(headers))
            self.lines.append("  " + "  ".join("-" * w for w in width))
            for row in marked:
                self.lines.append(line(row))
        self.lines.append("")

    def _strong(self, text: str) -> str:
        return f"**{text}**" if self.md else f"{text} *"

    def _blank(self) -> None:
        if self.lines and self.lines[-1] != "":
            self.lines.append("")

    def text(self) -> str:
        while self.lines and self.lines[-1] == "":
            self.lines.pop()
        return "\n".join(self.lines) + "\n"


# ---------------------------------------------------------------- the document itself

def report(kept: Sequence[Mapping[str, Any]], *, fits: Sequence[Any] = (),
           elsewhere: Sequence[tuple[str, Sequence[Any]]] = (), at: int = 32768,
           min_n: int = 6, full_n: int = 0, md: bool = True, noise: float = NOISE,
           room: str = "", store: str = "",
           extracted: Sequence[Mapping[str, Any]] = (),
           min_msgs: int = MIN_MESSAGES,
           ingested: Sequence[str] = ()) -> str:
    """Every measurement there is, as one document.

    ``fits`` are the memory records for this machine, ``elsewhere`` the same records asked
    about another room -- ``[(name, fits), ...]``, one per ``--room`` -- and ``at`` the
    per-user context the "how many fit" column answers at.

    ``kept`` is the answering runs and ``extracted`` the extraction runs, narrowed by the
    same window; ``ingested`` names the stores a ``--sources`` pointed at, read off those
    stores' own files. Each of the three prints no section when it is empty.
    """
    doc = Doc(md)
    doc.head(1, "What has been measured")
    if not kept and not extracted:
        doc.para("Nothing kept yet. `ml-stack-bench run LABEL` measures one asking; "
                 "`ml-stack-bench sweep` measures every model every way.")
        if ingested:
            _ingest(doc, ingested)
        _memory(doc, fits, elsewhere, at=at, room=room)
        return doc.text()

    everything = [*kept, *extracted]
    spans = sorted(str(one.get("at") or "") for one in everything if one.get("at"))
    machines = sorted(hosts_of(everything))
    doc.para(f"{len(kept)} run(s)"
             + (f" and {len(extracted)} extraction run(s)" if extracted else "")
             + (f" from `{store}`" if store else "")
             + (f", {spans[0]} to {spans[-1]}" if spans else "")
             + (f", on {', '.join(machines)}" if machines else "")
             + ". A conclusion drawn from kept runs, not a measurement: re-run "
               "`ml-stack-bench sweep` after any model release.")

    tables = answering(kept, min_n=min_n)
    _answering(doc, tables, min_n=min_n)
    _across(doc, kept, full_n=full_n)
    if extracted:
        _extraction(doc, extracted, min_msgs=min_msgs)
    if ingested:
        _ingest(doc, ingested)
    _drafts(doc, kept, noise=noise)
    _memory(doc, fits, elsewhere, at=at, room=room)
    _serving(doc, kept, tables, fits=fits, at=at, noise=noise)
    return doc.text()


def _answering(doc: Doc, tables: Mapping[str, tuple[list[Mapping[str, Any]], int]], *,
               min_n: int) -> None:
    doc.head(2, "Answering, per model")
    doc.para("One row per way a model was asked. `s/q` is the wall clock over the scored "
             "questions, so a short run compares with a full one; `made` is what F1 cannot "
             "see -- entries an answer named that no tool call ever found or read.")
    for model, (rows, short) in sorted(tables.items()):
        doc.head(3, f"`{model}`" if doc.md else model)
        if not rows:
            doc.para(f"Nothing measured at {min_n} questions or more.")
        else:
            doc.table(
                ("asking", "thinking", "cache", "draft", "n", "s/q", "F1", "recall",
                 "precision", "made"),
                [(asking_of(one.get("label")),
                  thinking_of(one.get("server") or {}),
                  cache_of(one.get("server") or {}),
                  of(one).head_said,
                  f"{derived(one)['questions']:.0f}",
                  f"{per_question(one):.1f}",
                  _pct(derived(one)["right"]),
                  _pct(derived(one)["recall"]),
                  _pct(derived(one)["precision"]),
                  of(one).made) for one in rows],
                best=0)
        if short:
            doc.note(f"{short} smoke and short run(s) of {model} left out: fewer than "
                     f"{min_n} scored questions, which proves the path works rather than "
                     f"how well it answers.")


def _across(doc: Doc, kept: Sequence[Mapping[str, Any]], *, full_n: int) -> None:
    ranked = across(kept, full_n=full_n)
    doc.head(2, "Across models")
    if not ranked:
        doc.para("No model has a run to rank.")
        return
    doc.para("Each model at its longest run, best F1 first -- what `ml-stack-bench show "
             "--rank` writes. `F1/1k tok` is accuracy over the tokens actually paid for "
             "(read plus written), so a model that is right more often for fewer tokens "
             "reads higher whatever its wall clock."
             + (f" Only runs of {full_n} question(s) or more." if full_n else ""))
    several = len(hosts_of([one for _, one in ranked])) > 1
    doc.table(
        ("model", "asking", "thinking", "draft", "n", "F1", "recall", "precision", "s/q",
         "F1/1k tok", *(("host",) if several else ())),
        [(f"`{model}`" if doc.md else model,
          asking_of(one.get("label")),
          thinking_of(one.get("server") or {}),
          of(one).head_said,
          f"{derived(one)['questions']:.0f}",
          _pct(derived(one)["right"]),
          _pct(derived(one)["recall"]),
          _pct(derived(one)["precision"]),
          f"{per_question(one):.1f}",
          (f"{derived(one)['right_per_1k']:.4f}" if "right_per_1k" in derived(one) else "-"),
          *((host_of(one) or "-",) if several else ()))
         for model, one in ranked],
        best=0)


def _extraction(doc: Doc, extracted: Sequence[Mapping[str, Any]], *, min_msgs: int) -> None:
    from ml_stack.bench.extract import detail
    from ml_stack.hub import pretty_name

    rows, short = extractions(extracted, min_msgs=min_msgs)
    doc.head(2, "Extraction")
    doc.para("One row per `ml-stack-bench extract` run -- reading a graph *out of* "
             "messages rather than answering questions about one -- newest first, so a "
             "row reads against the row under it. Coverage and precision stay separate "
             "columns because they are fixed by opposite changes to the asking: a model "
             "that misses half the relations and one that invents twice as many can share "
             "an F1. `invented` is the share of extracted people and organisations naming "
             "nothing in the world -- the hallucination rate, and the number that moved "
             "most when the instructions were given the vocabulary to use.")
    if not rows:
        doc.para(f"Nothing read at {min_msgs} message(s) or more.")
    else:
        best = best_extractor(rows)
        doc.table(
            ("run", "model", "msgs", "s/msg", "tok/msg", "n-F1", "r-F1", "top-prec",
             "rel-cov", "rel-prec", "invented", "resident"),
            [_extraction_row(one, doc=doc, pretty=pretty_name) for one in rows],
            best=next((n for n, one in enumerate(rows) if one is best), -1))
        for one in rows:
            said = [line for line in detail(one.get("scores") or {})
                    if line.startswith(("topology:", "conformance:"))]
            if said:
                label = str(one.get("label") or "?")
                doc.bullet(f"{f'`{label}`' if doc.md else label} — " + "; ".join(said))
        doc.lines.append("")
        if best is not None:
            rel = _scores(best, "relations")
            name = pretty_name(extract_model_of(best))
            doc.para(
                f"Best at relations: {f'**{name}**' if doc.md else name} at "
                f"{_pct(rel.get('f1'))} relation F1 over {len(read_messages(best))} "
                f"message(s) (`{best.get('label') or '?'}`), read among the runs that read "
                "the most messages -- a coverage over ten messages and one over forty are "
                "not the same measurement, and ranking them together would name whichever "
                "model was asked less.")
    if short:
        doc.note(f"{short} smoke and short extraction run(s) left out: fewer than "
                 f"{min_msgs} messages read, which proves the path works rather than how "
                 f"well it reads.")


def _extraction_row(one: Mapping[str, Any], *, doc: Doc, pretty: Any) -> tuple[str, ...]:
    """One run's cells. Nothing is derived that the run does not carry: a rate the scores
    left out prints "-" rather than being recomputed from counts that may have been scored
    against a different gold."""
    rows = read_messages(one)
    n = len(rows)
    seconds = sum(float(r.get("seconds") or 0) for r in rows)
    tokens = sum(int(r.get("prompt_tokens") or 0) + int(r.get("completion_tokens") or 0)
                 for r in rows)
    nodes, rel = _scores(one, "nodes"), _scores(one, "relations")
    topics = (_scores(one, "by_kind") or {}).get("topics") or {}
    made_up = _scores(one, "invented")
    label, model = str(one.get("label") or "?"), pretty(extract_model_of(one))
    return (label,
            f"`{model}`" if doc.md else model,
            f"{n}",
            f"{(seconds / n if n else 0):.1f}",
            f"{(tokens / n if n else 0):.0f}",
            _pct(nodes.get("f1")),
            _pct(rel.get("f1")),
            _pct(topics.get("precision")),
            _pct(rel.get("coverage")),
            _pct(rel.get("precision")),
            _pct(made_up.get("rate")),
            _gb((one.get("server") or {}).get("resident_bytes")))


# ---------------------------------------------------------------- the ingested sources

def _run_nodes(view: Any) -> dict[str, Mapping[str, Any]]:
    """``{run id: its attrs}`` off the store -- the hidden ``run`` nodes
    ``ml_stack.ingest.write_run`` hangs units on. Empty where there is no store yet."""
    if not view.out.exists():
        return {}
    try:
        with view.store() as store:
            return {str(node["id"]): node.get("attrs") or {} for node in store.nodes(kind="run")}
    except Exception:
        return {}


def _decisions_count(view: Any) -> int | None:
    """How many name pairs the store's hygiene pass has judged, or None for no such
    document -- a store never tidied says so rather than showing a 0 it never measured."""
    from ml_stack.graph.verdicts import DECISIONS

    if not view.out.exists():
        return None
    try:
        with view.store() as store:
            held = store.get_doc(DECISIONS)
    except Exception:
        return None
    pairs = (held or {}).get("pairs") if isinstance(held, Mapping) else None
    return len(pairs) if isinstance(pairs, Mapping) else None


def _runs_said(run_nodes: Mapping[str, Mapping[str, Any]], run_ids: Sequence[str]) -> str:
    """The run(s) that read a source: model, serving and when, off the run node each names.
    A run id the store holds no node for -- a unit read before the fold caught up with it
    -- prints the bare id rather than nothing."""
    said = []
    for run_id in run_ids:
        attrs = run_nodes.get(run_id)
        if attrs:
            said.append(f"{attrs.get('model') or '?'} / {attrs.get('serving') or '?'} "
                        f"({attrs.get('started') or '?'})")
        else:
            said.append(run_id)
    return "; ".join(said) if said else "-"


def _ingest_row(view: Any, one: Any, run_nodes: Mapping[str, Mapping[str, Any]]
               ) -> tuple[tuple[str, ...], dict[str, float]]:
    """One source's row, and what it adds to the store's total line.

    Tokens are summed from the calls each read made -- ``read.calls``, telemetry the model
    server itself reported -- never estimated from a rate: a unit whose reply carried no
    usage counts as zero rather than the store's average.
    """
    rows = view.reads(one.slug)
    calls = [call for row in rows for call in (row.get("calls") or ())]
    prompt = sum(int(c.get("prompt_tokens") or 0) for c in calls)
    completion = sum(int(c.get("completion_tokens") or 0) for c in calls)
    run_ids = sorted({str(r.get("run") or "") for r in rows if r.get("run")})
    row = (one.title or one.slug,
           f"{one.read}/{one.wanted or '?'}",
           str(one.failed),
           str(one.given_up),
           str(one.folded_nodes),
           str(one.folded_edges),
           f"{one.seconds:.0f}",
           f"{one.per_unit:.1f}",
           str(prompt),
           str(completion),
           f"{((prompt + completion) / one.units if one.units else 0):.0f}",
           _runs_said(run_nodes, run_ids))
    totals = {"read": one.read, "wanted": one.wanted, "failed": one.failed,
             "given_up": one.given_up, "units": one.units, "seconds": one.seconds,
             "prompt": prompt, "completion": completion}
    return row, totals


def _ingest_store(doc: Doc, where: str, view: Any, listed: Sequence[Any]) -> None:
    doc.head(3, f"`{where}`" if doc.md else where)
    run_nodes = _run_nodes(view)
    rows, totalled = [], []
    for one in sorted(listed, key=lambda s: (s.title or s.slug).lower()):
        row, totals = _ingest_row(view, one, run_nodes)
        rows.append(row)
        totalled.append(totals)
    doc.table(("source", "read", "failed", "given up", "nodes", "edges", "seconds", "s/unit",
              "prompt tok", "completion tok", "tok/unit", "run(s)"), rows)
    total = {key: sum(t[key] for t in totalled)
            for key in ("read", "wanted", "failed", "given_up", "units", "seconds",
                        "prompt", "completion")}
    tokens = total["prompt"] + total["completion"]
    line = (f"{len(listed)} source(s), {total['read']}/{total['wanted'] or '?'} unit(s) read, "
           f"{total['failed']} failed ({total['given_up']} given up), "
           f"{total['seconds']:.0f}s "
           f"({(total['seconds'] / total['units'] if total['units'] else 0):.1f} s/unit), "
           f"{total['prompt']} prompt and {total['completion']} completion token(s) "
           f"({(tokens / total['units'] if total['units'] else 0):.0f} tok/unit).")
    decisions = _decisions_count(view)
    if decisions is not None:
        line += f" {decisions} pair(s) of names judged for merge."
    doc.para(line)


def _ingest(doc: Doc, ingested: Sequence[str]) -> None:
    from ml_stack.ingest import Sources

    named = [(where, Sources(where)) for where in ingested]
    named = [(where, view, view.sources()) for where, view in named]
    named = [one for one in named if one[2]]
    if not named:
        return
    doc.head(2, "Ingest")
    doc.para("One row per source in a `--sources` store, from `ml_stack.ingest` rather than "
             "from a bench run: how many of its units are read, what is in the store as of "
             "the last fold, what reading it cost, and the run(s) -- model, serving, when "
             "-- that read it. `tok/unit` is summed from every call's own usage, never "
             "estimated from a rate.")
    for where, view, listed in named:
        _ingest_store(doc, where, view, listed)


def _drafts(doc: Doc, kept: Sequence[Mapping[str, Any]], *, noise: float) -> None:
    grouped = by_model(kept)
    with_heads = {model: mine for model, mine in grouped.items()
                  if any(of(one).head for one in mine)}
    doc.head(2, "Draft heads, per model")
    if not with_heads:
        doc.para("No draft head measured. `ml-stack-bench drafts MODEL --draft HEAD` "
                 "measures one, and `--draft \"\"` measures the baseline it is read "
                 "against.")
        return
    doc.para("What each head was worth against the same model's newest undrafted run of "
             "the same size on the same build. A head cannot change an answer -- the "
             "target verifies every token -- so one whose F1 moved changed something else, "
             "and it is on the table rather than in the recommendation.")
    for model, mine in sorted(with_heads.items()):
        doc.head(3, f"`{model}`" if doc.md else model)
        doc.pre(drafted(mine, among=list(kept), noise=noise))


def _memory(doc: Doc, fits: Sequence[Any],
            elsewhere: Sequence[tuple[str, Sequence[Any]]], *, at: int,
            room: str) -> None:
    from ml_stack.serve import fit as fit_mod

    doc.head(2, "Memory")
    if not fits and not any(rows for _, rows in elsewhere):
        doc.para("Nothing measured. `ml-stack-serve fit MODEL --measure` serves it once "
                 "and records what it allocated.")
        return
    if fits:
        doc.para(f"How many conversations fit at {at:,} tokens each"
                 + (f", on this machine ({room})" if room else "") + ", and what one "
                 "more costs. The weights are paid for once; the cache is paid for per "
                 "slot per token, which is what decides the head count.")
        doc.table(("model", "cache", "draft", "loaded", "each user", f"users at {at:,}",
                   "one user, longest"),
                  [(f"`{one.model}`" if doc.md else one.model,
                    one.cache_type,
                    one.spec or "-",
                    human_bytes(one.loaded()),
                    human_bytes(one.cost(at)),
                    str(one.users(at)),
                    f"{one.longest(1):,}") for one in fits],
                  best=-1)
    for name, rows in elsewhere:
        if not rows:
            continue
        doc.head(3, f"A machine with {name}")
        doc.table(("model", "cache", "draft", f"users at {at:,}", "one user, longest"),
                  [(f"`{one.model}`" if doc.md else one.model, one.cache_type,
                    one.spec or "-", str(one.users(at)), f"{one.longest(1):,}")
                   for one in rows],
                  best=-1)
    if fits:
        doc.para("Every record in full -- what was read off the load itself, not "
                 "estimated:")
        if doc.md:
            doc.lines += fit_mod.render(fits, md=True).splitlines() + [""]
        else:
            doc.pre(fit_mod.render(fits, md=False))


def _serving(doc: Doc, kept: Sequence[Mapping[str, Any]],
             tables: Mapping[str, tuple[list[Mapping[str, Any]], int]], *,
             fits: Sequence[Any], at: int, noise: float) -> None:
    doc.head(2, "What to serve")
    doc.para("One line per model, composed from the three tables above. A part nothing "
             "measured says so: a guess here would be read as a measurement.")
    grouped = by_model(kept)
    for model, (rows, _short) in sorted(tables.items()):
        if not rows:
            doc.bullet(f"{f'`{model}`' if doc.md else model} — not measured at length.")
            continue
        best = rows[0]
        server = best.get("server") or {}
        mine = grouped.get(model, [])
        head_run = recommended_head(mine, list(kept))
        # three different states, and only one of them is a head: nothing measured at all,
        # measured and none worth serving, and one to serve. Printing the first two the
        # same way would read as "no head is best" when nothing had been tried
        if not any(of(one).head for one in mine):
            head = "not measured"
        elif head_run is None:
            head = "none -- no head held its baseline's F1 and beat serving none"
        else:
            head = of(head_run).head_said
        cost = head_run if head_run is not None else best
        fit = fit_for(model, fits)
        got = derived(best)
        cache = cache_of(server)
        doc.bullet(
            f"{f'`{model}`' if doc.md else model} — serve with: "
            f"{asking_of(best.get('label'))}, "
            f"thinking {thinking_of(server)}, "
            f"head {head}, "
            f"cache {cache if cache != '-' else 'f16'}; "
            f"expect ~{per_question(cost):.1f} s/q at F1 {_pct(got['right'])} "
            f"(n={got['questions']:.0f}); "
            + (f"{fit.users(at)} users at {at:,} tokens on this machine."
               if fit is not None else f"users at {at:,} tokens: not measured."))
    doc.lines.append("")


# ---------------------------------------------------------------- the subcommand

def fits_named(fits: Iterable[Any], wanted: Sequence[str]) -> list[Any]:
    """The fit records whose model file matches one of ``wanted``; all of them for none."""
    if not wanted:
        return list(fits)
    low = [w.lower() for w in wanted]
    return [f for f in fits if any(w in str(f.model).lower() for w in low)]


def main(args: Any) -> int:
    """``ml-stack-bench report``. Reads the store and the fit records; serves nothing."""
    from ml_stack.bench import extract as bench_extract
    from ml_stack.bench.ops import newest
    from ml_stack.serve import fit as fit_mod

    store = str(getattr(args, "kept", "") or "")
    everything = bench.runs(store) if store and Path(store).expanduser().exists() else []
    # an extraction run is kept in the same store and is not an answering run: it has no
    # question to score and no F1 to rank, so the two are narrowed apart and tabled apart.
    # They used to be *dropped* here instead, which is why the document could say nothing
    # about extraction at all
    kept = [r for r in everything if r.get("kind") != bench_extract.KIND]
    extracted = bench_extract.only(everything)
    # the window is applied to each kind on its own: `--last 3` means the three newest of
    # each, not three rows shared out between them, where a busy afternoon of answering
    # runs would silently empty the extraction table
    last, since = (int(getattr(args, "last", 0) or 0),
                   str(getattr(args, "since", "") or ""))
    kept = newest(kept, last=last, since=since)
    extracted = newest(extracted, last=last, since=since)
    wanted = [str(w).lower() for w in (getattr(args, "model", None) or [])]
    if wanted:
        kept = [r for r in kept
                if any(w in model_of(r).lower() or w in str(r.get("label") or "").lower()
                       for w in wanted)]
        extracted = [r for r in extracted
                     if any(w in extract_model_of(r).lower()
                            or w in str(r.get("label") or "").lower() for w in wanted)]

    if getattr(args, "profile", False):
        # the file the serve path and the asking path read, set from the store rather than
        # by hand -- see `write_profiles`. Nothing is printed but what was written, because
        # a record quietly rewritten is the one thing here nobody would notice
        written = write_profiles(kept, full_n=int(getattr(args, "full_n", 0) or 0),
                                 path=(Path(str(getattr(args, "profiles", "") or "")).expanduser()
                                       if getattr(args, "profiles", "") else None))
        if not written:
            warn("no model has a run to write a profile from")
            return 1
        for one, where in written:
            say(f"{one.model}: {one.label or '?'} "
                f"({one.questions} q, {one.right * 100:.0f}% F1, "
                f"{one.seconds_per_question:.1f} s/q) -> {where}")
        profiled = {one.model for one, _where in written}
        for model, mine in by_model(kept).items():
            longest = max((derived(o)["questions"] for o in mine if derived(o)), default=0)
            if model not in profiled and 0 < longest < SHORT:
                warn(f"{model}: not profiled -- its longest run is {int(longest)} question(s), "
                     f"and a record is never set from fewer than {SHORT}")
        return 0

    rooms: list[int] = []
    for said in (getattr(args, "room", None) or []):
        try:
            rooms.append(fit_mod.parse_room(said))
        except ValueError as exc:
            warn(f"error: {exc}")
            return 2

    from ml_stack.hub import room as machine_room

    here = machine_room()
    fits = fits_named(fit_mod.records(room=here), wanted)
    elsewhere = [(human_bytes(size),
                  [f.at_room(size) for f in fits]) for size in rooms if size != here]

    body = report(kept, fits=fits, elsewhere=elsewhere,
                  at=int(getattr(args, "at", 32768) or 32768),
                  min_n=int(getattr(args, "min_n", 6) or 0),
                  full_n=int(getattr(args, "full_n", 0) or 0),
                  md=not bool(getattr(args, "text", False)),
                  noise=float(getattr(args, "noise", NOISE * 100) or 0) / 100,
                  room=human_bytes(here) if here else "", store=store,
                  extracted=extracted, ingested=list(getattr(args, "sources", None) or ()),
                  min_msgs=int(getattr(args, "min_msgs", MIN_MESSAGES) or MIN_MESSAGES))

    where = str(getattr(args, "md", "") or "")
    if not where:
        say(body, end="")
        return 0
    out = Path(where).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(body, encoding="utf-8")
    say(f"wrote {out}")
    if getattr(args, "open", False):
        from ml_stack.platform import open_path

        say(f"opened with {open_path(out)}")
    return 0

