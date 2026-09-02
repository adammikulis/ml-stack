"""Everything measured so far, as one document: how each model was asked, what a draft
head was worth, how much memory it wants, and what to serve.

`show` prints evidence -- one line per run, every column that could matter. This prints the
*conclusion a person asked for*: "a nice table with all the stats you have so far" was
composed by hand once, out of `show`, the `drafts` summaries in a log and
`ml-stack-serve fit`, and composing it by hand is exactly the step that becomes a command.

Nothing here measures anything or serves anything. It reads the kept runs and the fit
records and arranges them:

1. **Answering, per model** -- one table per model file, a row per way it was asked;
2. **Across models** -- the best row of each, which is what `show --rank` writes;
3. **Draft heads** -- the `drafts` summary and its recommendation, per model;
4. **Memory** -- the fit records, at this machine's room and at each ``--room``;
5. **What to serve** -- one line per model composing 1, 3 and 4.

A part that was never measured says "not measured". Nothing here guesses: a report that
filled a gap with a plausible number would be read as a measurement, and the whole point of
the store is that a number in it was paid for.
"""

from __future__ import annotations

import re
import sys
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

# The package is the namespace the tests and `selfcheck` patch -- `bench.runs`, `bench.HOME`
# -- so anything patchable is looked up there at call time, never bound here at import.
from ml_stack.graph import bench
from ml_stack.graph.bench.score import (
    NOISE,
    _head_of,
    derived,
    host_of,
    hosts_of,
    per_question,
)
from ml_stack.graph.bench.show import drafted, kv_short, made

__all__ = ["ASKINGS", "Doc", "across", "answering", "asking_of", "by_model",
           "cache_of", "fit_for", "fits_named", "head_of", "model_of", "recommended_head",
           "report", "thinking_of"]


# The words a sweep puts in a label for the way it asked (`bench.halves`, `bench._ways`).
# ``shortlist`` is here beside ``plain`` although it is a half rather than a way: without
# it `shortlist-terse` and `plain-terse` read as the same row, which is two measurements
# printed as one -- the mistake every column in `show`'s table exists to prevent.
ASKINGS = ("plain", "shortlist", "terse", "card", "greedy", "rich", "tight")

_WORD = re.compile(r"[-_:@.]+")

# `drafted`'s recommendation, whose label may itself contain a colon
_RECOMMENDED = re.compile(r"^serve (.+?): fastest whose F1 held")


def model_of(one: Mapping[str, Any]) -> str:
    """The model file a run was served from -- what groups runs into tables. "?" when the
    run was kept before the server record named one."""
    return str((one.get("server") or {}).get("model") or "?")


def asking_of(label: Any) -> str:
    """The way a run was asked, read out of its label: ``plain+terse``, ``shortlist``.

    The label is where this lives and the only place it lives -- `served` composes it from
    the half and the ``--also`` and keeps no separate record -- so it is read back the same
    way, by whole word, never by substring: a model called ``tightfit`` is not a ``tight``
    asking.
    """
    words = [w for w in _WORD.split(str(label or "").lower()) if w in ASKINGS]
    return "+".join(dict.fromkeys(words)) or "-"


def thinking_of(server: Mapping[str, Any]) -> str:
    """``on`` when nothing bound the model's thinking, ``off`` at a budget of zero, else
    the budget itself. A run served with a reasoning budget is another configuration, and
    a budget of 0 is not the same measurement as no budget at all."""
    budget = (server or {}).get("reasoning_budget")
    if budget is None:
        return "on"
    return "off" if int(budget) == 0 else str(int(budget))


def cache_of(server: Mapping[str, Any]) -> str:
    """The KV cache type when it was quantised, "-" at f16. A quantised cache against an
    f16 one is two configurations, not two models."""
    kind = str((server or {}).get("cache_type") or "")
    return kv_short(kind) if kind and kind != "f16" else "-"


def head_of(one: Mapping[str, Any]) -> str:
    """The draft head a run was served with and how far it guessed -- ``mtp-a.gguf@n4`` --
    or "-" for none."""
    head = _head_of(one)
    if not head:
        return "-"
    ahead = (one.get("server") or {}).get("spec_draft_max")
    return f"{head}@n{int(ahead)}" if ahead is not None else head


def _pct(value: float | None) -> str:
    return f"{value * 100:.0f}%" if value is not None else "-"


def by_model(kept: Sequence[Mapping[str, Any]]) -> dict[str, list[Mapping[str, Any]]]:
    """The scored runs grouped by the model file they were served from, newest last.

    A run with no scored question is not an answering run and is left out: it has no F1,
    and a row of dashes said nothing about why.
    """
    out: dict[str, list[Mapping[str, Any]]] = {}
    for one in kept:
        if not derived(one):
            continue
        out.setdefault(model_of(one), []).append(one)
    return out


def answering(kept: Sequence[Mapping[str, Any]], *, min_n: int = 6
              ) -> dict[str, tuple[list[Mapping[str, Any]], int]]:
    """Per model: the runs long enough to read, best F1 first, and how many were too short.

    A run of two questions is a smoke run proving the path works; its score is a coin toss
    by construction, and put in the table beside a full run it is read as a measurement.
    So it is counted in a footnote instead of printed.
    """
    out: dict[str, tuple[list[Mapping[str, Any]], int]] = {}
    for model, mine in by_model(kept).items():
        long_enough = [o for o in mine if derived(o)["questions"] >= min_n]
        long_enough.sort(key=lambda o: (-derived(o)["right"], str(o.get("at") or "")))
        out[model] = (long_enough, len(mine) - len(long_enough))
    return out


def across(kept: Sequence[Mapping[str, Any]], *, full_n: int = 0
           ) -> list[tuple[str, Mapping[str, Any]]]:
    """The best run of each model, most accurate first -- the ranking, as data.

    "Best" is the best F1 among the model's *longest* runs, because a score is only
    comparable with another over the same questions: twenty scored questions make each one
    worth five points of F1 and fifty make it two, so the short run and the full one are
    two measurements that must not be sorted against each other. ``full_n`` fixes the floor
    across every model; unset, each model is read at the largest run it has.
    """
    out = []
    for model, mine in by_model(kept).items():
        floor = full_n or max(derived(o)["questions"] for o in mine)
        pool = [o for o in mine if derived(o)["questions"] >= floor]
        if not pool:
            continue
        out.append((model, max(pool, key=lambda o: (derived(o)["right"],
                                                    str(o.get("at") or "")))))
    return sorted(out, key=lambda pair: -derived(pair[1])["right"])


def recommended_head(mine: Sequence[Mapping[str, Any]],
                     among: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    """The drafted run `drafted` recommends serving, or None for "serve no head".

    Read out of `drafted`'s own last line rather than re-derived here: the rule -- the
    fastest head whose F1 held within the noise of its baseline, and only if it beat no
    head at all -- belongs in one place, and a second copy of it would be a second thing to
    keep in step. A summary that stops saying "serve LABEL" makes this return None, which
    reads as "not measured" rather than as a wrong recommendation.
    """
    if not any(_head_of(o) for o in mine):
        return None
    said = _RECOMMENDED.match(drafted(mine, among=among).splitlines()[-1])
    if not said:
        return None
    # matched to the end of the label rather than to its first colon: a drafts label is
    # `draft:mtp-alder@n4`, and splitting on the colon recommended a run called "draft"
    return next((o for o in mine if str(o.get("label") or "") == said.group(1)), None)


def fit_for(model: str, fits: Sequence[Any]) -> Any | None:
    """The fit record measured for this model file, or None.

    By file name first and by substring after, because the two sides name a model from
    different ends: a run records what ``/props`` called the file it served, and a fit
    record is keyed on the basename it was measured under.
    """
    name = str(model).lower()
    for one in fits:
        if str(getattr(one, "model", "")).lower() == name:
            return one
    for one in fits:
        held = str(getattr(one, "model", "")).lower()
        if held and (held in name or name in held):
            return one
    return None


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
           room: str = "", store: str = "") -> str:
    """Every measurement there is, as one document. See the module docstring for the parts.

    ``fits` are the memory records for this machine, ``elsewhere`` the same records asked
    about another room -- ``[(name, fits), ...]``, one per ``--room``. ``at`` is the
    per-user context the "how many fit" column answers at.
    """
    doc = Doc(md)
    doc.head(1, "What has been measured")
    if not kept:
        doc.para("Nothing kept yet. `ml-stack-bench run LABEL` measures one asking; "
                 "`ml-stack-bench sweep` measures every model every way.")
        _memory(doc, fits, elsewhere, at=at, room=room)
        return doc.text()

    spans = sorted(str(one.get("at") or "") for one in kept if one.get("at"))
    machines = sorted(hosts_of(kept))
    doc.para(f"{len(kept)} run(s)"
             + (f" from `{store}`" if store else "")
             + (f", {spans[0]} to {spans[-1]}" if spans else "")
             + (f", on {', '.join(machines)}" if machines else "")
             + ". A conclusion drawn from kept runs, not a measurement: re-run "
               "`ml-stack-bench sweep` after any model release.")

    tables = answering(kept, min_n=min_n)
    _answering(doc, tables, min_n=min_n)
    _across(doc, kept, full_n=full_n)
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
                  head_of(one),
                  f"{derived(one)['questions']:.0f}",
                  f"{per_question(one):.1f}",
                  _pct(derived(one)["right"]),
                  _pct(derived(one)["recall"]),
                  _pct(derived(one)["precision"]),
                  made(one) or "-") for one in rows],
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
          head_of(one),
          f"{derived(one)['questions']:.0f}",
          _pct(derived(one)["right"]),
          _pct(derived(one)["recall"]),
          _pct(derived(one)["precision"]),
          f"{per_question(one):.1f}",
          (f"{derived(one)['right_per_1k']:.4f}" if "right_per_1k" in derived(one) else "-"),
          *((host_of(one) or "-",) if several else ()))
         for model, one in ranked],
        best=0)


def _drafts(doc: Doc, kept: Sequence[Mapping[str, Any]], *, noise: float) -> None:
    grouped = by_model(kept)
    with_heads = {model: mine for model, mine in grouped.items()
                  if any(_head_of(one) for one in mine)}
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
                    fit_mod._human(one.loaded()),
                    fit_mod._human(one.cost(at)),
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
        if not any(_head_of(one) for one in mine):
            head = "not measured"
        elif head_run is None:
            head = "none -- no head held its baseline's F1 and beat serving none"
        else:
            head = head_of(head_run)
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
    from ml_stack.graph.bench import extract as bench_extract
    from ml_stack.graph.bench.run import newest
    from ml_stack.serve import fit as fit_mod

    store = str(getattr(args, "kept", "") or "")
    everything = bench.runs(store) if store and Path(store).expanduser().exists() else []
    # an extraction run is kept in the same store and is not an answering run: it has no
    # question to score, and `extract`'s own table is where it belongs
    kept = [r for r in everything if r.get("kind") != bench_extract.KIND]
    kept = newest(kept, last=int(getattr(args, "last", 0) or 0),
                  since=str(getattr(args, "since", "") or ""))
    wanted = [str(w).lower() for w in (getattr(args, "model", None) or [])]
    if wanted:
        kept = [r for r in kept
                if any(w in model_of(r).lower() or w in str(r.get("label") or "").lower()
                       for w in wanted)]

    rooms: list[int] = []
    for said in (getattr(args, "room", None) or []):
        try:
            rooms.append(fit_mod.parse_room(said))
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2

    from ml_stack.hub import room as machine_room

    here = machine_room()
    fits = fits_named(fit_mod.records(room=here), wanted)
    elsewhere = [(fit_mod._human(size),
                  [f.at_room(size) for f in fits]) for size in rooms if size != here]

    body = report(kept, fits=fits, elsewhere=elsewhere,
                  at=int(getattr(args, "at", 32768) or 32768),
                  min_n=int(getattr(args, "min_n", 6) or 0),
                  full_n=int(getattr(args, "full_n", 0) or 0),
                  md=not bool(getattr(args, "text", False)),
                  noise=float(getattr(args, "noise", NOISE * 100) or 0) / 100,
                  room=fit_mod._human(here) if here else "", store=store)

    where = str(getattr(args, "md", "") or "")
    if not where:
        print(body, end="")
        return 0
    out = Path(where).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(body, encoding="utf-8")
    print(f"wrote {out}")
    if getattr(args, "open", False):
        from ml_stack.platform import open_path

        print(f"opened with {open_path(out)}")
    return 0

