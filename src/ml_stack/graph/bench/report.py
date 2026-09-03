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
3. **Extraction** -- one row per `extract` run, newest first, with the topology and the
   conformance under it; printed only when the window holds one;
4. **Draft heads** -- the `drafts` summary and its recommendation, per model;
5. **Memory** -- the fit records, at this machine's room and at each ``--room``;
6. **What to serve** -- one line per model composing 1, 4 and 5.

Extraction is here because the day it was left out is the day the document could not hold
the cause-then-fix record it exists for: a model read topics at 19% precision and relations
at 0% F1 with 26% invented ids, the extraction instructions were given the topic and the
relation vocabulary, and the same model read 67% / 62% / 7% after. That is a measurement, a
cause we controlled and a re-measurement, and it lived only in a terminal because the report
filtered the runs out before it read them.

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
from ml_stack.graph.bench.keep import SHORT
from ml_stack.graph.bench.score import (
    NOISE,
    _head_of,
    derived,
    held_up,
    host_of,
    hosts_of,
    per_question,
)
from ml_stack.graph.bench.show import _gb, drafted, kv_short, made

__all__ = ["ASKINGS", "Doc", "MIN_MESSAGES", "WAYS", "across", "answering", "asking_of",
           "best_extractor", "build_of", "by_model", "cache_of", "extract_model_of",
           "extractions", "fit_for", "fits_named", "head_of", "measured_best", "model_of",
           "profile_of", "read_messages", "recommended_head", "report", "thinking_of",
           "ways_of", "write_profiles"]


# The words a sweep puts in a label for the way it asked (`bench.halves`, `bench._ways`).
# ``shortlist`` is here beside ``plain`` although it is a half rather than a way: without
# it `shortlist-terse` and `plain-terse` read as the same row, which is two measurements
# printed as one -- the mistake every column in `show`'s table exists to prevent.
ASKINGS = ("plain", "shortlist", "terse", "card", "greedy", "rich", "tight", "reach")

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


# ---------------------------------------------------------------- the extraction runs

# How many messages an extraction run reads before its scores are read as a measurement,
# and `min_n`'s opposite number for the other half of the bench. `extract.SMOKE_MESSAGES`
# is three, and three messages fix every coverage to a third: a run that missed one thing
# reads 67%, which is not a rate but an arithmetic accident of how few it was asked.
MIN_MESSAGES = 10


def extract_model_of(one: Mapping[str, Any]) -> str:
    """The model an extraction run read with, "?" for a run that names none.

    Its own top-level ``model`` first, because that is where `extract.save` writes it --
    already the file's basename with the ``.gguf`` off -- and the server record only after.
    An answering run keeps the same fact under ``server.model`` and `model_of` reads it
    there; the two are separate functions rather than one that guesses, since a run that
    named neither would otherwise be grouped under whatever the other kind happened to say.
    """
    return str(one.get("model") or (one.get("server") or {}).get("model") or "?")


def read_messages(one: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """The rows of an extraction run whose gold is exact -- the messages its scores were
    measured over.

    `extract.measure` scores the template-written messages and puts the model-written ones
    in ``lower_bound``, and `extract.table` counts the run's messages the same way. So does
    this: a run's `s/msg` counted over rows its coverage was not measured over is two
    numbers over two different sets printed as one row.
    """
    return [r for r in (one.get("rows") or ()) if r.get("exact", True)]


def _scores(one: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    return (one.get("scores") or {}).get(key) or {}


def extractions(kept: Sequence[Mapping[str, Any]], *, min_msgs: int = MIN_MESSAGES
                ) -> tuple[list[Mapping[str, Any]], int]:
    """The extraction runs among ``kept``, newest first, and how many were too short.

    Newest first rather than best first, unlike `answering`: an extraction run is read as a
    record of what changed -- an instruction rewritten, a vocabulary defined -- and the
    order that shows a change is the order it happened in, latest at the top. Which model
    read best is a separate sentence, `best_extractor`, so the ordering never has to carry
    two jobs at once.

    A run of three messages is a smoke run proving the path works, for `answering`'s reason
    exactly: it is counted here and footnoted rather than tabled beside a full one.

    The key breaks a tie, not the order the store gave them back: `bench.runs` returns runs
    sorted by key, and a key begins with the label, so two runs kept inside the same second
    would be ordered by whatever they were called. The key's own tail is the run's stamp
    and the suffix `save` adds when one second held two, which is the only record of which
    came second.
    """
    from ml_stack.graph.bench.extract import only

    mine = only(kept)
    long_enough = [one for one in mine if len(read_messages(one)) >= min_msgs]
    long_enough.sort(key=lambda one: (str(one.get("at") or ""), str(one.get("key") or "")),
                     reverse=True)
    return long_enough, len(mine) - len(long_enough)


def best_extractor(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    """The run that read a graph out of messages best: the highest relation F1 among those
    that read the most messages. None for nothing to read.

    By relations rather than by nodes because naming the right people and joining none of
    them is the failure this half of the bench exists to catch -- a model can list every
    name in a message and state no relation at all, and its node F1 will not say so.

    Among the longest runs only, for `across`'s reason: a coverage over ten messages and one
    over forty are not the same measurement, and sorting them against each other rewards
    whichever was asked less. Ties go to the later run, which is the one measured against
    whatever changed last.
    """
    if not rows:
        return None
    floor = max(len(read_messages(one)) for one in rows)
    pool = [one for one in rows if len(read_messages(one)) >= floor]
    return max(pool, key=lambda one: (float(_scores(one, "relations").get("f1") or 0.0),
                                      str(one.get("at") or "")))


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


# ---------------------------------------------------------------- the record it all sets

# Every word a label can carry about the asking, and what it means to `converse`. Wider
# than `ASKINGS`, which is only what the tables print: `batch`, `kinds` and `summary` ride
# on a way rather than naming one, and a profile has to carry them or a model measured with
# all three would be served with none.
WAYS = ("tight", "batch", "single", "few", "kinds", "summary", "rich", "terse", "reach",
        "rounds")


def ways_of(one: Mapping[str, Any]) -> dict[str, Any]:
    """The asking a run records, as the fields of a profile.

    A run kept since `asked_with` carries ``asking`` -- the keywords `converse` was actually
    handed -- and that is taken as it is: it is the record, and reading a label instead
    would be inferring what is already written down.

    Older runs have only the label, so it is read by whole word, never by substring -- a
    model called ``tightfit`` is not a ``tight`` asking. ``loose`` is the one word that
    means the *absence* of a way: it is the control the ranking runs were measured with,
    and it is how ``tight=False`` is said.
    """
    said = one.get("asking")
    if isinstance(said, Mapping) and said:
        out: dict[str, Any] = {"tight": bool(said.get("tight", True))}
        for way in ("batch", "kinds", "summary", "rich", "terse", "single", "few"):
            out[way] = bool(said.get(way, False))
        if said.get("reach"):
            out["reach"] = int(said["reach"])
        if said.get("rounds"):
            out["rounds"] = int(said["rounds"])
        return out
    words = {w for w in _WORD.split(str(one.get("label") or "").lower()) if w}
    out = {"tight": "loose" not in words}
    for way in ("batch", "kinds", "summary", "rich", "terse", "single", "few"):
        out[way] = way in words
    if "reach" in words:
        from ml_stack.graph.bench.run import REACH

        # the label says a run reached; it does not say how far, and `--also reach` is the
        # only thing that puts the word there, so its own figure is what was measured
        out["reach"] = int(REACH)
    return out


def build_of(server: Mapping[str, Any]) -> str:
    """The named llama.cpp build a run was served on, "" for the managed current one.

    A run records the binary's path, because that is what it started; a profile records the
    *name*, because that is what `ml-stack-serve up --build` takes. A head withheld from
    mainline loads on one build and no other, so this is not decoration.
    """
    from ml_stack.serve.build import NAMED_DIR

    binary = Path(str((server or {}).get("binary") or ""))
    try:
        named = binary.resolve().relative_to(Path(NAMED_DIR).resolve())
    except (OSError, ValueError):
        # not under the named builds: `current`, a hand-named binary, or a path that no
        # longer exists -- none of which is a build name anything could be asked for
        try:
            named = binary.relative_to(Path(NAMED_DIR))
        except ValueError:
            return ""
    return named.parts[0] if named.parts else ""


def measured_best(mine: Sequence[Mapping[str, Any]], *, full_n: int = 0
                  ) -> Mapping[str, Any] | None:
    """The run one model's record should be written from: the fastest whose F1 held.

    `across` ranks by F1 alone, which is the right question for "which model answers best"
    and the wrong one for "how should this model be asked". Two askings the questions
    cannot tell apart are not two accuracies -- their 95% bands overlap, which is all
    twenty questions can say -- and between them the record takes the cheaper one, because
    the seconds are a difference the questions *can* see. Held is `score.held_up`: it did
    not fall at all, or the fall is inside what the questions can account for (`bands` and
    `separated`, with the fixed `NOISE` where a run carries no interval).

    Compared only among a model's longest runs, for `across`'s reason: a score means
    nothing beside a score over a different number of questions. Ties -- two rows at the
    same seconds -- go to the higher F1, then to the later run.

    Never from fewer than `SHORT` questions: the ranking refuses to rank a smoke, and a
    record set from one would send every later serve of that model the shape a coin toss
    chose (a two-question row once wrote a 27B's profile). Such a model gets no record,
    and `main` says so.
    """
    pool = [one for one in mine if derived(one)]
    if not pool:
        return None
    floor = full_n or max(derived(one)["questions"] for one in pool)
    if floor < SHORT:
        return None
    pool = [one for one in pool if derived(one)["questions"] >= floor]
    if not pool:
        return None
    best = max(pool, key=lambda o: (derived(o)["right"], str(o.get("at") or "")))
    held = [one for one in pool if held_up(one, best)] or [best]
    return min(held, key=lambda o: (per_question(o), -derived(o)["right"],
                                    str(o.get("at") or "")))


def profile_of(model: str, one: Mapping[str, Any]) -> Any:
    """The measured shape a run records, as a `ml_stack.serve.profile.Profile`.

    One row sets one record, and the record says which row: a shape composed from the
    accuracy of one run and the speed of another is a configuration nobody ever served.
    Nothing is guessed -- a field the run does not carry is left at its default, and `add`
    keeps whatever the older record knew about the two fields a kept run cannot see (the
    extra llama-server flags and the vision projector).
    """
    from ml_stack.hub import spec_for
    from ml_stack.serve.profile import record

    server = one.get("server") or {}
    got = derived(one)
    head = str(server.get("draft_model") or server.get("draft") or "")
    slots = int(server.get("slots") or 0) or 1
    context = int(server.get("context") or 0)
    asked = one.get("asking") if isinstance(one.get("asking"), Mapping) else {}
    sampling = asked.get("sampling") or server.get("sampling")
    return record(
        model,
        build=build_of(server),
        draft=head,
        spec_type=spec_for(head) if head else "",
        spec_draft_max=(int(server["spec_draft_max"])
                        if server.get("spec_draft_max") is not None else None),
        cache_type=str(server.get("cache_type") or ""),
        reasoning_budget=(int(server["reasoning_budget"])
                          if server.get("reasoning_budget") is not None else None),
        seat_context=(context // slots) if context else 32768,
        parallel=slots,
        sampling=dict(sampling) if isinstance(sampling, Mapping) else {},
        measured_at=str(one.get("at") or "")[:10],
        label=str(one.get("label") or ""),
        questions=int(got.get("questions") or 0),
        right=float(got.get("right") or 0.0),
        recall=float(got.get("recall") or 0.0),
        precision=float(got.get("precision") or 0.0),
        seconds_per_question=float(per_question(one)),
        host=host_of(one),
        note=(f"set from the fastest row whose F1 held: `{one.get('label') or '?'}`, "
              f"{int(got.get('questions') or 0)} question(s) at "
              f"{float(got.get('right') or 0.0) * 100:.0f}% F1, "
              f"{float(per_question(one)):.1f} s/question"),
        **ways_of(one))


def write_profiles(kept: Sequence[Mapping[str, Any]], *, full_n: int = 0,
                   path: Path | None = None) -> list[tuple[Any, Path]]:
    """Write one record per model, from the row `across` ranks it by. Returns what it wrote.

    The ranking fixes the *order* and `measured_best` fixes the *row*. They are not the
    same question: the ranking asks which model answers best, and a record asks how this
    model should be asked, where two askings the questions cannot tell apart should be
    settled by the seconds rather than by a hundredth of an F1. Both read only a model's
    longest runs, so a profile is never a shape chosen by a coin toss over two questions.
    """
    from dataclasses import replace

    from ml_stack.serve.profile import WAYS, add, profile_for, records_in, writable_file

    grouped = by_model(kept)
    out = []
    for model, _ranked in across(kept, full_n=full_n):
        one = measured_best(grouped.get(model) or (), full_n=full_n)
        if one is None:
            continue
        made_one = profile_of(model, one)
        if not asked_recorded(one):
            # The run predates asking records, so its label is all `ways_of` could read,
            # and a label says nothing about the globals the sweep rode on every way. The
            # evening this was learned, the hundred-question row asked with batch, kinds
            # and summary rewrote the record as asked with none of them, and the page
            # served the 70% shape under the 80% number. What the record already says
            # about the asking is measured too, and it is kept.
            older = profile_for(model, records=records_in(path or writable_file()))
            if older is not None:
                asked = {way: getattr(older, way) for way in WAYS}
                asked.update(reach=older.reach, rounds=older.rounds)
                made_one = replace(made_one, **asked,
                                   note=(made_one.note + " -- asked as the record already "
                                         "said: this run predates asking records"))
        out.append((made_one, add(made_one, path=path)))
    return out


def asked_recorded(one: Mapping[str, Any]) -> bool:
    """Whether a run carries the asking record `asked_with` keeps -- the keywords
    `converse` was handed -- rather than only a label to read words from."""
    said = one.get("asking")
    return isinstance(said, Mapping) and bool(said)


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
           min_msgs: int = MIN_MESSAGES) -> str:
    """Every measurement there is, as one document. See the module docstring for the parts.

    ``fits` are the memory records for this machine, ``elsewhere`` the same records asked
    about another room -- ``[(name, fits), ...]``, one per ``--room``. ``at`` is the
    per-user context the "how many fit" column answers at.

    ``kept`` is the answering runs and ``extracted`` the extraction runs, narrowed by the
    same window: they are kept in one store and are two different measurements, and mixing
    them cost an "Extraction" section that never printed. An empty ``extracted`` prints no
    such section -- a heading over nothing reads as a model that scored nothing.
    """
    doc = Doc(md)
    doc.head(1, "What has been measured")
    if not kept and not extracted:
        doc.para("Nothing kept yet. `ml-stack-bench run LABEL` measures one asking; "
                 "`ml-stack-bench sweep` measures every model every way.")
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


def _extraction(doc: Doc, extracted: Sequence[Mapping[str, Any]], *, min_msgs: int) -> None:
    from ml_stack.graph.bench.extract import detail
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
            print("no model has a run to write a profile from", file=sys.stderr)
            return 1
        for one, where in written:
            print(f"{one.model}: {one.label or '?'} "
                  f"({one.questions} q, {one.right * 100:.0f}% F1, "
                  f"{one.seconds_per_question:.1f} s/q) -> {where}")
        profiled = {one.model for one, _where in written}
        for model, mine in by_model(kept).items():
            longest = max((derived(o)["questions"] for o in mine if derived(o)), default=0)
            if model not in profiled and 0 < longest < SHORT:
                print(f"{model}: not profiled -- its longest run is {int(longest)} question(s), "
                      f"and a record is never set from fewer than {SHORT}", file=sys.stderr)
        return 0

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
                  room=fit_mod._human(here) if here else "", store=store,
                  extracted=extracted,
                  min_msgs=int(getattr(args, "min_msgs", MIN_MESSAGES) or MIN_MESSAGES))

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

