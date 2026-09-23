"""What a measuring run will cost before it is paid for, and the ceiling it is refused over.

`estimate` says, per model a command serves or measures, how long it should take: seconds
per question from that model's newest kept run (at the same context when one is kept),
else a guess from its weights, times the questions and the askings, plus a load per model
served. Over `--ceiling` without ``--yes`` the run is refused; a smoke run never is. The
``estimate:`` lines are what `history` reads back, the total last.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ml_stack import hub
from ml_stack.bench.askings import _asked, halves
from ml_stack.bench.extract import SMOKE_MESSAGES, only
from ml_stack.bench.keep import SMOKE
from ml_stack.bench.questions import _how_many, read_questions, sample
from ml_stack.bench.speed import PROMPTS, STREAMS, _ints
from ml_stack.bench.underway import wants_smoke
from ml_stack.graph.community import QUESTIONS
from ml_stack.serve.weights import weight_of
from ml_stack.units import span

# A model nothing is known about: no run kept and no weights on disk to size it by.
GUESS_S = 15.0
# With weights on disk and no run: seconds per question per gigabyte of weights. A 4G
# model answering in three seconds and a 60G one in forty is what this reproduces.
GUESS_PER_GB_S = 0.7
# A load nothing recorded: what a mid-sized model takes to come up from a warm page cache.
GUESS_LOAD_S = 30.0
# Over this many minutes a run is refused unless --yes; MLSTACK_BENCH_CEILING overrides.
CEILING_MIN = 30.0
CEILING_ENV = "MLSTACK_BENCH_CEILING"


def ceiling_default() -> float:
    """The ceiling in minutes: the environment's, else `CEILING_MIN`."""
    try:
        return float(os.environ.get(CEILING_ENV, "") or CEILING_MIN)
    except ValueError:
        return CEILING_MIN


@dataclass
class ModelEstimate:
    """One model (or one served configuration of it) and what asking it should take."""

    name: str
    questions: int                  # asked of each asking, the smoke's included
    askings: int                    # of one load: plain, terse, shortlist...
    per_question: float             # seconds
    load_s: float                   # 0 for a server already up
    source: str                     # where per_question came from, said in the line
    guessed: bool = False           # True when no kept run supplied per_question

    @property
    def seconds(self) -> float:
        return self.questions * self.askings * self.per_question + self.load_s

    def line(self) -> str:
        return (f"estimate: {span(self.seconds)} ({self.name} {self.questions} q x "
                f"{self.askings} asking{'s' if self.askings != 1 else ''} x "
                f"{self.per_question:.0f} s/q"
                + (f" + load {self.load_s:.0f} s" if self.load_s else "")
                + f"; {self.source})")


@dataclass
class Estimate:
    """Every model's estimate, the ceiling, and whether the run is ``--smoke``."""

    models: list[ModelEstimate]
    ceiling_min: float = CEILING_MIN
    smoke: bool = False

    @property
    def seconds(self) -> float:
        return sum(m.seconds for m in self.models)

    @property
    def over(self) -> bool:
        """Past the ceiling -- never for a smoke run, whose two questions prove the path."""
        return not self.smoke and self.seconds > self.ceiling_min * 60

    def lines(self) -> list[str]:
        """One line per model, then the total -- last, since `history` takes the last
        ``estimate:`` line, and with no second duration on it, since `parse_duration`
        adds up every one it finds (the ceiling is on the refusal and in --help)."""
        out = [m.line() for m in self.models]
        n = len(self.models)
        guessed = sum(1 for m in self.models if m.guessed)
        out.append(f"estimate: {span(self.seconds)} in all for {n} model{'s' if n != 1 else ''}"
                   + (f", {guessed} guessed with no run kept" if guessed else "")
                   + (" (a smoke run, never refused)" if self.smoke
                      else " (over the ceiling)" if self.over else ""))
        return out

    def refusal(self) -> str:
        return (f"error: estimated {span(self.seconds)}, over the {self.ceiling_min:g} min "
                f"ceiling -- no more eight-hour tests. Ask fewer questions (--sample N, "
                f"--short), raise the ceiling (--ceiling MINUTES, or {CEILING_ENV}), or pass "
                f"--yes to run it anyway.")


# What a served model reads and writes at, in tokens a second, when nothing has measured
# it: a large model on Apple silicon, for the speed grid's estimate.
GUESS_PREFILL_TPS = 300.0
GUESS_DECODE_TPS = 20.0


def _stem(named: Any) -> str:
    """``models/Flash-Next-Q4.gguf`` and ``flash-next-q4`` are the same model."""
    return str(named or "").rsplit("/", 1)[-1].removesuffix(".gguf").lower()


def _of_model(one: Mapping[str, Any], model: str, labels: Sequence[str]) -> bool:
    server = one.get("server") or {}
    label = str(one.get("label") or "")
    stem = _stem(model)
    if stem and _stem(server.get("model")) == stem:
        return True
    return any(label == want or label.startswith(want + "-") for want in labels if want)


def measured(kept: Sequence[Mapping[str, Any]], *, model: str = "", labels: Sequence[str] = (),
             context: int = 0) -> tuple[float, float | None, str] | None:
    """``(seconds per question, load_s, where it came from)`` from the newest kept run of
    ``model`` -- by its ``server.model``, or a label among ``labels`` (``name`` or
    ``name-...``) -- at ``context`` when one is kept there, else the newest at any; None
    when nothing of it is kept. The mean of the run's rows' seconds, timeouts included:
    a cap paid is time spent."""
    mine = [one for one in kept if (one.get("rows") and _of_model(one, model, labels))]
    if not mine:
        return None
    same = [one for one in mine if context and int((one.get("server") or {}).get("context")
                                                    or 0) == int(context)]
    newest = max(same or mine, key=lambda one: str(one.get("at") or ""))
    rows = newest.get("rows") or []
    per = sum(float(r.get("seconds") or 0) for r in rows) / len(rows)
    load = (newest.get("server") or {}).get("load_s")
    return (per, float(load) if load is not None else None,
            f"from {newest.get('label', '')} kept {newest.get('at', '?')}"
            + (f" at {int(context) // 1024}k" if same else ""))


def guessed(model: str) -> tuple[float, str]:
    """``(seconds per question, why)`` for a model with no run kept: from its weights on
    disk when they are here, else `GUESS_S` -- and the line says which."""
    size = weight_of(model) if model else 0
    if size > 0:
        return (max(1.0, size / 1e9 * GUESS_PER_GB_S),
                f"a guess from {size / 2**30:.1f}G of weights, no run of it kept")
    return GUESS_S, "a guess, no run of it kept and no weights on disk to size it by"


@dataclass(frozen=True, slots=True)
class _Who:
    """A model an estimate is for: its name in the line, its weights, the labels its kept
    runs go by, and the context it is served at."""

    name: str
    model: str = ""
    labels: Sequence[str] = ()
    context: int = 0


def _one(kept: Sequence[Mapping[str, Any]], who: _Who, *, questions: int, askings: int,
         served: bool) -> ModelEstimate:
    got = measured(kept, model=who.model, labels=who.labels, context=who.context)
    if got is not None:
        per, load, source = got
        load_s = (GUESS_LOAD_S if load is None else load) if served else 0.0
        return ModelEstimate(who.name, questions, askings, per, load_s, source)
    per, source = guessed(who.model)
    return ModelEstimate(who.name, questions, askings, per, GUESS_LOAD_S if served else 0.0,
                         source, guessed=True)


def _questions(args: Any) -> int:
    """How many questions each way is asked: the sample, plus the smoke a real run makes
    first."""
    named = getattr(args, "questions", "") or ""
    everything = read_questions(named) if named else QUESTIONS
    asked = len(sample(everything, _how_many(args)))
    return asked + (SMOKE if wants_smoke(args) else 0)


def _located(wanted: str) -> tuple[str, str]:
    """``(model path, the stem a label is made from)`` for a model a command names."""
    model = str(hub.located(wanted, loose=True) or wanted)
    return model, str(model).rsplit("/", 1)[-1].removesuffix(".gguf")


def _sweep(args: Any, kept: Sequence[Mapping[str, Any]]) -> list[ModelEstimate]:
    q = _questions(args)
    context = int(getattr(args, "context", 0) or 0) or 32768 * max(
        1, int(getattr(args, "parallel", 1) or 1))
    out = []
    for wanted in getattr(args, "serve", None) or []:
        model, stem = _located(wanted)
        askings = len(_asked(args, halves(args, f"{wanted} {model}")))
        out.append(_one(kept, _Who(stem[:14], model, [stem[:14]], context), questions=q,
                        askings=askings, served=True))
    for one in getattr(args, "on", None) or []:
        name, _, url = one.partition("=")
        if name and url:
            out.append(_one(kept, _Who(name, labels=[name]), questions=q,
                            askings=len(halves(args, name)), served=False))
    return out


def _drafts(args: Any, kept: Sequence[Mapping[str, Any]]) -> list[ModelEstimate]:
    model, stem = _located(getattr(args, "model", ""))
    asked = SMOKE if getattr(args, "smoke", False) else int(getattr(args, "sample", 0) or 0)
    q = asked + (SMOKE if wants_smoke(args) else 0)
    lengths = list(getattr(args, "n_max", None) or []) or [None]
    context = int(getattr(args, "context", 0) or 0)
    out = []
    for head in getattr(args, "draft", None) or [""]:
        name = "none" if not head else str(head).rsplit("/", 1)[-1].removesuffix(".gguf")
        for length in (lengths if head else [None]):
            tagged = f"{name}@n{length}" if length is not None else name
            out.append(_one(kept, _Who(f"{stem[:14]} draft:{tagged}", model,
                                       [f"draft:{tagged}"], context),
                            questions=q, askings=1, served=True))
    return out


def _speed(args: Any, kept: Sequence[Mapping[str, Any]]) -> list[ModelEstimate]:
    prompts = _ints(getattr(args, "prompts", ""), PROMPTS)
    streams = _ints(getattr(args, "streams", ""), STREAMS)
    if getattr(args, "smoke", False):
        prompts, streams = [min(prompts)], [min(streams)]
    cells = len(prompts) * len(streams) + (1 if wants_smoke(args) else 0)
    generate = int(getattr(args, "generate", 256) or 256)
    # each prompt is read twice (calibration, then the cell) and ``generate`` tokens written
    per = sum(p / GUESS_PREFILL_TPS * 2 + generate / GUESS_DECODE_TPS for p in prompts) \
        / max(1, len(prompts))
    why = "a guess from the grid, no run of it timed"
    out = [ModelEstimate(_located(wanted)[1][:14], cells, 1, per, GUESS_LOAD_S, why,
                         guessed=True)
           for wanted in getattr(args, "serve", None) or []]
    out += [ModelEstimate(name, cells, 1, per, 0.0, why, guessed=True)
            for name in (one.partition("=")[0] for one in getattr(args, "on", None) or [])
            if name]
    return out


def _concurrent(args: Any, kept: Sequence[Mapping[str, Any]]) -> list[ModelEstimate]:
    many, long = ((2, 1) if getattr(args, "smoke", False)
                  else (int(getattr(args, "conversations", 1) or 1),
                        int(getattr(args, "turns", 1) or 1)))
    q = many * long + (2 if wants_smoke(args) else 0)
    label = str(getattr(args, "label", "") or "")
    return [_one(kept, _Who(label or "the server", labels=[label]), questions=q, askings=1,
                 served=False)]


def _extract(args: Any, kept: Sequence[Mapping[str, Any]]) -> list[ModelEstimate]:
    serving = list(getattr(args, "serve", None) or [])
    model, stem = _located(serving[0]) if serving else ("", "")
    n = SMOKE_MESSAGES if getattr(args, "smoke", False) else int(getattr(args, "sample", 0) or 0)
    askings = 2 if getattr(args, "twice", False) else 1
    load = GUESS_LOAD_S if serving else 0.0
    # an extraction run keeps `model` at its top and its rows are messages
    seen = [float(r.get("seconds") or 0) for one in only(kept)
            if stem and str(one.get("model") or "") == stem
            for r in (one.get("rows") or ())]
    name = stem or str(getattr(args, "label", "") or "the server")
    if seen:
        return [ModelEstimate(name, n, askings, sum(seen) / len(seen), load,
                              f"from {len(seen)} earlier messages of {stem}")]
    per, source = guessed(model)
    return [ModelEstimate(name, n, askings, per, load, source, guessed=True)]


def _run(args: Any, kept: Sequence[Mapping[str, Any]]) -> list[ModelEstimate]:
    label = str(getattr(args, "label", "") or "")
    client = str(getattr(args, "client", "") or "")
    return [_one(kept, _Who(label or client or "the server", labels=[label]),
                 questions=_questions(args), askings=1, served=False)]


_BY_COMMAND = {"sweep": _sweep, "drafts": _drafts, "speed": _speed,
               "concurrent": _concurrent, "extract": _extract}


def estimate(args: Any, kept: Sequence[Mapping[str, Any]], *,
             ceiling_min: float | None = None) -> Estimate:
    """What ``args`` will cost, model by model, from the runs in ``kept``; ``ceiling_min``
    defaults to ``args.ceiling``, then the environment's, then `CEILING_MIN`."""
    if ceiling_min is None:
        held = getattr(args, "ceiling", None)
        ceiling_min = float(held) if held is not None else ceiling_default()
    models = _BY_COMMAND.get(str(getattr(args, "cmd", "") or ""), _run)(args, kept)
    return Estimate(models, ceiling_min=ceiling_min, smoke=bool(getattr(args, "smoke", False)))
