"""One model measured for one workload: how to serve it, how to ask it, and what said so.

`fit` answers "how many people fit"; this answers the question that came before it -- *what
shape*. A model does not have one good configuration and a list of flags, and it does not
have one per model either: the best shape depends on what the model is doing, because the
share of the output that is predictable differs. A tool call is mostly JSON skeleton and
repeated key names; a document extraction is a thin skeleton around free strings; prose has
no skeleton at all. So a record is keyed on the model *and the workload*, `WORKLOADS` names
the three this repository drives a model with, and both ends read it: the serve path takes
:meth:`Profile.shape`, the asking path takes :meth:`Profile.asked`, and the per-call path
takes :meth:`Profile.talking`.

A record has a startup half and a request half. `--context`, the draft head file, the cache
type, the seat count and the build are what a server is told once; the draft depth, the
draft p-min and the sampling ride on each call, so two workloads can share one served model
and still each get what they measured. The startup half is written under ``serve`` and the
request half under ``request``.

The file is `ml_stack/data/profiles.json`, beside `fit.json` and layered the same way:
what ships, with `~/.ml-stack/profiles.json` (or ``$MLSTACK_PROFILES_FILE``) over it, so a
machine that measured a model again keeps its own answer without editing the package. A
record written before workloads is read as the graph asking, and a record that keeps its
draft depth under ``serve`` is read the same as one that keeps it under ``request``.

Nothing here measures anything. `ml-stack-bench report --profile` writes the records from
the store's best row per model and workload, which is the only way a record should ever
appear.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from ml_stack.records import Records

__all__ = [
    "ASK",
    "WORKLOADS",
    "Profile",
    "add",
    "local_file",
    "package_file",
    "profile_for",
    "profiles",
    "record",
    "resolved",
    "said",
    "whole_context",
    "workload_named",
    "writable_file",
]


# What this repository drives a model with, and one line saying what each is, in the order
# a person meets them. Three, not one per module: `graph.ask`, `do`, `harness` and the
# page all call `converse`; `ingest` and `Client.extract` and the judge all fill a schema;
# `fleet.chat`, the world's writer and a question synthesiser all write prose.
WORKLOADS = {
    "ask": "tool-calling over a graph",
    "ingest": "documents into JSON under a schema",
    "chat": "prose, under no schema",
}

ASK = "ask"
"""The workload a record with none belongs to, and what `--for` means when unsaid."""


def workload_named(name: str) -> str:
    """``name`` as one of `WORKLOADS`, or `ASK` when it is empty."""
    named = str(name or "").strip().lower() or ASK
    if named not in WORKLOADS:
        raise ValueError(f"no such workload: {named}; there are "
                         f"{', '.join(WORKLOADS)}")
    return named


# The asking fields whose value is a plain on/off, in the order a person reads them out.
# `single` and `few` sit beside `batch` because they are the same question answered the
# other way: how much one read carries, and how many tools there are to choose between.
# Nothing here is a default -- a record says what *this* model measured, and two records
# disagreeing about every one of these is the intended outcome, not a mistake.
WAYS = ("tight", "batch", "single", "few", "kinds", "summary", "rich", "terse",
        "constrain_ids")

# The sampler settings a record keeps, and what each is called when it is read out. In the
# order a publisher's card lists them, so a record and a card can be compared by eye.
SAMPLERS = (("temperature", "temperature"), ("top_p", "top-p"), ("top_k", "top-k"),
            ("min_p", "min-p"))

# What a kept bench run cannot see about the serving, and so must never erase when it
# rewrites a record: llama-server's own extra flags and the vision projector are in the
# spec that started the server, not in anything `/props` reports back.
UNSEEN = ("extra_args", "mmproj")


@dataclass(frozen=True)
class Profile:
    """One model file doing one workload, in the shape that measured best.

    Four groups, and they stay apart because they are read by different code: ``serve`` is
    :meth:`shape`'s and is what a server is told once, ``request`` is :meth:`talking`'s and
    rides on each call, ``ask`` is :meth:`asked`'s, and ``measured`` is the provenance.
    """

    model: str
    workload: str = ASK

    # -- serving, at startup ------------------------------------------------------------------------
    build: str = ""                      # a named llama.cpp build, "" for the managed one
    draft: str = ""                      # the head's file name, path, or hf: reference
    spec_type: str = ""                  # draft-mtp, draft-eagle3; "" reads it off the name
    cache_type: str = ""                 # "" leaves the shape's own, q8_0
    reasoning_budget: int | None = None  # 0 turns the thinking off; None leaves it alone
    mmproj: str = ""                     # a path, or "auto" to find it beside the weights
    extra_args: tuple[str, ...] = ()     # -ub 2048, --spec-draft-p-min 0.5
    seat_context: int = 32768            # what one conversation gets
    parallel: int = 1                    # how many conversations at once

    # -- serving, per request -----------------------------------------------------------
    spec_draft_max: int | None = None    # tokens guessed ahead
    spec_p_min: float | None = None      # the draft's confidence floor

    # -- asking -------------------------------------------------------------------------
    tight: bool = True
    batch: bool = False
    single: bool = False                 # one entry to a read, more turns -- batch's opposite
    few: bool = False                    # three tools offered, not eight
    kinds: bool = False
    summary: bool = False                # the `summarise` tool, named as the bench is
    rich: bool = False
    terse: bool = False                  # `tools_for`'s, not `converse`'s -- see `asking`
    constrain_ids: bool = False          # id arguments held to the graph's ids by grammar
    reach: int | None = None
    rounds: int | None = None            # tool-calling turns one question may spend
    sampling: Mapping[str, Any] = field(default_factory=dict)

    # -- what measured it ---------------------------------------------------------------
    measured_at: str = ""
    label: str = ""                      # the row of the store that set this record
    questions: int = 0
    right: float = 0.0                   # F1
    recall: float = 0.0
    precision: float = 0.0
    seconds_per_question: float = 0.0
    host: str = ""
    note: str = ""

    # Not part of the record: the reference this profile was asked about, so a shape is
    # built with the `hf:` reference or path the caller has rather than the basename the
    # record is keyed on. `profile_for` fills it in.
    served: str = ""

    # -- reading it ---------------------------------------------------------------------

    @property
    def family(self) -> str:
        """The model without its quantisation -- ``Qwen3.8-Flash-Next``. What a record is
        matched by when the exact file is not the one on this machine."""
        return family_of(self.model)

    @property
    def quant(self) -> str:
        """The quantisation the record was measured at -- ``Q4_K_XL``, "" when unnamed."""
        return quant_of(self.model)

    def shape(self, *, port: int = 8080, seats: int | None = None,
              model: str = "", resolve: bool = True) -> Any:
        """The :class:`~ml_stack.serve.Shape` this model measured best in.

        One seat alone (no ``seats``, or ``seats=1``) gets the model's own trained context
        length when this machine's room holds it, the longest context that room does hold
        when it does not, and the measured shape's whole cache -- ``seat_context *
        parallel``, the record's own -- only when neither can be computed.
        :attr:`~ml_stack.serve.Shape.note` says which. ``seats`` above one each get what
        one measured seat got, from the record's own ``parallel``. ``model`` overrides the
        reference served, which otherwise is what :func:`profile_for` was asked about, and
        the record's own file name failing that.

        ``resolve`` answers 'auto' and a bare head file name the way `ml-stack-serve up`
        does -- a record names the head it measured, and where that file is is this
        machine's question. Off, the strings are handed on as they are, which is what a
        test wants and what a caller resolving them itself wants.
        """
        from ml_stack.serve.shape import Shape

        served = str(model or self.served or self.model)
        draft, seeing = self.draft, self.mmproj
        if resolve:
            draft, seeing = resolved(served, draft, seeing, build=self.build)
        taken = max(1, int(seats or 1))
        each, note = (_alone_context(self, served) if taken == 1
                     else (self.seat_context, ""))
        return Shape(model=served, port=port, seats=taken,
                     seat_context=each, cache_type=self.cache_type,
                     draft=draft, draft_n_max=self.spec_draft_max,
                     spec_type=self.spec_type, mmproj=seeing,
                     reasoning_budget=self.reasoning_budget, build=self.build,
                     extra_args=tuple(self.extra_args), note=note)

    def asked(self) -> Any:
        """The ways this record measured, as an :class:`~ml_stack.graph.Asking`: every one
        of `WAYS`, ``terse`` included, plus ``reach`` and ``rounds``."""
        from ml_stack.graph.asking import Asking

        return Asking(**{way: bool(getattr(self, way)) for way in WAYS},
                      reach=self.reach, rounds=self.rounds)

    def talking(self, *, n_predict: int = 16384, timeout: float = 300.0) -> Any:
        """The client this record measured with, as a :class:`~ml_stack.serve.Talking`.

        The sampling and the speculative depth are the record's, and both go out with each
        call. The ceiling and the timeout are the caller's: how long a machine will wait
        for one call is not what a measurement decided.
        """
        from ml_stack.serve.shape import Talking

        return Talking(n_predict=n_predict, timeout=timeout, sampling=dict(self.sampling),
                       spec_draft_max=self.spec_draft_max, spec_p_min=self.spec_p_min)

    def alone(self, *, port: int = 8080, model: str = "", resolve: bool = True,
              n_predict: int = 16384, timeout: float = 300.0) -> Any:
        """This record as one conversation: one seat holding the whole cache the record
        measured across its ``parallel`` seats. The same as :meth:`run` with no ``seats``.
        """
        return self.run(port=port, seats=1, model=model, resolve=resolve,
                        n_predict=n_predict, timeout=timeout)

    def run(self, *, port: int = 8080, seats: int | None = None, model: str = "",
            resolve: bool = True, n_predict: int = 16384, timeout: float = 300.0) -> Any:
        """This record whole, as a :class:`~ml_stack.serve.Run`: the shape to serve it in,
        the ways to ask it, and the client to ask it with.

        One object built once and handed on, so a bench row, a page answer and a seated
        client for this model are the same lease and the same asking. ``port``, ``seats``,
        ``model`` and ``resolve`` are :meth:`shape`'s: no ``seats`` is one seat holding the
        whole measured cache.
        """
        from ml_stack.serve.shape import Run

        return Run(shape=self.shape(port=port, seats=seats, model=model, resolve=resolve),
                   asking=self.asked(),
                   talking=self.talking(n_predict=n_predict, timeout=timeout))

    # -- the file -----------------------------------------------------------------------

    def as_dict(self) -> dict[str, Any]:
        """The record as it is written: four groups, because a person reads it."""
        serve: dict[str, Any] = {"build": self.build, "draft": self.draft,
                                 "spec_type": self.spec_type,
                                 "cache_type": self.cache_type,
                                 "reasoning_budget": self.reasoning_budget,
                                 "mmproj": self.mmproj,
                                 "extra_args": list(self.extra_args),
                                 "seat_context": self.seat_context,
                                 "parallel": self.parallel}
        request: dict[str, Any] = {"spec_draft_max": self.spec_draft_max,
                                   "spec_p_min": self.spec_p_min,
                                   "sampling": dict(self.sampling)}
        ask: dict[str, Any] = {**{way: bool(getattr(self, way)) for way in WAYS},
                               "reach": self.reach, "rounds": self.rounds}
        measured = {"measured_at": self.measured_at, "label": self.label,
                    "questions": self.questions, "right": self.right,
                    "recall": self.recall, "precision": self.precision,
                    "seconds_per_question": self.seconds_per_question,
                    "host": self.host, "note": self.note}
        return {"model": self.model, "workload": self.workload, "serve": serve,
                "request": request, "ask": ask, "measured": measured}

    @classmethod
    def from_dict(cls, row: Mapping[str, Any]) -> Profile:
        """One record read back, under the workload it names or the graph asking.

        Every group is flattened into one namespace, so a record that keeps its draft
        depth under ``serve`` and its sampling under ``ask`` -- where they were written
        before the startup and request halves were told apart -- reads the same as one
        that keeps them under ``request``. A key this version does not know is ignored and
        one it wants but does not find takes its default.
        """
        flat: dict[str, Any] = {}
        for group in ("serve", "request", "ask", "measured"):
            part = row.get(group)
            if isinstance(part, Mapping):
                flat.update(part)
        known = {f for f in cls.__dataclass_fields__
                 if f not in ("model", "workload", "served")}
        taken = {k: v for k, v in flat.items() if k in known}
        if "extra_args" in taken:
            taken["extra_args"] = tuple(str(a) for a in (taken["extra_args"] or ()))
        if "sampling" in taken and not isinstance(taken["sampling"], Mapping):
            taken.pop("sampling")
        named = str(row.get("workload") or "").strip().lower() or ASK
        return cls(model=str(row.get("model") or ""),
                   workload=named if named in WORKLOADS else ASK, **taken)

    def carrying(self, older: Profile | None) -> Profile:
        """This record, keeping from ``older`` the serving fields a measurement cannot see.

        A kept bench run records what `/props` reports; it does not record the extra flags
        the spec was built with or the projector it was served with. A rewrite that set
        those to nothing would silently delete two measured facts, so they are carried.
        """
        if older is None:
            return self
        keep = {name: getattr(older, name) for name in UNSEEN
                if not getattr(self, name) and getattr(older, name)}
        return replace(self, **keep) if keep else self


# ---------------------------------------------------------------- matching a model to one

def _basename(name: str) -> str:
    """A model reference reduced to its file name, as it is spelt."""
    return str(name or "").replace("\\", "/").rsplit("/", 1)[-1].strip()


def _plain(name: str) -> str:
    """A model reference reduced to its file name, lower case: what a record is keyed on."""
    return _basename(name).lower()


def family_of(name: str) -> str:
    """``Qwen3.8-Flash-Next`` out of a file name, path or hf: reference -- the pretty name
    with its quantisation taken off."""
    from ml_stack.hub import pretty_name

    return pretty_name(str(name or "")).split(" (")[0].strip()


def quant_of(name: str) -> str:
    """``Q4_K_XL`` out of a file name, "" when the name does not say."""
    from ml_stack.hub import pretty_name

    pretty = pretty_name(str(name or ""))
    return pretty.split(" (", 1)[1].rstrip(")") if " (" in pretty else ""


def _matched(every: Sequence[Profile], model: str) -> Profile | None:
    """The record for this model among ``every``, by file name, then family and
    quantisation, then family alone -- the last with ``note`` saying it is a different
    quantisation of the same model."""
    asked = _plain(model)
    for one in every:
        if _plain(one.model) == asked:
            return replace(one, served=str(model))
    family, quant = family_of(model).lower(), quant_of(model).lower()
    if not family:
        return None
    for one in every:
        if one.family.lower() == family and one.quant.lower() == quant:
            return replace(one, served=str(model))
    for one in every:
        if one.family.lower() == family:
            return replace(one, served=str(model),
                           note=_and(one.note, f"measured on {one.model}, not on "
                                               f"{_basename(model)}: same model, another "
                                               f"quantisation"))
    return None


def _and(note: str, said: str) -> str:
    """``note`` with ``said`` after it."""
    return f"{note}; {said}" if note else said


def profile_for(model: str, *, workload: str = ASK,
                records: Sequence[Profile] | None = None) -> Profile | None:
    """The measured shape for this model doing this workload, or None when nothing
    measured it.

    A model with no record for the workload asked falls back to its graph-asking record,
    returned with ``note`` saying which workload measured it and which one it was asked
    for, so no caller serves a shape measured for something else without being told.
    """
    named = workload_named(workload)
    every = list(records if records is not None else profiles())
    found = _matched([one for one in every if one.workload == named], model)
    if found is not None:
        return found
    if named == ASK:
        return None
    general = _matched([one for one in every if one.workload == ASK], model)
    if general is None:
        return None
    return replace(general, note=_and(
        general.note, f"measured for {ASK} ({WORKLOADS[ASK]}), not for {named} "
                      f"({WORKLOADS[named]}); nothing has measured this model for "
                      f"{named}"))


def resolved(model: str, draft: str, mmproj: str, *, build: str = "") -> tuple[str, str]:
    """A head and a projector as llama-server can be handed them.

    'auto' is answered the way `ml-stack-serve up` answers it. A bare file name -- which is
    what a bench run records and so what a record keeps -- is looked for in the Hub cache,
    the same way `up` resolves a bare model name. Anything already a path or an ``hf:``
    reference is left exactly as it is, and anything that cannot be found is served without
    rather than handed on as a file name llama-server would try to open.
    """
    from ml_stack.serve.shape import draft_for, projector_for

    head = str(draft or "")
    if head.lower() == "auto":
        head = draft_for(model, "auto", build=build)
    elif head and "/" not in head and not head.startswith("hf:"):
        from ml_stack.hub import located

        try:
            found = located(head)
        except Exception:  # noqa: BLE001 - a head we cannot find is served without
            found = None
        head = str(found) if found is not None else ""
    seeing = str(mmproj or "")
    if seeing.lower() == "auto":
        seeing = projector_for(model, "auto")
    return head, seeing


# ---------------------------------------------------------------- the file it lives in

_STORE: Records[Profile] = Records(
    "profiles.json", env="MLSTACK_PROFILES_FILE",
    build=Profile.from_dict, unbuild=lambda p: p.as_dict(),
    key=lambda p: (_plain(p.model), p.workload),
    order=lambda p: (p.model.lower(), list(WORKLOADS).index(p.workload)))


def package_file() -> Path:
    """The profiles that ship with ml-stack."""
    return _STORE.package_path()


def local_file() -> Path:
    """This machine's own records, layered over the shipped ones.
    ``$MLSTACK_PROFILES_FILE`` moves it."""
    return _STORE.local_path()


def records_in(path: Path) -> list[Profile]:
    """The records one file holds -- the shipped file, this machine's, or one named by
    `--profiles` -- so a writer can see what it is about to supersede."""
    return _STORE.read(path)


def profiles(*, package: Path | None = None, local: Path | None = None) -> list[Profile]:
    """Every measured shape: what ships, with this machine's own layered over it.

    A local record for the same model file replaces the shipped one rather than sitting
    beside it.
    """
    return _STORE.all(package=package or package_file(), local=local or local_file())


def writable_file() -> Path:
    """Where a new record goes: the shipped file in a checkout somebody can write to, and
    this machine's own file otherwise."""
    return _STORE.writable_path()


def add(profile: Profile, *, path: Path | None = None) -> Path:
    """Write one record into the source of truth, replacing the one it supersedes.

    Returns where it was written, which is what `--profile` prints.
    """
    return _STORE.add(profile, path=path or writable_file(),
                      merge=lambda new, older: new.carrying(older))


def record(model: str, **fields: Any) -> Profile:
    """One record, built by name and keyword -- what `add` is handed.

    The same shape as `fit`'s `Fit.of`: everything optional, the model file the only thing
    that must be said, so a caller writing a record from a measurement names the fields it
    measured and nothing else.
    """
    known = {f for f in Profile.__dataclass_fields__ if f != "model"}
    unknown = sorted(set(fields) - known)
    if unknown:
        raise TypeError(f"no such profile field: {', '.join(unknown)}")
    if "workload" in fields:
        fields["workload"] = workload_named(str(fields["workload"]))
    if "extra_args" in fields:
        fields["extra_args"] = tuple(str(a) for a in (fields["extra_args"] or ()))
    return Profile(model=str(model), **fields)


# ---------------------------------------------------------------- saying it to a person

def whole_context(profile: Profile) -> int:
    """The cache the record measured, summed across its seats: one seat's worth."""
    return profile.seat_context * max(1, profile.parallel)


def _model_path(served: str) -> Path | None:
    """Where ``served`` already sits on this machine, or None. Never fetches it: a bare
    name or an ``hf:`` reference is looked up in the Hub cache by its file name alone."""
    from ml_stack.hub import located

    text = str(served or "")
    candidate = Path(text).expanduser()
    if candidate.is_file():
        return candidate
    return located(text.rsplit("/", 1)[-1])


def _trained_context(served: str) -> int:
    """The context length a model's own GGUF header names, or 0 when the file is not on
    this machine or the header does not say."""
    path = _model_path(served)
    if path is None:
        return 0
    from ml_stack.serve.layout import layout

    try:
        return int(layout(path).context_length)
    except Exception:  # noqa: BLE001 - a header that cannot be read names no context
        return 0


def _fit_for(profile: Profile, *, room: int) -> Any | None:
    """The measured :class:`~ml_stack.serve.fit.Fit` for this profile's own model, cache
    type and speculation, asked about ``room``. None when nothing measured this model."""
    from ml_stack.serve.fit import records as fit_records

    name = _plain(profile.model)
    want_cache = profile.cache_type or "f16"
    every = [f for f in fit_records(room=room) if _plain(f.model) == name]
    matched = [f for f in every if f.cache_type == want_cache and f.spec == profile.spec_type]
    if matched:
        return matched[0]
    same_cache = [f for f in every if f.cache_type == want_cache]
    return same_cache[0] if same_cache else (every[0] if every else None)


def _alone_context(profile: Profile, served: str) -> tuple[int, str]:
    """What one seat gets when it is alone, and which of three ways decided it.

    The model's trained context length when a measured :class:`Fit` for this shape says
    this machine's room holds it; the longest context that room does hold when it does
    not; the record's own measured shape -- :func:`whole_context` -- when the trained
    context or a fit record cannot be had at all.
    """
    from ml_stack import hub

    fallback = whole_context(profile)
    trained = _trained_context(served)
    if not trained:
        return fallback, (f"no trained context read off the model's header; served the "
                          f"measured shape, {fallback:,} tokens")
    room = hub.room()
    fit = _fit_for(profile, room=room) if room else None
    if fit is None:
        return fallback, (f"no fit record for this shape; served the measured shape, "
                          f"{fallback:,} tokens")
    if fit.cost(trained) <= fit.free():
        return trained, (f"the model's trained context, {trained:,} tokens, fits this "
                         f"machine's room")
    longest = fit.longest(1)
    if longest > 0:
        return longest, (f"the model's trained context, {trained:,} tokens, does not fit "
                         f"this machine's room; served the longest that does, "
                         f"{longest:,} tokens")
    return fallback, (f"nothing fits this machine's room beyond the measured shape; "
                      f"served it, {fallback:,} tokens")


def _flags(profile: Profile) -> str:
    """The serving line: what `ml-stack-serve up` would be told, in its own flags. One
    seat holding the whole measured cache; `--parallel` is left to the caller.

    The draft depth is here as well as on the request line: it is the server's default,
    which is what a call that cannot override it gets.
    """
    parts = [f"--context {whole_context(profile)}"]
    if profile.build:
        parts.append(f"--build {profile.build}")
    if profile.draft:
        parts.append(f"--draft {profile.draft}")
    if profile.spec_type:
        parts.append(f"--spec {profile.spec_type}")
    if profile.spec_draft_max is not None:
        parts.append(f"--spec-n-max {profile.spec_draft_max}")
    if profile.cache_type:
        parts.append(f"--kv {profile.cache_type}")
    if profile.mmproj:
        parts.append(f"--mmproj {profile.mmproj}")
    if profile.reasoning_budget is not None:
        parts.append(f"--reasoning-budget {profile.reasoning_budget}")
    return " ".join(parts)


def _sampled(sampling: Mapping[str, Any] | None) -> str:
    """The sampler settings a record measured, read out: ``at temperature 1.0 / top-p 0.95
    / top-k 20``, or the one word ``greedy``.

    Greedy says the whole thing -- at temperature 0 no other sampler can change an argument
    -- so it is one word. Anything else is read out in full, every setting the record
    carries, because "the card asks for 1.0 and the measurement agreed" is exactly the
    thing a person serving this model needs to see rather than infer.
    """
    held = dict(sampling or {})
    temperature = held.get("temperature")
    if temperature is not None and float(temperature) == 0:
        return "greedy"
    parts = [f"{name} {held[key]}" for key, name in SAMPLERS if held.get(key) is not None]
    return "at " + " / ".join(parts) if parts else ""


def _ways(profile: Profile) -> str:
    """The asking line, as the words the bench and `converse` both use."""
    said = [way.replace("_", "-") for way in WAYS if getattr(profile, way)]
    if not profile.tight:
        said.insert(0, "loose")
    if profile.reach is not None:
        said.append(f"reach {profile.reach}")
    if profile.rounds is not None:
        said.append(f"rounds {profile.rounds}")
    sampled = _sampled(profile.sampling)
    if sampled == "greedy":
        said.append("greedy")
        sampled = ""
    line = " + ".join(said) or "the defaults"
    return f"{line} {sampled}".rstrip()


def _per_request(profile: Profile) -> str:
    """The request line: what travels with each call rather than with the server."""
    parts = []
    if profile.spec_draft_max is not None:
        parts.append(f"draft {profile.spec_draft_max} ahead")
    if profile.spec_p_min is not None:
        parts.append(f"draft p-min {profile.spec_p_min}")
    sampled = _sampled(profile.sampling)
    if sampled:
        parts.append(sampled if sampled == "greedy" else sampled.removeprefix("at "))
    return ", ".join(parts)


def said(profile: Profile) -> str:
    """One record as a person reads it: what it is for, serve with, per request, ask with,
    measured.

    A few lines and no table. A person asking `ml-stack-serve profile MODEL` is about to
    serve it, and what they need is the workload, the flags, the ways, and enough of the
    provenance to know whether to believe them.
    """
    lines = [profile.model,
             f"  for         {profile.workload} -- "
             f"{WORKLOADS.get(profile.workload, 'unknown')}",
             f"  serve with  {_flags(profile)}"]
    if profile.extra_args:
        # llama-server's own flags: `up` has none of its own for them, and `--profile` is
        # the only thing that passes them, so the line says so rather than reading as
        # something a person could type
        lines.append(f"              and {' '.join(profile.extra_args)} "
                     f"-- llama-server's own, passed by --profile")
    lines.append(f"  measured at --parallel {max(1, profile.parallel)}, "
                 f"{profile.seat_context} per seat")
    asked = _per_request(profile)
    if asked:
        lines.append(f"  per request {asked} -- sent with each call, where the build "
                     f"takes it")
    lines.append(f"  ask with    {_ways(profile)}")
    if profile.questions:
        lines.append(
            f"  measured    {profile.right * 100:.0f}% F1 "
            f"({profile.recall * 100:.0f}% recall, {profile.precision * 100:.0f}% precision) "
            f"at {profile.seconds_per_question:.1f} s/question over "
            f"{profile.questions} question(s)")
    where = ", ".join(part for part in (profile.measured_at, profile.host) if part)
    if where or profile.label:
        lines.append(f"              {where}"
                     + (f", from `{profile.label}`" if profile.label else ""))
    if profile.note:
        lines.append(f"  note        {profile.note}")
    return "\n".join(lines)
