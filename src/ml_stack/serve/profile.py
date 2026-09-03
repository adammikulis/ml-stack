"""One model's measured shape: how to serve it, how to ask it, and what said so.

`fit` answers "how many people fit"; this answers the question that came before it -- *what
shape*. A model does not have one good configuration and a list of flags: it has one that
was measured, and the numbers live in a bench store nobody reads at serving time. So the
conclusion is written down here, one record per model file, and both ends read it: the
serve path takes :meth:`Profile.shape`, the asking path takes :meth:`Profile.asking`.

Why a record and not a default. Qwen3.8-Flash-Next answers well only in a shape nothing
else wants -- a fork build, the shared MTP head at four, a q8_0 cache, thinking off,
``-ub 2048``, ``--spec-draft-p-min 0.5``, and three ways of asking at once -- and every one
of those was a measurement. Written as defaults they would be wrong for gemma-4, which
wants its thinking left on and no such flags at all. Written per model they are what they
are: this model, measured on this machine, on that date, by that row of the store.

The file is `ml_stack/data/profiles.json`, beside `fit.json` and layered the same way:
what ships, with `~/.ml-stack/profiles.json` (or ``$MLSTACK_PROFILES_FILE``) over it, so a
machine that measured a model again keeps its own answer without editing the package.

Nothing here measures anything. `ml-stack-bench report --profile` writes the records from
the store's best row per model, which is the only way a record should ever appear: a shape
typed in by hand is a remembered one, and the whole point is that this one was paid for.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

__all__ = ["Profile", "add", "local_file", "package_file", "profile_for", "profiles",
           "record", "resolved", "said", "writable_file"]


# The asking fields whose value is a plain on/off, in the order a person reads them out.
# `single` and `few` sit beside `batch` because they are the same question answered the
# other way: how much one read carries, and how many tools there are to choose between.
# Nothing here is a default -- a record says what *this* model measured, and two records
# disagreeing about every one of these is the intended outcome, not a mistake.
WAYS = ("tight", "batch", "single", "few", "kinds", "summary", "rich", "terse")

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
    """One model file, in the shape that measured best.

    Three groups, and they stay apart because they are read by different code: ``serve``
    is :meth:`shape`'s, ``ask`` is :meth:`asking`'s, and ``measured`` is neither -- it is
    the provenance, so a person reading the record can tell what paid for it and when it
    goes stale.
    """

    model: str

    # -- serving ------------------------------------------------------------------------
    build: str = ""                      # a named llama.cpp build, "" for the managed one
    draft: str = ""                      # the head's file name, path, or hf: reference
    spec_type: str = ""                  # draft-mtp, draft-eagle3; "" reads it off the name
    spec_draft_max: int | None = None    # tokens guessed ahead
    cache_type: str = ""                 # "" is the server's f16
    reasoning_budget: int | None = None  # 0 turns the thinking off; None leaves it alone
    mmproj: str = ""                     # a path, or "auto" to find it beside the weights
    extra_args: tuple[str, ...] = ()     # -ub 2048, --spec-draft-p-min 0.5
    seat_context: int = 32768            # what one conversation gets
    parallel: int = 1                    # how many conversations at once

    # -- asking -------------------------------------------------------------------------
    tight: bool = True
    batch: bool = False
    single: bool = False                 # one entry to a read, more turns -- batch's opposite
    few: bool = False                    # three tools offered, not eight
    kinds: bool = False
    summary: bool = False                # `converse`'s summary_tool, named as the bench is
    rich: bool = False
    terse: bool = False                  # `tools_for`'s, not `converse`'s -- see `asking`
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

        ``seats`` overrides the measured ``parallel`` -- how many conversations a machine
        wants is that machine's business, and the rest of the shape is not. ``model``
        overrides the reference served, which otherwise is what :func:`profile_for` was
        asked about, and the record's own file name failing that.

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
        return Shape(model=served, port=port, seats=int(seats or self.parallel or 1),
                     seat_context=self.seat_context, cache_type=self.cache_type,
                     draft=draft, draft_n_max=self.spec_draft_max,
                     spec_type=self.spec_type, mmproj=seeing,
                     reasoning_budget=self.reasoning_budget, build=self.build,
                     extra_args=tuple(self.extra_args))

    def asked(self) -> Any:
        """The ways this record measured, as an :class:`~ml_stack.serve.Asking`: every one
        of `WAYS`, ``terse`` included, plus ``reach`` and ``rounds``."""
        from ml_stack.serve.shape import Asking

        return Asking(**{way: bool(getattr(self, way)) for way in WAYS},
                      reach=self.reach, rounds=self.rounds)

    def talking(self, *, n_predict: int = 16384, timeout: float = 300.0) -> Any:
        """The client this record measured with, as a :class:`~ml_stack.serve.Talking`.

        The sampling is the record's. The ceiling and the timeout are the caller's: how
        long a machine will wait for one call is not what a measurement decided.
        """
        from ml_stack.serve.shape import Talking

        return Talking(n_predict=n_predict, timeout=timeout, sampling=dict(self.sampling))

    def run(self, *, port: int = 8080, seats: int | None = None, model: str = "",
            resolve: bool = True, n_predict: int = 16384, timeout: float = 300.0) -> Any:
        """This record whole, as a :class:`~ml_stack.serve.Run`: the shape to serve it in,
        the ways to ask it, and the client to ask it with.

        One object built once and handed on, so a bench row, a page answer and a seated
        client for this model are the same lease and the same asking. ``port``, ``seats``,
        ``model`` and ``resolve`` are :meth:`shape`'s.
        """
        from ml_stack.serve.shape import Run

        return Run(shape=self.shape(port=port, seats=seats, model=model, resolve=resolve),
                   asking=self.asked(),
                   talking=self.talking(n_predict=n_predict, timeout=timeout))

    def asking(self) -> dict[str, Any]:
        """The keyword arguments :func:`ml_stack.graph.ask.converse` takes.

        Only what this record actually asks for, so a profile that measured nothing about
        the asking leaves `converse` exactly as it was. ``summary`` becomes
        ``summary_tool`` -- `converse`'s own ``summary`` is a thread's rolling summary and
        the two must never be confused; the bench renames it at the same hop and for the
        same reason.

        ``terse`` and ``sampling`` are not here and cannot be: ``terse`` chooses the tool
        *schemas*, which is :func:`~ml_stack.graph.ask.tools_for`'s argument, and sampling
        is the client's. Both are on the record because both were measured, and
        :meth:`run` is what carries all three of them together.
        """
        return dict(self.asked().converse())

    # -- the file -----------------------------------------------------------------------

    def as_dict(self) -> dict[str, Any]:
        """The record as it is written: three groups, because a person reads it."""
        serve: dict[str, Any] = {"build": self.build, "draft": self.draft,
                                 "spec_type": self.spec_type,
                                 "spec_draft_max": self.spec_draft_max,
                                 "cache_type": self.cache_type,
                                 "reasoning_budget": self.reasoning_budget,
                                 "mmproj": self.mmproj,
                                 "extra_args": list(self.extra_args),
                                 "seat_context": self.seat_context,
                                 "parallel": self.parallel}
        ask: dict[str, Any] = {**{way: bool(getattr(self, way)) for way in WAYS},
                               "reach": self.reach, "rounds": self.rounds,
                               "sampling": dict(self.sampling)}
        measured = {"measured_at": self.measured_at, "label": self.label,
                    "questions": self.questions, "right": self.right,
                    "recall": self.recall, "precision": self.precision,
                    "seconds_per_question": self.seconds_per_question,
                    "host": self.host, "note": self.note}
        return {"model": self.model, "serve": serve, "ask": ask, "measured": measured}

    @classmethod
    def from_dict(cls, row: Mapping[str, Any]) -> Profile:
        """One record read back. A key this version does not know is ignored and one it
        wants but does not find takes its default, so an older file still loads."""
        flat: dict[str, Any] = {}
        for group in ("serve", "ask", "measured"):
            part = row.get(group)
            if isinstance(part, Mapping):
                flat.update(part)
        known = {f for f in cls.__dataclass_fields__ if f not in ("model", "served")}
        taken = {k: v for k, v in flat.items() if k in known}
        if "extra_args" in taken:
            taken["extra_args"] = tuple(str(a) for a in (taken["extra_args"] or ()))
        if "sampling" in taken and not isinstance(taken["sampling"], Mapping):
            taken.pop("sampling")
        return cls(model=str(row.get("model") or ""), **taken)

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


def profile_for(model: str, *, records: Sequence[Profile] | None = None) -> Profile | None:
    """The measured shape for this model, or None when nothing measured it.

    Three ways, narrowing: the file's own name; then the family and quantisation together,
    so ``hf:owner/repo/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf`` finds the record
    kept under the bare file name; then the family alone, which is a *different*
    quantisation of the same model and is returned with ``note`` saying so. A shape
    measured on Q4_K_XL is the right starting point for IQ4_XS and is not a measurement of
    it, and a caller that cannot see the difference would report one as the other.
    """
    every = list(records if records is not None else profiles())
    asked = _plain(model)
    for one in every:
        if _plain(one.model) == asked:
            return replace(one, served=str(model))
    family, quant = family_of(model).lower(), quant_of(model).lower()
    if family:
        for one in every:
            if one.family.lower() == family and one.quant.lower() == quant:
                return replace(one, served=str(model))
        for one in every:
            if one.family.lower() == family:
                said = (f"measured on {one.model}, not on {_basename(model)}: same model, "
                        f"another quantisation")
                return replace(one, served=str(model),
                               note=f"{one.note}; {said}" if one.note else said)
    return None


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

def package_file() -> Path:
    """The profiles that ship with ml-stack. A function rather than a constant so a test
    can point it at somewhere with nothing in it."""
    return Path(__file__).resolve().parent.parent / "data" / "profiles.json"


def local_file() -> Path:
    """This machine's own records, layered over the shipped ones.
    ``$MLSTACK_PROFILES_FILE`` moves it, which is how the tests keep out of a real
    ``~/.ml-stack``."""
    named = os.environ.get("MLSTACK_PROFILES_FILE")
    if named:
        return Path(named).expanduser()
    return Path.home() / ".ml-stack" / "profiles.json"


def _read(path: Path) -> list[Profile]:
    """Every record in one file. A file that is absent, unreadable or not a list of objects
    contributes nothing: half a profile is a shape nobody measured."""
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    if not isinstance(parsed, list):
        return []
    out: list[Profile] = []
    for row in parsed:
        if not isinstance(row, Mapping) or not row.get("model"):
            continue
        try:
            out.append(Profile.from_dict(row))
        except (TypeError, ValueError):
            continue
    return out


def records_in(path: Path) -> list[Profile]:
    """The records one file holds -- the shipped file, this machine's, or one named by
    `--profiles` -- so a writer can see what it is about to supersede."""
    return _read(path)


def profiles(*, package: Path | None = None, local: Path | None = None) -> list[Profile]:
    """Every measured shape: what ships, with this machine's own layered over it.

    A local record for the same model file replaces the shipped one rather than sitting
    beside it -- two shapes for one model is a choice nobody can make from the outside, and
    the newer measurement is the one this machine paid for.
    """
    merged: dict[str, Profile] = {}
    for one in _read(package or package_file()) + _read(local or local_file()):
        merged[_plain(one.model)] = one
    return sorted(merged.values(), key=lambda p: p.model.lower())


def writable_file() -> Path:
    """Where a new record goes: the shipped file in a checkout somebody can write to, and
    this machine's own file otherwise -- an installed wheel is not a place to keep a
    measurement, since the next upgrade takes it away."""
    shipped = package_file()
    if "site-packages" in shipped.parts or "dist-packages" in shipped.parts:
        return local_file()
    parent = shipped.parent
    try:
        parent.mkdir(parents=True, exist_ok=True)
        if os.access(parent, os.W_OK):
            return shipped
    except OSError:
        pass
    return local_file()


def add(profile: Profile, *, path: Path | None = None) -> Path:
    """Write one record into the source of truth, replacing the one it supersedes.

    Returns where it was written, which is what `--profile` prints: a person who measured
    a model in a checkout and expected the record in the repository should be told when it
    went to this machine's own file instead.
    """
    where = path or writable_file()
    held = _read(where)
    older = next((one for one in held if _plain(one.model) == _plain(profile.model)), None)
    kept = [one for one in held if _plain(one.model) != _plain(profile.model)]
    kept.append(profile.carrying(older))
    kept.sort(key=lambda p: p.model.lower())
    where.parent.mkdir(parents=True, exist_ok=True)
    where.write_text(json.dumps([one.as_dict() for one in kept], indent=2) + "\n",
                     encoding="utf-8")
    return where


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
    if "extra_args" in fields:
        fields["extra_args"] = tuple(str(a) for a in (fields["extra_args"] or ()))
    return Profile(model=str(model), **fields)


# ---------------------------------------------------------------- saying it to a person

def _flags(profile: Profile) -> str:
    """The serving line: what `ml-stack-serve up` would be told, in its own flags."""
    parts = [f"--context {profile.seat_context * max(1, profile.parallel)}",
             f"--parallel {max(1, profile.parallel)}"]
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
    said = [way for way in WAYS if getattr(profile, way)]
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


def said(profile: Profile) -> str:
    """One record as a person reads it: serve with, ask with, measured.

    Three lines and no table. A person asking `ml-stack-serve profile MODEL` is about to
    serve it, and what they need is the flags, the ways, and enough of the provenance to
    know whether to believe them.
    """
    lines = [profile.model, f"  serve with  {_flags(profile)}"]
    if profile.extra_args:
        # llama-server's own flags: `up` has none of its own for them, and `--profile` is
        # the only thing that passes them, so the line says so rather than reading as
        # something a person could type
        lines.append(f"              and {' '.join(profile.extra_args)} "
                     f"-- llama-server's own, passed by --profile")
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
