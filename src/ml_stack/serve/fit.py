"""How many people fit on one machine at a given context -- from what llama.cpp allocated.

A formula over a GGUF's header cannot answer this. `preflight._kv_estimate_bytes` counts
every layer as full attention, and no interesting model is built that way any more:

* `qwen4exp` (Qwen3.8-Flash-Next) carries `full_attention_interval = 4` -- one layer in four
  holds a token cache and the other three are recurrent, with a fixed state per *sequence*
  rather than per token -- and those attention layers then compress their keys again.
* `gemma4` has a sliding-window pattern (a bool per layer, a 512-token window, its own
  `key_length_swa`) and `shared_kv_layers = 18`: eighteen layers own no cache at all.
* `gpt-oss` has a 128-token window with no pattern, and llama.cpp alternates -- even layers
  slide, odd ones do not.

Each of those is a different multiplier on the same header, and the header does not say
which. llama.cpp does: at load it prints exactly what it allocated, per cache, in MiB.
`ml_stack.serve.loadlog` reads those lines into **bytes per token of context** and **bytes
fixed per sequence**, and this module keeps them, per model, in one file:
`ml_stack/data/fit.json`, with `~/.ml-stack/fit.json` layered over it for a machine's own
additions. `ml-stack-serve fit` is the command over this.

The two numbers are what make the question answerable in either direction::

    cost(context)  = per_token * context + per_seq
    users(context) = (room - weights - draft - compute) // cost(context)
    longest(n)     = ((room - weights - draft - compute) // n - per_seq) // per_token

Measured, never assumed: `ml_stack.serve.measuring.measure` serves the model once at `-lv 4`
(the llama.cpp library's own INFO lines are LOG_LEVEL_TRACE, so verbosity 3 -- the server's
default -- prints the server's lines and none of these), reads the log the backend already
writes, and stops the server again.

Why a model is smaller in memory than it is on disk
---------------------------------------------------

A 103.7G file that never costs more than about 90G of "Real Mem" is not a mystery and not a
mis-measurement -- it is mmap. llama.cpp maps the GGUF and then, per tensor, decides which
backend buffer the tensor belongs in. A tensor the GPU backend takes is copied into a device
buffer and is resident; a tensor it does not take is left **where it already is** -- in the
mapped file -- and is read through the page cache. The same load log says which is which,
one line per backend::

    load_tensors:   CPU_Mapped model buffer size =  1872.00 MiB
    load_tensors:  MTL0_Mapped model buffer size =  4005.31 MiB

Four things end up in the `CPU_Mapped` half, and they are what to look for:

* **A lookup table that is gathered rather than multiplied.** This is the big one, and it
  is what Flash-Next's missing gigabytes are. `Qwen3.8-Flash-Next-UD-Q4_K_XL` is 103.7G on
  disk, of which a *single* tensor -- `per_layer_token_embd.weight`, IQ4_NL, shape
  (160, 320001536) -- is 51.2B parameters and 26.8G of the file. The header says what it is
  for: `qwen4exp.ple.ngram_size = 3`, `heads_per_ngram = 8`,
  `embedding_length_per_layer_input = 160` -- sixteen heads of roughly twenty million rows,
  an n-gram embedding table. A gather touches only the rows whose n-grams actually occur,
  so with mmap the other rows' pages never become resident at all. The process settles at
  about 90G -- the ~77G of everything else (the 512 experts are Q8_0; `ffn_down_exps` alone
  is 0.8G a layer) plus however much of the table has been walked so far -- and climbs
  slowly, bounded above by 26.8G of table.
* **Tensors the backend has no kernel for at that type.** llama.cpp says so out loud --
  ``tensor 'X' (q6_K) (and N others) cannot be used with preferred buffer type ..., using
  CPU instead`` -- and a mixed quantisation is where this bites. `UD-Q4_K_XL` is not one
  type: it is Q4_K for most of the weights and something wider for the parts that matter,
  and the wide parts are the ones a backend is most likely to decline.
* **The token embeddings, and on some architectures the output/`lm_head`.** These are a
  lookup and a single matmul at the ends of the graph; several architectures leave them
  mapped on purpose. ``load_tensors: offloading output layer to GPU`` is llama.cpp saying
  it did *not* do that here -- its absence is the tell.
* **Anything past ``--n-gpu-layers``.** ``load_tensors: offloaded 43/43 layers to GPU`` is
  the whole story in one line; ``offloaded 39/43`` means four layers are being paged.

None of that is free -- a mapped tensor is read from the page cache on every token that
touches it -- but none of it is counted in Activity Monitor's "Real Mem" either, which is
why a model appears to shrink. So the file size is never the intercept. `Fit.loaded()`
takes, in order of preference: a **measured resident** total (`weights_resident`, from a
peak RSS after a real run, with that run's own caches taken back off), else the
**GPU-resident** total off the load log (`weights_gpu`), else the old file-size sum. The
first two are the part that has to fit under `iogpu.wired_limit_mb` beside the KV cache; a
paged lookup table competes for ordinary page cache instead, and `table_bytes` records how
much of it is out there as an upper bound on the drift.

For a specific model, ``ml-stack-serve fit MODEL --tensors`` (`ml_stack.serve.tensors`)
totals the GGUF header's own per-tensor sizes -- the largest tensors with their type and
shape, the lookup tables flagged, and experts against attention against table -- which
answers "what is the 15.7G *of*" without serving anything.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path

from ml_stack.hub import pretty_name
from ml_stack.records import Records
from ml_stack.serve.loadlog import Measured
from ml_stack.units import human_bytes

__all__ = [
    "DEFAULT_PER_USER", "Fit", "add", "local_file", "package_file", "parse_room",
    "records", "render", "writable_file",
]

# The per-user contexts the table asks about, unless told otherwise. 4k is a question with
# its tools; 128k is a whole conversation kept open.
DEFAULT_PER_USER: tuple[int, ...] = (4096, 8192, 16384, 32768, 65536, 131072)


# ------------------------------------------------------------------ one measured model

@dataclass(frozen=True, slots=True)
class Fit:
    """One model, measured once, and what it means for a machine with this much room.

    ``room`` is what a model may actually use here -- `hub.room()`, not the installed RAM.
    Everything else was read off a load. ``spec`` is the guessing-ahead kind it was measured
    with (``""`` for none): a draft *model* keeps its own cache, so the same weights at the
    same cache type are a different measurement with one and without.
    """

    model: str
    weights: int = 0
    draft: int = 0
    room: int = 0
    per_token: int = 0
    per_seq: int = 0
    compute: int = 0
    cache_type: str = "f16"
    spec: str = ""
    draft_per_token: int = 0
    """Bytes of KV cache the draft head keeps per token of context, per slot. A draft keeps
    its own cache at the target's context, so this is charged per user like the model's."""
    draft_per_seq: int = 0
    """Bytes the draft head costs a slot whatever its context is."""
    draft_cache_type: str = ""
    """What the head's cache stores. "" for a record with no head, or one measured before
    the head's cache was read apart from the model's."""
    build: str = ""
    measured_at: str = ""
    context: int = 0
    parallel: int = 0
    kv_layers: int = 0
    recurrent_layers: int = 0
    swa_cells: int = 0
    weights_gpu: int = 0
    """Weights resident on a device, off the load log -- target, draft and projector
    together. 0 for a record measured before this was read."""
    weights_cpu: int = 0
    """Weights left mapped in the file and paged. The difference between the file size and
    what a process appears to hold."""
    cpu_tensors: str = ""
    """What the load log said is on the CPU, when it said: layers past ``--n-gpu-layers``,
    an output that was not offloaded, a tensor a backend had no kernel for."""
    table_bytes: int = 0
    """Bytes of gathered lookup table in the file -- `per_layer_token_embd`, n-gram and
    engram tensors. Read from the GGUF header, not the log. This is the part that is paged
    in a row at a time as distinct keys are seen, so it is an *upper bound* on how much the
    resident figure can still climb, never a cost paid at load."""
    weights_resident: int = 0
    """What the process actually held for weights, from a peak RSS after a real run with
    that run's own caches and compute buffers taken back off. The truest intercept there
    is, and 0 until somebody measures one -- see ``Fit.of``'s ``resident_peak``."""
    resident_after: int = 0
    """How many questions had been answered when that peak was taken. A resident figure
    with no run length beside it cannot be argued with: a table paged in a row at a time
    reads low after two questions and high after two hundred."""

    @property
    def weights_file(self) -> int:
        """What the model weighs on disk: the target's file(s) and any draft head's.

        Never the intercept. A GGUF is mmapped, and the tensors a backend declines are read
        through the page cache rather than copied anywhere -- so this is reliably the
        largest of the three numbers and reliably the wrong one to plan memory with.
        """
        return self.weights + self.draft

    @property
    def key(self) -> tuple[str, str, str, str]:
        """What a record is keyed by in the file: the model file's basename, the cache type
        it was measured with, the speculation kind, and the head's own cache type."""
        return (self.model, self.cache_type, self.spec, self.draft_cache_type)

    @property
    def token_bytes(self) -> int:
        """Bytes one slot pays per token of context: the model's cache and the head's."""
        return self.per_token + self.draft_per_token

    @property
    def seq_bytes(self) -> int:
        """Bytes one slot pays whatever its context: the model's and the head's."""
        return self.per_seq + self.draft_per_seq

    def free(self) -> int:
        """Bytes left for caches once the weights, a draft and the compute buffers are in.
        Never negative: a model that does not fit at all has no room for anyone."""
        return max(0, self.room - self.loaded())

    def cost(self, per_user_context: int) -> int:
        """What one more user at ``per_user_context`` tokens costs, in bytes -- the model's
        cache and, where a head is served, the head's own."""
        return self.token_bytes * max(0, int(per_user_context)) + self.seq_bytes

    def draft_cost(self, per_user_context: int) -> int:
        """What of `cost` is the draft head's own cache. 0 where none is served."""
        return self.draft_per_token * max(0, int(per_user_context)) + self.draft_per_seq

    def loaded(self) -> int:
        """What the model costs with nobody on it: the weights that are actually resident,
        plus the compute buffers. The number a person means by "how big is it", and the one
        that does *not* grow -- which is why a large model with a cheap cache overtakes a
        small one with an expensive cache somewhere, rather than never.

        Which weights count, in order of what has been measured:

        1. ``weights_resident`` -- a peak RSS after a real run, less that run's caches.
        2. ``weights_gpu`` -- what the load log says landed in a device buffer.
        3. ``weights_file`` -- the file size, for a record measured before either existed.

        The file size is the fallback and not the answer: with mmap, a tensor the backend
        declines stays in the mapped file and is paged, so a 103.7G Flash-Next settles
        around 90G resident and the file size overstates the intercept by the part of the
        n-gram table nobody has walked yet. The module docstring has the whole story.
        """
        return (self.weights_resident or self.weights_gpu or self.weights_file
                ) + self.compute

    def line(self, per_user_context: int) -> tuple[int, int]:
        """``(bytes with nobody on it, bytes each user adds)`` at that context.

        The whole memory story as two numbers, so a chart of it is a straight line and a
        test of that chart is an equality rather than a picture. Everything the second
        panel draws is ``loaded() + users * cost(context)``.
        """
        return self.loaded(), self.cost(per_user_context)

    def users(self, per_user_context: int) -> int:
        """How many users fit at that context. 0 when even one does not."""
        each = self.cost(per_user_context)
        return self.free() // each if each > 0 else 0

    def longest(self, parallel: int = 1) -> int:
        """The longest context ``parallel`` users can each be given. 0 when they do not fit.

        The whole answer read the other way round -- the same two numbers, solved for the
        context rather than the head count.
        """
        parallel = max(1, int(parallel))
        if self.token_bytes <= 0:
            return 0
        each = self.free() // parallel - self.seq_bytes
        return max(0, each // self.token_bytes)

    def at_room(self, room: int) -> Fit:
        """The same measurement, asked about a machine with this much room instead."""
        return replace(self, room=int(room))

    def as_dict(self) -> dict:
        return {
            "model": self.model, "weights": self.weights, "draft": self.draft,
            "room": self.room, "per_token": self.per_token, "per_seq": self.per_seq,
            "compute": self.compute, "cache_type": self.cache_type, "spec": self.spec,
            "draft_per_token": self.draft_per_token, "draft_per_seq": self.draft_per_seq,
            "draft_cache_type": self.draft_cache_type,
            "build": self.build, "measured_at": self.measured_at, "context": self.context,
            "parallel": self.parallel, "kv_layers": self.kv_layers,
            "recurrent_layers": self.recurrent_layers, "swa_cells": self.swa_cells,
            "weights_gpu": self.weights_gpu, "weights_cpu": self.weights_cpu,
            "cpu_tensors": self.cpu_tensors, "table_bytes": self.table_bytes,
            "weights_resident": self.weights_resident,
            "resident_after": self.resident_after,
        }

    @classmethod
    def from_dict(cls, row: dict) -> Fit:
        """One record read back. Unknown keys are ignored so an older file still loads,
        and a missing one takes the default rather than raising."""
        fields = {f for f in cls.__slots__}
        return cls(**{k: v for k, v in row.items() if k in fields and k != "model"},
                   model=str(row.get("model") or ""))

    @classmethod
    def of(cls, measured: Measured, *, model: str, weights: int = 0, draft: int = 0,
           room: int = 0, cache_type: str = "", spec: str = "", context: int = 0,
           parallel: int = 0, build: str = "", when: str = "", table_bytes: int = 0,
           resident_peak: int = 0, resident_after: int = 0) -> Fit:
        """A record from one measurement and the sizes around it.

        ``resident_peak`` is a whole process's peak RSS after a real run -- the bench's
        ``resident_peak``, or anything else that watched the server. It is turned into a
        *weights* figure here rather than stored raw, because an RSS holds the caches of
        the run it was measured in as well: the compute buffers and one cache per slot come
        back off, using the numbers this same load produced. Never below what the log said
        was resident on the GPU, because that part cannot be paged out.
        """
        per_token = measured.per_token + measured.draft_per_token
        per_seq = measured.per_seq + measured.draft_per_seq
        resident = 0
        if resident_peak > 0:
            held = max(1, int(parallel or 1)) * (per_token * max(0, int(context)) + per_seq)
            resident = max(measured.weights_gpu,
                           int(resident_peak) - measured.compute - held)
        return cls(
            model=model or measured.model_file,
            weights=weights, draft=draft, room=room,
            per_token=measured.per_token, per_seq=measured.per_seq,
            compute=measured.compute,
            cache_type=cache_type or measured.cache_type or "f16",
            spec=spec, build=build or measured.build,
            draft_per_token=measured.draft_per_token, draft_per_seq=measured.draft_per_seq,
            draft_cache_type=measured.draft_cache_type,
            measured_at=when or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            context=context, parallel=parallel, kv_layers=measured.kv_layers,
            recurrent_layers=measured.recurrent_layers, swa_cells=measured.swa_cells,
            weights_gpu=measured.weights_gpu, weights_cpu=measured.weights_cpu,
            cpu_tensors=measured.why_cpu(), table_bytes=int(table_bytes),
            weights_resident=resident, resident_after=int(resident_after))


# ------------------------------------------------------------------ the file it lives in

_STORE: Records[Fit] = Records(
    "fit.json", env="MLSTACK_FIT_FILE",
    build=Fit.from_dict, unbuild=lambda f: f.as_dict(), key=lambda f: f.key,
    order=lambda f: (f.model.lower(), f.cache_type, f.spec, f.draft_cache_type))


def package_file() -> Path:
    """The measurements that ship with ml-stack -- the single source of truth."""
    return _STORE.package_path()


def local_file() -> Path:
    """This machine's own additions, layered over the shipped ones. `$MLSTACK_FIT_FILE`
    moves it."""
    return _STORE.local_path()


def records(*, package: Path | None = None, local: Path | None = None,
            room: int | None = None) -> list[Fit]:
    """Every measured model: the shipped file, with this machine's own layered over it.

    A local record with the same (model, cache type, speculation) key replaces the shipped
    one. ``room`` overrides the room every record was recorded with.
    """
    out = _STORE.all(package=package or package_file(), local=local or local_file())
    if room is not None:
        out = [fit.at_room(room) for fit in out]
    return out


def writable_file() -> Path:
    """Where a new measurement goes: the shipped file in a checkout somebody can write to,
    this machine's own file otherwise."""
    return _STORE.writable_path()


def add(fit: Fit, *, path: Path | None = None) -> Path:
    """Write one measurement into the source of truth, replacing any it supersedes.

    Returns where it was written, which is what `--measure` prints.
    """
    return _STORE.add(fit, path=path or writable_file())


# ------------------------------------------------------------------ saying it

_ROOM = re.compile(r"^\s*([\d.]+)\s*([KMGT]?)(?:i?B?)?\s*$", re.IGNORECASE)
_SCALE = {"": 1, "K": 1024, "M": 1024 ** 2, "G": 1024 ** 3, "T": 1024 ** 4}


def parse_room(text: str) -> int:
    """``24G``, ``24GiB``, ``24576M``, ``25769803776`` -- all the same number of bytes.

    A bare number is bytes, because that is what every other number in this module is.
    Raises ``ValueError`` on anything else rather than guessing: a room misread by a factor
    of 1024 answers the question confidently and wrongly.
    """
    match = _ROOM.match(str(text))
    if not match:
        raise ValueError(f"cannot read {text!r} as an amount of memory; try 24G")
    return int(float(match.group(1)) * _SCALE[match.group(2).upper()])


def _tokens(count: int) -> str:
    return f"{count:,}"


def render(fits: Iterable[Fit], per_user: Sequence[int] = DEFAULT_PER_USER,
           room: int | None = None, md: bool = False) -> str:
    """One block per model: what it costs, and who fits.

    ``room`` re-asks every record against a different machine. ``md`` writes the same thing
    as Markdown, for `--write docs/fit.md`.
    """
    rows = [fit.at_room(room) for fit in fits] if room is not None else list(fits)
    if not rows:
        return ("No model has been measured yet. `ml-stack-serve fit MODEL --measure` "
                "serves it once and records what it allocated.")
    contexts = [int(c) for c in per_user if int(c) > 0] or list(DEFAULT_PER_USER)
    return ("\n\n".join(_block_md(f, contexts) for f in rows) if md
            else "\n\n".join(_block(f, contexts) for f in rows))


def _headline(fit: Fit) -> str:
    bits = [f"{fit.cache_type} cache"]
    if fit.spec:
        bits.append(f"guessing ahead by {fit.spec}")
    if fit.context:
        bits.append(f"measured at {_tokens(fit.context)} tokens"
                    + (f" over {fit.parallel} slots" if fit.parallel else ""))
    if fit.build:
        bits.append(f"build {fit.build}")
    if fit.measured_at:
        bits.append(fit.measured_at)
    return ", ".join(bits)


def _drafts(fit: Fit) -> str:
    """What the draft head's own cache costs a slot, or "" where none is served."""
    if not (fit.draft_per_token or fit.draft_per_seq):
        return ""
    said = [f"the draft head keeps its own cache: {human_bytes(fit.draft_per_token)} per "
            f"token of context"]
    if fit.draft_per_seq:
        said.append(f"{human_bytes(fit.draft_per_seq)} fixed per sequence")
    if fit.draft_cache_type:
        said.append(f"stored as {fit.draft_cache_type}")
    return ", ".join(said)


def _shape(fit: Fit) -> str:
    bits = [f"{fit.kv_layers} layers with a cache"]
    if fit.recurrent_layers:
        bits.append(f"{fit.recurrent_layers} recurrent (a fixed state per sequence, not "
                    f"per token)")
    if fit.swa_cells:
        bits.append(f"a {_tokens(fit.swa_cells)}-cell sliding window per sequence")
    return "; ".join(bits)


def _where_it_went(fit: Fit) -> list[str]:
    """The file size, and what of it is actually resident -- at most two lines.

    Said out loud because the two numbers differ by tens of gigabytes on exactly the models
    worth serving, and every reasonable-looking assumption about the difference is wrong.
    A record measured before any of this was read says nothing extra rather than saying
    zeroes, which would read as "none of it is on the GPU".
    """
    if not (fit.weights_gpu or fit.weights_cpu or fit.weights_resident):
        return []
    said = [f"{human_bytes(fit.weights_file)} on disk: {human_bytes(fit.weights_gpu)} in GPU memory, "
            f"{human_bytes(fit.weights_cpu)} mapped on the CPU"
            + (f" ({fit.cpu_tensors})" if fit.cpu_tensors
               else " (`fit MODEL --tensors` says what of)")]
    after = []
    if fit.table_bytes:
        after.append(f"a {human_bytes(fit.table_bytes)} lookup table is paged on demand, a row "
                     "at a time")
    if fit.weights_resident:
        after.append(f"resident after {_tokens(fit.resident_after)} questions "
                     f"{human_bytes(fit.weights_resident)} (measured)"
                     if fit.resident_after
                     else f"resident {human_bytes(fit.weights_resident)} (measured)")
    if after:
        said.append("of which " + "; ".join(after) if fit.table_bytes
                    else "; ".join(after))
    return said


def _block(fit: Fit, contexts: list[int]) -> str:

    lines = [f"{pretty_name(fit.model)}", f"  {_headline(fit)}"]
    lines.append(
        f"  weights {human_bytes(fit.weights)}"
        + (f", draft {human_bytes(fit.draft)}" if fit.draft else "")
        + f", compute {human_bytes(fit.compute)}"
        + f" -- of {human_bytes(fit.room)} room, {human_bytes(fit.free())} is left for caches")
    lines += [f"  {said}" for said in _where_it_went(fit)]
    lines.append(f"  {human_bytes(fit.per_token)} per token of context, "
                 f"{human_bytes(fit.per_seq)} fixed per sequence")
    if _drafts(fit):
        lines.append(f"  {_drafts(fit)}")
    shape = _shape(fit)
    if shape:
        lines.append(f"  {shape}")
    lines.append("")
    lines.append("  per user context   users that fit   each costs   of which the head")
    for context in contexts:
        lines.append(f"  {_tokens(context):>16}   {fit.users(context):>14}   "
                     f"{human_bytes(fit.cost(context)):>10}   "
                     f"{(human_bytes(fit.draft_cost(context)) if fit.draft_per_token else '-'):>17}")
    lines.append(f"  one user, longest context: {_tokens(fit.longest(1))} tokens")
    return "\n".join(lines)


def _block_md(fit: Fit, contexts: list[int]) -> str:
    lines = [f"### {pretty_name(fit.model)}", "", f"{_headline(fit)}.", ""]
    lines.append(
        f"- weights {human_bytes(fit.weights)}"
        + (f", draft {human_bytes(fit.draft)}" if fit.draft else "")
        + f", compute {human_bytes(fit.compute)}")
    lines += [f"- {said}" for said in _where_it_went(fit)]
    lines.append(f"- room {human_bytes(fit.room)}, of which {human_bytes(fit.free())} is left for "
                 f"caches")
    lines.append(f"- **{human_bytes(fit.per_token)} per token of context**, "
                 f"**{human_bytes(fit.per_seq)} fixed per sequence**")
    if _drafts(fit):
        lines.append(f"- {_drafts(fit)}")
    shape = _shape(fit)
    if shape:
        lines.append(f"- {shape}")
    lines += ["", "| per user context | users that fit | each costs | of which the head |",
              "| --- | --- | --- | --- |"]
    for context in contexts:
        lines.append(f"| {_tokens(context)} | {fit.users(context)} | "
                     f"{human_bytes(fit.cost(context))} | "
                     f"{human_bytes(fit.draft_cost(context)) if fit.draft_per_token else '-'} |")
    lines += ["", f"One user, longest context: **{_tokens(fit.longest(1))} tokens**."]
    return "\n".join(lines)
