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
which. llama.cpp does: at load it prints exactly what it allocated, per cache, in MiB. This
module reads those lines, turns them into two numbers that compose -- **bytes per token of
context** and **bytes fixed per sequence** -- and keeps them, per model, in one file:
`ml_stack/data/fit.json`, with `~/.ml-stack/fit.json` layered over it for a machine's own
additions. `ml-stack-serve fit` is the command over this.

The two numbers are what make the question answerable in either direction::

    cost(context)  = per_token * context + per_seq
    users(context) = (room - weights - draft - compute) // cost(context)
    longest(n)     = ((room - weights - draft - compute) // n - per_seq) // per_token

Measured, never assumed: `measure()` serves the model once at `-lv 4` (the llama.cpp
library's own INFO lines are LOG_LEVEL_TRACE, so verbosity 3 -- the server's default --
prints the server's lines and none of these), reads the log the backend already writes, and
stops the server again.

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

For a specific model, ``ml-stack-serve fit MODEL --tensors`` totals the GGUF header's own
per-tensor sizes -- the largest tensors with their type and shape, the lookup tables
flagged, and experts against attention against table -- which answers "what is the 15.7G
*of*" without serving anything.
"""

from __future__ import annotations

import json
import os
import re
import struct
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from ml_stack.hub import pretty_name

__all__ = [
    "DEFAULT_PER_USER", "Fit", "Measured", "PLOT_CONTEXTS", "Segment", "Tensor", "add",
    "label_of", "local_file", "measure", "package_file", "parse_load_log", "parse_room",
    "plot", "records", "render", "render_tensors", "table_bytes", "tensors_of",
    "totals_by_role", "totals_by_type", "writable_file",
]

_MIB = 1024 * 1024

# The per-user contexts the table asks about, unless told otherwise. 4k is a question with
# its tools; 128k is a whole conversation kept open.
DEFAULT_PER_USER: tuple[int, ...] = (4096, 8192, 16384, 32768, 65536, 131072)


# ------------------------------------------------------------------ reading a load log

# `llama_kv_cache: size = 1234.00 MiB ( 65536 cells,  12 layers,  2/1 seqs), K (f16): ...`
# (llama-kv-cache.cpp, the constructor's own summary). A model with sliding-window layers
# is an `llama_kv_cache_iswa`, which builds two of these and so prints the line twice: the
# base cache first, at the full context, then the SWA one at a few hundred cells.
_KV_SIZE = re.compile(
    r"llama_kv_cache:\s*size\s*=\s*([\d.]+)\s*MiB\s*\(\s*(\d+)\s*cells,\s*(\d+)\s*layers,"
    r"\s*(\d+)\s*/\s*(\d+)\s*seqs\).*?K\s*\(([^)]*)\):\s*([\d.]+)\s*MiB,"
    r"\s*V\s*\(([^)]*)\):\s*([\d.]+)\s*MiB")

# `llama_memory_recurrent: size = 12.00 MiB ( 2 cells, 36 layers, 2 seqs 2 rs_seq), R ...`
# One cell per sequence: a recurrent layer keeps a state, not a history, so its cost does
# not grow with the context at all -- which is the whole reason a hybrid model is worth
# serving at a long one.
_RS_SIZE = re.compile(
    r"llama_memory_recurrent:\s*size\s*=\s*([\d.]+)\s*MiB\s*\(\s*(\d+)\s*cells,"
    r"\s*(\d+)\s*layers,\s*(\d+)\s*seqs")

# `sched_reserve:      Metal compute buffer size =   304.00 MiB` -- one line per backend,
# printed once after every graph has been reserved. The function name in front of it has
# moved between releases (it was `llama_context:`), so nothing here depends on it.
_COMPUTE = re.compile(r"(\S+)\s+compute buffer size\s*=\s*([\d.]+)\s*MiB")

# `llama_model_loader: loaded meta data with 40 key-value pairs and 300 tensors from PATH`
# -- printed once per model, and the only reliable boundary between the target's numbers
# and a draft head's, which are otherwise the same lines again a few hundred lines later.
_LOADED = re.compile(
    r"llama_model_loader:\s*loaded meta data with\s*\d+\s*key-value pairs and\s*\d+\s*"
    r"tensors from\s*(\S+)")

# `clip_model_loader: model name:  ...` -- the projector is loaded by mtmd's own reader,
# which prints neither `llama_model_loader` nor a per-backend buffer line. It is still a
# model that arrives in memory, so it is still a segment; its size comes from
# `clip_model_loader: model size: X MiB` and its backend from `clip_ctx: CLIP using X`.
_CLIP_START = re.compile(r"clip_model_loader:\s*model name:")
_CLIP_SIZE = re.compile(r"clip_model_loader:\s*model size:\s*([\d.]+)\s*MiB")
_CLIP_FILE = re.compile(r"clip_model_loader:\s*loaded\s+\d+\s+tensors from\s+(\S+)")
_CLIP_BACKEND = re.compile(r"clip_ctx:\s*CLIP using (\S+) backend")

# `load_tensors:   CPU_Mapped model buffer size =  1872.00 MiB` -- one line per backend the
# weights landed in, and the only place llama.cpp says where they went. `_COMPUTE` above
# matches "compute buffer size"; this one must not, which is why "model" is in the pattern.
_TENSOR_BUFFER = re.compile(r"load_tensors:\s*(\S+)\s+model buffer size\s*=\s*([\d.]+)\s*MiB")

# `load_tensors: offloaded 43/43 layers to GPU`, and the two lines above it. `offloaded
# 39/43` is four layers being paged; a missing "offloading output layer" is the output
# staying mapped on the CPU, which several architectures do on purpose.
_OFFLOADED = re.compile(r"load_tensors:\s*offloaded\s+(\d+)\s*/\s*(\d+)\s+layers to GPU")
_OUTPUT_ON_GPU = re.compile(r"load_tensors:\s*offloading output layer to GPU")

# `tensor 'X' (q6_K) (and 12 others) cannot be used with preferred buffer type Metal, using
# CPU instead`, and `tensor X (48 MiB q6_K) buffer type overridden to CPU` -- llama.cpp
# naming, out loud, a tensor the backend declined. Exactly the "which tensors" a person
# wants when a model is smaller in memory than on disk.
_NO_KERNEL = re.compile(
    r"tensor '([^']+)'\s*\(([^)]*)\)\s*\(and (\d+) others\) cannot be used with preferred "
    r"buffer type (\S+), using (\S+) instead")
_OVERRIDDEN = re.compile(
    r"tensor\s+(\S+)\s*\(\s*\d+\s*MiB\s+([^)]*)\)\s*buffer type overridden to\s+(\S+)")

# `llama_model_loader: - type q4_0:  345 tensors` -- the mixture a quantisation actually is.
_TYPE_COUNT = re.compile(r"llama_model_loader:\s*-\s*type\s+(\S+):\s*(\d+)\s+tensors")

# `cmn  common_param: build N (<commit>) with Apple clang ...` -- LOG_TRC, so it is in the
# log at `-lv 4` for the same reason everything else here is.
_BUILD = re.compile(r"\bbuild\s+\d+\s+\(([0-9A-Za-z._-]+)\)")

# The backends that are the CPU wearing a hat. Everything else -- MTL0, Metal, CUDA0,
# ROCm0, Vulkan0, SYCL0 -- holds a device buffer whose bytes are resident. Matched on the
# part before the underscore, so `CPU_Mapped` and `CPU_REPACK` are the CPU and
# `MTL0_Mapped` is not.
_HOST_BACKENDS = frozenset({"CPU", "AMX", "BLAS", "ACCELERATE", "HOST"})


def _on_gpu(backend: str) -> bool:
    """Whether a buffer named by llama.cpp is resident on a device rather than mapped."""
    return backend.split("_", 1)[0].upper() not in _HOST_BACKENDS


def _mib(text: str) -> int:
    return int(round(float(text) * _MIB))


@dataclass(frozen=True, slots=True)
class Segment:
    """One model's arrival in memory, as its own load log says it went.

    A server load is more than one model: the target, then a draft head, then a projector,
    each printing the same lines again. They are kept apart and in load order, because
    "where did the weights go" has a different answer for each and a summed one is not
    checkable against anything.
    """

    kind: str = "target"
    """``target``, ``draft`` or ``mmproj`` -- the first model loaded is the target."""
    model_file: str = ""
    buffers: tuple[tuple[str, int], ...] = ()
    """``(backend, bytes)`` in the order llama.cpp printed them: `CPU_Mapped`,
    `MTL0_Mapped`, `CUDA0`. The whole answer to where the weights went."""
    offloaded: int = 0
    """Layers put on the GPU, from ``offloaded N/M layers to GPU``."""
    layers: int = 0
    """The M of that line. ``offloaded < layers`` is layers past ``--n-gpu-layers``."""
    output_on_gpu: bool = False
    """Whether ``offloading output layer to GPU`` was printed. Its absence is the output
    staying mapped, which is a real and frequently large part of the gap."""
    types: tuple[tuple[str, int], ...] = ()
    """``llama_model_loader: - type q4_0: 345 tensors`` -- what the quantisation is a
    mixture of, which is where a backend finds something it has no kernel for."""
    declined: tuple[str, ...] = ()
    """Tensors llama.cpp said a backend would not take, verbatim enough to grep for."""

    @property
    def gpu(self) -> int:
        """Bytes resident on a device."""
        return sum(size for name, size in self.buffers if _on_gpu(name))

    @property
    def cpu(self) -> int:
        """Bytes left mapped in the file and paged through the page cache."""
        return sum(size for name, size in self.buffers if not _on_gpu(name))

    @property
    def total(self) -> int:
        return sum(size for _, size in self.buffers)

    def why_cpu(self) -> str:
        """What the log itself says is on the CPU, or ``""`` when it does not say.

        Never a guess: every clause here is a line llama.cpp printed. A load that explains
        nothing gets an empty string and the caller points at `--tensors` instead, which is
        an honest "go and look" rather than a plausible wrong answer.
        """
        said: list[str] = []
        if self.layers and self.offloaded < self.layers:
            said.append(f"{self.layers - self.offloaded} of {self.layers} layers past "
                        "--n-gpu-layers")
        if self.offloaded and not self.output_on_gpu:
            said.append("the output layer, which was not offloaded")
        said += list(self.declined)
        return "; ".join(said)


@dataclass(frozen=True, slots=True)
class Measured:
    """What one load actually allocated, read off llama.cpp's own log.

    ``per_token`` and ``per_seq`` are the two numbers that compose; everything else is what
    was seen while working them out, kept so a surprising answer can be argued with.
    """

    per_token: int = 0
    """Bytes of KV cache per token of context. The base cache's size divided by its cells;
    the cells are the context, whether or not the slots share them."""
    per_seq: int = 0
    """Bytes every sequence costs no matter how long its context is: the recurrent state,
    and the sliding-window cache, each divided by the sequences it was sized for."""
    compute: int = 0
    """The compute buffers, summed over the backends. Paid once, not per user."""
    cache_type: str = ""
    """What the base cache stores, as llama.cpp names it: `f16`, `q8_0`. K and V are joined
    with a `/` when they differ."""
    kv_layers: int = 0
    recurrent_layers: int = 0
    cells: int = 0
    seqs: int = 0
    swa_cells: int = 0
    kv_bytes: int = 0
    swa_bytes: int = 0
    recurrent_bytes: int = 0
    model_file: str = ""
    build: str = ""
    segments: tuple[Segment, ...] = ()
    """Every model the load brought in, in load order -- the target, a draft head, a
    projector -- and where each one's weights ended up, per backend."""

    @property
    def measured(self) -> bool:
        """Whether the log said anything at all. False for a log written at the server's
        default verbosity, where none of these lines exist."""
        return bool(self.per_token or self.per_seq or self.compute)

    @property
    def weights_gpu(self) -> int:
        """Bytes of weights resident on a device, over every model the load brought in."""
        return sum(one.gpu for one in self.segments)

    @property
    def weights_cpu(self) -> int:
        """Bytes of weights left mapped on the CPU, over every model the load brought in.
        The part "Real Mem" does not count and the file size does."""
        return sum(one.cpu for one in self.segments)

    def why_cpu(self) -> str:
        """What the log says is on the CPU, named per model when more than one is."""
        said = []
        for one in self.segments:
            because = one.why_cpu()
            if because:
                said.append(because if one.kind == "target" else f"{one.kind}: {because}")
        return "; ".join(said)

    def said(self) -> str:
        """One line per fact, for a person reading a `--measure` that surprised them."""
        parts = [
            f"per token {_human(self.per_token)}",
            f"per sequence {_human(self.per_seq)}",
            f"compute {_human(self.compute)}",
            f"{self.kv_layers} layers with a cache",
        ]
        if self.recurrent_layers:
            parts.append(f"{self.recurrent_layers} recurrent")
        if self.swa_cells:
            parts.append(f"a {self.swa_cells}-cell sliding window")
        if self.cache_type:
            parts.append(f"cache {self.cache_type}")
        if self.segments:
            parts.append(f"weights {_human(self.weights_gpu)} on the GPU, "
                         f"{_human(self.weights_cpu)} mapped on the CPU")
        return ", ".join(parts)


def _segments(text: str) -> list[str]:
    """The log split at each model load, so a draft head's cache is not read as the
    target's. The text before the first load is dropped: nothing is allocated yet there.

    A projector is a load too, and mtmd's reader announces itself differently, so
    `clip_model_loader` starts a segment as well as `llama_model_loader`.
    """
    bounds = sorted(m.start() for m in
                    [*_LOADED.finditer(text), *_CLIP_START.finditer(text)])
    if not bounds:
        return [text]
    bounds.append(len(text))
    return [text[bounds[i]:bounds[i + 1]] for i in range(len(bounds) - 1)]


def _kind_of(name: str, index: int) -> str:
    """target / draft / mmproj, from the file's own name and the order it loaded in.

    Order alone is not enough -- a projector can load before or after a draft head -- and
    the name alone is not either, since a target is named whatever somebody named it. The
    first model loaded is always the target; the rest are read off their names.
    """
    low = name.lower()
    if "mmproj" in low or low.startswith("clip") or "-clip" in low:
        return "mmproj"
    if index == 0:
        return "target"
    return "draft"


def _segment_of(text: str, index: int) -> Segment | None:
    """One model's part of the log, as a ``Segment`` -- or None when it brought nothing."""
    named = _LOADED.search(text) or _CLIP_FILE.search(text)
    model_file = Path(named.group(1)).name if named else ""

    buffers: list[tuple[str, int]] = []
    for one in _TENSOR_BUFFER.finditer(text):
        buffers.append((one.group(1), _mib(one.group(2))))
    if not buffers:
        # A projector prints one total and the backend it chose, not a line per buffer.
        size, backend = _CLIP_SIZE.search(text), _CLIP_BACKEND.search(text)
        if size:
            buffers.append((backend.group(1) if backend else "CPU", _mib(size.group(1))))
    if not buffers:
        return None

    offloaded = layers = 0
    found = _OFFLOADED.search(text)
    if found:
        offloaded, layers = int(found.group(1)), int(found.group(2))

    declined: list[str] = []
    for one in _NO_KERNEL.finditer(text):
        others = int(one.group(3))
        declined.append(
            f"{one.group(1)} ({one.group(2)})"
            + (f" and {others} others" if others else "")
            + f" -- no {one.group(4)} kernel, on the {one.group(5)}")
    for one in _OVERRIDDEN.finditer(text):
        declined.append(f"{one.group(1)} ({one.group(2)}) overridden to {one.group(3)}")

    return Segment(
        kind=_kind_of(model_file, index), model_file=model_file,
        buffers=tuple(buffers), offloaded=offloaded, layers=layers,
        output_on_gpu=bool(_OUTPUT_ON_GPU.search(text)),
        types=tuple((one.group(1), int(one.group(2)))
                    for one in _TYPE_COUNT.finditer(text)),
        declined=tuple(declined))


def parse_load_log(text: str) -> Measured:
    """What llama.cpp allocated, from the log it wrote at `-lv 4`.

    Robust to every line being absent: a model with no recurrent layers prints no
    `llama_memory_recurrent` line, one with no sliding window prints one `llama_kv_cache`
    line rather than two, and a log written at the default verbosity prints none of them --
    which comes back as an all-zero ``Measured`` whose ``measured`` is False, never as a
    raise. Only the *first* model's **cache** is read: a draft head loads after the target
    and prints the same lines again.

    The **weights** are read from every model in the log, kept apart as ``segments`` -- the
    target, then a draft head, then a projector -- because "how big is it in memory" is not
    the file size and llama.cpp is the only thing that knows the difference. See this
    module's own docstring for why the two numbers differ and by how much.
    """
    build = ""
    found = _BUILD.search(text)
    if found:
        build = found.group(1)

    parts = _segments(text)
    loaded = tuple(seg for seg in
                   (_segment_of(part, i) for i, part in enumerate(parts))
                   if seg is not None)

    for segment in parts:
        kv = list(_KV_SIZE.finditer(segment))
        rs = list(_RS_SIZE.finditer(segment))
        if not kv and not rs:
            continue

        source = _LOADED.search(segment)
        model_file = Path(source.group(1)).name if source else ""

        per_token = kv_bytes = cells = seqs = kv_layers = 0
        swa_bytes = swa_cells = 0
        cache_type = ""
        if kv:
            # llama_kv_cache_iswa builds the base cache first and the SWA one second; the
            # base is the one whose cells are the context.
            base, *sliding = kv
            kv_bytes, cells = _mib(base.group(1)), int(base.group(2))
            kv_layers, seqs = int(base.group(3)), int(base.group(4))
            # llama-kv-cache.cpp prints "n_seq_max/n_stream" -- one stream is one shared
            # buffer sized for every sequence; more than one is a separate full-size copy
            # per sequence, and kv_bytes is already the sum over every stream.
            streams = max(1, int(base.group(5)))
            type_k, type_v = base.group(6), base.group(8)
            cache_type = type_k if type_k == type_v else f"{type_k}/{type_v}"
            per_token = kv_bytes // (cells * streams) if cells else 0
            for other in sliding:
                swa_bytes += _mib(other.group(1))
                swa_cells = max(swa_cells, int(other.group(2)))
                kv_layers += int(other.group(3))

        recurrent_bytes = recurrent_layers = 0
        rs_seqs = 0
        if rs:
            for one in rs:
                recurrent_bytes += _mib(one.group(1))
                recurrent_layers += int(one.group(3))
                rs_seqs = max(rs_seqs, int(one.group(4)))

        # Both fixed costs were sized for however many sequences were served; one
        # sequence's share is what a user costs.
        share = max(seqs, rs_seqs, 1)
        per_seq = (recurrent_bytes + swa_bytes) // share

        # The compute buffers are one per backend and the reserve can run more than once,
        # so the last figure for each backend is the one that stands.
        by_backend: dict[str, int] = {}
        for one in _COMPUTE.finditer(segment):
            by_backend[one.group(1)] = _mib(one.group(2))

        return Measured(
            per_token=per_token, per_seq=per_seq, compute=sum(by_backend.values()),
            cache_type=cache_type, kv_layers=kv_layers, recurrent_layers=recurrent_layers,
            cells=cells, seqs=share, swa_cells=swa_cells, kv_bytes=kv_bytes,
            swa_bytes=swa_bytes, recurrent_bytes=recurrent_bytes, model_file=model_file,
            build=build, segments=loaded)

    return Measured(build=build, segments=loaded)


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
    def key(self) -> tuple[str, str, str]:
        """What a record is keyed by in the file: the model file's basename, the cache type
        it was measured with, and the speculation kind."""
        return (self.model, self.cache_type, self.spec)

    def free(self) -> int:
        """Bytes left for caches once the weights, a draft and the compute buffers are in.
        Never negative: a model that does not fit at all has no room for anyone."""
        return max(0, self.room - self.loaded())

    def cost(self, per_user_context: int) -> int:
        """What one more user at ``per_user_context`` tokens costs, in bytes."""
        return self.per_token * max(0, int(per_user_context)) + self.per_seq

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
        if self.per_token <= 0:
            return 0
        each = self.free() // parallel - self.per_seq
        return max(0, each // self.per_token)

    def at_room(self, room: int) -> Fit:
        """The same measurement, asked about a machine with this much room instead."""
        return replace(self, room=int(room))

    def as_dict(self) -> dict:
        return {
            "model": self.model, "weights": self.weights, "draft": self.draft,
            "room": self.room, "per_token": self.per_token, "per_seq": self.per_seq,
            "compute": self.compute, "cache_type": self.cache_type, "spec": self.spec,
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
        per_token, per_seq = measured.per_token, measured.per_seq
        resident = 0
        if resident_peak > 0:
            held = max(1, int(parallel or 1)) * (per_token * max(0, int(context)) + per_seq)
            resident = max(measured.weights_gpu,
                           int(resident_peak) - measured.compute - held)
        return cls(
            model=model or measured.model_file,
            weights=weights, draft=draft, room=room,
            per_token=per_token, per_seq=per_seq,
            compute=measured.compute,
            cache_type=cache_type or measured.cache_type or "f16",
            spec=spec, build=build or measured.build,
            measured_at=when or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            context=context, parallel=parallel, kv_layers=measured.kv_layers,
            recurrent_layers=measured.recurrent_layers, swa_cells=measured.swa_cells,
            weights_gpu=measured.weights_gpu, weights_cpu=measured.weights_cpu,
            cpu_tensors=measured.why_cpu(), table_bytes=int(table_bytes),
            weights_resident=resident, resident_after=int(resident_after))


# ------------------------------------------------------------------ the file it lives in

def package_file() -> Path:
    """The measurements that ship with ml-stack -- the single source of truth. A function
    rather than a constant so a test can point it somewhere with nothing in it."""
    return Path(__file__).resolve().parent.parent / "data" / "fit.json"


def local_file() -> Path:
    """This machine's own additions, layered over the shipped ones. `$MLSTACK_FIT_FILE`
    moves it, which is how the tests keep out of a real `~/.ml-stack`."""
    named = os.environ.get("MLSTACK_FIT_FILE")
    if named:
        return Path(named).expanduser()
    return Path.home() / ".ml-stack" / "fit.json"


def _read(path: Path) -> list[Fit]:
    """Every record in one file. A file that is absent, unreadable or not a list of objects
    contributes nothing -- there is no such thing as a half-measured model."""
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    if not isinstance(parsed, list):
        return []
    out: list[Fit] = []
    for row in parsed:
        if not isinstance(row, dict) or not row.get("model"):
            continue
        try:
            out.append(Fit.from_dict(row))
        except (TypeError, ValueError):
            continue
    return out


def records(*, package: Path | None = None, local: Path | None = None,
            room: int | None = None) -> list[Fit]:
    """Every measured model: the shipped file, with this machine's own layered over it.

    A local record with the same (model, cache type, speculation) key replaces the shipped
    one rather than appearing beside it -- a machine that measured a model again means the
    newer number, and a listing that showed both would make a person choose between two
    facts that are not in disagreement about anything but the date.

    ``room`` overrides the room every record was recorded with, which is how a 24 GB card
    is asked about from a machine that is not one.
    """
    merged: dict[tuple[str, str, str], Fit] = {}
    for fit in _read(package or package_file()) + _read(local or local_file()):
        merged[fit.key] = fit
    out = list(merged.values())
    if room is not None:
        out = [fit.at_room(room) for fit in out]
    return sorted(out, key=lambda f: (f.model.lower(), f.cache_type, f.spec))


def writable_file() -> Path:
    """Where a new measurement goes: the shipped file when this is a checkout somebody can
    write to, and this machine's own file otherwise. An installed wheel is not a place to
    keep a measurement -- the next upgrade would take it away."""
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


def add(fit: Fit, *, path: Path | None = None) -> Path:
    """Write one measurement into the source of truth, replacing any it supersedes.

    Returns where it was written, which is what `--measure` prints: a person who measured a
    model on a laptop and expected it in the repository should be told it went elsewhere.
    """
    where = path or writable_file()
    kept = [row for row in _read(where) if row.key != fit.key]
    kept.append(fit)
    kept.sort(key=lambda f: (f.model.lower(), f.cache_type, f.spec))
    where.parent.mkdir(parents=True, exist_ok=True)
    where.write_text(json.dumps([row.as_dict() for row in kept], indent=2) + "\n",
                     encoding="utf-8")
    return where


# ------------------------------------------------------------------ measuring one

def _load_log(spec, *, backend=None, timeout: float | None = None) -> str:
    """Serve it once, read the log the backend already writes, stop it again.

    The seam `measure` replaces in a test. Nothing is faked here: the same `ServerManager`
    every other caller leases through, so a spec that a load would refuse is refused the
    same way, by the same preflight, before anything is spawned.
    """
    from ml_stack.serve.backend import LOG_DIR, LlamaServerBackend, ServerFailed
    from ml_stack.serve.manager import ServerManager

    manager = ServerManager(backend or LlamaServerBackend())
    info = manager.lease(spec, timeout=timeout)
    try:
        if info.adopted:
            raise ServerFailed(
                f"{info.base_url} was already serving that model, and an adopted server's "
                "log is from a load that may not have been asked for -lv 4. Stop it "
                "(`ml-stack-serve down --port %d`) and measure again." % info.port)
        where = info.log_path or LOG_DIR / f"llama-server-{info.port}.log"
        return Path(where).read_text(encoding="utf-8", errors="replace")
    finally:
        manager.release(info)


def measure(spec, *, backend=None, timeout: float | None = None,
            serve: Callable[..., str] | None = None) -> Measured:
    """Serve ``spec`` once at `-lv 4`, read what it allocated, and stop it.

    The verbosity is not decoration: every line this reads is an `LLAMA_LOG_INFO` from the
    library, which `common_log_get_verbosity` maps to LOG_LEVEL_TRACE -- so the server's own
    default of 3 prints the server's lines and none of the model's. A measurement taken
    without it comes back empty and truthfully says so.
    """
    if "-lv" not in spec.extra_args:
        spec = replace(spec, extra_args=tuple(spec.extra_args) + ("-lv", "4"))
    text = (serve or _load_log)(spec, backend=backend, timeout=timeout)
    return parse_load_log(text)


# ------------------------------------------------------------------ saying it

def _human(size: float) -> str:
    value = float(size)
    for unit in ("B", "K", "M", "G", "T"):
        if value < 1024 or unit == "T":
            return f"{value:.0f}{unit}" if unit == "B" else f"{value:.1f}{unit}"
        value /= 1024.0
    return f"{value:.1f}T"


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
    said = [f"{_human(fit.weights_file)} on disk: {_human(fit.weights_gpu)} in GPU memory, "
            f"{_human(fit.weights_cpu)} mapped on the CPU"
            + (f" ({fit.cpu_tensors})" if fit.cpu_tensors
               else " (`fit MODEL --tensors` says what of)")]
    after = []
    if fit.table_bytes:
        after.append(f"a {_human(fit.table_bytes)} lookup table is paged on demand, a row "
                     "at a time")
    if fit.weights_resident:
        after.append(f"resident after {_tokens(fit.resident_after)} questions "
                     f"{_human(fit.weights_resident)} (measured)"
                     if fit.resident_after
                     else f"resident {_human(fit.weights_resident)} (measured)")
    if after:
        said.append("of which " + "; ".join(after) if fit.table_bytes
                    else "; ".join(after))
    return said


def _block(fit: Fit, contexts: list[int]) -> str:

    lines = [f"{pretty_name(fit.model)}", f"  {_headline(fit)}"]
    lines.append(
        f"  weights {_human(fit.weights)}"
        + (f", draft {_human(fit.draft)}" if fit.draft else "")
        + f", compute {_human(fit.compute)}"
        + f" -- of {_human(fit.room)} room, {_human(fit.free())} is left for caches")
    lines += [f"  {said}" for said in _where_it_went(fit)]
    lines.append(f"  {_human(fit.per_token)} per token of context, "
                 f"{_human(fit.per_seq)} fixed per sequence")
    shape = _shape(fit)
    if shape:
        lines.append(f"  {shape}")
    lines.append("")
    lines.append("  per user context   users that fit   each costs")
    for context in contexts:
        lines.append(f"  {_tokens(context):>16}   {fit.users(context):>14}   "
                     f"{_human(fit.cost(context)):>10}")
    lines.append(f"  one user, longest context: {_tokens(fit.longest(1))} tokens")
    return "\n".join(lines)


def _block_md(fit: Fit, contexts: list[int]) -> str:
    lines = [f"### {pretty_name(fit.model)}", "", f"{_headline(fit)}.", ""]
    lines.append(
        f"- weights {_human(fit.weights)}"
        + (f", draft {_human(fit.draft)}" if fit.draft else "")
        + f", compute {_human(fit.compute)}")
    lines += [f"- {said}" for said in _where_it_went(fit)]
    lines.append(f"- room {_human(fit.room)}, of which {_human(fit.free())} is left for "
                 f"caches")
    lines.append(f"- **{_human(fit.per_token)} per token of context**, "
                 f"**{_human(fit.per_seq)} fixed per sequence**")
    shape = _shape(fit)
    if shape:
        lines.append(f"- {shape}")
    lines += ["", "| per user context | users that fit | each costs |",
              "| --- | --- | --- |"]
    for context in contexts:
        lines.append(f"| {_tokens(context)} | {fit.users(context)} | "
                     f"{_human(fit.cost(context))} |")
    lines += ["", f"One user, longest context: **{_tokens(fit.longest(1))} tokens**."]
    return "\n".join(lines)


# ------------------------------------------------------ what the file is made of

# ggml's own block quantisation, as (name, values per block, bytes per block). Straight out
# of `ggml.c`'s type_traits table; the arithmetic below is ggml_nbytes' arithmetic, which is
# how a per-tensor size can be had from a header without reading a byte of tensor data.
_GGML_TYPES: dict[int, tuple[str, int, int]] = {
    0: ("f32", 1, 4), 1: ("f16", 1, 2), 2: ("q4_0", 32, 18), 3: ("q4_1", 32, 20),
    6: ("q5_0", 32, 22), 7: ("q5_1", 32, 24), 8: ("q8_0", 32, 34), 9: ("q8_1", 32, 40),
    10: ("q2_K", 256, 84), 11: ("q3_K", 256, 110), 12: ("q4_K", 256, 144),
    13: ("q5_K", 256, 176), 14: ("q6_K", 256, 210), 15: ("q8_K", 256, 292),
    16: ("iq2_xxs", 256, 66), 17: ("iq2_xs", 256, 74), 18: ("iq3_xxs", 256, 98),
    19: ("iq1_s", 256, 50), 20: ("iq4_nl", 32, 18), 21: ("iq3_s", 256, 110),
    22: ("iq2_s", 256, 82), 23: ("iq4_xs", 256, 136), 24: ("i8", 1, 1), 25: ("i16", 1, 2),
    26: ("i32", 1, 4), 27: ("i64", 1, 8), 28: ("f64", 1, 8), 29: ("iq1_m", 256, 56),
    30: ("bf16", 1, 2), 34: ("tq1_0", 256, 54), 35: ("tq2_0", 256, 66),
    39: ("mxfp4", 32, 17),
}

# A gathered table, not a matmul: only the rows whose keys occur are ever touched, so with
# mmap the rest of it never becomes resident. `per_layer_token_embd` is Flash-Next's n-gram
# table -- 26.8G of a 103.7G file -- and the other two names are what the same idea is
# called elsewhere.
_TABLE_NAMES = ("per_layer_token_embd", "ngram", "engram")


@dataclass(frozen=True, slots=True)
class Tensor:
    """One tensor as the GGUF header describes it: what it is called, what it is stored
    as, how big it is, and how many bytes of the file it takes."""

    name: str
    type: str
    shape: tuple[int, ...]
    bytes: int

    @property
    def elements(self) -> int:
        count = 1
        for dim in self.shape:
            count *= dim
        return count

    @property
    def role(self) -> str:
        """``table``, ``experts``, ``attention``, ``embedding`` or ``other``.

        The point of the grouping is the first one: a table is paged a row at a time and an
        expert is not, so two files of the same size can hold very different amounts of
        resident memory and only the names say which.
        """
        low = self.name.lower()
        if any(word in low for word in _TABLE_NAMES):
            return "table"
        if "exps" in low or ".experts" in low:
            return "experts"
        if "attn" in low or "attention" in low:
            return "attention"
        # `output.weight` is the lm_head; `output_norm.weight` is a norm and is not
        # one, so the dot is load-bearing rather than tidiness.
        if "token_embd" in low or low.startswith("output."):
            return "embedding"
        return "other"


def _shard_paths(model: str | Path) -> list[Path]:
    """Every shard of a model that is on this machine, from its first file's name."""
    from ml_stack.serve.preflight import shard_names

    first = Path(model).expanduser()
    found = [first.parent / name for name in shard_names(first.name)]
    return [one for one in found if one.is_file()] or [first]


def _tensors_in(path: Path) -> list[Tensor]:
    """Every tensor one GGUF file's header names, without reading any tensor data.

    `preflight.read_gguf_header` deliberately stops before the tensor table -- nothing a
    preflight needs is in it, and walking it is the expensive half. This walks the same
    metadata block, with the same value-kind table, only to get past it to the table that
    answers "what is this file made *of*". Still a header read: the offsets are read, the
    bytes they point at never are.
    """
    from ml_stack.serve.preflight import _GGUF_MAGIC, _SCALAR_FMT

    out: list[Tensor] = []
    with Path(path).expanduser().open("rb") as f:
        if f.read(4) != _GGUF_MAGIC:
            raise ValueError(f"{path}: not a GGUF file (no GGUF magic at the start)")
        struct.unpack("<I", f.read(4))                        # version -- unused here
        (tensor_count,) = struct.unpack("<Q", f.read(8))
        (kv_count,) = struct.unpack("<Q", f.read(8))

        def text() -> str:
            (n,) = struct.unpack("<Q", f.read(8))
            return f.read(n).decode("utf-8", "replace")

        def skip(kind: int) -> None:
            if kind == 8:
                text()
                return
            if kind == 9:
                (item_kind,) = struct.unpack("<I", f.read(4))
                (count,) = struct.unpack("<Q", f.read(8))
                for _ in range(count):
                    skip(item_kind)
                return
            f.read(struct.calcsize(_SCALAR_FMT[kind]))

        for _ in range(kv_count):
            text()
            (kind,) = struct.unpack("<I", f.read(4))
            skip(kind)

        for _ in range(tensor_count):
            name = text()
            (dims,) = struct.unpack("<I", f.read(4))
            shape = struct.unpack(f"<{dims}Q", f.read(8 * dims)) if dims else ()
            (kind,) = struct.unpack("<I", f.read(4))
            struct.unpack("<Q", f.read(8))                     # offset -- unused here
            named, block, per_block = _GGML_TYPES.get(kind, (f"type{kind}", 1, 0))
            count = 1
            for dim in shape:
                count *= dim
            out.append(Tensor(name=name, type=named, shape=tuple(shape),
                              bytes=count // block * per_block if block else 0))
    return out


def tensors_of(model: str | Path) -> list[Tensor]:
    """Every tensor in a model, over all of its shards, largest first."""
    found: list[Tensor] = []
    for shard in _shard_paths(model):
        found += _tensors_in(shard)
    return sorted(found, key=lambda t: (-t.bytes, t.name))


def totals_by_type(found: Iterable[Tensor]) -> list[tuple[str, int, int]]:
    """``(type, how many tensors, how many bytes)``, largest first.

    The listing that answers "what is the 15.7G of": a `UD-Q4_K_XL` is not one type, and
    the types a backend declines are exactly the ones that are not the majority.
    """
    counts: dict[str, list[int]] = {}
    for one in found:
        row = counts.setdefault(one.type, [0, 0])
        row[0] += 1
        row[1] += one.bytes
    return sorted(((name, n, size) for name, (n, size) in counts.items()),
                  key=lambda row: -row[2])


def totals_by_role(found: Iterable[Tensor]) -> list[tuple[str, int, int]]:
    """The same bytes grouped by what the tensor is *for*, largest first -- table,
    experts, attention, embedding, other. The one grouping that predicts residency."""
    counts: dict[str, list[int]] = {}
    for one in found:
        row = counts.setdefault(one.role, [0, 0])
        row[0] += 1
        row[1] += one.bytes
    return sorted(((name, n, size) for name, (n, size) in counts.items()),
                  key=lambda row: -row[2])


def table_bytes(model: str | Path) -> int:
    """Bytes of gathered lookup table in a model's file, or 0 when it has none or cannot
    be read. Never raises: a record is worth writing without this number."""
    try:
        return sum(one.bytes for one in tensors_of(model) if one.role == "table")
    except (OSError, ValueError, struct.error):
        return 0


def _dims(shape: Sequence[int]) -> str:
    return "(" + ", ".join(f"{int(d):,}" for d in shape) + ")"


def render_tensors(model: str | Path, *, top: int = 12) -> str:
    """What a model file is made of, from its header alone -- the answer to "why is it
    smaller in memory than on disk" that needs no server and no GPU.

    Three parts: the largest tensors with their type and shape (a lookup table marked, so
    the one tensor that is 26% of Flash-Next is not just another row), the totals per
    tensor type, and the totals per role. Sums every shard.
    """
    found = tensors_of(model)
    if not found:
        return f"{Path(model).name}: no tensors in the header"
    shards = _shard_paths(model)
    total = sum(one.bytes for one in found)

    lines = [f"{pretty_name(Path(model).name)}"
             + (f"  ({len(shards)} shards)" if len(shards) > 1 else ""),
             f"  {len(found):,} tensors, {_human(total)} of weights in the header"]
    table = sum(one.bytes for one in found if one.role == "table")
    if table:
        lines.append(f"  {_human(table)} of that is a gathered lookup table: paged a row "
                     "at a time, so most of it never becomes resident")
    lines += ["", f"  the {min(top, len(found))} largest tensors", ""]
    for one in found[:top]:
        lines.append(f"  {_human(one.bytes):>8}  {one.type:<8} {one.name:<40} "
                     f"{_dims(one.shape)}"
                     + ("   <- gathered table" if one.role == "table" else ""))
    lines += ["", "  by type", ""]
    for name, count, size in totals_by_type(found):
        lines.append(f"  {_human(size):>8}  {name:<8} {count:>5} tensors")
    lines += ["", "  by what it is for", ""]
    for name, count, size in totals_by_role(found):
        lines.append(f"  {_human(size):>8}  {name:<10} {count:>5} tensors")
    return "\n".join(lines)


# ------------------------------------------------------------------ drawing it

# Where the first panel's x axis runs: 2k is a question with its tools, 128k is a whole
# conversation kept open, and everything interesting about a cache happens between them.
PLOT_CONTEXTS: tuple[int, ...] = (2048, 4096, 8192, 16384, 32768, 65536, 131072)

# One line style per room asked about, so a model keeps its colour across all of them and
# the rooms are told apart by the line rather than by a second set of colours.
_ROOM_STYLES = ("-", "--", ":", "-.")

# The cards and machines people actually have, in GB, drawn faintly behind the second panel.
# A chart that only knows about *this* machine answers "will it fit here"; these are what
# turn it into "and what would I need". Nothing is measured about them -- they are gridlines
# with names on.
COMMON_VRAM_GB: tuple[int, ...] = (6, 8, 12, 16, 24, 32, 48, 64, 96, 128)


def _pyplot():
    """matplotlib, or a refusal that says how to get it.

    Optional the way every other heavy dependency here is optional -- `torch_ops`,
    `vision.payloads`, `fleet.app` all raise with the install line rather than failing on
    an ImportError somebody has to interpret. Note that `ml-stack-bench show --plot` is
    *not* the precedent it looks like: that one writes hand-built SVG with no library at
    all, on purpose, because it has to open on a machine with no packages. A PNG cannot be
    written that way, which is why this one has a dependency and that one does not.
    """

    try:
        import matplotlib

        matplotlib.use("Agg")           # a file, never a window: nothing here is interactive
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - the install line is the whole message
        raise RuntimeError(
            "drawing the fit chart needs matplotlib: pip install 'ml-stack[plot]'") from exc
    return plt


def label_of(fit: Fit) -> str:
    """What a line is called in the legend: the model, what its cache is stored as, and
    whether it was measured with a draft head -- the three things that make two records of
    the same weights different measurements."""

    parts = [pretty_name(fit.model)]
    if fit.cache_type and fit.cache_type != "f16":
        parts.append(fit.cache_type)
    if fit.spec:
        parts.append("+draft")
    return " ".join(parts)


def _k(tokens: int) -> str:
    """32768 -> `32k`, for a legend that has to say a context without spending a line on it."""
    return f"{tokens // 1024}k" if tokens >= 1024 and tokens % 1024 == 0 else f"{tokens:,}"


def _rooms_of(fits: Sequence[Fit], rooms: Sequence[int]) -> list[int]:
    """The rooms to draw, in the order asked for. None asked means whatever the records
    were recorded against, which is this machine."""
    asked = [int(r) for r in rooms if int(r) > 0]
    if asked:
        return asked
    return [max((f.room for f in fits), default=0)] if fits else [0]


def plot(fits: Iterable[Fit], where: str | Path, *, rooms: Sequence[int] = (),
         at: int = 32768, contexts: Sequence[int] = PLOT_CONTEXTS,
         machine: str = "") -> str:
    """Two panels over the measured records: who fits, and what it costs.

    The panels answer the same question from the two ends, and the second is the one worth
    having. A model is usually chosen by its weights, and the weights are the part that does
    *not* grow: a 30B whose cache costs 4 KiB a token overtakes an 8B whose cache costs 32
    KiB somewhere around the fourth or fifth user, and no table of per-model numbers makes
    that crossing visible. Drawn as memory against users at one context, it is where the
    lines cross.

    1. **Users that fit against the per-user context** (log2 x, log y): one line per record,
       one line style per room. A record whose weights do not leave room for a single cache
       draws nothing and says "does not fit" in the legend, which is a fact about the
       machine rather than an empty space to be interpreted.
    2. **Memory against users** at ``at`` tokens each: a straight line per record, starting
       at zero users -- where the height is the model with an empty cache -- and climbing
       by `cost(at)` per user. The rooms in force are drawn across it, and the familiar card
       sizes (`COMMON_VRAM_GB`) faintly behind, so the chart answers "what would I need"
       as well as "does it fit here". Where a line leaves a room is the last user that
       machine holds. `Fit.line(context)` is the pair of numbers each line is drawn from.

    ``where`` names the format: `.png`, `.svg`, `.pdf`. Returns the path written.
    """
    plt = _pyplot()

    rows = [f for f in fits]
    if not rows:
        raise ValueError("nothing to plot: no model has been measured")
    out = Path(where).expanduser()
    if out.suffix.lower() not in {".png", ".svg", ".pdf"}:
        raise ValueError(f"cannot draw a {out.suffix or 'nameless'} file; "
                         "name it .png, .svg or .pdf")
    spans = [int(c) for c in contexts if int(c) > 0] or list(PLOT_CONTEXTS)
    spans.sort()
    drawn_rooms = _rooms_of(rows, rooms)
    at = max(1, int(at))

    figure, (left, right) = plt.subplots(1, 2, figsize=(13.5, 5.6))
    colours = plt.rcParams["axes.prop_cycle"].by_key().get("color") or ["C0"]

    # -- panel one: who fits, at every context, in every room -------------------------
    for index, fit in enumerate(rows):
        colour = colours[index % len(colours)]
        drew = False
        for style, room in zip(_ROOM_STYLES, drawn_rooms):
            here = fit.at_room(room)
            points = [(c, here.users(c)) for c in spans]
            points = [(c, n) for c, n in points if n > 0]
            if not points:
                continue
            drew = True
            left.plot([c for c, _ in points], [n for _, n in points], style, color=colour,
                      marker="o", markersize=3.5,
                      label=(f"{label_of(fit)} ({_human(fit.loaded())} loaded)"
                             if style == _ROOM_STYLES[0] else None))
        if not drew:
            left.plot([], [], "-", color=colour,
                      label=f"{label_of(fit)} ({_human(fit.loaded())}) -- does not fit")

    left.set_xscale("log", base=2)
    left.set_yscale("log")
    left.set_xticks(spans)
    left.set_xticklabels([f"{c // 1024}k" for c in spans])
    left.set_xlabel("context each user gets (tokens)")
    left.set_ylabel("users that fit")
    left.set_title("How many fit, and at what context")
    left.grid(True, which="both", alpha=0.25)
    people = left.legend(fontsize=8, loc="upper right")
    if len(drawn_rooms) > 1:
        left.add_artist(people)
        left.legend(handles=[plt.Line2D([], [], color="0.35", linestyle=style,
                                        label=f"{_human(room)} of room")
                             for style, room in zip(_ROOM_STYLES, drawn_rooms)],
                    fontsize=8, loc="lower left")

    # -- panel two: what it costs as the users arrive ---------------------------------
    #
    # Every line here is `loaded() + users * cost(at)` -- a straight line from the model
    # sitting there with an empty cache, climbing by one user's worth of cache each step.
    # Starting at zero users is the point: the intercept is the part a person already knows
    # (the weights) and the slope is the part nobody does, and the crossing between a heavy
    # model with a cheap cache and a light one with an expensive cache is only visible when
    # both are drawn from the axis rather than from the first user.
    most = max((f.at_room(max(drawn_rooms)).users(at) for f in rows), default=0)
    most = min(max(most, 8), 256)
    people_axis = list(range(0, most + 1))
    gb = float(1024 ** 3)
    reach = 0.0
    for index, fit in enumerate(rows):
        colour = colours[index % len(colours)]
        intercept, each = fit.line(at)
        heights = [(intercept + n * each) / gb for n in people_axis]
        reach = max(reach, heights[-1])
        right.plot(people_axis, heights, "-", color=colour, linewidth=1.8,
                   label=f"{label_of(fit)}: {_human(intercept)} + "
                         f"{each / gb:.2f}G/user at {_k(at)}")

    # A little above whichever is higher: the rooms actually being asked about, or the
    # largest familiar card any of these lines climbs past. Below that and a crossing is
    # cut off the top; far above it and every line is squashed into the bottom inch.
    crossed = [size for size in COMMON_VRAM_GB if size <= reach]
    top = max([room / gb for room in drawn_rooms] + [max(crossed, default=0)]) * 1.08
    right.set_xlim(0, most)
    right.set_ylim(0, top or 1.0)

    for size in COMMON_VRAM_GB:
        if size > top:
            continue
        right.axhline(size, color="0.75", linewidth=0.8, zorder=0)
        right.annotate(f"{size}G", xy=(most, size), xytext=(-3, 2),
                       textcoords="offset points", fontsize=7, color="0.55",
                       va="bottom", ha="right", zorder=0)
    for style, room in zip(_ROOM_STYLES, drawn_rooms):
        right.axhline(room / gb, color="#b0413e", linestyle=style, linewidth=1.4)
        right.annotate(f"{_human(room)} of room", xy=(0, room / gb), xytext=(4, 3),
                       textcoords="offset points", fontsize=8, color="#b0413e",
                       va="bottom", ha="left")

    right.set_xlabel(f"users, each with {at:,} tokens")
    right.set_ylabel("memory needed (GB)")
    right.set_title("What it costs as they arrive")
    right.grid(True, axis="x", alpha=0.25)
    right.legend(fontsize=8, loc="upper left")

    builds = sorted({f.build for f in rows if f.build})
    figure.suptitle(
        (machine or "this machine") + f" -- {_human(drawn_rooms[0])} of room"
        + (f", measured on llama.cpp {', '.join(builds)}" if builds else ""),
        fontsize=11)
    figure.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out, dpi=140)
    plt.close(figure)
    return str(out)
