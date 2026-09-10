"""Reading what llama.cpp printed at load, at ``-lv 4``, into two structured records: one
model's arrival in memory (``Segment``) and what a load allocated overall (``Measured``).

The verbosity is not decoration: every line here is an ``LLAMA_LOG_INFO`` from the library,
which ``common_log_get_verbosity`` maps to LOG_LEVEL_TRACE -- so the server's own default
of 3 prints the server's lines and none of the model's.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from pathlib import Path

from ml_stack.units import human_bytes

__all__ = ["Measured", "Segment", "parse_load_log"]

_MIB = 1024 * 1024

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
    draft_per_token: int = 0
    """Bytes of KV cache the draft head keeps per token of context, per sequence. 0 for a
    load with no head, and for one whose head kept no cache of its own."""
    draft_per_seq: int = 0
    """Bytes the draft head costs a sequence whatever its context is."""
    draft_cache_type: str = ""
    """What the draft head's cache stores, as llama.cpp names it. llama.cpp's own default
    is f16 whatever the target's cache type is."""
    draft_kv_layers: int = 0
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
            f"per token {human_bytes(self.per_token)}",
            f"per sequence {human_bytes(self.per_seq)}",
            f"compute {human_bytes(self.compute)}",
            f"{self.kv_layers} layers with a cache",
        ]
        if self.recurrent_layers:
            parts.append(f"{self.recurrent_layers} recurrent")
        if self.swa_cells:
            parts.append(f"a {self.swa_cells}-cell sliding window")
        if self.cache_type:
            parts.append(f"cache {self.cache_type}")
        if self.draft_per_token or self.draft_per_seq:
            parts.append(f"the draft head's own cache {human_bytes(self.draft_per_token)} "
                         f"per token"
                         + (f", {human_bytes(self.draft_per_seq)} per sequence"
                            if self.draft_per_seq else "")
                         + (f" ({self.draft_cache_type})" if self.draft_cache_type else ""))
        if self.segments:
            parts.append(f"weights {human_bytes(self.weights_gpu)} on the GPU, "
                         f"{human_bytes(self.weights_cpu)} mapped on the CPU")
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


def _cache_of(segment: str) -> Measured | None:
    """What one model's part of the log says it allocated, or None when it allocated none."""
    kv = list(_KV_SIZE.finditer(segment))
    rs = list(_RS_SIZE.finditer(segment))
    if not kv and not rs:
        return None

    source = _LOADED.search(segment)
    model_file = Path(source.group(1)).name if source else ""

    per_token = kv_bytes = cells = seqs = kv_layers = 0
    swa_bytes = swa_cells = 0
    cache_type = ""
    if kv:
        # llama_kv_cache_iswa builds the base cache first and the SWA one second; the
        # base is the one whose cells are the context.
        base, *others = kv
        kv_bytes, cells = _mib(base.group(1)), int(base.group(2))
        kv_layers, seqs = int(base.group(3)), int(base.group(4))
        # llama-kv-cache.cpp prints "n_seq_max/n_stream" -- one stream is one shared
        # buffer sized for every sequence; more than one is a separate full-size copy
        # per sequence, and kv_bytes is already the sum over every stream.
        streams = max(1, int(base.group(5)))
        type_k, type_v = base.group(6), base.group(8)
        cache_type = type_k if type_k == type_v else f"{type_k}/{type_v}"
        per_token = kv_bytes // (cells * streams) if cells else 0
        for other in others:
            each, wide = _mib(other.group(1)), int(other.group(2))
            if cells and wide >= cells:
                # a second cache spanning the whole context -- a sparse layer's indexer
                # keeps one beside its K/V -- costs per token, not per sequence
                kv_bytes += each
                per_token += each // (wide * streams)
            else:
                swa_bytes += each
                swa_cells = max(swa_cells, wide)
            kv_layers += int(other.group(3))

    recurrent_bytes = recurrent_layers = 0
    rs_seqs = 0
    for one in rs:
        recurrent_bytes += _mib(one.group(1))
        recurrent_layers += int(one.group(3))
        rs_seqs = max(rs_seqs, int(one.group(4)))

    # Both fixed costs were sized for however many sequences were served; one sequence's
    # share is what a user costs.
    share = max(seqs, rs_seqs, 1)

    # The compute buffers are one per backend and the reserve can run more than once, so
    # the last figure for each backend is the one that stands.
    by_backend: dict[str, int] = {}
    for one in _COMPUTE.finditer(segment):
        by_backend[one.group(1)] = _mib(one.group(2))

    return Measured(
        per_token=per_token, per_seq=(recurrent_bytes + swa_bytes) // share,
        compute=sum(by_backend.values()),
        cache_type=cache_type, kv_layers=kv_layers, recurrent_layers=recurrent_layers,
        cells=cells, seqs=share, swa_cells=swa_cells, kv_bytes=kv_bytes,
        swa_bytes=swa_bytes, recurrent_bytes=recurrent_bytes, model_file=model_file)


def _loaded_file(segment: str) -> str:
    """The file name the load line in this part of the log names, or ""."""
    found = _LOADED.search(segment) or _CLIP_FILE.search(segment)
    return Path(found.group(1)).name if found else ""


def parse_load_log(text: str) -> Measured:
    """What llama.cpp allocated, from the log it wrote at `-lv 4`: the target's cache, a
    draft head's own cache beside it, and every model's weights."""
    build = ""
    found = _BUILD.search(text)
    if found:
        build = found.group(1)

    parts = _segments(text)
    loaded = tuple(seg for seg in
                   (_segment_of(part, i) for i, part in enumerate(parts))
                   if seg is not None)

    caches = [(i, one) for i, one in
              ((i, _cache_of(part)) for i, part in enumerate(parts)) if one is not None]
    if not caches:
        return Measured(build=build, segments=loaded)
    head = next((one for i, one in caches[1:]
                 if _kind_of(_loaded_file(parts[i]), i) == "draft"), None)
    return replace(caches[0][1], build=build, segments=loaded,
                   draft_per_token=head.per_token if head else 0,
                   draft_per_seq=head.per_seq if head else 0,
                   draft_cache_type=head.cache_type if head else "",
                   draft_kv_layers=head.kv_layers if head else 0)
