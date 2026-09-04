"""What must be true about a model before any process is started for it.

A benchmark that loads five models, some 87G, pays for every one of these mistakes at the
far end of a load rather than the start of one: an architecture the build does not read, a
flag a release renamed, a draft head that starts downloading inside the timed window, a
shard left behind by an interrupted pull. Each of those costs minutes to find out the slow
way and a fraction of a second to find out here -- a GGUF header is a few hundred bytes read
off the front of the file, never the tensors that make up the rest of it.

``Preflight`` runs every check and hands back one ``Report``; ``LlamaServerBackend.start``
raises before ``Popen`` when it fails.
"""

from __future__ import annotations

import re
import struct
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = ["Check", "Preflight", "PreflightFailed", "Report", "read_gguf_header",
           "shard_names"]


class PreflightFailed(RuntimeError):
    """A preflight check failed. Carries the report's own lines, one failure per line."""


@dataclass(frozen=True, slots=True)
class Check:
    """One question asked and answered before a load, never during one."""

    name: str
    ok: bool
    detail: str = ""


@dataclass
class Report:
    """Every check ``Preflight`` ran, and what it estimated along the way."""

    checks: list[Check] = field(default_factory=list)
    weights_bytes: int = 0
    kv_estimate_bytes: int = 0

    @property
    def ok(self) -> bool:
        return all(c.ok for c in self.checks)

    def said(self) -> str:
        """One line per check, ``ok``/``FAIL`` first -- what ``PreflightFailed`` carries
        and what ``up --preflight-only`` prints."""
        return "\n".join(
            f"{'ok  ' if c.ok else 'FAIL'}  {c.name}" + (f": {c.detail}" if c.detail else "")
            for c in self.checks
        )


# ---------------------------------------------------------------- a GGUF's own header

_GGUF_MAGIC = b"GGUF"
# The scalar value kinds the GGUF format defines, and the struct code that reads one.
# 8 is a string and 9 is an array, handled separately below; the rest are fixed-width.
_SCALAR_FMT = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?",
               10: "Q", 11: "q", 12: "d"}


def read_gguf_header(path: Path | str) -> dict[str, object]:
    """Every key/value metadata pair in a GGUF file: strings, ints, floats, bools, arrays.

    Reads the magic, the version, the tensor count, and then each of the key/value pairs
    that follow -- and stops there. The tensor *list* that comes after names every tensor's
    shape and file offset, and reading it costs nothing measurable for a small model and a
    real pause for an 87G one; nothing a preflight needs is in it, so nothing here reads it.

    A minimal reader rather than the ``gguf`` package's own, on purpose: that reader is
    built to open a file for inference and walks the tensor table as part of doing so. This
    one is built to answer one question cheaply before any inference is intended.
    """
    out: dict[str, object] = {}
    with Path(path).expanduser().open("rb") as f:
        if f.read(4) != _GGUF_MAGIC:
            raise ValueError(f"{path}: not a GGUF file (no GGUF magic at the start)")
        struct.unpack("<I", f.read(4))                       # version -- unused here
        struct.unpack("<Q", f.read(8))                        # tensor count -- unread
        (kv_count,) = struct.unpack("<Q", f.read(8))

        def text() -> str:
            (n,) = struct.unpack("<Q", f.read(8))
            return f.read(n).decode("utf-8", "replace")

        def value(kind: int) -> object:
            if kind == 8:
                return text()
            if kind == 9:
                (item_kind,) = struct.unpack("<I", f.read(4))
                (count,) = struct.unpack("<Q", f.read(8))
                return [value(item_kind) for _ in range(count)]
            code = _SCALAR_FMT[kind]
            return struct.unpack("<" + code, f.read(struct.calcsize(code)))[0]

        for _ in range(kv_count):
            name = text()
            (kind,) = struct.unpack("<I", f.read(4))
            out[name] = value(kind)
    return out


# ---------------------------------------------------------------- sharded models

_SHARD = re.compile(r"-(\d{5})-of-(\d{5})(?=\.gguf$)", re.IGNORECASE)


def shard_names(filename: str) -> list[str]:
    """Every shard name a sharded GGUF implies, generated from its first file's name.

    Reconstructed rather than globbed for, so a shard that is simply absent is named
    exactly -- ``model-00002-of-00003.gguf`` -- instead of a directory listing that is
    merely short one entry and does not say which. A name with no shard marker is not
    sharded, and is its own single-element answer.
    """
    match = _SHARD.search(filename)
    if not match:
        return [filename]
    total = int(match.group(2))
    prefix, suffix = filename[: match.start()], filename[match.end():]
    return [f"{prefix}-{i:05d}-of-{total:05d}{suffix}" for i in range(1, total + 1)]


def _local_index() -> dict[str, Path]:
    """Every model file this machine already holds, by filename -- the same disk scan
    ``ml_stack.hub.held()`` does, kept here with its path rather than only its size,
    because a preflight has to *open* the file to read its header."""
    from ml_stack.fleet.models import Models, default_roots

    try:
        found = Models(roots=default_roots(Path.home() / ".ml-stack"),
                       store=Path.home() / ".ml-stack").all()
    except Exception:  # noqa: BLE001 - a machine with no models has no models
        return {}
    return {m.path.name: m.path for m in found}


def _local_shards(path: Path) -> tuple[int, Path | None, Check]:
    """Presence and completeness of a local model's shards, from its first file's name."""
    names = shard_names(path.name)
    paths = [path.parent / n for n in names]
    missing: list[str] = []
    total = 0
    for p in paths:
        try:
            size = p.stat().st_size if p.is_file() else 0
        except OSError:
            size = 0
        if size <= 0:
            missing.append(p.name)
        total += size
    ok = not missing
    detail = "complete" if ok else "missing or empty: " + ", ".join(missing)
    first = paths[0] if paths and paths[0].is_file() else None
    return total, first, Check("shards", ok, detail)


def _hf_shards(repo: str, name: str) -> tuple[int, Path | None, Check]:
    """The same check, for a repository not necessarily downloaded yet -- resolved through
    the Hub cache the way ``ml-stack-models files`` reports what is already on this machine."""
    if not name:
        return 0, None, Check(
            "shards", True,
            "no file named in the reference; the server will resolve one, so there is "
            "nothing yet to check")

    from ml_stack.hub import files as hub_files

    names = shard_names(name)
    try:
        listing = dict(hub_files(repo))
    except Exception as exc:  # noqa: BLE001 - the Hub is somebody else's machine
        # What the build should hold could not even be asked, which is not the same as
        # asking and finding a shard absent -- unknown, not a measured contradiction.
        return 0, None, Check("shards", True,
                              f"could not read {repo} from the Hub ({exc}); no opinion")

    local = _local_index()
    missing: list[str] = []
    total = 0
    first_path: Path | None = None
    for shard in names:
        total += listing.get(shard, 0)
        basename = shard.rsplit("/", 1)[-1]
        found = local.get(basename)
        if found is None or not found.is_file() or found.stat().st_size <= 0:
            missing.append(shard)
        elif shard == names[0]:
            first_path = found
    ok = not missing
    detail = "complete on this machine" if ok else "not on this machine yet: " + ", ".join(missing)
    return total, first_path, Check("shards", ok, detail)


def _shards_of(spec) -> tuple[int, Path | None, Check]:
    """The weights' size, the first shard's path when it is on this machine, and the
    shards check -- off the disk for a path, through the Hub cache for an `hf:` reference.
    The default ``shards_of`` reader; the one fact a preflight cannot ask without touching
    a file or the Hub."""
    if spec.is_hf_ref:
        repo, name = spec.hf_parts(spec.model)
        return _hf_shards(repo, name)
    return _local_shards(Path(spec.model))


def _ref_bytes(ref: str | Path | None) -> int:
    """Roughly what a companion reference (a draft, an mmproj) will cost, local or on the
    Hub. Best-effort: a companion that cannot be sized costs nothing to the estimate rather
    than blocking it -- the shards check is what refuses to load, not this."""
    if not ref:
        return 0
    from ml_stack.serve.backend import ServerSpec

    parts = ServerSpec.hf_parts(ref)
    if parts is None:
        try:
            return Path(ref).stat().st_size
        except OSError:
            return 0
    repo, name = parts
    if not name:
        return 0
    try:
        from ml_stack.hub import files as hub_files

        return dict(hub_files(repo)).get(name, 0)
    except Exception:  # noqa: BLE001
        return 0


# ---------------------------------------------------------------- architecture

def _architecture_check(first_file: Path | None, binary: str | Path, *,
                        read_header: Callable[[Path | str], dict[str, object]] | None = None,
                        arches: Callable[[str | Path], set[str]] | None = None) -> Check:
    """Same philosophy as ``flags_of``/``unknown_flags``: a fact that cannot be read is
    unknown, and an unknown build (or an unreadable file) is given no opinion rather than
    told it is wrong. Only a fact that *was* read, and contradicts what the build reads,
    fails -- a missing shard is still a fact; a file this cannot even open is not.

    ``read_header`` and ``arches`` are `Preflight`'s seams of the same names."""
    if first_file is None:
        return Check("architecture", True,
                     "no local copy to read yet; nothing to check until the shards check "
                     "above has something on disk")
    try:
        meta = (read_header or read_gguf_header)(first_file)
    except Exception as exc:  # noqa: BLE001
        return Check("architecture", True,
                     f"could not read {first_file} ({exc}); no opinion")

    arch = str(meta.get("general.architecture") or "")
    if not arch:
        return Check("architecture", True,
                     f"{first_file} names no general.architecture; no opinion")

    known = (arches or known_architectures)(binary)
    if not known:
        return Check("architecture", True,
                     f"{arch!r} (this build's own architectures could not be read; no opinion)")
    if _plain(arch) in {_plain(k) for k in known}:
        return Check("architecture", True, arch)
    shown = ", ".join(sorted(known)[:6]) + (" ..." if len(known) > 6 else "")
    return Check("architecture", False, f"{arch!r} -- this build reads {shown}")


def _plain(name: str) -> str:
    """An architecture name with its punctuation gone, for comparing: llama.cpp writes
    `gpt-oss`, a symbol table may say `gptoss`, and `strings` keeps whichever it finds."""
    return "".join(c for c in str(name).lower() if c.isalnum())


def known_architectures(binary: str | Path) -> set[str]:
    """The architectures a build reads: the names in its source checkout's llama-arch.cpp
    when the managed build's source is at hand, else what `strings` finds in libllama.

    Measured 2026-09-01: the strings guess keeps only alphanumeric words beginning with a
    family prefix, so `gpt-oss` was never in it and a preflight refused a model the same
    build had served all afternoon. The source table is the truth when it exists."""
    from ml_stack.setup import _arches

    found: set[str] = set()
    try:
        from ml_stack.serve.build import _arches_from_source

        found |= _arches_from_source(Path(source_dir()))
    except Exception:  # noqa: BLE001 - no source checkout, or a table that moved
        pass
    found |= _arches(binary)
    return found


def source_dir() -> Path:
    """Where the managed build's source checkout lives; a seam so tests can point elsewhere."""
    from ml_stack.serve.build import SRC_DIR

    return Path(SRC_DIR)


# ---------------------------------------------------------------- fit (weights + kv + runtime)

# Bytes per cached element for the K/V cache types llama.cpp accepts on --cache-type-k/-v.
# Block-quantised types carry a scale per 32 elements, so the average is not the nominal
# quant width -- q8_0 stores 34 bytes per 32 elements, not 32. f16 (2 bytes) is what a
# server serves with unless told otherwise, which is why it is also the default here.
_CACHE_BYTES = {
    "f32": 4.0, "f16": 2.0, "bf16": 2.0,
    "q8_0": 34 / 32, "q6_0": 28 / 32, "q5_1": 24 / 32, "q5_0": 22 / 32,
    "q4_1": 20 / 32, "q4_0": 18 / 32, "iq4_nl": 18 / 32,
}

# A load's own buffers -- the compute graph, the output buffer, the scratch space around
# the KV cache -- cost real memory beyond the weights and the KV cache itself, and none of
# it is in the GGUF's metadata. Not measured per model; a fixed floor so the fit check is
# conservative rather than reporting a load that just barely fails as one that just fits.
RUNTIME_ALLOWANCE_BYTES = 512 * 1024 * 1024


def _per_layer(value: object, n_layer: int) -> list[float]:
    """A metadata value as one number per layer.

    gemma-4-26B-A4B (measured 2026-09-01) stores `attention.head_count_kv` as an array, one
    entry per block, where a dense model stores one integer; multiplying the array by a
    float crashed the preflight and with it the load it existed to protect."""
    if isinstance(value, (list, tuple)):
        rows = [float(v) for v in value]
        if not rows:
            return []
        return (rows + [rows[-1]] * n_layer)[:n_layer]
    return [float(value)] * n_layer


def _recurrent_layers(key: Callable[[str], object], n_layer: int) -> list[bool]:
    """Which layers keep a *state* rather than a history, and so cost nothing per token.

    Two ways a GGUF says so, both read the way llama.cpp's own loader reads them
    (`models/qwen4exp.cpp`): an explicit `attention.recurrent_layers` array wins, and
    otherwise `full_attention_interval = N` means every Nth layer holds a cache and the
    other N-1 do not -- `(il + 1) % N != 0` is recurrent. Qwen3.8-Flash-Next is 4, so three
    layers in four cost nothing at all as the context grows.
    """
    explicit = key("attention.recurrent_layers")
    if isinstance(explicit, (list, tuple)) and explicit:
        rows = [bool(v) for v in explicit]
        return (rows + [rows[-1]] * n_layer)[:n_layer]
    interval = key("full_attention_interval")
    try:
        every = int(interval) if interval is not None else 0
    except (TypeError, ValueError):
        every = 0
    if every <= 0:
        return [False] * n_layer
    return [(il + 1) % every != 0 for il in range(n_layer)]


def _sliding_layers(key: Callable[[str], object], n_layer: int) -> list[bool]:
    """Which layers see only a window, read the way `llama_hparams::set_swa_pattern` writes it.

    `attention.sliding_window_pattern` is a bool per layer in gemma4 and a *period* in
    gemma3 and cohere2, and llama.cpp reads either through the same `get_key_or_arr`. A
    model that names a window and no pattern gets llama.cpp's own default period of 2 --
    which is gpt-oss, where the even layers slide and the odd ones do not.
    """
    pattern = key("attention.sliding_window_pattern")
    if isinstance(pattern, (list, tuple)) and pattern:
        rows = [bool(v) for v in pattern]
        return (rows + [rows[-1]] * n_layer)[:n_layer]
    period = 0
    if isinstance(pattern, bool):
        period = 0
    elif isinstance(pattern, (int, float)):
        period = int(pattern)
    elif key("attention.sliding_window"):
        period = 2
    if period <= 0:
        return [False] * n_layer
    return [il % period < (period - 1) for il in range(n_layer)]


def _kv_estimate_bytes(meta: dict[str, object], context: int,
                       cache_type_k: str, cache_type_v: str) -> int:
    """``sum over the layers that hold one of n_kv_heads * head_dim * span * bytes`` -- 0
    when the GGUF does not carry the keys this needs, which is a real answer, not a failure
    to read. Never raises: an estimate that cannot be made is unknown, and a preflight that
    crashes a load has defeated itself.

    Not every layer holds a cache and not every cache spans the context, which is the whole
    difference between this and the flat multiplication it used to be:

    * a **recurrent** layer keeps a fixed state per sequence, so it costs nothing per token
      -- three layers in four of Qwen3.8-Flash-Next (`full_attention_interval = 4`);
    * a **sliding-window** layer holds its window and no more, in its own `key_length_swa`
      -- gemma4 (a bool per layer, a 512-token window) and gpt-oss (128, every other layer);
    * a **shared-KV** layer holds nothing of its own -- gemma4's `shared_kv_layers = 18`
      says the last eighteen read the cache the layers before them wrote.

    It is still an estimate, and still only the fallback: the compute buffers are not in the
    header at all, a recurrent layer's per-sequence state is not either, and the indexer
    key cache of a sparse-attention layer (Qwen3.8-Flash-Next: one head of
    `attention.indexer.key_length` per attention layer) is not counted.
    `attention.compress_ratios` is the block size that indexer scores; the cache itself
    spans the whole context (`llama_memory_hybrid_idx`). `ml-stack-serve fit` is the
    measured answer; this is what there is before anybody has measured one.
    """
    try:
        arch = str(meta.get("general.architecture") or "")
        if not arch:
            return 0

        def key(suffix: str) -> object:
            return meta.get(f"{arch}.{suffix}")

        n_layer = int(key("block_count") or 0)
        if not (n_layer and context):
            return 0
        kv_heads = _per_layer(key("attention.head_count_kv") or 0, n_layer)

        fallback = key("attention.key_length")
        if not fallback:
            embed, n_head = key("embedding_length"), key("attention.head_count")
            if embed and n_head:
                heads = _per_layer(n_head, n_layer)
                fallback = [float(embed) / h if h else 0.0 for h in heads]
        key_dim = _per_layer(fallback or 0, n_layer)
        value_dim = _per_layer(key("attention.value_length") or fallback or 0, n_layer)
        key_swa = _per_layer(key("attention.key_length_swa") or fallback or 0, n_layer)
        # A model that names a narrower key for its windowed layers has a narrower value
        # there too (gemma4 refuses to load if they differ), so the SWA key is a better
        # fallback for the SWA value than the full-attention one is.
        value_swa = _per_layer(
            key("attention.value_length_swa") or key("attention.key_length_swa")
            or key("attention.value_length") or fallback or 0, n_layer)
        if not any(kv_heads) or not any(key_dim):
            return 0

        recurrent = _recurrent_layers(key, n_layer)
        sliding = _sliding_layers(key, n_layer)
        window = int(key("attention.sliding_window") or 0)
        shared = int(key("attention.shared_kv_layers") or 0)
        holds_kv = n_layer - shared if shared > 0 else n_layer

        bytes_k = _CACHE_BYTES.get(cache_type_k.lower(), 2.0) if cache_type_k else 2.0
        bytes_v = _CACHE_BYTES.get(cache_type_v.lower(), 2.0) if cache_type_v else 2.0

        total = 0.0
        for il in range(n_layer):
            if recurrent[il] or il >= holds_kv:
                continue
            swa = sliding[il] and window > 0
            span = min(context, window) if swa else context
            k_dim = key_swa[il] if swa else key_dim[il]
            v_dim = value_swa[il] if swa else value_dim[il]
            total += kv_heads[il] * (k_dim * bytes_k + v_dim * bytes_v) * span
        return int(total)
    except Exception:  # noqa: BLE001 - an estimate that cannot be made is 0, never a crash
        return 0


def _human(size: int) -> str:
    value = float(size)
    for unit in ("B", "K", "M", "G"):
        if value < 1024 or unit == "G":
            return f"{value:.0f}{unit}" if unit == "B" else f"{value:.1f}{unit}"
        value /= 1024.0
    return f"{value:.1f}G"


def _fit_check(weights_bytes: int, draft_bytes: int, mmproj_bytes: int, kv_bytes: int,
              limit_bytes: int) -> Check:
    total = weights_bytes + draft_bytes + mmproj_bytes + kv_bytes + RUNTIME_ALLOWANCE_BYTES
    pieces = (f"weights {_human(weights_bytes)}"
              + (f", draft {_human(draft_bytes)}" if draft_bytes else "")
              + (f", mmproj {_human(mmproj_bytes)}" if mmproj_bytes else "")
              + f", kv+runtime est {_human(kv_bytes + RUNTIME_ALLOWANCE_BYTES)}")
    if not limit_bytes:
        return Check("fit", True, f"{_human(total)} estimated ({pieces}); "
                                  "no machine memory limit is known to compare against")
    ok = total <= limit_bytes
    verb = "fits under" if ok else "exceeds"
    return Check("fit", ok,
                f"{_human(total)} estimated {verb} the {_human(limit_bytes)} this machine "
                f"may use ({pieces})")


# ---------------------------------------------------------------- flags (reuses backend.py)

def _flags_check(spec, binary: str | Path, *,
                 flags: Callable[[str | Path], frozenset[str]] | None = None) -> Check:
    """Every flag ``command(spec)`` emits is one the build's ``--help`` lists. ``flags``
    stands in for `flags_of` when handed in (`Preflight`'s seam); the argv is always built
    for real, because building it is where a spec the backend refuses is found out."""
    from ml_stack.serve.backend import LlamaServerBackend, flags_of, unknown_flags

    known = (flags or flags_of)(binary)
    if not known:
        return Check("flags", True, "could not read this build's --help; no opinion")
    # A draft named by hf: file is fetched and served by path at start(); this check is
    # about flags, not files, so it builds the argv with a stand-in path rather than
    # refusing the reference (measured 2026-09-02: it refused, and E4B's heads never ran).
    parts = spec.hf_parts(spec.draft) if spec.draft else None
    if parts and parts[1]:
        from dataclasses import replace

        spec = replace(spec, draft=Path("draft-head.gguf"))
    argv = LlamaServerBackend(binary=binary).command(spec)
    lacking = unknown_flags(argv, known)
    if lacking:
        detail = "; ".join(f"no {flag}" + (f", it has {near}" if near else "")
                           for flag, near in lacking)
        return Check("flags", False, detail)
    return Check("flags", True, "every flag this spec would emit is one this build accepts")


# ---------------------------------------------------------------- the whole thing

def Preflight(spec, *, binary: str | Path, limit_bytes: int = 0,
              shards_of: Callable[[Any], tuple[int, Path | None, Check]] | None = None,
              read_header: Callable[[Path | str], dict[str, object]] | None = None,
              arches: Callable[[str | Path], set[str]] | None = None,
              flags: Callable[[str | Path], frozenset[str]] | None = None,
              ref_bytes: Callable[[str | Path | None], int] | None = None) -> Report:
    """Everything worth knowing about ``spec`` before a process is started for it.

    Every check runs and is recorded even after one fails -- a report that stopped at the
    first failure would hide a second, unrelated one behind it, and the whole point of
    asking before the load is asking everything at once rather than one slow round at a time.

    The five keyword seams are where the facts come from, and each defaults to the real
    reader: ``shards_of(spec)`` is `_shards_of` (the disk, or the Hub cache),
    ``read_header(path)`` is `read_gguf_header`, ``arches(binary)`` is
    `known_architectures`, ``flags(binary)`` is `flags_of`, ``ref_bytes(ref)`` is
    `_ref_bytes` (a companion's size, local or on the Hub). They exist for one caller: the
    bench's self-check, which hands in facts that touch nothing so that every check's own
    code -- the shard arithmetic, the KV estimate, the fit, and above all the argv
    `_flags_check` builds -- runs over the exact spec a run is about to serve. Twice on
    2026-09-02 a self-check that replaced this whole function said ok, and the run then
    died in here. A fake that skips the checks is not a check of the checks.
    """
    report = Report()

    weights_bytes, first_file, shards = (shards_of or _shards_of)(spec)
    report.checks.append(shards)
    report.weights_bytes = weights_bytes

    report.checks.append(_architecture_check(first_file, binary, read_header=read_header,
                                             arches=arches))

    meta: dict[str, object] = {}
    if first_file is not None:
        try:
            meta = (read_header or read_gguf_header)(first_file)
        except Exception:  # noqa: BLE001 - already reported by the architecture check
            meta = {}

    kv_bytes = _kv_estimate_bytes(meta, spec.context, spec.cache_type_k, spec.cache_type_v)
    report.kv_estimate_bytes = kv_bytes
    sized = ref_bytes or _ref_bytes
    draft_bytes = sized(spec.draft)
    mmproj_bytes = sized(spec.mmproj)
    report.checks.append(
        _fit_check(weights_bytes, draft_bytes, mmproj_bytes, kv_bytes, limit_bytes))

    report.checks.append(_flags_check(spec, binary, flags=flags))
    return report
