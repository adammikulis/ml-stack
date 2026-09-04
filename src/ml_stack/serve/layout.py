"""The attention layout of a GGUF, read off its header: which layers hold a full cache,
which slide, which are recurrent, which share, plus experts, indexers and lookup tables."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ml_stack.hub import pretty_name
from ml_stack.serve.fit import _shard_paths, tensors_of, totals_by_role
from ml_stack.serve.preflight import (_per_layer, _recurrent_layers, _sliding_layers,
                                      read_gguf_header)

__all__ = ["Layout", "layout", "render"]

_SSM_KEYS = ("inner_size", "state_size", "group_count", "conv_kernel", "time_step_rank")
_INDEXER_KEYS = ("head_count", "key_length", "top_k", "block_size", "local_blocks")


@dataclass(frozen=True)
class Layout:
    """What a GGUF header says about a model's layers, and what its tensor table holds."""

    model: str
    arch: str
    n_layer: int
    context_length: int
    embedding_length: int
    head_count: int
    kv_heads: tuple[int, ...]
    key_length: int
    value_length: int
    key_length_swa: int
    value_length_swa: int
    kinds: tuple[str, ...]
    shared: tuple[bool, ...]
    window: int
    pattern: str
    shared_kv_layers: int
    compress_ratios: tuple[int, ...]
    indexer: dict[str, int] = field(default_factory=dict)
    ssm: dict[str, int] = field(default_factory=dict)
    expert_count: int = 0
    expert_used_count: int = 0
    expert_feed_forward_length: int = 0
    expert_shared_feed_forward_length: int = 0
    tables: tuple[tuple[str, str, tuple[int, ...], int], ...] = ()
    by_role: tuple[tuple[str, int, int], ...] = ()
    shards: int = 1

    def layers(self, kind: str) -> list[int]:
        """The layer indices of one kind: ``full``, ``sliding``, ``recurrent`` or ``shared``."""
        if kind == "shared":
            return [il for il, on in enumerate(self.shared) if on]
        return [il for il, k in enumerate(self.kinds) if k == kind and not self.shared[il]]

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_json(cls, text: str) -> Layout:
        raw = json.loads(text)
        raw["kv_heads"] = tuple(raw["kv_heads"])
        raw["kinds"] = tuple(raw["kinds"])
        raw["shared"] = tuple(raw["shared"])
        raw["compress_ratios"] = tuple(raw["compress_ratios"])
        raw["tables"] = tuple((n, t, tuple(s), b) for n, t, s, b in raw["tables"])
        raw["by_role"] = tuple(tuple(row) for row in raw["by_role"])
        return cls(**raw)


def _int(value: object, default: int = 0) -> int:
    try:
        return int(value) if value is not None and not isinstance(value, (list, tuple)) else default
    except (TypeError, ValueError):
        return default


def _ints(value: object, n_layer: int) -> tuple[int, ...]:
    return tuple(int(v) for v in _per_layer(value or 0, n_layer))


def _pattern(key, n_layer: int) -> str:
    """How the header names its sliding layers, in words."""
    pattern = key("attention.sliding_window_pattern")
    if isinstance(pattern, (list, tuple)) and pattern:
        return "one bool per layer"
    if isinstance(pattern, (int, float)) and not isinstance(pattern, bool) and int(pattern) > 0:
        return f"period {int(pattern)}"
    if key("attention.sliding_window"):
        return "no pattern named; period 2"
    return ""


def layout(model: str | Path) -> Layout:
    """The layout of a model on this machine, from its header and its tensor table."""
    path = Path(model).expanduser()
    meta = read_gguf_header(path)
    arch = str(meta.get("general.architecture") or "")

    def key(suffix: str) -> object:
        return meta.get(f"{arch}.{suffix}")

    n_layer = _int(key("block_count"))
    kv_heads = _ints(key("attention.head_count_kv"), n_layer)
    head_count = _int(key("attention.head_count"))
    key_length = _int(key("attention.key_length"))
    if not key_length and head_count:
        key_length = _int(key("embedding_length")) // head_count
    recurrent = _recurrent_layers(key, n_layer)
    sliding = _sliding_layers(key, n_layer)
    ssm = {name: _int(key(f"ssm.{name}")) for name in _SSM_KEYS if key(f"ssm.{name}")}
    window = _int(key("attention.sliding_window"))
    shared_n = _int(key("attention.shared_kv_layers"))
    kinds = []
    for il in range(n_layer):
        if recurrent[il] or (ssm and kv_heads[il] == 0):
            kinds.append("recurrent")
        elif sliding[il] and window > 0:
            kinds.append("sliding")
        else:
            kinds.append("full")
    shared = tuple(kinds[il] != "recurrent" and il >= n_layer - shared_n
                   for il in range(n_layer))

    tensors = tensors_of(path)
    tables = tuple((t.name, t.type, t.shape, t.bytes) for t in tensors if t.role == "table")
    return Layout(
        model=pretty_name(path.name),
        arch=arch,
        n_layer=n_layer,
        context_length=_int(key("context_length")),
        embedding_length=_int(key("embedding_length")),
        head_count=head_count,
        kv_heads=kv_heads,
        key_length=key_length,
        value_length=_int(key("attention.value_length")) or key_length,
        key_length_swa=_int(key("attention.key_length_swa")),
        value_length_swa=_int(key("attention.value_length_swa")),
        kinds=tuple(kinds),
        shared=shared,
        window=window,
        pattern=_pattern(key, n_layer),
        shared_kv_layers=shared_n,
        compress_ratios=_ints(key("attention.compress_ratios"), n_layer)
        if key("attention.compress_ratios") else (),
        indexer={name: _int(key(f"attention.indexer.{name}")) for name in _INDEXER_KEYS
                 if key(f"attention.indexer.{name}") is not None},
        ssm=ssm,
        expert_count=_int(key("expert_count")),
        expert_used_count=_int(key("expert_used_count")),
        expert_feed_forward_length=_int(key("expert_feed_forward_length")),
        expert_shared_feed_forward_length=_int(key("expert_shared_feed_forward_length")),
        tables=tables,
        by_role=tuple(totals_by_role(tensors)),
        shards=len(_shard_paths(path)),
    )


# ---------------------------------------------------------------------------- words

def _ranges(layers: list[int]) -> str:
    """``0-3, 5, 7-9`` for ``[0, 1, 2, 3, 5, 7, 8, 9]``."""
    if not layers:
        return "none"
    out: list[str] = []
    start = prev = layers[0]
    for il in layers[1:] + [None]:
        if il is not None and il == prev + 1:
            prev = il
            continue
        out.append(str(start) if start == prev else f"{start}-{prev}")
        if il is not None:
            start = prev = il
    return ", ".join(out)


def _human(size: float) -> str:
    for unit in ("B", "K", "M", "G"):
        if size < 1024 or unit == "G":
            return f"{size:.0f}{unit}" if unit == "B" else f"{size:.1f}{unit}"
        size /= 1024.0
    return f"{size:.1f}G"


def _count(n: int, word: str) -> str:
    return f"{n} {word}" if n == 1 else f"{n} {word}s"


def _paragraph(lay: Layout) -> str:
    full = lay.layers("full")
    sliding = lay.layers("sliding")
    recurrent = lay.layers("recurrent")
    shared = lay.layers("shared")
    heads = {lay.kv_heads[il] for il in full + sliding + shared} or set(lay.kv_heads)
    kv = (f"{_count(min(heads), 'KV head')}" if len(heads) == 1
          else f"{min(heads)}-{max(heads)} KV heads per layer")
    dims = (f"head size {lay.key_length}" if lay.key_length == lay.value_length
            else f"key {lay.key_length}, value {lay.value_length}")
    said = [f"{lay.model} is {lay.arch}: {_count(lay.n_layer, 'layer')}, "
            f"{lay.head_count} heads, {kv}, {dims}, {lay.context_length:,} context."]

    parts = []
    if full:
        parts.append(f"{_count(len(full), 'layer')} hold{'s' if len(full) == 1 else ''} "
                     "a full cache")
    if sliding:
        swa = (f", key {lay.key_length_swa}" if lay.key_length_swa
               and lay.key_length_swa != lay.key_length else "")
        parts.append(f"{len(sliding)} slide over a {lay.window}-token window{swa}")
    if recurrent:
        parts.append(f"{len(recurrent)} are recurrent")
    if shared:
        parts.append(f"the last {len(shared)} read the cache of the layers before them "
                     "and own none")
    if parts:
        said.append("; ".join(parts) + ".")
    if lay.compress_ratios and lay.indexer:
        ratio = sorted({r for r in lay.compress_ratios if r})
        said.append(f"The attention layers are sparse: an indexer "
                    f"({_count(lay.indexer.get('head_count', 0), 'head')}, key "
                    f"{lay.indexer.get('key_length', 0)}) scores blocks of "
                    f"{', '.join(map(str, ratio))} tokens and each token attends to the top "
                    f"{lay.indexer.get('top_k', 0):,}.")
    elif lay.indexer:
        said.append(f"An indexer ({_count(lay.indexer.get('head_count', 0), 'head')}, key "
                    f"{lay.indexer.get('key_length', 0)}) picks the top "
                    f"{lay.indexer.get('top_k', 0):,} tokens.")
    if lay.expert_count:
        wide = f", {lay.expert_feed_forward_length} wide" if lay.expert_feed_forward_length else ""
        plus = (f" plus a shared expert of {lay.expert_shared_feed_forward_length}"
                if lay.expert_shared_feed_forward_length else "")
        said.append(f"{lay.expert_count} experts, {lay.expert_used_count} used{wide}{plus}.")
    for name, kind, shape, size in lay.tables:
        cells = " x ".join(f"{d:,}" for d in shape)
        said.append(f"A lookup table in one tensor: {name}, {cells}, {kind}, {_human(size)}, "
                    "gathered a row at a time.")
    return " ".join(said)


def render(lay: Layout) -> str:
    """The paragraph, then one bullet per group."""
    lines = [_paragraph(lay), ""]
    full = lay.layers("full")
    sliding = lay.layers("sliding")
    recurrent = lay.layers("recurrent")
    shared = lay.layers("shared")
    lines.append(f"- full cache: {len(full)} -- {_ranges(full)}")
    if sliding or lay.window:
        swa = ""
        if lay.key_length_swa or lay.value_length_swa:
            swa = f"; key_length_swa {lay.key_length_swa}, value_length_swa {lay.value_length_swa}"
        lines.append(f"- sliding: {len(sliding)} -- {_ranges(sliding)}; window {lay.window}"
                     + (f"; pattern: {lay.pattern}" if lay.pattern else "") + swa)
    else:
        lines.append("- sliding: none")
    ssm = ", ".join(f"{k} {v}" for k, v in lay.ssm.items())
    lines.append(f"- recurrent: {len(recurrent)} -- {_ranges(recurrent)}"
                 + (f"; ssm {ssm}" if recurrent and ssm else ""))
    lines.append(f"- shared KV: {len(shared)} -- {_ranges(shared)}"
                 + (f"; shared_kv_layers {lay.shared_kv_layers}" if shared else ""))
    if lay.compress_ratios:
        by_ratio: dict[int, list[int]] = {}
        for il, r in enumerate(lay.compress_ratios):
            if r:
                by_ratio.setdefault(r, []).append(il)
        said = "; ".join(f"{r} on {_count(len(ils), 'layer')} ({_ranges(ils)})"
                         for r, ils in sorted(by_ratio.items())) or "all 0"
        lines.append(f"- compress ratios: {said}")
    if lay.indexer:
        lines.append("- indexer: " + ", ".join(f"{k} {v:,}" for k, v in lay.indexer.items()))
    if lay.expert_count:
        lines.append(f"- experts: {lay.expert_count}, {lay.expert_used_count} used"
                     + (f", {lay.expert_feed_forward_length} wide"
                        if lay.expert_feed_forward_length else "")
                     + (f", shared expert {lay.expert_shared_feed_forward_length}"
                        if lay.expert_shared_feed_forward_length else ""))
    for name, kind, shape, size in lay.tables:
        cells = " x ".join(f"{d:,}" for d in shape)
        lines.append(f"- table: {name}  {kind}  ({cells})  {_human(size)}")
    if lay.by_role:
        roles = ", ".join(f"{name} {n} ({_human(size)})" for name, n, size in lay.by_role)
        lines.append(f"- tensors by role: {roles}"
                     + (f"; {lay.shards} shards" if lay.shards > 1 else ""))
    return "\n".join(lines)
