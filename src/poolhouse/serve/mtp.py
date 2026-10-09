"""Multi-token prediction on by default: the head a served model drafts with, whether it may be
trusted and loaded, and the one line saying why a lease is served without one."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path

from poolhouse import hub, sentinel
from poolhouse.serve.backend import ServerSpec, flags_of, help_of
from poolhouse.serve.mlx_tree import is_mlx
from poolhouse.serve.preflight import read_gguf_header
from poolhouse.serve.tensors import tensors_of

__all__ = ["DEPTH", "ENV", "KIND", "Plan", "applied", "depth_for", "enabled", "failed", "plan",
           "supported"]

ENV = "POOLHOUSE_MTP"
KIND = "draft-mtp"
_OFF = frozenset({"0", "off", "no", "false", "none", "disable", "disabled"})
_SAME = ("embedding_length",)
# Draft depth per model architecture where kept runs favour one (docs/serving.md, "What has been
# measured"); an architecture not listed is served at the server's own default.
DEPTH = {"gemma4": 2}
_FAILED: set[tuple[str, str, str]] = set()


@dataclass(frozen=True, slots=True)
class Plan:
    """What a lease does about multi-token prediction: the head, the method, and one line."""

    draft: str = ""
    spec_type: str = ""
    head: str = ""
    note: str = ""
    loud: bool = False
    depth: int = 0

    @property
    def worth_saying(self) -> bool:
        """Whether the note belongs on the console: a head in use or a head turned away."""
        return self.active or self.loud

    @property
    def active(self) -> bool:
        """Whether the server is started drafting."""
        return bool(self.spec_type)


def depth_for(model: Path) -> int:
    """The draft depth kept runs favour for ``model``'s architecture, or 0 for the default."""
    try:
        arch = read_gguf_header(model).get("general.architecture")
    except (OSError, ValueError):
        return 0
    return DEPTH.get(str(arch), 0)


def failed(model: str | Path, draft: str | Path | None, binary: str | Path | None) -> None:
    """Remember that ``model`` would not start with ``draft`` (or its own layer) on ``binary``."""
    _FAILED.add((str(model), str(draft or ""), str(binary)))


def enabled(environ: Mapping[str, str] | None = None) -> bool:
    """Whether ``POOLHOUSE_MTP`` leaves multi-token prediction on (it does unless set to off)."""
    env = os.environ if environ is None else environ
    return str(env.get(ENV, "")).strip().lower() not in _OFF


def supported(binary: str | Path | None) -> bool:
    """Whether this llama-server build lists ``draft-mtp`` among its ``--spec-type`` kinds."""
    if binary is None:
        return False
    block: list[str] = []
    for line in help_of(binary).splitlines():
        if line.startswith("--spec-type"):
            block = [line]
        elif block and line[:1] in (" ", "\t"):
            block.append(line)
        elif block:
            break
    return any(KIND in line for line in block)


@lru_cache(maxsize=64)
def _embeds(path: str, mtime_ns: int, size: int) -> bool:
    try:
        return any(".nextn." in t.name for t in tensors_of(path))
    except (OSError, ValueError):
        return False


def embeds_head(model: Path) -> bool:
    """Whether the weights carry their own prediction layer (``blk.N.nextn.*`` tensors)."""
    try:
        info = model.stat()
    except OSError:
        return False
    return _embeds(str(model), info.st_mtime_ns, info.st_size)


def mismatch(model: Path, head: Path) -> str:
    """Why ``head`` cannot draft for ``model`` from the two headers, or ''."""
    try:
        a, b = read_gguf_header(model), read_gguf_header(head)
    except (OSError, ValueError) as exc:
        return f"{head.name}: header unreadable ({exc})"
    arch = a.get("general.architecture")
    if b.get("general.architecture") != arch:
        return (f"{head.name}: architecture {b.get('general.architecture')} is not the "
                f"model's {arch}")
    for suffix in _SAME:
        if a.get(f"{arch}.{suffix}") != b.get(f"{arch}.{suffix}"):
            return f"{head.name}: {suffix} differs from the model's"
    blocks = a.get(f"{arch}.block_count")
    if isinstance(blocks, int) and b.get(f"{arch}.block_count") not in (blocks, blocks + 1):
        return f"{head.name}: block_count differs from the model's"
    for key in ("tokenizer.ggml.pre", "tokenizer.ggml.model"):
        if a.get(key) != b.get(key):
            return f"{head.name}: {key} differs from the model's"
    mine, theirs = a.get("tokenizer.ggml.tokens"), b.get("tokenizer.ggml.tokens")
    if isinstance(mine, list) and isinstance(theirs, list) and len(mine) != len(theirs):
        return f"{head.name}: vocabulary size differs from the model's"
    if not int(b.get(f"{arch}.nextn_predict_layers") or 0):
        return f"{head.name}: carries no prediction layer"
    return ""


def _repo(path: Path) -> str:
    return next((part for part in path.parts if part.startswith("models--")), "")


def provenance(model: Path, head: Path) -> str:
    """Why ``head`` is not trusted to draft for ``model``, or ''.

    A head is trusted when it sits in the model's own repository (Hub cache) or beside the
    weights (same folder, the folder above, or an ``MTP/`` folder there); a head from
    anywhere else only when sentinel holds a pin a download made for it.
    """
    repo = _repo(model)
    if repo and _repo(head) == repo:
        return ""
    here, there = model.parent, head.parent
    if not repo and (there in (here, here.parent)
                     or (there.name.lower() == "mtp" and there.parent in (here, here.parent))):
        return ""
    pin = sentinel.default().manifest.pins().get(str(head))
    if pin is not None and pin.source and pin.source != "first-use":
        return ""
    return (f"{head.name} comes from a different repository than the model and nothing "
            f"pinned it; name it with draft= to use it")


def _unsuited(spec: ServerSpec, escalate: bool) -> str:
    if spec.embedding:
        return "an embedding server does not draft"
    if spec.engine or is_mlx(spec.model):
        return "this engine does not use llama.cpp speculation"
    if spec.spec_tree:
        return "tree decoding is its own speculation"
    if escalate or spec.slot_save_path:
        return ("slot save and restore keep the target's cache only, so a restored slot "
                "would draft from nothing")
    return ""


def plan(spec: ServerSpec, *, binary: str | Path | None, escalate: bool = False,
         environ: Mapping[str, str] | None = None) -> Plan:
    """The multi-token-prediction head ``spec`` is served with, or why it is served without."""
    if spec.mtp is False:
        return Plan(note="MTP off: this lease asked for none")
    if not enabled(environ):
        return Plan(note=f"MTP off: {ENV} is off")
    if spec.draft or spec.spec_type:
        return Plan()
    if why := _unsuited(spec, escalate):
        return Plan(note=f"MTP off: {why}")
    model = Path(str(spec.model)).expanduser()
    if not model.is_file():
        return Plan(note="MTP off: the weights are not on this machine yet")
    if not supported(binary):
        return Plan(note=f"MTP off: this llama-server build has no --spec-type {KIND}")
    depth = 0 if spec.spec_draft_max is not None else depth_for(model)
    if embeds_head(model):
        if (str(model), "", str(binary)) in _FAILED:
            return Plan(note="MTP off: the server failed to start with the weights' own layer")
        return Plan(spec_type=KIND, head="embedded", depth=depth,
                    note="MTP on: the weights carry their own prediction layer")
    if binary is None or "-md" not in flags_of(binary):
        return Plan(note="MTP off: this llama-server build takes no draft model")
    ahead = sorted((h for h in hub.heads_for(model, binary=binary)
                    if h.spec_type == KIND and not h.build
                    and (str(model), h.path, str(binary)) not in _FAILED),
                   key=lambda h: ("q8_0" not in h.name.lower(), h.bytes, h.name))
    refused: list[str] = []
    for one in ahead:
        path = Path(one.path)
        why = provenance(model, path) or mismatch(model, path)
        if not why:
            return Plan(draft=one.path, spec_type=KIND, head=one.name, depth=depth,
                        note=f"MTP on: draft head {one.name}")
        refused.append(why)
    return Plan(note="MTP off: " + ("; ".join(refused) if refused
                                    else "no MTP head ships beside the weights"),
                loud=bool(refused))


def applied(spec: ServerSpec, chosen: Plan) -> ServerSpec:
    """``spec`` served with ``chosen``; ``mtp=True`` marks a head the default picked."""
    if not chosen.active:
        return spec
    return replace(spec, draft=chosen.draft or None, spec_type=chosen.spec_type, mtp=True,
                   spec_draft_max=chosen.depth or spec.spec_draft_max)
