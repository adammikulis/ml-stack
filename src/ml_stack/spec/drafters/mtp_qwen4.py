"""Qwen4-Exp's multi-token-prediction head, and its weights read from a llama.cpp GGUF.

One step fuses a token's embedding with the target's residual streams, runs one hyper-connected
decoder layer (dense attention, the MoE block), and returns the streams; ``readout`` mixes
them down for the target's own output head. The ``shared`` GGUF carries no embedding or head,
so both are always the target's.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
from mlx_lm.models.cache import KVCache
from mlx_vlm.models.qwen4_exp.language import (
    Qwen4ExpDecoderLayer,
    Qwen4ExpGatedResidual,
    Qwen4ExpRMSNorm,
    _qwen4_inject,
)

from ml_stack import home
from ml_stack.spec.layout import Layout
from ml_stack.spec.qwen4 import attend

__all__ = ["Qwen4MtpHead", "load_gguf_head"]

#: GGUF tensor name after ``blk.<n>.`` -> the head's parameter, for tensors that map one to one
NAMES = {
    "nextn.enorm.weight": "pre_fc_norm_embedding.weight",
    "nextn.hnorm.weight": "pre_fc_norm_hidden.weight",
    "nextn.hc_head_norm.weight": "hyper_connection_mixer.hc_norm.weight",
    "nextn.hc_head_down.weight": "hyper_connection_mixer.input_mix_weight_down.weight",
    "nextn.hc_head_up.weight": "hyper_connection_mixer.input_mix_weight_up.weight",
    "attn_q.weight": "layer.self_attn.q_proj.weight",
    "attn_k.weight": "layer.self_attn.k_proj.weight",
    "attn_v.weight": "layer.self_attn.v_proj.weight",
    "attn_output.weight": "layer.self_attn.o_proj.weight",
    "attn_q_norm.weight": "layer.self_attn.q_norm.weight",
    "attn_k_norm.weight": "layer.self_attn.k_norm.weight",
    "ffn_gate_inp.weight": "layer.mlp.gate.weight",
    "ffn_gate_inp_shexp.weight": "layer.mlp.shared_expert_gate.weight",
    "ffn_gate_exps.weight": "layer.mlp.switch_mlp.gate_proj.weight",
    "ffn_up_exps.weight": "layer.mlp.switch_mlp.up_proj.weight",
    "ffn_down_exps.weight": "layer.mlp.switch_mlp.down_proj.weight",
    "ffn_gate_shexp.weight": "layer.mlp.shared_expert.gate_proj.weight",
    "ffn_up_shexp.weight": "layer.mlp.shared_expert.up_proj.weight",
    "ffn_down_shexp.weight": "layer.mlp.shared_expert.down_proj.weight",
    **{f"hc_{side}_{part}.weight": f"layer.{name}_hyper_connection.{field}.weight"
       for side, name in (("attn", "attn"), ("ffn", "mlp"))
       for part, field in (("norm", "hc_norm"), ("down", "input_mix_weight_down"),
                           ("up", "input_mix_weight_up"), ("inject", "block_inject_weight"))},
}


class Qwen4MtpHead(nn.Module):
    def __init__(self, args: object) -> None:
        super().__init__()
        width, streams = args.hidden_size, args.hc_count
        self.streams, self.width = streams, width
        self.pre_fc_norm_embedding = Qwen4ExpRMSNorm(width, eps=args.rms_norm_eps)
        self.pre_fc_norm_hidden = Qwen4ExpRMSNorm(streams * width, group_size=width,
                                                  eps=args.rms_norm_eps)
        self.fc_embedding = nn.Linear(width, width, bias=False)
        self.fc_hidden = nn.Linear(width, width, bias=False)
        one = replace(args, num_hidden_layers=1, layer_types=["qwen_sparse_attention"],
                      full_attention_interval=1, ple_layer_ids=[])
        self.layer = Qwen4ExpDecoderLayer(one, layer_idx=0)
        self.hyper_connection_mixer = Qwen4ExpGatedResidual(one, use_combine=False)

    def step(self, layout: Layout, cache: KVCache, pair: tuple[mx.array, mx.array],
             positions: mx.array, mask: mx.array | str | None) -> mx.array:
        """The head's residual streams [W, hc*D] for (token, streams) pairs at ``positions``."""
        tokens, hidden = pair
        count = hidden.shape[0]
        embedded = self.fc_embedding(self.pre_fc_norm_embedding(layout.embed(tokens)))
        streams = self.fc_hidden(self.pre_fc_norm_hidden(hidden).reshape(count, self.streams,
                                                                          self.width))
        x = (embedded[:, None, :] + streams).reshape(count, -1)
        mixed, kept, weights = self.layer.attn_hyper_connection(x)
        x = _qwen4_inject(attend(self.layer.self_attn, cache, mixed, positions, mask), kept,
                          weights)
        mixed, kept, weights = self.layer.mlp_hyper_connection(x)
        return _qwen4_inject(self.layer.mlp(mixed), kept, weights)

    def readout(self, out: mx.array) -> mx.array:
        return self.hyper_connection_mixer(out)


def _routing(path: str, module: nn.Module) -> bool | dict:
    if not hasattr(module, "to_quantized") or "indexer" in path:
        return False
    if path.endswith(("mlp.gate", "shared_expert_gate")):
        return {"group_size": 64, "bits": 8}
    return True


def gguf_weights(path: Path) -> dict[str, mx.array]:
    """The head's parameters from a GGUF MTP block: norms back to zero-centred gammas and
    ``eh_proj`` split into its embedding and hidden halves."""
    out: dict[str, mx.array] = {}
    for name, value in mx.load(str(path)).items():
        _, _, rest = name.partition(".")
        _, _, rest = rest.partition(".")
        if rest == "nextn.eh_proj.weight":
            half = value.shape[1] // 2
            out["fc_embedding.weight"], out["fc_hidden.weight"] = value[:, :half], value[:, half:]
        elif rest in NAMES:
            target = NAMES[rest]
            if value.ndim == 1 and target.endswith("norm.weight"):
                value = value - 1.0
            out[target] = value.reshape(-1, value.shape[-1]) if "shared_expert_gate" in target \
                else value
    return out


def load_gguf_head(path: Path, args: object, bits: int = 4, group_size: int = 32) -> Qwen4MtpHead:
    """The head from a llama.cpp MTP GGUF, quantized to ``bits``; cached once quantized."""
    head = Qwen4MtpHead(args)
    cached = home.cache("spec", "mtp", f"{path.stem}-{bits}bit.safetensors")
    if not cached.is_file():
        found = gguf_weights(path)
        wanted = {name for name, _ in tree_flatten(head.parameters()) if ".indexer." not in name}
        missing = sorted(wanted - set(found))
        if missing:
            raise ValueError(f"{path.name} has no tensor for {', '.join(missing)}")
        head.load_weights(list(found.items()), strict=False)
        head.set_dtype(mx.bfloat16)
        nn.quantize(head, group_size=group_size, bits=bits, class_predicate=_routing)
        mx.eval(head.parameters())
        cached.parent.mkdir(parents=True, exist_ok=True)
        mx.save_safetensors(str(cached), dict(tree_flatten(head.parameters())))
        return head
    nn.quantize(head, group_size=group_size, bits=bits, class_predicate=_routing)
    head.load_weights(list(mx.load(str(cached)).items()))
    mx.eval(head.parameters())
    return head
