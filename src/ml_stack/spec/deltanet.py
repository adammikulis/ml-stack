"""Gated DeltaNet over a draft tree, in parallel form, and the replay that commits a path.

For node n with ancestors a_0 = n, a_1, ... up to the root, the output is
``y_n = S_prefix p_n + sum_j c_j v_j`` where ``p`` starts as ``q_n`` and each ancestor applies
``c = beta <k, p>; p = alpha (p - c k)``. The prefix state is read once per layer.
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.gated_delta import compute_g, gated_delta_update

__all__ = ["Pending", "commit", "conv_taps", "tree_mix", "walk"]

_WALK = r"""
    const uint g = thread_position_in_grid.x;
    const int n = (int)(g / HV), h = (int)(g % HV);
    if (n >= N) return;
    const int hk = h / (HV / HK);
    float p[DK];
    for (int d = 0; d < DK; ++d) p[d] = q[(n * HK + hk) * DK + d];
    for (int d = 0; d < DV; ++d) y[(n * HV + h) * DV + d] = 0.0f;
    for (int s = 0; s < S; ++s) {
        const int j = anc[n * S + s];
        if (j < 0) break;
        const int kb = (j * HK + hk) * DK;
        const int vb = (j * HV + h) * DV;
        float dot = 0.0f;
        for (int d = 0; d < DK; ++d) dot += k[kb + d] * p[d];
        const float c = dot * beta[j * HV + h];
        for (int d = 0; d < DV; ++d) y[(n * HV + h) * DV + d] += c * v[vb + d];
        const float a = alpha[j * HV + h];
        for (int d = 0; d < DK; ++d) p[d] = a * (p[d] - c * k[kb + d]);
    }
    for (int d = 0; d < DK; ++d) p_out[(n * HV + h) * DK + d] = p[d];
"""

_kernel = mx.fast.metal_kernel(
    name="spec_deltanet_walk",
    input_names=["q", "k", "v", "alpha", "beta", "anc"],
    output_names=["p_out", "y"],
    source=_WALK,
)


def walk(q: mx.array, k: mx.array, v: mx.array, gates: tuple[mx.array, mx.array],
         ancestors: mx.array) -> tuple[mx.array, mx.array]:
    """``(p [N, Hv, Dk], y [N, Hv, Dv])`` for q, k [N, Hk, Dk], v [N, Hv, Dv] and the
    decay and write gates ``(alpha, beta)`` [N, Hv]."""
    alpha, beta = gates
    count, key_heads, key_dim = q.shape
    value_heads, value_dim = v.shape[1], v.shape[2]
    p, y = _kernel(
        inputs=[q.astype(mx.float32), k.astype(mx.float32), v.astype(mx.float32),
                alpha.astype(mx.float32), beta.astype(mx.float32), ancestors],
        template=[("N", count), ("HV", value_heads), ("HK", key_heads), ("DK", key_dim),
                  ("DV", value_dim), ("S", ancestors.shape[1])],
        grid=(count * value_heads, 1, 1),
        threadgroup=(min(256, count * value_heads), 1, 1),
        output_shapes=[(count, value_heads, key_dim), (count, value_heads, value_dim)],
        output_dtypes=[mx.float32, mx.float32],
    )
    return p, y


@dataclass
class Pending:
    """What one tree pass computed that a commit replays along the accepted path."""

    history: mx.array
    q: mx.array
    k: mx.array
    v: mx.array
    a: mx.array
    b: mx.array


def conv_taps(depth: list[int], ancestors: list[list[int]], width: int) -> mx.array:
    """Row index into ``[prefix conv rows | tree rows]`` for every node's convolution taps."""
    pad = width - 1
    rows = []
    for node, d in enumerate(depth):
        row = []
        for s in range(pad, -1, -1):
            row.append(pad + ancestors[node][s] if s <= d else pad - 1 - (s - d - 1))
        rows.append(row)
    return mx.array(rows, mx.int32)


def tree_mix(mod: nn.Module, cache: list, x: mx.array, taps: mx.array,
             ancestors: mx.array) -> tuple[mx.array, Pending]:
    """The mixer output [N, D] for normalised input x [N, D], and what a commit needs."""
    count = x.shape[0]
    qkv = mod.in_proj_qkv(x)
    z = mod.in_proj_z(x).reshape(count, mod.num_v_heads, mod.head_v_dim)
    b = mod.in_proj_b(x)
    a = mod.in_proj_a(x)
    pad = mod.conv_kernel_size - 1
    held = cache[0][0] if cache[0] is not None else mx.zeros((pad, qkv.shape[-1]), dtype=x.dtype)
    history = mx.concatenate([held, qkv], axis=0)
    weight = mod.conv1d.weight[:, :, 0].T
    mixed = nn.silu((history[taps].astype(mx.float32) * weight.astype(mx.float32)).sum(axis=1))
    mixed = mixed.astype(x.dtype)
    q, k, v = mx.split(mixed, [mod.key_dim, 2 * mod.key_dim], -1)
    q = q.reshape(count, mod.num_k_heads, mod.head_k_dim)
    k = k.reshape(count, mod.num_k_heads, mod.head_k_dim)
    v = v.reshape(count, mod.num_v_heads, mod.head_v_dim)
    inv_scale = mod.head_k_dim ** -0.5
    q = (inv_scale ** 2) * mx.fast.rms_norm(q, None, 1e-6)
    k = inv_scale * mx.fast.rms_norm(k, None, 1e-6)
    beta = mx.sigmoid(b)
    alpha = compute_g(mod.A_log, a, mod.dt_bias)
    p, y = walk(q, k, v, (alpha, beta), ancestors)
    state = cache[1][0] if cache[1] is not None else mx.zeros(
        (mod.num_v_heads, mod.head_v_dim, mod.head_k_dim), dtype=mx.float32)
    y = y + mx.matmul(p.transpose(1, 0, 2), state.transpose(0, 2, 1)).transpose(1, 0, 2)
    out = mod.norm(y.astype(x.dtype), z)
    return mod.out_proj(out.reshape(count, -1)), Pending(history, q, k, v, a, b)


def commit(mod: nn.Module, cache: list, pending: Pending, path: mx.array) -> None:
    """Advance the conv and recurrent state along ``path`` (node indices from the root)."""
    pad = mod.conv_kernel_size - 1
    rows = mx.concatenate([pending.history[:pad], pending.history[pad + path]], axis=0)
    cache[0] = rows[-pad:][None]
    _, cache[1] = gated_delta_update(
        pending.q[path][None], pending.k[path][None], pending.v[path][None],
        pending.a[path][None], pending.b[path][None], mod.A_log, mod.dt_bias, cache[1], None)
