"""Sampling a token per tree node: temperature, top-k, then top-p and min-p, in one kernel.

Each row selects its top K logits exactly by radix select on the bf16 bit patterns, sorts
them, applies temperature, softmax over the top-k, top-p and min-p, and draws once by inverse
CDF. With stats it also returns the distribution the draw was taken from.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import mlx.core as mx

__all__ = ["Sampling", "sample_rows"]

#: the most candidates the kernel keeps per row
MAX_TOP_K = 64

_SRC = r"""
    const uint row = threadgroup_position_in_grid.x;
    const uint tid = thread_position_in_threadgroup.x;    // 256 threads per row
    if (row >= N) return;
    const device bfloat16_t* x = logits + (size_t)row * V;
    const device ushort* xb = (const device ushort*)x;
    threadgroup atomic_uint hist[256];
    threadgroup uint sel_i[K];
    threadgroup float sel_v[K];
    threadgroup atomic_uint counters[2];     // [0] slots used, [1] ties taken
    threadgroup uint thr_key;
    threadgroup uint need_ties;

    // ---- pass 1: histogram of the high byte of the order-preserving key ----
    atomic_store_explicit(&hist[tid], 0u, memory_order_relaxed);
    if (tid < 2) atomic_store_explicit(&counters[tid], 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int j = (int)tid; j < V; j += 256) {
        uint b = xb[j];
        uint key = (b & 0x8000u) ? (~b & 0xFFFFu) : (b | 0x8000u);
        atomic_fetch_add_explicit(&hist[key >> 8], 1u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        uint acc = 0; uint hi = 0;
        for (int b = 255; b >= 0; --b) {
            uint c = atomic_load_explicit(&hist[b], memory_order_relaxed);
            if (acc + c >= (uint)K) { hi = (uint)b; break; }
            acc += c;
        }
        thr_key = hi;
        need_ties = acc;          // elements strictly above the selected high-byte bin
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint hi_sel = thr_key;
    const uint above_hi = need_ties;
    atomic_store_explicit(&hist[tid], 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // ---- pass 2: histogram of the low byte within the selected high-byte bin ----
    for (int j = (int)tid; j < V; j += 256) {
        uint b = xb[j];
        uint key = (b & 0x8000u) ? (~b & 0xFFFFu) : (b | 0x8000u);
        if ((key >> 8) == hi_sel) atomic_fetch_add_explicit(&hist[key & 0xFFu], 1u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        uint acc = above_hi; uint lo = 0;
        for (int b = 255; b >= 0; --b) {
            uint c = atomic_load_explicit(&hist[b], memory_order_relaxed);
            if (acc + c >= (uint)K) { lo = (uint)b; break; }
            acc += c;
        }
        thr_key = (hi_sel << 8) | lo;
        need_ties = (uint)K - acc;   // how many elements equal to the threshold we take
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint T = thr_key; const uint ties = need_ties;
    // ---- pass 3: collect the top-K (unsorted) ----
    for (int j = (int)tid; j < V; j += 256) {
        uint b = xb[j];
        uint key = (b & 0x8000u) ? (~b & 0xFFFFu) : (b | 0x8000u);
        if (key > T) {
            uint s = atomic_fetch_add_explicit(&counters[0], 1u, memory_order_relaxed);
            if (s < (uint)K) { sel_i[s] = (uint)j; sel_v[s] = (float)x[j]; }
        } else if (key == T) {
            uint t = atomic_fetch_add_explicit(&counters[1], 1u, memory_order_relaxed);
            if (t < ties) {
                uint s = atomic_fetch_add_explicit(&counters[0], 1u, memory_order_relaxed);
                if (s < (uint)K) { sel_i[s] = (uint)j; sel_v[s] = (float)x[j]; }
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // ---- thread 0: sort the K selected (descending), softmax over the top-k, filter, draw ----
    if (tid == 0) {
        for (int a = 1; a < K; ++a) {
            float v = sel_v[a]; uint ii = sel_i[a]; int p = a - 1;
            while (p >= 0 && sel_v[p] < v) { sel_v[p + 1] = sel_v[p]; sel_i[p + 1] = sel_i[p]; --p; }
            sel_v[p + 1] = v; sel_i[p + 1] = ii;
        }
        const float inv_t = 1.0f / temp;
        const int kk = min(top_k, K);
        const float m = sel_v[0] * inv_t;
        float tot = 0.0f;
        for (int i = 0; i < kk; ++i) tot += metal::exp(sel_v[i] * inv_t - m);
        const float lse = m + metal::log(tot);      // softmax over the top-k: tokens outside it are dropped first
        float cum = 0.0f; int keep = 0; const float p0 = metal::exp(sel_v[0] * inv_t - lse);
        for (int i = 0; i < kk; ++i) {
            float pi = metal::exp(sel_v[i] * inv_t - lse);
            if (top_p < 1.0f && i > 0 && cum >= top_p) break;
            if (min_p > 0.0f && pi < min_p * p0) break;
            cum += pi; keep = i + 1;
        }
        float u = uniforms[row] * cum;
        float acc = 0.0f; int pick = keep - 1;
        for (int i = 0; i < keep; ++i) { acc += metal::exp(sel_v[i] * inv_t - lse); if (u < acc) { pick = i; break; } }
        out[row] = sel_i[pick];
        if (STATS) {          // processed distribution over the top-K (renormalized kept set; zero outside it)
            for (int i = 0; i < K; ++i) {
                top_ids[row * K + i] = sel_i[i];
                top_probs[row * K + i] = i < keep ? metal::exp(sel_v[i] * inv_t - lse) / cum : 0.0f;
            }
        }
    }
"""

_kernels: dict[int, object] = {}


def _kernel(width: int) -> object:
    if width not in _kernels:
        _kernels[width] = mx.fast.metal_kernel(
            name=f"spec_sample_{width}",
            input_names=["logits", "uniforms", "temp", "top_k", "top_p", "min_p"],
            output_names=["out", "top_ids", "top_probs"], source=_SRC)
    return _kernels[width]


@dataclass(frozen=True)
class Sampling:
    """How a token is drawn; temperature 0 is greedy."""

    temperature: float = 0.0
    top_k: int = 20
    top_p: float = 1.0
    min_p: float = 0.0

    def __post_init__(self) -> None:
        if not (math.isfinite(self.temperature) and self.temperature >= 0):
            raise ValueError(f"temperature must be >= 0, got {self.temperature}")
        if not 1 <= self.top_k <= MAX_TOP_K:
            raise ValueError(f"top_k must be between 1 and {MAX_TOP_K}, got {self.top_k}")
        if not 0 < self.top_p <= 1:
            raise ValueError(f"top_p must be in (0, 1], got {self.top_p}")
        if not 0 <= self.min_p <= 1:
            raise ValueError(f"min_p must be in [0, 1], got {self.min_p}")

    @property
    def greedy(self) -> bool:
        return self.temperature == 0

    @property
    def width(self) -> int:
        """Candidates kept per row: at least 32, so the relaxed rules see the drafted children."""
        return max(self.top_k, 32)


def sample_rows(logits: mx.array, sampling: Sampling,
                stats: bool = False) -> tuple[mx.array, mx.array, mx.array]:
    """(tokens [N], ids [N, K], probs [N, K]) drawn from logits [N, V]; ids and probs are
    the processed distribution, sorted by logit, and are empty unless stats."""
    if logits.dtype != mx.bfloat16:
        logits = logits.astype(mx.bfloat16)
    count, vocab = logits.shape
    width = sampling.width
    uniforms = mx.random.uniform(shape=(count,))
    out, ids, probs = _kernel(width)(
        inputs=[logits, uniforms, mx.array(float(sampling.temperature)),
                mx.array(int(sampling.top_k)), mx.array(float(sampling.top_p)),
                mx.array(float(sampling.min_p))],
        template=[("N", count), ("V", vocab), ("K", width), ("STATS", int(stats))],
        grid=(count * 256, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(count,), (count, width), (count, width)] if stats
        else [(count,), (1,), (1,)],
        output_dtypes=[mx.uint32, mx.uint32, mx.float32],
    )
    return out, ids, probs
