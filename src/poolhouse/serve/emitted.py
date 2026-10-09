"""The flags ``LlamaServerBackend.command`` can emit."""

from __future__ import annotations

from poolhouse.serve.backend import LlamaServerBackend, ServerSpec

__all__ = ["emitted_flags"]


def emitted_flags(backend: LlamaServerBackend) -> list[str]:
    """Every flag ``backend.command`` can emit, from specs with every field set.

    Several shapes are needed because they exclude each other: an embedding server drops
    ``--jinja`` for ``--embeddings``, an ``hf:`` reference swaps ``-m`` for ``--hf-repo``,
    and a ``--no-`` flag is only emitted where its positive form is not. Values are harmless placeholders; nothing here is run.
    """
    full = ServerSpec(
        model="model.gguf", parallel=2, mmproj="mmproj-model.gguf", draft="draft.gguf",
        spec_type="draft-simple", spec_draft_max=3, spec_p_min=0.5, spec_draft_min=1, spec_ngram_min=48,
        spec_ngram_max=64, spec_draft_ngl=99, spec_draft_type_k="q8_0",
        spec_draft_type_v="q8_0", lookup_static="static.bin", lookup_dynamic="dynamic.bin",
        cache_reuse=256, warmup=False, context_per_slot=4096, override_tensor=("x=CPU",),
        cpu_moe=True, n_cpu_moe=1, kv_unified=True, cache_ram_mb=8192, cache_idle_slots=True,
        slot_prompt_similarity=0.5, slot_save_path="slots", chat_template_file="t.jinja",
        cache_type_k="q8_0", api_key="k",
        cache_type_v="q8_0", mlock=True, reasoning_budget=2048, rope_scaling="yarn",
        rope_scale=4.0, yarn_orig_ctx=32768, yarn_ext_factor=1.0, yarn_attn_factor=1.0,
        yarn_beta_fast=32.0, yarn_beta_slow=1.0)
    # the `--no-` forms are a third shape for the same reason: True and False exclude
    # each other on one spec
    shapes = (full,
              ServerSpec(model="model.gguf", kv_unified=False, cache_idle_slots=False,
                         mmap=False),
              ServerSpec(model="model.gguf", embedding=True),
              ServerSpec(model="hf:owner/repo/model.gguf", mmproj="hf:owner/repo/mmproj.gguf",
                         draft="hf:owner/repo"))
    flags: list[str] = []
    for shape in shapes:
        for token in backend.command(shape)[1:]:
            if token.startswith("-") and token.lstrip("-")[:1].isalpha() and token not in flags:
                flags.append(token)
    return flags
