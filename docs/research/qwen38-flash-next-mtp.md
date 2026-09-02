# Qwen3.8-Flash-Next MTP heads: why they fail on mainline, and what runs them

Written 2026-09-01. Everything below is either **measured** (this machine, or a cited source
that ran it) or **claimed** (a README or a comment). Dates are UTC. People are not named:
every source is the PR, comment, discussion or repository it came from, linked.

Local facts, measured 2026-09-01 on this machine (mainline master `3466812` = b10751, built
today; last week's `62acc89` = b10707 behaved the same): all six heads under
`unsloth/Qwen3.8-Flash-Next-GGUF/MTP/` fail to load as `-md` with
`llama_model_load: error loading model: check_tensor_dims: tensor 'output_hc_norm.weight' not found`.
The main model `UD-IQ4_XS` serves fine without a draft. Header read of the local files
(GGUF KV and tensor names only, no weights):

| file | tensors | `block_count` | `nextn_predict_layers` | `nextn_shared_target_tensors` | has `output.weight` / `token_embd` | has `output_hc_*` |
|---|---|---|---|---|---|---|
| `UD-IQ4_XS` (3 shards) | 1224 | 48 | absent | absent | yes (shard 2) | yes (shard 2) |
| `MTP/mtp-...-Q8_0.gguf` | 34 | 49 | 1 | absent | yes | **no** — carries `blk.48.nextn.hc_head_{norm,down,up}` instead |
| `MTP/mtp-...-shared-Q8_0.gguf` | 32 | 49 | 1 | `true` | **no** | no |

Every head tensor sits at `blk.48.*` (attention, indexer, 512-expert MoE, `hc_attn_*`,
`hc_ffn_*`, `nextn.{eh_proj,enorm,hnorm,hc_head_norm,hc_head_down,hc_head_up}`).
One caveat on the local note: the loader creates `token_embd` before `output_hc_norm`
(code below), so the three `shared-*` heads should die on `token_embd.weight` first — which
is exactly what a Hub user saw on b10731
([discussion #54](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/discussions/54),
2026-09-01). Re-read the log line for the shared heads before quoting it.

## 1. Why the heads fail on mainline

Mainline loads a `-md` file as a complete model: `common/speculative.cpp`
`common_speculative_init_result` calls `llama_model_load_from_file(...)` when `has_draft`
(master `3466812`, lines 2540-2556), and the server hands it the draft path through
`common_base_params_to_speculative` (`tools/server/server-context.cpp:1126-1132`). The
qwen4exp loader then demands the whole trunk plus the output mixer, `src/models/qwen4exp.cpp`
`load_arch_tensors`, master `3466812` lines 157-164:

```cpp
tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab }, 0);

// there is no output_norm: the final hyper-connection mixer carries it
hc_head_norm = create_tensor(tn(LLM_TENSOR_HC_HEAD_NORM, "weight"), { hc_dim }, 0);
hc_head_down = create_tensor(tn(LLM_TENSOR_HC_HEAD_DOWN, "weight"), { hc_dim, hc_lr }, 0);
hc_head_up   = create_tensor(tn(LLM_TENSOR_HC_HEAD_UP,   "weight"), { hc_lr, hc_dim }, 0);

output = create_tensor(tn(LLM_TENSOR_OUTPUT, "weight"), { n_embd, n_vocab }, TENSOR_NOT_REQUIRED);
```

Flag `0` means required; `src/llama-arch.cpp:521` maps `LLM_TENSOR_HC_HEAD_NORM` to
`"output_hc_norm"`, and `llama_model_loader::create_tensor` (`src/llama-model-loader.cpp:1330`)
calls `check_tensor_dims(name, ne, required=!(flags & TENSOR_NOT_REQUIRED), ...)`, which throws
the message above. Had that passed, the per-layer loop (lines 206-213) requires
`blk.0.hc_attn_norm.weight` .. `blk.47.*` with flag `0` as well — the error every
detached-head user hit on PR #27836 (see §2). And even a file that loaded would draft
nothing: mainline `qwen4exp.cpp` has no `graph_mtp` and no `nextn` reference at all (grep,
2026-09-01), which is the "context type MTP requested but model doesn't contain MTP layers"
warning in [discussion #47](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/discussions/47).
Mainline does list `LLM_ARCH_QWEN4EXP` in `llm_arch_supports_rs_rollback` since
[#28123](https://github.com/ggml-org/llama.cpp/pull/28123) (merged 2026-09-01 04:24, b10731).

What has to give, one of:

- **The loader accepts a head-only file.** The unsloth fork's PR #144 (§3) detects
  `mtp_only = n_layer_nextn > 0 && blk.0.hc_attn_norm.weight absent`, marks the trunk and
  `output_hc_*` `TENSOR_NOT_REQUIRED`, uses `blk.48.nextn.hc_head_*` as the head's mixer, and
  borrows `token_embd`/`output` from the target when `nextn_shared_target_tensors` is set.
  [#28097](https://github.com/ggml-org/llama.cpp/pull/28097) and the detached-head loader
  commit [a82a58a](https://github.com/crusaderky/llama.cpp/commit/a82a58a57fc307e5cec0dc68db64d143339be4f2)
  (2026-08-28, from a commenter on #27836) do the first half the same way.
- **The head carries everything.** There is no metadata the current export could emit that
  the current loader accepts standalone: `block_count=1, nextn=1` trips
  `GGML_ASSERT(n_layer_nextn < n_layer_all)` (a commenter on #27836, 2026-08-28). The only
  loader-free route is grafting the head into the target as a 49th block (the merge script in
  alternative repository 2, §6) — and that still needs a build with the qwen4exp MTP graph.

## 2. ggml-org/llama.cpp PR #27836

[#27836](https://github.com/ggml-org/llama.cpp/pull/27836) "qwen4exp : add NextN/MTP draft
head (--spec-type draft-mtp) for Qwen3.8-Flash-Next", branch `qwen4exp-mtp`.
**Open, still a Draft**, created 2026-08-27 23:24, last updated 2026-09-01 15:58, not merged.
+403/−52 over `conversion/qwen4exp.py`, `gguf-py/gguf/{constants,tensor_mapping}.py`,
`src/llama-arch.{cpp,h}`, `src/llama-model.{cpp,h}`, `src/models/models.h`,
`src/models/qwen4exp.cpp`. Bot flagged "PR template not respected" and "AI-generated
content" (2026-08-27); 26 comments, none from the PR's author as of 2026-09-01.

What it adds (from the diff): removes `supports_mtp_export = False` / `no_mtp = True` from
the converter so `--mtp` exports the head; renames `mtp.hyper_connection_mixer.*` to
`blk.{n}.nextn.hc_head_{norm,down,up}` (new `LLM_TENSOR_NEXTN_HC_HEAD_*`); fuses
`mtp.fc_embedding|fc_hidden` into one `blk.{n}.nextn.eh_proj` (`[W_e|W_h] @ concat(e,h)`);
pads `compress_ratios` to `block_count`; reads `nextn_predict_layers`; adds a `graph_mtp`
that runs the combiner per hyper-connection stream on the 4x-wide residual (the PR text: mean
pooling first "drops acceptance catastrophically"). No new CLI flag — `--spec-type draft-mtp`
already exists on master for other archs. Head must carry the `blk.48.*` block plus
`nextn.{enorm,hnorm,eh_proj,hc_head_*}`; the main model is unchanged. It does **not** relax
the trunk, so a detached head fails on `blk.0.hc_attn_norm.weight` (three separate
commenters, 2026-08-28 and 2026-08-31).

The PR's own numbers (claimed, M3 Max 128 GB, UD-IQ4_XS, temp 0): 27.43 t/s bare; n-max 2 →
37.22 (+35.7%, 89.2% acceptance); n-max 3 → 38.83 (+41.6%, 85.7%).

Open problems in the thread:

- Detached heads need the loader commit above; the `--mtp` export whitelist drops the head
  mixer (a commenter, with a fix at [PR #1 on the author's fork](https://github.com/rmonsurate/llama.cpp/pull/1)).
  Heads from other converters are not interchangeable ("expected 35, got 30", 2026-09-01).
- Rebased on current master it breaks twice: `h_nextn` rows are cropped before the export
  (reported with [#28104](https://github.com/ggml-org/llama.cpp/pull/28104)), and post-#27941
  `n_head_kv_arr[48] == 0` trips a `ggml_set_rows` assert (a commenter, 2026-09-01, with a
  6-line fix in the comment).
- Every speculative round took a full host checkpoint of the recurrent state, so on Vulkan
  and dual-GPU CUDA MTP was a 2-4x **loss** despite 85-95% acceptance (three reports).
  Fixed upstream by #28123 (rollback support), alternatively
  [#28118](https://github.com/ggml-org/llama.cpp/pull/28118) (on-device checkpoints, open).
  Metal was unaffected by that fix; its cost is small-batch `mul_mat_id` coverage
  (an M5 Pro report: −12% at n-max 2, +5% at n-max 3 with p-min 0.7).
- Greedy output is not byte-identical with the head on (Metal n-max ≥ 3, HIP, and on the
  fork's PR #144 with all four unsloth heads) — suspected conv-state rollback numerics.

Related PRs: [#27739](https://github.com/ggml-org/llama.cpp/pull/27739) (the original
Qwen4-Exp PR with `--mtp`, closed 2026-08-26 for #27742); [#27842](https://github.com/ggml-org/llama.cpp/pull/27842)
(closed 2026-08-28); [#27956](https://github.com/ggml-org/llama.cpp/pull/27956)
(closed 2026-08-29); [#28104](https://github.com/ggml-org/llama.cpp/pull/28104) (a port of a
third-party implementation, closed); [#28097](https://github.com/ggml-org/llama.cpp/pull/28097)
(open draft 2026-08-31: `mtp_only` for the unsloth layout on top of #27836, CPU-measured +29%
with Q8_0 at n-max 4); [#28136](https://github.com/ggml-org/llama.cpp/pull/28136) (PLE direct
reads, open). No mainline issue mentions `output_hc_norm` (search 2026-09-01).

## 3. The unsloth fork: PR #144 and release b10715-mix-86bd2d3

[unslothai/llama.cpp#144](https://github.com/unslothai/llama.cpp/pull/144) "MTP for
Qwen3.8-Flash-Next", opened 2026-08-30 08:05, **open**, branch `mtp/qwen4exp-nextn` onto
`base/upstream-662a0b012` (= mainline b10715). Seven commits: the three from #27836
cherry-picked with their original authorship; "let an MTP draft borrow the target's
embeddings and lm head" (fork #142); "allow loading a draft-only MTP export"; "cover the
recurrent conv state for rollback" (`c6e318e3e`, the same fix landed upstream as #28123);
"key the CUDA graph cache by shape". Versus mainline it adds: converter `--mtp-shared-embd`;
KV `{arch}.nextn_shared_target_tensors`; `llama_model_params.model_shared` and
`llama_model_loader::borrow_shared_tensor` (only `token_embd`, `output`, `output_norm`;
mismatched shapes throw); `mtp_only` detection in `qwen4exp.cpp`; `common/speculative.cpp`
sets `mparams.model_shared = model_tgt`. Files: 24, including `ggml-cuda`.

Claimed in the PR (one B200, greedy, 512 tokens, Q8_0 shared head): UD-Q4_K_XL 83.22 →
138.75 t/s (1.667x), UD-IQ1_S 90.07 → 120.90 (1.342x); before the two fixes MTP was 0.531x.
Acceptance by head precision (±0.91 pp): bf16 66.50, Q8_0 66.14, Q6_K 65.85, Q5_K_M 65.19,
Q4_K_M 64.35, Q3_K_M 63.22, Q2_K 54.08. Concurrency 8 is a net loss (0.81-0.87x). Open
items: a `--kv-unified` NaN under ≥4 sequences inherited from #27941, and
`--ctx-checkpoints 0` is needed for any byte-identical comparison. A commenter on #144
(RTX PRO 6000, 2026-09-01) measured off 98.80 t/s; shared-Q8_0 176.18 (76.1%); shared-Q4_K_M
177.14; **Q8_0 191.14 (1.935x)**; shared-BF16 167.35 — the borrow path costs ~8%, and greedy
diverged on 3 of 4 prompts with every head.

Release [`b10715-mix-86bd2d3`](https://github.com/unslothai/llama.cpp/releases/tag/b10715-mix-86bd2d3),
published 2026-08-31 17:36, the newest as of 2026-09-01 (the tag before it, `b10698-mix-67dfc8b`,
predates #144). Its manifest: upstream tag b10715 (`662a0b012`, 2026-08-31 10:11) plus 14
pinned PRs — #144 at `586b15e`, mainline #27941 (qwen4exp follow-up fixes, merged upstream
2026-09-01 as b10737), #27754 (GLM-5-Next), fork #137/#152/#154 (mmap and kv-cells), #91
(IQ1_XS family), #95, #107, #149, #157, #158, #25731. The mixed source commit `92cedc8` is
**not** in the repository (only in `llama.cpp-source-commit-92cedc8...tar.gz`); the tag itself
points at the pin-manifest branch. Assets that matter here:

| platform | asset |
|---|---|
| macOS arm64 | `llama-b10715-mix-86bd2d3-bin-macos-arm64.tar.gz` (11.3 MB) |
| macOS x64 | `llama-b10715-mix-86bd2d3-bin-macos-x64.tar.gz` |
| Windows CUDA 12 | `app-b10715-mix-86bd2d3-windows-x64-cuda12-{legacy,older,newer,portable}.zip` |
| Windows CUDA 13 | `app-b10715-mix-86bd2d3-windows-x64-cuda13-{older,newer,portable}.zip` |
| Windows Vulkan | `app-b10715-mix-86bd2d3-windows-x64-vulkan.zip` |
| Windows CPU | `app-b10715-mix-86bd2d3-windows-x64-cpu.zip`, `...-windows-arm64-cpu.zip` |
| checksums | `llama-prebuilt-sha256.json`, `llama-prebuilt-manifest.json` |

(Linux and ROCm follow the same `app-<tag>-linux-<arch>-<backend>.tar.gz` pattern.)

Behind/ahead: base b10715; mainline master `3466812` (b10751) is 37 commits ahead of b10715.
Of those, the fork already carries #27941 and the #28123 fix; it lacks the rest (e.g. #25952,
today's fused CUDA MoE reduction). Nothing in mainline's 37 adds the MTP graph, the borrow
path, or `mtp_only`. A Hub user measured the Windows CUDA 13 asset on an RTX PRO 6000 Max-Q,
UD-Q4_K_XL, temp 0.7, `--parallel 2`: 70 → 120 t/s at n-max 3, 140 at n-max 5
([discussion #55](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/discussions/55)).

## 4. Does the main Flash-Next GGUF already embed the nextn tensors?

No — measured on the local UD-IQ4_XS: `block_count 48`, no `nextn_predict_layers`, and none
of the 1224 tensors is `blk.48.*` or `*.nextn.*` (table at the top). Mainline's converter still
declares `supports_mtp_export = False` / `no_mtp = True` (the lines #27836 deletes), so every
Hub quant made with it dropped the checkpoint's 31 `mtp.*` tensors; the alternative head
repositories in §6 say the same ("the popular GGUFs have none", "converted `--no-nextn`"). The
Hub API exposes no per-tensor metadata for split files, so the local header read is the check.

The "one flag unlocks MTP" recipe repository
([link](https://github.com/sudoingX/qwen38-mtp)) is about **Qwen3.8-27B**, arch `qwen35`: its
GGUFs carry `blk.*.nextn.*` (the README verifies `qwen35.nextn_predict_layers`), and llama.cpp
has had a qwen35 MTP graph since [#22673](https://github.com/ggml-org/llama.cpp/pull/22673)
(July 2026), so "the flag was free" there. None of that transfers to Flash-Next: different arch,
no head in the file, no graph on master. That README never mentions Flash-Next (grep, 2026-09-01).

## 5. Recommended head, precision, published numbers, and the flags

Head: unsloth's [MTP/README](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/blob/main/MTP/README.md)
(2026-09-01) says use `mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf` (2.60 GB), `--spec-draft-n-max 2`,
"1.3x to 1.7x at low concurrency", skip it for concurrent serving, and "these do not work on
mainline `ggml-org/llama.cpp`" as of `0eadefebd` (= b10731). Precision: Q8_0 is the optimum
in PR #144 (bf16 larger *and* slower, "the LM head is cheaper at 8 bits") and in #28097's CPU
ladder; the #144 commenter above found the non-shared Q8_0 faster than shared by 8% at
identical acceptance. Alternative repository 2 in §6 measured the opposite for a Q4 target
(Q4_K_M head beat Q8_0, "a draft quantized like its target agrees with it more often") — so
measure both against UD-IQ4_XS.

Published acceptance and speed (all greedy unless stated): #27836 itself 85.7-89.2% / +36-42%
(M3 Max, Metal); PR #144 66% / 1.67x (B200); the #144 commenter 76% / 1.9x (RTX PRO 6000);
alternative repository 2: 0.90 code / 0.74 prose, 20.3 → 35.8 t/s (Strix Halo ROCm);
alternative repository 1: 0.62, 24.2 → 29.3 (Strix Halo Vulkan); an M5 Max report on #27836
+68% short / +13-18% at 4-32K (n-max 6, p-min 0.7, backend sampling); an M5 Pro report −12%
to +5%; #28097 +29% (CPU, n-max 4). Apple Silicon is the weakest platform in every report,
and the wins there came from gating (`p-min 0.7`) rather than depth.

Flags, identical in `common/arg.cpp` on master `3466812` (lines 4140-4255) and on the fork
tag (lines 4096-4179) — no new option was added, the fork changes only what loads:

| flag | default (`common/common.h`) | env |
|---|---|---|
| `--spec-type draft-mtp[,ngram-mod,...]` | comma list; repeated flag is allowed | |
| `-md`, `--spec-draft-model`, `--model-draft FNAME` | unused; also sets `hf_file` for `--spec-draft-hf` | `LLAMA_ARG_SPEC_DRAFT_MODEL` |
| `--spec-draft-n-max N` | 3 | `LLAMA_ARG_SPEC_DRAFT_N_MAX` |
| `--spec-draft-n-min N` | 0 | `LLAMA_ARG_SPEC_DRAFT_N_MIN` |
| `--spec-draft-p-min P` (`--draft-p-min`) | 0.00 | `LLAMA_ARG_SPEC_DRAFT_P_MIN` |
| `--spec-draft-ngl`, `-ngld` | auto | |
| `--spec-draft-backend-sampling` | on | |

`--spec-draft-n-max 2` is unsloth's default; PR #144 used it; #28097 found 4 best on CPU;
the M5 Max report used 6 with p-min 0.7. `--ctx-checkpoints 0` for identity checks (PR #144).

## 6. Alternative head repositories

| repo | files | built for | loads on |
|---|---|---|---|
| alternative 1 ([link](https://huggingface.co/agentionai/Qwen3.8-Flash-Next-MTP-Q8_0-GGUF), 2026-08-30) | one self-contained Q8_0, 4.14 GB; a sibling repo has a 2.28 GiB ROCmFP4 head for a fork's quant types | a Vulkan fork branch `vulkan/qwen4exp-rocmfpx` ([link](https://github.com/LaurentZuijdwijk/llama.cpp)), graph from #27739 | that fork; a #27836 commenter ran it via `spec-draft-hf` on #27836 + a82a58a (CUDA) |
| alternative 2 ([link](https://huggingface.co/dzannotti/Qwen3.8-Flash-Next-MTP-GGUF), 2026-08-27) | BF16 7.8 GB, Q4_K_M 2.5 GB (34 tensors incl. `output_hc_*`, `output`, `token_embd`); two shards that graft the head into unsloth UD-Q4_K_XL as a 5-shard set; `patches/*.patch`, `merge-mtp-shard.py` | b10612 + #27742 + its patch (the #27739 graph), or the Vulkan fork above; wants `LLAMA_ATTN_ROT_DISABLE=1` on that base | #27836 + a82a58a ("loads cleanly", a commenter 2026-09-01; #28123 was tested with it); **not** #27836 alone (a DGX Spark report); the grafted shards were rejected by the strict reader (a HIP report) |
| alternative 3 ([link](https://huggingface.co/drluoto/Qwen3.8-Flash-Next-MTP-GGUF)) | Q8_0 4.14 GB, Q4_K_M, bf16; 37 tensors from #27836's `--mtp` + the whitelist fix | #27836 + a82a58a (a pre-assembled branch is linked from its card) | same |
| alternative 4 ([link](https://huggingface.co/quimmedes/Qwen3.8-Flash-Next-MTP-GGUF), 2026-08-28) | Q4_K_M/Q6_K/Q8_0/BF16 | its own llama.cpp fork (linked from the card) | that fork |

None of these loads on mainline master (every one lacks the trunk; §1), and none is known to
load on the unsloth fork prebuilt — the fork's `mtp_only` path looks for `nextn.hc_head_*`, and
only heads exported by #27836/#144's converter have that name (alternatives 1 and 2 carry
`output_hc_*`). Untested either way; a header read would settle it.

## 7. Recommendation for ml-stack

Measure first, on this machine: the fork asset `llama-b10715-mix-86bd2d3-bin-macos-arm64.tar.gz`
(sha256 in `llama-prebuilt-sha256.json`; or a Metal build of #144 at `586b15e`) against the
local `UD-IQ4_XS` with the head files already in
`~/.cache/huggingface/hub/models--unsloth--Qwen3.8-Flash-Next-GGUF/snapshots/5d16c05.../MTP/`.
Arms, all with `--spec-type draft-mtp --spec-draft-backend-sampling --ctx-checkpoints 0`:
`shared-Q8_0` (unsloth's pick), `Q8_0` (the 8% seen on #144), `shared-Q4_K_M`
(the match-the-target result from alternative 2), each at n-max 2 / 3 / 6 with
`--spec-draft-p-min 0.7`, plus one bare arm. Smoke it (`ml-stack-bench drafts --smoke`) before
the sweep, and check greedy identity off/on, because Apple Silicon is where every published
number is weakest and where output diverged. Do not spend time on mainline for this until §2
moves.

`ml-stack-serve --draft auto` picks "the `mtp-*.gguf` beside the weights"; here there are six
in `MTP/`, none beside the shards, and three need a binary that can borrow — the chooser needs
to know which binary is serving before it picks `shared-*`.

Re-check, weekly until one of them closes the loop: #27836 state (still Draft, no reply from
its author); #28097 and #28118 (open); whether mainline `src/models/qwen4exp.cpp` gains
`graph_mtp` and `mtp_only`, and `src/llama-arch.cpp` gains `nextn_shared_target_tensors` —
that is the signal to drop the fork; a newer `unslothai/llama.cpp` release tag (the README's
"or newer"); and the `MTP/README.md` last-commit date (2026-09-01 09:13). If mainline lands a
different loader contract, the unsloth heads may need re-export: watch `conversion/qwen4exp.py`
for `no_mtp = True` disappearing.
