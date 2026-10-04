# Harness comparison

Rows are appended by `scripts/compare-harnesses`. No scored run is recorded yet.

## 2026-10-03: smoke, Codex only, not scored

- command: `scripts/compare-harnesses --model Qwen3.5-2B --ctx 262144 --port 8791` (stopped by hand after about fifteen minutes)
- model: Qwen3.5-2B (Q4_K_M, `unsloth/Qwen3.5-2B-MTP-GGUF`), served by the broker lease, 1 slot x 262,144 tokens, q8_0 KV cache, no separate draft head
- result: Codex 0.160 reached the server through `/v1/responses`, the PreToolUse hook ran and refused the shell edits it classified as unsure, and the model made 34 tool calls without fixing the bug. The fixture's test was not passed; the run produced no row. Claude Code did not finish a run before it was stopped. The `/slots` output of the server reported `n_prompt_tokens_cache` against `n_prompt_tokens` per request (15,608 cached of 61,739 at the moment it was read), which the comparison table records as `last_prompt_tokens` and `last_cached_tokens`.
