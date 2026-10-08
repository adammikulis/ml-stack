# Model lineups and published benchmarks

Retrieved 2026-10-08. Every figure carries its source URL, the source's own date and the settings
the source states. Nothing here was measured by this repository. Where the vendor's figure and a
third party's differ, both are shown. `unverified` means the page could not be fetched or only a
secondary write-up was available. The repository's own measurements (JevBench, the decider sets,
`docs/bench.md`) override every number in this file.

Sources fetched directly: Anthropic platform docs and announcements, OpenAI developer docs, Vals AI
model pages. `openai.com/index/...` returned HTTP 403 to the fetch tool, so OpenAI launch-page figures
come from secondary coverage and are marked.

## 1. Lineups and exact API ids

### 1.1 Anthropic (Claude 5 family)

Source: https://platform.claude.com/docs/en/models/overview (retrieved 2026-10-08), model pages
`.../models/{haiku,sonnet,opus,fable}-5-5|5-1/overview`. Every Claude id is a pinned snapshot; the
dateless id is its own alias.

| Tier | Name | Claude API id | Bedrock id | Released | $/MTok in / out | Context | Max out | Default effort |
|---|---|---|---|---|---|---|---|---|
| 1 (lowest) | Claude Haiku 5.5 | `claude-haiku-5-5` | `anthropic.claude-haiku-5-5` | 2026-10-07 | 0.10 / 0.50 (prompt up to 100k); 0.50 / 2.50 above | 1M | 128K | medium |
| 2 | Claude Sonnet 5.5 | `claude-sonnet-5-5` | `anthropic.claude-sonnet-5-5` | 2026-09-28 | 2 / 10 | 1M | 128K | high |
| 3 | Claude Opus 5.5 | `claude-opus-5-5` | `anthropic.claude-opus-5-5` | 2026-09-22 | 4 / 20 | 1M | 128K | medium |
| 4 | Claude Fable 5.1 | `claude-fable-5-1` | `anthropic.claude-fable-5-1` | not on fetched page | 10 / 50 | 1M | 128K | high |

- Batch API is 50% off. Cache reads cost 10% of input (5% on Opus 5.5 and Sonnet 5.5, 2.5% on Fable 5.1).
  Opus 5.5 fast mode: $8 / $40 (https://www.anthropic.com/news/claude-opus-5-5, 2026-09-22).
- Claude Mythos 5.1 is named on the overview page. Its id was not fetched; it is not in the draft tier table.
- Legacy and still available: Fable 5, Opus 5, Opus 4.8/4.7/4.6/4.5, Sonnet 5, Sonnet 4.6, Haiku 4.5.
  `claude-haiku-4-5-20251001` is the Haiku 4.5 id (third-party, https://www.gradually.ai/en/claude-models/,
  unverified against the platform page).
- Anthropic: Haiku 5.5 is "for high-volume, latency-sensitive tasks such as classification, extraction,
  and routing" and "pairs well with Opus 5.5 and Sonnet 5.5 as a subagent on coding work", while "Sonnet 5.5
  and Opus 5.5 remain better choices for complex agentic coding tasks" (https://www.anthropic.com/claude-haiku-5-5,
  2026-10-07).
- Haiku 5.5 uses the newer tokenizer: about 30% more tokens for the same text than Haiku 4.5.

### 1.2 OpenAI (GPT-6 family)

Source: https://developers.openai.com/api/docs/models (retrieved 2026-10-08; page undated).

| Tier | Name | API id | $/MTok in / out | Context | Max out |
|---|---|---|---|---|---|
| 1 (lowest) | GPT-6 Luna | `gpt-6-luna` | 0.10 / 0.50 | 1.05M | 128K |
| 2 | GPT-6.1 Sol | `gpt-6.1-sol` | 2 / 10 | 1.05M | 128K |
| 2 (earlier) | GPT-6 Sol | `gpt-6-sol` | 2 / 10 (secondary, iClarified, 2026-09-22) | 1.05M | 128K |
| 3 | GPT-6 Astra | `gpt-6-astra` | 10 / 50 | 1.05M | 128K |

- Page descriptions: Astra "our most capable model for the most demanding work"; Sol balances
  intelligence and cost; Luna "our most efficient model for focused, high-volume tasks".
- GPT-6 Sol and Luna launched 2026-09-22 (https://www.iclarified.com/102351/openai-cuts-api-prices-50-with-gpt-6-sol-and-luna,
  secondary; the primary page https://openai.com/index/introducing-gpt-6-sol-and-luna/ was not fetchable).
  GPT-6.1 Sol launched 2026-09-29 (https://www.vals.ai/models/openai_gpt-6.1-sol, release date field).
  Earlier family: GPT-5.6 Sol, Terra, Luna (ids not verified).
- Codex-facing names (https://learn.chatgpt.com/docs/models, redirect of developers.openai.com/codex/models,
  retrieved 2026-10-08): the same ids. `gpt-6.1-sol` is recommended for complex coding and agentic work and is
  the example `model` in `config.toml`; Luna for "focused, repeatable tasks" (extraction, classification,
  transformation, structured summaries); Astra for the hardest end-to-end work. The page does not say which
  model Codex subagents use.

### 1.3 Local (Qwen3.8)

Names as they appear in this repository's docs, not looked up externally: `Qwen3.8-27B`
(`Qwen3.8-27B-UD-Q4_K_XL.gguf` is the owner's default 27B quant), `Qwen3.8-Flash-Next`, and the decider
base `Qwen3.5-2B`. No public benchmark was collected for them.

## 2. Benchmarks

Scores from different harnesses and effort levels are not comparable to each other.

### 2.1 Agentic coding

Terminal-Bench 4.0.

| Model id | Vendor figure | Third-party figure |
|---|---|---|
| `claude-sonnet-5-5` | 70.6%, effort not stated (https://www.anthropic.com/claude-sonnet-5-5, 2026-09-28) | Vals 64.14% +/-1.01, rank 2/45; update post 53.03% (https://www.vals.ai/models/anthropic_claude-sonnet-5-5, undated, effort max) |
| `claude-opus-5-5` | 66.4%, xhigh, +/-2.6 (https://www.anthropic.com/news/claude-opus-5-5, 2026-09-22) | Vals 65.15%, rank 1/45; update post 61.62%, or 53.54% counting fallback-assisted tasks as failures (https://www.vals.ai/models/anthropic_claude-opus-5-5) |
| `claude-fable-5-1` | 55.8% (Opus 5.5 page) | not collected |
| `claude-haiku-5-5` | 39.2%, effort not stated (https://www.anthropic.com/claude-haiku-5-5, 2026-10-07) | Vals 35.35% +/-2.20, rank 12/45 (https://www.vals.ai/models/anthropic_claude-haiku-5-5) |
| `gpt-6-astra` | 57.9%, high, "as reported by OpenAI", quoted on the Opus 5.5 page | not collected |
| `gpt-6.1-sol` | not collected from OpenAI | Vals 55.05% +/-1.82, rank 6/45 (https://www.vals.ai/models/openai_gpt-6.1-sol) |
| `gpt-6-luna` | 16.4% (Haiku 5.5 page, 2026-10-07) | Vals 13.64% +/-1.51, rank 29/45; update post 9.60% (https://www.vals.ai/models/openai_gpt-6-luna) |

Vals pages state that table and update-post figures differ without explanation, and that fallback
models completed some tasks. Vals pages retrieved 2026-10-08 and are undated.

FrontierCode 1.1 (Main), vendor tables:

| Model | Score | Source |
|---|---|---|
| Opus 5.5 | 54.4% (text: 54.6% at medium) | Opus 5.5 page, 2026-09-22 |
| Sonnet 5.5 | 52.1% xhigh; 46.2% at max, penalised for out-of-scope edits | Sonnet 5.5 page, 2026-09-28 |
| Haiku 5.5 | 46.4% | Haiku 5.5 page, 2026-10-07 |
| Fable 5.1 | 50.3% | Opus 5.5 page |
| GPT-6 Astra | 53.3% | Opus 5.5 page |
| GPT-6 Sol | 49.3% | Sonnet 5.5 page |
| GPT-6 Luna | 42.4% | Haiku 5.5 page |

- CursorBench 4.0: Opus 5.5 57.8%, Sonnet 5.5 55.5%, Fable 5.1 51.8% (Opus 5.5 and Sonnet 5.5 pages).
- DeepSWE v1.1, OpenAI-reported, secondary (https://www.iclarified.com/102351/openai-cuts-api-prices-50-with-gpt-6-sol-and-luna,
  2026-09-22): GPT-6 Sol (max) 68.8%, GPT-6 Luna (max) 66.6%. GPT-6.1 Sol: secondary coverage cites 75.2% at high
  and BenchLM lists 71.9% at max; unverified.
- Vibe Code Bench v1.1 (Vals): Sonnet 5.5 92.39%, Haiku 5.5 90.44%, Opus 5.5 90.29%, GPT-6.1 Sol 88.93%, GPT-6 Luna 81.65%.
- SWE-bench Verified and SWE-bench Pro: no score for these models appeared on the vendor pages fetched. The
  Sonnet 5.5 system card lists SWE-Bench Pro; its scores were not read. unverified.

### 2.2 Reasoning, computer use, knowledge work (vendor tables)

| Benchmark | Haiku 5.5 | Sonnet 5.5 | Opus 5.5 | Fable 5.1 | Luna | Others |
|---|---|---|---|---|---|---|
| Humanity's Last Exam, with tools | 57.4% | 64.5% | 67.7% | 65.6% | n/a | Astra 57.2% |
| Humanity's Last Exam, no tools | 45.9% | 56.9% | n/a | n/a | n/a | n/a |
| OSWorld 2.1 (partial or offline subset) | 72.4% | 83.9% (80.1% on Sonnet page) | 81.8% | 80.7% | 48.9% | n/a |
| GDPval-AA v2.1 (Elo) | 1620 | 1840 (1844 on Sonnet page) | 1846 | 1735 | 1437 | Astra 1542, Sol 1487 |
| AA-Briefcase v1.1 (Elo) | 1578 | 1824 (1811 on Sonnet page) | 1822 | n/a | 1336 | Sol 1483 |
| Terminal-Bench-Science 0.1 | n/a | n/a | 58.7% | 52.6% | n/a | Astra 64.6% |

Sources: Haiku 5.5 page (2026-10-07), Sonnet 5.5 page (2026-09-28), Opus 5.5 page (2026-09-22). Pages disagree
with each other on the same cell; the Haiku page's reference column is shown first, the Sonnet page's value in
brackets. Artificial Analysis ran GDPval-AA and AA-Briefcase on a pre-release deployment with a
structured-output bug since fixed, and OpenAI fixed a GPT-6 Sol image-understanding bug after those runs
(Sonnet 5.5 page footnotes).

### 2.3 Tool use and long-horizon agents

- AutomationBench (Zapier-run): Opus 5.5 40.0%, Astra 41.4%, Fable 5.1 31.4% (Opus 5.5 page, 2026-09-22);
  GPT-6 Sol (xhigh) 33.2% at $0.27 per task (secondary, iClarified). Agents' Last Exam: GPT-6 Sol (max) 56.4%
  (secondary; unverified).
- Vals SRE Bench: Opus 5.5 33.59%, Sonnet 5.5 30.15%, GPT-6.1 Sol 50.76% (rank 2/32), GPT-6 Luna 2.67%. Opus 5.5's
  score drops to 5.34% when fallback-assisted tasks (82.82% of them) count as failures.
- Vals Index and cost per test, effort max: Sonnet 5.5 67.04% at $21.34; Opus 5.5 66.97% at $32.14; GPT-6.1 Sol
  61.15% at $3.237; Haiku 5.5 54.31% at $2.991; GPT-6 Luna 51.22% at $0.431. Update posts give different values
  (Opus 5.5 69.69%, Sonnet 5.5 69.22%, Luna 58.45%).

### 2.4 Long context, cost, latency

Context and output limits are in section 1 (1M or 1.05M context, 128K output). No long-context retrieval
benchmark was found on the pages fetched; unverified. Latency for the full Vals Index run at max effort (not per
call): Sonnet 5.5 78 min, Opus 5.5 79 min, GPT-6.1 Sol 43 min, Haiku 5.5 47 min, GPT-6 Luna 31 min. Anthropic's
comparative latency labels: Haiku 5.5 "Fastest", Sonnet 5.5 "Fast", Opus 5.5 "Moderate", Fable 5.1 "Slower"
(models overview). Anthropic claims Opus 5.5 costs about 40% less than Opus 5 on typical workloads and
generates output over 30% faster (Opus 5.5 page).

## 3. What to launch when

Cheapest adequate tier, using the repo order: Haiku 5.5 for read-only lookup and for ratchet-lowering with a named
metric and test selector, Sonnet 5.5 for anything else that writes or reviews code, Opus 5.5 only after Sonnet
fails, Fable only when the owner asks.

| Work | Launch | Evidence |
|---|---|---|
| Read-only search, lookup, summarising, compaction, classification | `claude-haiku-5-5` | Vendor positioning; $0.10 / $0.50 |
| Small, well-specified edit (one function, one doc) | Haiku 5.5; escalate after the first failed test run | Vibe Code Bench 90.44% |
| Routine implementation with tests in a claimed worktree | `claude-sonnet-5-5` | Terminal-Bench 4.0 39.2% (Haiku) vs 70.6% (Sonnet); FrontierCode 46.4% vs 52.1%; Anthropic names Sonnet/Opus the better choice for complex agentic coding |
| Architecture, security diagnosis, hard debugging | Sonnet 5.5 first; Opus 5.5 after it fails | Opus leads HLE (67.7 vs 64.5) and FrontierCode (54.4 vs 52.1) by small margins at twice the price |
| Independent review | A different model from the author; Sonnet 5.5 or higher | Same-model review shares blind spots |
| Coordination (main session) | Sonnet 5.5 or higher; never Haiku 5.5 or Luna | Lowest tier never coordinates |
| Codex-side work | `gpt-6.1-sol` for coding; `gpt-6-luna` for read-only and extraction | OpenAI Codex docs; Luna Terminal-Bench 4.0 13.64% to 16.4% |
| Hardest end-to-end work, on request | `claude-fable-5-1` or `gpt-6-astra` | Fable 5.1 scores below Opus 5.5 on every vendor-table benchmark collected, at 2.5 times the price |

Effort is a separate setting from the model. Anthropic reports Sonnet 5.5 at max effort scoring lower on
FrontierCode than at xhigh, so max is not a default.

### Where the evidence argues against a rule

- "Default to Haiku 5.5 for most subagent work" (former CLAUDE.md rule): replaced. Haiku 5.5 holds for lookup,
  summarising and narrow edits, and the in-repo run below shows it matching Sonnet 5.5 on ratchet-lowering with a
  named metric. For open-ended implementation the Terminal-Bench 4.0 gap (39.2% vs 70.6%, vendor) argues for
  Sonnet; a failed Haiku attempt costs a full rerun at Sonnet price.
- "Opus only after Sonnet fails": consistent. Sonnet 5.5 matches Opus 5.5 on Terminal-Bench 4.0 (70.6 vs 66.4
  vendor; 64.14 vs 65.15 Vals) at half the price.
- "Fable only when the owner asks": consistent.
- "Lowest tier never coordinates": nothing collected argues against it; Haiku 5.5 and Luna are the weakest
  agentic scores above.

## 4. Limits of public benchmarks

- Vendor numbers use vendor-chosen effort and harnesses; footnotes state that safeguards intervened and
  fallback models completed some tasks, lowering Opus 5.5's scores.
- Third-party pages disagree with their own update posts and with the vendor, and ranks move between visits.
- Terminal-Bench, FrontierCode and OSWorld measure public tasks. This repository's work is claims, hooks,
  graph code, gates and long-lived worktrees; no benchmark here measures obeying AGENTS.md or reporting honestly.
- Prices change; the vendor pages cited are the source, not this table.
- Repository measurements (JevBench, decider sets, `docs/bench.md`) override this file. Re-collect it when a
  tier ships or a vendor retires an id.

## 5. In-repo run: ratchet-lowering, Haiku 5.5 against Sonnet 5.5

Run 2026-10-08 by the lead session. One attempt per model per task, each in its own throwaway worktree cut
from `0.2dev` at `b9f28265`, identical brief per pair, no branch landed. Models `claude-haiku-5-5` and
`claude-sonnet-5-5`. Source of the targets: `scripts/budgets --show ruff-other` on that commit. Checks the
lead re-ran in each worktree (not the workers' reports): `ruff check <files> --output-format concise` and
`scripts/test all -n 1 <test files>`. A blind Sonnet reviewer (model labels shuffled, commit messages hidden)
scored the diffs and ran a differential test of old against new code.

| Task | Target | Haiku 5.5 | Sonnet 5.5 |
|---|---|---|---|
| T1 | 11 `RUF059` in `tests/test_graph_cache.py`, `tests/test_graph_bench_standard.py` | 11 fixed, 30 tests pass, 10+/10-, 123 s, 59k tokens | 11 fixed, 30 tests pass, 10+/10-, 101 s, 55k tokens |
| T2 | 10 `RUF012`/`RUF007` in `tests/test_fleet_work.py` | 10 fixed, 34 tests pass, 19+/14- (reflowed lines), 145 s, 62k | 10 fixed, 34 tests pass, 12+/10-, 168 s, 58k |
| T4 | `PLR0915` (58 > 50 statements), `UP035`, `RUF046` in `src/ml_stack/bench/history.py` | all fixed, 16 tests pass, 71+/42-, largest function 26 statements, 157 s, 68k | all fixed, 16 tests pass, 51+/36-, largest function 32 statements, 104 s, 57k |

- Objective checks tie: every target finding fixed, no new finding, every test file passes, the same
  out-of-scope findings remain (3 in T1, 2 in T2, 1 in T4).
- T4 differential test, 316 inputs: 0 differences against the original for both models.
- Blind review, with the labels resolved afterwards: T1 prefers Haiku moderately (keeps the discarded names,
  `_calls`); T2 prefers Haiku strongly; T4 prefers Sonnet weakly (smaller diff; Haiku's decomposition read
  better but added docstrings and renamed `exit`). The reviewer's T2 and T4 line-length penalties do not
  apply to the repo's gates: `pyproject.toml` sets `line-length = 100` but does not select `E501`.
- Haiku's diffs carried more unrequested churn on T2 and T4 (reflowed continuation lines, docstrings, one
  rename). Price at the vendor's figures is about twenty times lower for Haiku.
- Limits: three tasks, one attempt each, one reviewer. Not a measurement of design-heavy refactors.
