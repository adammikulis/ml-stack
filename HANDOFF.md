# Handoff

**Every item here is a task.** A finished task is deleted, not marked done — what exists and
why is in `README.md`, `docs/`, the code and `git log`. Each carries the context to pick it
up cold. Rules: invented names only, everywhere (`tests/known-fixtures.txt`, or a rule in
`contracts/name-shapes.json` when the refusal is a code fragment); tests build their own
fixtures and never read `~/.ml-stack`; a measurement is estimated before it runs and smoked
before it is paid for; nothing is pushed without Adam's go-ahead (a push cuts a release).
The app that drives this library is `~/ai_ceo`; its `HANDOFF.md` holds what is
Slack-specific. What was measured on 2026-09-02 and what it settled is
`docs/report-2026-09-02.md`, `docs/model-ranking.md`, `docs/architectures/` and
`src/ml_stack/data/profiles.json` (a model's measured shape, read by `ml-stack-serve up
--profile`, `sweep`, `extract` and `converse`).

## The next measurements (queued or waiting, 2026-09-02 evening)

- [ ] **Read the evening's queues into the record.** Queues 9–12 (per-way comparison of
  batch/kinds/summary/rich; the fit load of Flash-Next for its GPU/CPU split; extraction on
  the relation-bearing world in each model's measured shape; every other model's knob pass)
  were running when this was written; `ml-stack-bench status`, `show --since
  2026-09-02T21`, `show --extract` and `report --since 2026-09-02T09 --profile` read them.
  Then commit `docs/report-2026-09-02.md`, `docs/model-ranking.md`,
  `src/ml_stack/data/{profiles,fit}.json` and, if the split came through,
  `docs/architectures/qwen4exp.md`'s memory paragraph with the measured numbers.
- [ ] **Measure the single and few ways** (`--also single --also few --rounds N`, landing
  from the asking-space agent) on E2B, E4B and gpt-oss-20b -- the models whose tool choice
  may degrade with the number of schemas -- ten questions each, then `report --profile`.
- [ ] **The fine-tuned tool caller.** `ml-stack-train-tools from-bench` over the traced runs
  (traces are on by default at ≤20 questions since tonight; the hundred-question runs
  before that carry none -- rerun Flash-Next's hundred with `--trace` for ~5,000 turns),
  then `ml-stack-train-run --recipe tool-calls --size e4b --lora --export-gguf --yes` (~18 h
  here; Adam's go-ahead), then the measure in `docs/research/tool-caller-finetune.md`.
- [ ] **Watch ggml-org/llama.cpp#27836.** When the qwen4exp MTP graph merges, `ml-stack-serve
  build` and `ml-stack-bench drafts` Flash-Next on mainline: the PR reports 86–89% acceptance
  on an M3 Max against the fork's 73–79%. Then the profile's build field can go.

## Library

- [ ] **A `Run` object instead of twenty keyword arguments.** `served()` forwards most of what
  it takes and grew five more tonight (`serving`, `trace`, `reach`, the profile's fields);
  what is about the serving, the asking and the client is implicit, which is how `tight`
  leaked to `Client.__init__` once. Build one typed object from argv with three sections
  (`Shape`, the asking record, the client) and pass it; `measure.asking()`'s kwarg list is
  the seam to collapse.
- [ ] **`ml-stack-models layout MODEL`**: the attention layout off the GGUF header in one
  paragraph -- which layers hold a cache, which are recurrent, sliding (window, pattern,
  `key_length_swa`), shared, plus compress ratios, indexers, experts, and any lookup-table
  tensor (`fit --tensors` has the tensor side; `preflight._recurrent_layers` /
  `_sliding_layers` the layer side). It is how a `docs/architectures/` note starts.
- [ ] **`only_one(wait=False)` truncates the holder's pid when refused** (found 2026-09-02):
  the message read `pid 5` for pid 55017. Read the whole lock file before reporting it.
- [ ] **ladybug 0.20.2 returns nothing from a fresh store's scans** (CI on Linux; ten store
  tests read zero where one was written; 0.18.x pinned). Characterise with `ml-stack-store
  check` on a scratch store under 0.20 and either adapt the queries or keep the pin with
  the reason written down.
- [ ] **The tests, tidied.** ai_ceo's suite got the pass tonight (shared fakes in conftest,
  duplicates merged, machine state isolated, slow marked); this suite (~2,950 tests, ~3 min
  on every core) has the same duplication across `tests/test_graph_bench*.py`,
  `test_serve_*.py` and `test_fleet_*.py`: shared `_serving`/`_measured`/`a_row` fakes,
  `_fit_files_in_tmp`, the Scripted handler. Same brief; run with `-n 4` while anything
  measures. `tests/test_mcp.py::test_the_sdk_server_carries_the_same_tools` flakes under
  xdist (`asyncio.run` in a worker whose loop another test left running) -- isolate it.

## Measuring across the fleet

- [ ] **Run it for real across two machines.** Everything is tested against fakes and
  loopback; nothing has crossed a real network or a real Windows box. One visit:
  `irm https://raw.githubusercontent.com/adammikulis/ml-stack/main/packaging/install.ps1 |
  iex` (the app) or the `--headless` mode, then from here `ml-stack-fleet status` and a
  `sweep --fleet --serve gemma-4-E2B-it-qat-UD-Q4_K_XL.gguf --sample 6`. Expect bugs; the
  daemon's log and `ml-stack-doctor` are the first two places to look. After that the
  Windows box follows releases or main on its own (`fleet status` shows COMMIT/UPDATES).
- [ ] **Place users across the fleet by what fits.** Adam: "if we have more users we can
  sacrifice quality and use a smaller model with more simultaneous kv caches ... some users
  will get the bigger model and some smaller". Inputs exist: `fit` (per-token and
  per-sequence bytes, measured), `profiles.json` (each model's best shape and score),
  discovery (each peer's room). Build `ml-stack-fleet plan --users N --context C`: for every
  peer the best-ranked model whose loaded size plus N_i caches fit, N_i summed to N, best
  models to the most users; `--apply` serves it through each daemon. Then a router that
  sends a new session to a free slot on the best model. Tests on fakes.
- [ ] **`ml-stack-serve status --every` and `ml-stack-setup` should say which model cache
  is in use and its size** (the installer's `--system` mode can point a service at a user's
  cache; nothing reports it yet).

## Verifying

```bash
python3 -m pytest tests -q -n 4 > /tmp/out.txt; echo $?   # never pipe into tail; -n 4 while a bench runs
ml-stack-setup                                             # the machine
ml-stack-bench status                                      # measuring, serving, what the job kept
ml-stack-serve profile                                     # every model's measured shape
```
