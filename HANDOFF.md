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

## The next measurements (2026-09-03, small hours)

The queues of 2026-09-02 are in the record: `docs/report-2026-09-02.md` (with its
Extraction section), `docs/model-ranking.md`, `profiles.json`, `fit.json`,
`docs/architectures/qwen4exp.md`. Settled tonight: draft length 4 for answering *and*
extraction (6 equal, 8 slower, on both; and on textbook units, where the head is accepted
97% of the time, 4 / 8 / 12 decoded at 50.6 / 44.5 / 48.5 tok/s -- `ml-stack-ingest --n-max`
measures it); no per-workload draft field. Single and few on the
small models at ten questions: `single` +8 pts on E4B (59 vs 51), +3 on E2B, +2 on
gpt-oss-20b, all inside ±28 bands; `few` loses 20-30 pts everywhere; `--rounds 8` changed
nothing (the extra turns went unused).

- [ ] **The shelf is reading** (started 2026-09-02T23:44, detached, Flash-Next on port 8080,
  two workers): ten OpenStax PDFs from `~/Documents/Textbooks` plus
  `~/Downloads/Psychology2e_WEB.pdf` into `~/.ml-stack/shelf.ladybug`. Chapter 2 of
  Biology2e alone was 13 units in 19 min on one worker; the shelf is days of GPU, not
  hours. `ml-stack-ingest status --out ~/.ml-stack/shelf.ladybug` says where it is; the log
  is under `~/.ml-stack/ingest/logs/`; killed, the same command with `--resume` picks up
  (a failed unit is read again). When it is done or stopped: `ml-stack-serve down --port
  8080`, then ai_ceo's page back (`services/ui.sh fresh`). Then look at what it read --
  which needs a command: `ml-stack-ingest show --out STORE [--book B] [--sample N]` (a few
  concepts with their definitions, relations with page provenance, the folds it made);
  tonight that was a python read of `GraphStore.edges()` by hand, which is a missing
  command. First things seen: a person `created_by` a method (the verb's direction
  reversed by the model -- gloss it with the person on the right), and relations across
  books never joined (each book folds alone; a fold across the shelf is the next step).
- [ ] **Watch for units that still run to the ceiling.** The two that did on the first
  night were chapter-end question banks (one part carried 66 lettered answers); `pdf.units`
  now leaves those out and the ingest says how many. A unit that still fails keeps its
  whole reply in the reads file beside the store (`raw`), so read that before spending GPU
  on it. If any remains, try DRY sampling for extraction and measure it on the gold gate.
- [ ] **`single` on E4B at a hundred questions** (`sweep --serve gemma-4-E4B-it-qat-UD-Q4_K_XL.gguf
  --profile --plain-only --also single --yes`, ~10 min): the one asking-way change tonight
  that moved a small model, unconfirmed at ten. If it holds, `report --profile` sets it.
- [ ] **`ml-stack-ingest` serves on 8099 by default, the port the bench leaves its last
  model on**; one shape per port refused it twice tonight, silently inside a queue. Either
  the ingest takes the bench's lock and port discipline (`bench.serve` has both) or its
  default moves; and a queue step that dies in under a second should print why (`step`'s
  `tail` swallowed it).
- [ ] **The gold set is nine triples**, two of them (`center_of`, `independent_from`) outside
  the vocabulary on purpose, and precision against three triples a passage is not a model
  score (the model states true things the gold omits). Write twenty invented passages with
  everything they state written down (`tests/known-fixtures.txt` names), so the gate
  measures precision too; `ingest.INVERSES` covers the flipped ones.
- [ ] **A new entry point is invisible until `pip install -e . && pyenv rehash`**: every
  textbook step in two queues died in 0 s on `command not found`. `ml-stack-setup` (or the
  updater's dev track) should compare `[project.scripts]` with what is on PATH and say so.
- [ ] **The fine-tuned tool caller.** `ml-stack-train-tools from-bench` over the traced runs
  (traces are on by default at ≤20 questions; the hundred-question runs before that carry
  none -- rerun Flash-Next's hundred with `--trace` for ~5,000 turns), then
  `ml-stack-train-run --recipe tool-calls --size e4b --lora --export-gguf --yes` (~18 h
  here; Adam's go-ahead), then the measure in `docs/research/tool-caller-finetune.md`.
- [ ] **Watch ggml-org/llama.cpp#27836.** When the qwen4exp MTP graph merges, `ml-stack-serve
  build` and `ml-stack-bench drafts` Flash-Next on mainline: the PR reports 86–89% acceptance
  on an M3 Max against the fork's 73–79%. Then the profile's build field can go.
- [ ] **The 27B is unranked** -- a two-question smoke is all the store holds, and a record is
  never set from fewer than 20. Its twenty questions belong on the 3090 Ti through the
  fleet (below), not on this machine (Adam: "don't test the smaller models on this device").

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
