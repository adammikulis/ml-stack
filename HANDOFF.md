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

## Where things stand (2026-09-03, night)

The day's record: `docs/report-2026-09-02.md`, `docs/model-ranking.md`, `profiles.json`,
`fit.json`, `docs/architectures/qwen4exp.md`. Settled: Flash-Next answers (80% F1 at 27 s/q,
100 questions) and extracts (96% node / 76% relation F1); draft length 4 for both, on
textbook units too (97% acceptance, no faster at 8 or 12); one slot for extraction (two
workers measured slower in aggregate); `single` +8 pts on E4B at ten questions, unconfirmed.
Landed 2026-09-03: the ingest as a package with fold-as-you-go, provenance by pointer, a
hidden run node, `show`/`status`/`shelf`/`ask`; `graph.tidy` (the separate hygiene pass,
automated by a model judge that re-reads the source; `absorb` on the way in, wired into
the ingest, the Slack pipeline and the simulator); the lifecycle closed (a backend
launches nothing without a manager's `Lease`, a port is never reclaimed from a server we
did not record, one detached run at a time, `jobs` with `wait`); one `Run` across bench,
page, seat and ingest; the report's Ingest section; the isolation guards; ladybug 0.20.2
with a write guard; `ml-stack-claude` and `ml_stack.harness`. Suite: 3,226 green.

- [ ] **The shelf holds APBiology and Biology2e chapter 2, sound; nine books are unread.**
  `~/.ml-stack/shelf.ladybug` on ladybug 0.20.2: ~9,500 concepts with definitions, page
  provenance and the run that read each, after the judged pass (362 merges, 1,489 inverse
  pairs, 622 conflicts judged with 282 edges dropped, 186 definitions, 113 suspects),
  `ml-stack-store check` clean. Reads beside the store are the truth (`ml-stack-ingest
  fold` rebuilds; two damaged stores from the 0.18 delete bug are kept beside it as
  `shelf.ladybug.corrupt-*`). Whether the other nine books -- about four days of GPU at 86
  s a unit, one slot -- are worth it is Adam's call; the command is `ml-stack-ingest
  ~/Documents/Textbooks/<book>.pdf --out ~/.ml-stack/shelf.ladybug --model <flash-next>
  --images --resume --serve-port 8080`, one book at a time, and it tidies itself at the
  book's end. Two answers before that: what a question over the shelf scores
  (`ml-stack-ingest ask --out ... --gold FILE`, no gold questions written yet), and what
  `ml-stack-ingest shelf` says once a second full book is in.

- [ ] **Watch for units that still run to the ceiling.** Two causes found and fixed on the
  first night: chapter-end question banks (`pdf.units` leaves them out) and a greedy
  decode circling a long relations array (63 clean concepts, then 378 relations of which
  282 were distinct, until n_predict) -- the document schema now caps every list
  (`maxItems`), so the grammar closes the array. A unit that still fails is read once
  more, then given up on; `status` counts those, its whole reply is `raw` in the reads
  file beside the store, and `ml-stack-ingest retry --out STORE` frees them after a fix.
  If one still circles under the cap, try DRY sampling for extraction and measure it on
  the gold gate.
- [ ] **Score answering over the shelf.** Write twenty invented-free but real-book
  questions with expected concept ids (`ml-stack-ingest ask --gold FILE`, the bench's
  scorer) and run them through Flash-Next in its profile; until that number exists, "a
  usable bio graph" means queryable and sound, not scored. The judge's conflict verdicts
  are worth reading first: it kept both edges in cases where its own reason said one was
  a misreading ("Larynx part_of trachea" beside "Larynx precedes trachea") -- the schema
  offers keep both / keep one / unsure, and the instructions should say keep both only
  when both are true.
- [ ] **A fold across books.** Every book folds alone; `tidy` joins duplicates across the
  shelf after the fact, but nothing yet says "this concept in Biology2e is that one in
  APBiology" with a weight a person can read. `Shelf.graph()` per book plus `tidy`'s merge
  log is the material; a `shelf` view (books, shared concepts, the edges between books'
  vocabularies) is the command.
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

## Improvements queued 2026-09-03 afternoon (Adam: "knock them out")

Extraction quality
Store integrity
- [ ] **A recorded server whose owner has gone should be stopped by the lifecycle, not by
  hand.** Twice on 2026-09-03 `ml-stack-serve status` showed Flash-Next on 8080 with "the
  process that started it (pid N) has gone" -- a judged pass's `_serving` lease whose
  owner ended without releasing (a monitor's shell killed, once; the other unexplained),
  ~90G of memory held until a person ran `down`. The record makes it ours, so: `status`
  says "orphaned" and the next `lease` on that port (or `ml-stack-serve down --orphans`)
  stops it; and find why `_serving`'s exit did not release -- a `Stopped`/SIGTERM path
  that skips the context manager's exit, most likely.
- [ ] **Two ladybug reports to file upstream, with Adam's go-ahead.** 0.18.x: a single
  `DETACH DELETE` in a ~10k-node store blanks other nodes' string columns (reproduction:
  `tests/test_graph_store_scale.py`). 0.20.2: the cached-physical-plan fast path
  re-executes a parameterized MERGE against a table rewritten since and segfaults, and
  the text index returns a node once per version written (reproduction: the store's
  `_written` docstring; two lines). ml-stack is on 0.20.2 with the per-write guard and
  the pin `>=0.19,<0.21`; the probes gate any bump.

## Library

- [ ] **`ml-stack-models layout MODEL`**: the attention layout off the GGUF header in one
  paragraph -- which layers hold a cache, which are recurrent, sliding (window, pattern,
  `key_length_swa`), shared, plus compress ratios, indexers, experts, and any lookup-table
  tensor (`fit --tensors` has the tensor side; `preflight._recurrent_layers` /
  `_sliding_layers` the layer side). It is how a `docs/architectures/` note starts.
- [ ] **`only_one(wait=False)` truncates the holder's pid when refused** (found 2026-09-02):
  the message read `pid 5` for pid 55017. Read the whole lock file before reporting it.
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
- [ ] **First real run of `ml-stack-claude` and `ml-stack-agent`.** Built and tested
  against fakes only (Adam had the GPU). Run `ml-stack-claude <flash-next> -- --print
  "say hello"` and one `ml-stack-agent "read README.md and say what this is" --model
  <flash-next> --allow Read`; watch the served alias the model variables carry, the
  stream-idle watchdog (five minutes of silence aborts -- `CLAUDE_STREAM_IDLE_TIMEOUT_MS`),
  and what `Usage` reports against the server's own `/metrics`; then measure a small task
  set the bench's way so the local harness has a number beside the page's.

## Verifying

```bash
python3 -m pytest tests -q -n 4 > /tmp/out.txt; echo $?   # never pipe into tail; -n 4 while a bench runs
ml-stack-setup                                             # the machine
ml-stack-bench status                                      # measuring, serving, what the job kept
ml-stack-serve profile                                     # every model's measured shape
```
