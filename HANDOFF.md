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

## Where things stand (2026-09-03, late night)

The day's record: `docs/report-2026-09-02.md`, `docs/model-ranking.md`, `profiles.json`,
`fit.json`, `docs/architectures/qwen4exp.md`. Settled: Flash-Next answers (80% F1 at 27 s/q,
100 questions) and extracts (96% node / 76% relation F1); draft length 4 for both; one slot
for extraction; `single` +8 pts on E4B at ten questions, unconfirmed.

Landed tonight, thirteen branches by thirteen agents in their own worktrees (the rule held;
the ai_ceo diff was tens of lines): the judge reads a conflict's and a suspect's passages
through the caller's pointers (it had read `provenance` directly, so a community graph
keeping its pointers under `messages` judged 208 verdicts blind; re-judged with passages,
76 changed), `tidy --rejudge`, `ml-stack-store doc KEY [--drop]`; the bench's frontier,
rates and plot compare per question, the ranking breaks ties on made-up ids, `finding()`
reads the store, `drafts --store/--embed-url`; `GraphStore.has`/`unset_attribute`,
`remove_edge(source, rel, target)`, `merge_nodes` raising on a missing node, each `search`
its own statement (ladybug 0.20.2 served a cached plan's stale answers), `_unjson` refusing
a non-object; a colour for every kind, `render(most_messages=)`, `ml-stack-graph serve`;
orphaned servers named, adopted when the shape matches and stopped when not, `down
--orphans`, the ingest under `_stopping()` and the measuring lock, a queue step's fast
death printed, `ml-stack-setup` checking every entry point against PATH; `ml-stack-world
check`; ids constrained through `response_format` (llama.cpp refuses a grammar beside
tools) with `sweep --constrain-ids`; the tool list never changing within a question (the
final turn used to drop the searching tools, rendering a different prefix and re-reading
the whole conversation -- the page's 49 s against the bench's 27); `Client(api="ollama")`
over `/api/chat` with a timing a server does not report as None, `served_by()`,
`processes()`; `ml-stack-bench standard` (lm-evaluation-harness), `animate` (manim),
`ml-stack-do` (a served model drives the commands, asks first), `ml-stack-audit` and
`scripts/encrypted-volume.sh`; the divider rule in CLAUDE.md. Suite: ~3,420 green.

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
  a misreading ("Larynx part_of trachea" beside "Larynx precedes trachea") -- the
  instructions now say keep both only when the passages support both (2026-09-03); the
  shelf's verdicts predate that and are worth a `tidy --rejudge` over the shelf, ~an hour.
- [ ] **A fold across books.** Every book folds alone; `tidy` joins duplicates across the
  shelf after the fact, but nothing yet says "this concept in Biology2e is that one in
  APBiology" with a weight a person can read. `Shelf.graph()` per book plus `tidy`'s merge
  log is the material; a `shelf` view (books, shared concepts, the edges between books'
  vocabularies) is the command.
- [ ] **`single` on E4B at a hundred questions** (`sweep --serve gemma-4-E4B-it-qat-UD-Q4_K_XL.gguf
  --profile --plain-only --also single --yes`, ~10 min): the one asking-way change tonight
  that moved a small model, unconfirmed at ten. If it holds, `report --profile` sets it.
- [ ] **The gold set is nine triples**, two of them (`center_of`, `independent_from`) outside
  the vocabulary on purpose, and precision against three triples a passage is not a model
  score (the model states true things the gold omits). Write twenty invented passages with
  everything they state written down (`tests/known-fixtures.txt` names), so the gate
  measures precision too; `ingest.INVERSES` covers the flipped ones.
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
- [ ] **Two ladybug reports to file upstream, with Adam's go-ahead.** 0.18.x: a single
  `DETACH DELETE` in a ~10k-node store blanks other nodes' string columns (reproduction:
  `tests/test_graph_store_scale.py`). 0.20.2: the cached-physical-plan fast path
  re-executes a parameterized MERGE against a table rewritten since and segfaults, and
  the text index returns a node once per version written (reproduction: the store's
  `_written` docstring; two lines). ml-stack is on 0.20.2 with the per-write guard and
  the pin `>=0.19,<0.21`; the probes gate any bump.

## Flash-Next, two builds: llama.cpp (unsloth GGUF Q4_K_XL, with and without the draft head) against Ollama (MLX, nvfp4)

Adam, 2026-09-03: which is better, faster, or both; a LinkedIn graphic; "if ollama can't
keep up due to lack of drafting head, points in llama.cpp's favor"; "make sure we're
measuring actual max mem usage during the test". The machine has 128 GB; the GGUF serves at
~90 GB and the Ollama model is 104 GB on disk (`qwen3.8-flash-next:125b-mlx`, 1658
safetensors tensors, `file_type nvfp4`, default context 262144), so the halves run one at a
time with the page's server down for the Ollama half.

- [ ] **Run the three configurations and draw them.** Everything is landed and tested on
  fakes; nothing has touched a real server yet, so `--smoke` each line first. Labels carry
  the profile's suffix as `served()` always did; `served_by` is recorded on every run.
  ```
  # 1. llama.cpp + draft head, against the page's server on 8080 (its profile shape)
  ml-stack-bench sweep --on flash=http://127.0.0.1:8080 --plain-only --short
  ml-stack-bench speed --on flash=http://127.0.0.1:8080
  # 2. llama.cpp without the head, served by the bench in the measured shape minus -md
  ml-stack-bench sweep --serve Qwen3.8-Flash-Next-UD-Q4_K_XL --serve-label flash --no-draft --plain-only --short
  ml-stack-bench speed --serve Qwen3.8-Flash-Next-UD-Q4_K_XL --serve-label flash --no-draft
  # 3. Ollama, after `ml-stack-serve down` on 8080 (104 GB does not fit beside 90)
  ml-stack-bench sweep --on flash-ollama=ollama://127.0.0.1:11434/qwen3.8-flash-next:125b-mlx --plain-only --short --context 32768
  ml-stack-bench speed --on flash-ollama=ollama://127.0.0.1:11434/qwen3.8-flash-next:125b-mlx --context 32768
  # the standard sets, per configuration (~40 min each at --limit 200, thinking off)
  ml-stack-bench standard --url http://127.0.0.1:8080/v1 --model flash --label flash-plain --no-think --limit 200 --out ~/.ml-stack/bench/standard/flash-plain.json
  # then
  ml-stack-bench show --speed
  ml-stack-bench compare flash-plain flash-nodraft-plain-kv-q8_0 flash-ollama-plain --standard ~/.ml-stack/bench/standard/*.json --export ~/flash-comparison.json --title "Flash-Next three ways"
  ml-stack-bench animate ~/flash-comparison.json --out ~/flash-comparison.mp4 --png ~/flash-comparison.png
  ```
  Memory is sampled over the serving process tree every second (Ollama: the listener's
  children hold the weights) and kept as `resident_peak`. `speed --serve` defaults
  `--parallel` to the most streams asked (4) and the per-seat context to the largest prompt
  plus the generation, so set `--context` for the 4-stream cells. Estimate before each:
  ~45 min the hundred-question graph bench, ~10 min speed, ~40 min the standard sets.
- [ ] **`ttft_s` is the server's prompt clock**, never a streamed first token: `gather_stream`
  drops `timings` and the Ollama client raises on `on_delta`. A streamed measurement needs
  the client to keep `timings` on a streamed reply (`client/chat.py`); the speed table says
  `ttft_from: prompt_ms` until then.
- [ ] **`ml-stack-do`'s `bench_standard` example does not match `standard`'s parser**
  (`--url`/`--model` are required); routing works, the example in `do.py` does not.
- [ ] **One measured call on Ollama first.** Whether `prompt_eval_count` includes a cached
  prefix is not in its docs; whether `think: false` holds for this model; what the runner's
  process is called on 0.33.3 (found from source, not seen). Then the Ollama half.

## Measure what landed tonight (each needs the GPU; sample first)

- [ ] **The stable prefix: reading fell, calls rose; measure again on a quiet machine.**
  Nine sampled questions on the page's Flash-Next against `Qwen3.8-Flash--all-plain-kv-q8_0-rb0`
  (kept as `flashprefix-plain`): the last call of a question reads ~50 tokens with the whole
  prefix cached where it read 3193 with none; prefix hits 100% from ~80%; uncached read
  2.1k a question from 5.2k; F1 81% against 85% inside the ±19 band. But calls went 6.2 to
  7.7 a question -- 22 `show` calls over nine questions where one is the design -- and
  written tokens 5.1k to 7.8k, so the wall clock (27.7 against 25.6 s/q) did not move; three
  agents' suites were running, so the wall clock is unreliable either way (Adam). Next:
  `show --trace flashprefix-plain`, find what invites the second `show` (the show nudge now
  offers every tool), then a quiet `--sample 20` of both.
- [ ] **Constrained ids on E2B and E4B.** `sweep --serve <gemma> --profile --constrain-ids
  --sample 20` against the kept plain runs: precision up, recall held. Then `Profile.WAYS`
  needs `constrain_ids` so a profile can record it, and `graph.cache`'s fingerprint should
  include it (a cached answer is returned regardless of the flag today).
- [ ] **Thinking off on the gemma family** (`--reasoning-budget 0` on a sampled sweep each).
- [ ] **`ml-stack-do` against a served model.** `ml-stack-do "benchmark qwen3.8-flash-next
  with llama.cpp (both with draft head and no draft head) and with ollama, make some
  animations" --url http://127.0.0.1:8080`: it must look both backends up, ask to confirm
  the files, ask the bench kind and the animation, plan, then act. Tested on a scripted
  model only. `bench_standard/speed/compare/animate` tools follow the CLI and error until
  the subcommands land.
- [ ] **First real `ml-stack-claude` and `ml-stack-agent`.** Built and tested against fakes
  only. `ml-stack-claude <flash-next> -- --print "say hello"`; one `ml-stack-agent "read
  README.md and say what this is" --model <flash-next> --allow Read`; watch the served
  alias, the stream-idle watchdog (`CLAUDE_STREAM_IDLE_TIMEOUT_MS`), `Usage` against
  `/metrics`.

## Found tonight, not fixed

- [ ] **No full-suite run on the final `main`.** The last branches landed on targeted suites
  as `main` kept moving; run `python packaging/build.py && python3 -m pytest tests -q -n 4
  > FILE; echo $?` once and watch `tests/test_harness.py` (below).

- [ ] **`tests/test_harness.py` calls `asyncio.run` bare** (`harness.py:117`) and fails under
  `-n 4` ordering when a neighbour leaves a loop running; three agents hit it. Wrap it the
  way `conftest.on_a_fresh_loop`/`test_mcp.py` do.
- [ ] **`tests/test_ingest.py::test_a_book_is_read_section_by_section_into_a_store`** asserts
  `'1.2' not in` output that contained `unit 2 in 1.2s` under load: the timing string
  collides with the section number; assert on the structured line instead.
- [ ] **`merge_state` still drops a dead-owner record the next time any process saves**, so a
  second orphan's record can vanish when a lease adopts or stops a first one (`down
  --orphans` reads everything up front, so its sweep is unaffected). Changing it breaks
  `tests/test_serve.py::TestMergeState`, which wants rewriting with it.
- [ ] **`_unjson` returns `{}` silently for invalid JSON** in a column (a valid non-object
  now raises); decide whether invalid should raise too.
- [ ] **`simulate`'s `decision` closer never names the decider**; `world check` counts a
  speaker in the closing thread as named there, which is what lets a made world pass.
  A closer that names `{first}` would let that rule go.
- [ ] **`ScriptedModel`'s docstring** in `testing/fakes.py` still says "the last turn taking
  the searching tools away"; the behaviour is fine, the sentence is stale.
- [ ] **A `stats` document written before a tidy pass is what the export reads back** (the
  Slack pipeline printed 475 edges after a pass that left 598). `GraphStore.write` keeps the
  caller's `stats` verbatim and the export at `store.py:~565` returns it; recount on export,
  or have `graph.tidy` rewrite `stats` after it drops. `GraphStore` has no cheap count query.

## Library

- [ ] **`ml-stack-models layout MODEL`**: the attention layout off the GGUF header in one
  paragraph -- which layers hold a cache, which are recurrent, sliding (window, pattern,
  `key_length_swa`), shared, plus compress ratios, indexers, experts, and any lookup-table
  tensor (`fit --tensors` has the tensor side; `preflight._recurrent_layers` /
  `_sliding_layers` the layer side). It is how a `docs/architectures/` note starts.
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
