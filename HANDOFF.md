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

Settled: Flash-Next answers (80% F1 at 27 s/q, 100 questions) and extracts (96% node / 76%
relation F1); draft length 4 for both; one slot for extraction; `single` +8 pts on E4B at
ten questions, unconfirmed.

## The shelf (each needs the GPU; Adam's call)

- [ ] **The shelf holds APBiology and Biology2e chapter 2, sound; nine books are unread.**
  `~/.ml-stack/shelf.ladybug` on ladybug 0.20.2: ~9,500 concepts with definitions, page
  provenance and the run that read each, after the judged pass (362 merges, 1,489 inverse
  pairs, 622 conflicts judged with 282 edges dropped, 186 definitions, 113 suspects),
  `ml-stack-store check` clean. Reads beside the store are the truth (`ml-stack-ingest fold` rebuilds). Whether the other nine books -- about four days of GPU at 86
  s a unit, one slot -- are worth it is Adam's call; the command is `ml-stack-ingest
  ~/Documents/Textbooks/<book>.pdf --out ~/.ml-stack/shelf.ladybug --model <flash-next>
  --images --resume --serve-port 8080`, one book at a time, and it tidies itself at the
  book's end. Two answers before that: what a question over the shelf scores
  (`ml-stack-ingest ask --out ... --gold FILE`, no gold questions written yet), and what
  `ml-stack-ingest sources` says once a second full book is in.
- [ ] **Watch for units that still run to the ceiling.** The document schema caps every
  list (`maxItems`), so the grammar closes the array; a unit that still fails is read once
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
- [ ] **Fill the shelf's "between books" log.** `ml-stack-ingest sources` reads
  `tidy:merges`, written by every fold and tidy from 2026-09-04 on; the shelf's 210 shared
  concepts were joined before it existed, so the section is empty. `ml-stack-ingest fold
  --out ~/.ml-stack/shelf.ladybug` re-folds every book from its reads (no model, minutes)
  and writes the log; it rewrites the store, so take a copy beside it first.
- [ ] **Run the shipped extraction gate on a model** (`ml-stack-ingest --gold
  tests/fixtures/extraction-gold.json --model <flash-next> --fail-under 0.7`, ~10 min):
  twenty invented passages with every triple written down, so the number is precision as
  well as recall. Then E4B and E2B the same way, and the numbers into
  `docs/model-ranking.md`.

## Measurements queued (each needs the GPU; sample first)

Adam, 2026-09-04: "we're never going to have that many users, so flash-next is the way to
go (with shared MTP) always." So ranking a second model is not worth GPU: `gpt-oss-20b`
(a profile, no fit record), `Qwen3.8-27B` (a fit record, no profile) and Flash-Next
`IQ4_XS` (a fit record, no profile) stay half-measured on purpose, and `ml-stack-fleet
plan` names them as unplaceable rather than guessing.

- [ ] **`single` on E4B at a hundred questions** (`sweep --serve gemma-4-E4B-it-qat-UD-Q4_K_XL.gguf
  --profile --plain-only --also single --yes`, ~10 min): the one asking-way change that
  moved a small model, unconfirmed at ten. If it holds, `report --profile` sets it.
- [ ] **Constrained ids on E2B and E4B.** `sweep --serve <gemma> --profile --constrain-ids
  --sample 20` against the kept plain runs: precision up, recall held. A profile records
  `constrain_ids` and the answer cache keys on it, so `report --profile` can set it.
- [ ] **Thinking off on the gemma family** (`--reasoning-budget 0` on a sampled sweep each).
- [ ] **Is a unified cache slower than one slot?** (Adam, 2026-09-04: "still serve one by
  default, but we need to test down the line whether unified cache is slower".) Every
  command now serves one seat, so this decides whether more seats cost speed as well as
  room. A unified cache holds every sequence in one pool and masks out the tokens
  belonging to the others, so attention may pay over the whole pool even when one
  conversation is live -- if it does, four slots is slower than one at a single stream,
  not merely wastier. Measure on a small model, since the shape is what is under test and
  not the model: serve `gemma-4-E2B-it-qat-UD-Q4_K_XL.gguf` on a scratch port four ways --
  one slot and four slots, each with `--kv-unified` and without -- at the same total
  cache, and run `ml-stack-bench speed` at one stream against each. Generation tokens a
  second is the number; prompt reading is the second one to watch. Minutes, and it wants a
  quiet machine. The page ran four unified slots all through 2026-09-04, so a difference
  here also says what that cost.
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
- [ ] **The fine-tuned tool caller.** `ml-stack-train-tools from-bench` over the traced runs
  (traces are on by default at ≤20 questions; the hundred-question runs before that carry
  none -- rerun Flash-Next's hundred with `--trace` for ~5,000 turns), then
  `ml-stack-train-run --recipe tool-calls --size e4b --lora --export-gguf --yes` (~18 h
  here; Adam's go-ahead), then the measure in `docs/research/tool-caller-finetune.md`.
- [ ] **`docs/architectures/qwen4exp.md` says 48K bytes a token on the 12 attention layers
  at f16; the header says 12 x 2 KV heads x 256 x (K+V) x 2 bytes = 24K.** The difference
  may be the indexer key cache (one head of `indexer.key_length` 128 per attention layer)
  or the MTP head; `ml-stack-serve fit` at two contexts on Flash-Next says which, and the
  note and `preflight._kv_estimate_bytes` follow the measurement.
- [ ] **Watch ggml-org/llama.cpp#27836** (open, last touched 2026-09-02, checked
  2026-09-04). When the qwen4exp MTP graph merges, `ml-stack-serve build` and
  `ml-stack-bench drafts` Flash-Next on mainline: the PR reports 86–89% acceptance on an
  M3 Max against the fork's 73–79%. Then the profile's build field can go.

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
  plus the generation, so set `--context` for the 4-stream cells. `ttft_s` is a streamed
  first token on llama.cpp (`ttft_from: stream`) and the server's prompt clock on Ollama
  (`prompt_ms`, marked `*`). Estimate before each: ~45 min the hundred-question graph
  bench, ~10 min speed, ~40 min the standard sets.
- [ ] **One measured call on Ollama first.** Whether `prompt_eval_count` includes a cached
  prefix is not in its docs; whether `think: false` holds for this model; what the runner's
  process is called on 0.33.3 (found from source, not seen). Then the Ollama half.

## Driving what is built (each needs the served model; minutes)

- [ ] **`ml-stack-do` against a served model, the acceptance prompt.** Driven once bare
  (2026-09-03): with no `--model` it chose Flash-Next from the profiles, used the page's
  server as it stood, called `serve_status` and answered. Untried: `ml-stack-do "benchmark
  qwen3.8-flash-next with llama.cpp (both with draft head and no draft head) and with
  ollama, make some animations"`: it must look both backends up, ask to confirm the files,
  ask the bench kind and the animation, plan, then act. Tested on a scripted model only.
- [ ] **First real `ml-stack-claude` and `ml-stack-agent`.** Built and tested against fakes
  only. `ml-stack-claude <flash-next> -- --print "say hello"`; one `ml-stack-agent "read
  README.md and say what this is" --model <flash-next> --allow Read`; watch the served
  alias the model variables carry, the stream-idle watchdog (five minutes of silence
  aborts -- `CLAUDE_STREAM_IDLE_TIMEOUT_MS`), and what `Usage` reports against the
  server's own `/metrics`; then measure a small task set the bench's way so the local
  harness has a number beside the page's.
- [ ] **The Slack page's answer cache keys on the asking ways only once the app passes
  them.** `cache.asked`/`fingerprint` take `ways=` (at least `{"constrain_ids": ...}`);
  the one caller is `~/ai_ceo/slack_graph/ask.py:424`, which does not pass it yet, so a
  cached answer there is returned regardless of the flag. One line in the app.
- [ ] **The Slack page's own `Config` keeps two seats** (a run plus a reader) while every
  command here serves one seat by default. Adam's to keep or drop, in `~/ai_ceo`.

## Store integrity

- [ ] **Two ladybug faults are worked around here and stay here.** Adam, 2026-09-04: no
  upstreaming to public repositories. 0.18.x: a single `DETACH DELETE` in a ~10k-node
  store blanks other nodes' string columns (reproduction: `tests/test_graph_store_scale.py`).
  0.20.2: the cached-physical-plan fast path re-executes a parameterized MERGE against a
  table rewritten since and segfaults, and the text index returns a node once per version
  written (reproduction: the store's `_written` docstring; two lines). ml-stack is on
  0.20.2 with the per-write guard and the pin `>=0.19,<0.21`; the probes gate any bump,
  which is the whole of what is left to do about them.

## Measuring across the fleet

- [ ] **Run it for real across two machines.** Everything is tested against fakes and
  loopback; nothing has crossed a real network or a real Windows box. One visit:
  `irm https://raw.githubusercontent.com/adammikulis/ml-stack/main/packaging/install.ps1 |
  iex` (the app) or the `--headless` mode, then from here `ml-stack-fleet status`, a
  `ml-stack-fleet plan --users 3 --context 16384 --apply`, and a `sweep --fleet --serve
  gemma-4-E2B-it-qat-UD-Q4_K_XL.gguf --sample 6`. Expect bugs; the daemon's log and
  `ml-stack-doctor` are the first two places to look. After that the Windows box follows
  releases or main on its own (`fleet status` shows COMMIT/UPDATES).
- [ ] **A router across the fleet.** `ml-stack-fleet plan --apply` serves the placement;
  nothing yet sends a new session to a free seat on the best model. The daemon's `/infer`
  proxies by model name on one machine; the router picks the machine.

## Verifying

```bash
python3 -m pytest tests -q -n 4 > /tmp/out.txt; echo $?   # never pipe into tail; -n 4 while a bench runs
ml-stack-setup                                             # the machine
ml-stack-bench status                                      # measuring, serving, what the job kept
ml-stack-serve profile                                     # every model's measured shape
```
