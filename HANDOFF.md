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

## The store (each needs the GPU; Adam's call)

- [ ] **The store holds APBiology and Biology2e chapter 2, sound; nine textbook PDFs are
  unread.** `~/.ml-stack/sources.ladybug` on ladybug 0.20.2: ~9,500 concepts with
  definitions, page provenance and the run that read each, after the judged pass (362
  merges, 1,489 inverse pairs, 622 conflicts judged with 282 edges dropped, 186 definitions,
  113 suspects), `ml-stack-store check` clean. The reads files beside the store are the
  truth (`ml-stack-ingest fold` rebuilds). Whether the other nine -- about four days of GPU
  at 86 s a unit, one slot -- are worth it is Adam's call; the command is `ml-stack-ingest
  ~/Documents/Textbooks/<pdf> --out ~/.ml-stack/sources.ladybug --model <flash-next>
  --images --resume --serve-port 8080`, one source at a time, and it tidies itself at the
  source's end. Two answers before that: what a question over the store scores
  (`ml-stack-ingest ask --out ... --gold FILE`, no gold questions written yet), and what
  `ml-stack-ingest sources` says once a second full source is in.
- [ ] **Watch for units that still run to the ceiling.** The document schema caps every
  list (`maxItems`), so the grammar closes the array; a unit that still fails is read once
  more, then given up on; `status` counts those, its whole reply is `raw` in the reads
  file beside the store, and `ml-stack-ingest retry --out STORE` frees them after a fix.
  If one still circles under the cap, try DRY sampling for extraction and measure it on
  the gold gate.
- [ ] **Score answering over the store.** Write twenty invented-free but real-source
  questions with expected concept ids (`ml-stack-ingest ask --gold FILE`, the bench's
  scorer) and run them through Flash-Next in its profile; until that number exists, "a
  usable bio graph" means queryable and sound, not scored. The judge's conflict verdicts
  are worth reading first: it kept both edges in cases where its own reason said one was
  a misreading ("Larynx part_of trachea" beside "Larynx precedes trachea") -- the
  instructions now say keep both only when the passages support both (2026-09-03); the
  store's verdicts predate that and are worth `ml-stack-store tidy --rejudge
  ~/.ml-stack/sources.ladybug`, ~an hour.
- [ ] **Fill the store's "between sources" section.** `ml-stack-ingest sources` reads
  `tidy:merges`, written by every fold and tidy from 2026-09-04 on; the store's 210 shared
  concepts were joined before it existed, so the section is empty. `ml-stack-ingest fold
  --out ~/.ml-stack/sources.ladybug` re-folds every source from its reads (no model,
  minutes) and writes the log; it rewrites the store, so take a copy beside it first.

## Measurements queued (each needs the GPU; sample first)

Adam, 2026-09-04: "we're never going to have that many users, so flash-next is the way to
go (with shared MTP) always." So ranking a second model is not worth GPU: `gpt-oss-20b`
(a profile, no fit record), `Qwen3.8-27B` (a fit record, no profile) and Flash-Next
`IQ4_XS` (a fit record, no profile) stay half-measured on purpose, and `ml-stack-fleet
plan` names them as unplaceable rather than guessing.

- [ ] **`single` on E4B at a hundred questions** (`sweep --serve gemma-4-E4B-it-qat-UD-Q4_K_XL.gguf
  --profile --plain-only --also single --yes`, ~10 min): the one asking-way change that
  moved a small model, unconfirmed at ten. If it holds, `report --profile` sets it.
- [ ] **Smoothed embeddings on the bench.** `graph.smooth` spreads every vector over the
  graph (`ml-stack-store embed PATH --smooth N`, or `remember(..., smooth_hops=N)`); nothing
  says whether it makes the answers better. The measure: `ml-stack-bench ready --embed-url
  ... --embed-model ...` to build the vectors, `embed --smooth 1` and `--smooth 2` over
  copies of that store, then a sampled `sweep --plain-only --sample 20` on each against the
  unsmoothed one. What to watch is recall on questions about entries nobody wrote about --
  the whole point -- against precision on the rest, since a vector pulled towards its
  neighbours is a vector less about itself.

- [ ] **Constrained ids on E2B and E4B.** `sweep --serve <gemma> --profile --constrain-ids
  --sample 20` against the kept plain runs: precision up, recall held. A profile records
  `constrain_ids` and the answer cache keys on it, so `report --profile` can set it.
- [ ] **Thinking off on the gemma family** (`--reasoning-budget 0` on a sampled sweep each).
- [ ] **Four unified slots decoded slower at one stream; say whether the head is why.**
  Measured 2026-09-05 on E2B (`ml-stack-bench speed --serve gemma-4-E2B... --streams 1
  --prompts 4096 --context 65536 --parallel N --serve-kv-unified|--no-serve-kv-unified`, two
  runs each): one slot 125/153 tok/s, one slot unified 143/152, four slots 152/156, four
  slots unified 121/122 -- and the draft head's acceptance on the four unified slots was 44%
  where every other shape reported 70%. Prefill ~2,600 tok/s in every shape. So one seat
  by default costs nothing at one stream, and the page's 2026-09-04 shape (four unified
  slots) was ~20% slower to decode, but whether the cache or the head under it is the
  reason is one more run: the same four shapes with `--no-draft`.
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
- [ ] **Watch ggml-org/llama.cpp#27836** (a draft, last touched 2026-09-04, checked
  2026-09-05). When the qwen4exp MTP graph merges, `ml-stack-serve build` and
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
  ml-stack-bench compare flash-plain flash-nodraft-plain flash-ollama-plain --standard ~/.ml-stack/bench/standard/*.json --export ~/flash-comparison.json --title "Flash-Next three ways"
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

- [ ] **A number for the local harness beside the page's.** `ml-stack-claude <flash-next>
  -- --print "say hello"` and `ml-stack-agent "read README.md and say in two sentences what
  this is" --model <flash-next> --allow Read` both ran for real on 2026-09-05 (the agent: 5
  turns in 382 s, 79k tokens read with 185k from the cache, 1.2k written, a right answer).
  Untried: the stream-idle watchdog (five minutes of silence aborts --
  `CLAUDE_STREAM_IDLE_TIMEOUT_MS`), and what `Usage` reports against the server's own
  `/metrics`. Then a small task set measured the bench's way.

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

## From beehavior

`~/Documents/repos/beehavior` is the other repository that grew these shapes, and its own
`ML_STACK_MIGRATION.md` lists what it should now delete rather than keep. Two pieces came
here on 2026-09-05 (`ml-stack-suite`, `ml-stack-serve limits|reclaim`); three are still
worth taking, in this order:

- [ ] **A runtime redactor** (`tooling/compliance/{text_sanitizer,llm_sanitization}.py`,
  ~380 lines): a Presidio sanitizer built once per configuration and applied to prompts,
  model output and the turns a page stores. `ml_stack.redact` audits a repository at commit
  time and has nothing for the ask path, which is where a graph page sends real text.
- [ ] **The copyright carve-out gate** (`tooling/graph/ingestion/copyright_carveout.py`,
  183 lines): refuses a verbatim provenance snippet taken from a source that embeds
  third-party material under a permission granted to somebody else. Every ingested node
  keeps a snippet, so this is the same problem here.
- [ ] **A terminal command centre** (`tooling/daemon/tui.py`, 477 lines, Textual): the
  facts the fleet app's cluster view already shows, in a terminal. Worth it only if a
  headless machine wants one.

## The tensor toolkit

- [ ] **Six of the toolkit's functions have no caller here.** `graph.tensors` (the mapping
  graph as arrays) is called by `graph.smooth`, and `topology.knn_edges` by
  `graph.places.geocode`. `batch_graphs`, `resolvent_sweep`, `decompose_to_dags`,
  `mst_edges`, `morton_codes` and `spatial_window_edges` are tested library API that nothing
  in this repository calls. Each has an obvious home when the work arrives -- `batch_graphs`
  when a graph model is trained on many graphs at once, `resolvent_sweep` for influence down
  a DAG (a `reports_to` tree, a `part_of` hierarchy), `mst_edges`/`morton_codes`/
  `spatial_window_edges` for a cheaper `--near` on a graph too big for the full distance
  matrix `knn_edges` builds. The last one is the near-term task: `geocode --near K` is
  O(n^2) in placed entries, which is nothing at 300 and a problem at 30,000.

## Speech

- [ ] **The beacon does not say which machines can hear.** `refresh(b)` inside
  `fleet/daemon.py::serve_forever` fills `b.device` with `serving` and `models`;
  `"speech": [names that probe ok]` beside them would let `ml-stack-fleet status` and the
  cluster view show which peer to send a recording to, and would make
  `Peer.find_one(require="speech")` work. `ml_stack.speech.service.providers()` already
  returns exactly that list. Left out because another branch was editing `serve_forever`.
- [ ] **Nothing streams.** `StreamingASR` in `speech/protocols.py` is a protocol no
  provider implements, and `POST /speech/transcribe` takes one whole file. A push-to-talk
  button wants partial text while the person is still speaking: a provider that yields
  `Transcript`s off a chunk iterator, and a route that streams them the way `/ask/stream`
  does.
- [ ] **No microphone anywhere in the interface.** The chat screen has no record button, so
  the only way to reach any of this is the command line. The route and `Peer.transcribe`
  are the far end; what is missing is a control that records in the browser and posts the
  audio.

## The interface

- [ ] **The fleet app as components.** `graph/page.py` assembles a page out of custom
  elements (`ml_stack.ui.assemble`, one file a component under `graph/web/components/`);
  `fleet/web/app.js` (1,500 lines of screens as functions) and `fleet/web/fit.html` (a
  second standalone page) are the same shape waiting for the same split: one element per
  screen, the `/ui/*` routes in `fleet/ui.py` as mixins, `index.html` as the wiring.
- [ ] **Nobody has drawn 3,000 nodes in the page.** The page ships the whole graph as one
  JSON blob and lays it out in the browser; the biggest graph it has held is a few hundred
  nodes. `ml-stack-world make --size 5000` gives one to try, and `most_messages` already
  trims the quotes; nothing trims the drawing.

## Layers

- [ ] **Seventeen imports still cross the layers `tests/test_layers.py` sets out.** The
  layers are core, model, graph, machine, tools; a package may import downwards, and
  sideways only when the other package does not import it back. `KNOWN` in that file lists
  every edge that breaks it, and the test fails both on a new violation and on a `KNOWN`
  entry that is no longer one, so the set only shrinks. Four are two-way cycles inside one
  layer: `serve` <-> `fleet` (3 files one way, 7 the other), `serve` <-> `setup` (2 and 1),
  `fleet` <-> `setup` (1 and 1), and `sources` <-> `world` (5 and 1, going when
  `world.Message` moves down). Nine reach up a layer: `doctor`, `fleet`, `ingest` and
  `serve` into `bench` (1, 2, 2 and 1 files); `gguf`, `hub`, `graph` and `ingest` into
  `serve` (1, 1, 3 and 1); and `graph.data` into `train.backend` for a device handle, where
  the ops that sit under both belong below `graph` rather than in the tools layer. The
  `graph` -> `serve` three are function-local reaches for `Asking`, `Run`, `Shape`, `seat`
  and `held` in `graph/ask.py`, `graph/requests.py` and `graph/serve.py`; most go when
  `Asking` moves down out of `serve/shape.py`.

## Verifying

```bash
python3 -m pytest tests -q -n 4 > /tmp/out.txt; echo $?   # never pipe into tail; -n 4 while a bench runs
ml-stack-setup                                             # the machine
ml-stack-bench status                                      # measuring, serving, what the job kept
ml-stack-serve profile                                     # every model's measured shape
```
