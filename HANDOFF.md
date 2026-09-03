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
  one slot, one unit at a time -- two workers measured slower in aggregate, 140 s a unit
  against 86, and Adam: "we should never be splitting the GPU like that"): ten OpenStax PDFs from `~/Documents/Textbooks` plus
  `~/Downloads/Psychology2e_WEB.pdf` into `~/.ml-stack/shelf.ladybug`. Chapter 2 of
  Biology2e alone was 13 units in 19 min; the shelf is days of GPU, not
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
- [ ] **Watch for units that still run to the ceiling.** Two causes found and fixed on the
  first night: chapter-end question banks (`pdf.units` leaves them out) and a greedy
  decode circling a long relations array (63 clean concepts, then 378 relations of which
  282 were distinct, until n_predict) -- the document schema now caps every list
  (`maxItems`), so the grammar closes the array. A unit that still fails is read once
  more, then given up on; `status` counts those, its whole reply is `raw` in the reads
  file beside the store, and `ml-stack-ingest retry --out STORE` frees them after a fix.
  If one still circles under the cap, try DRY sampling for extraction and measure it on
  the gold gate.
- [ ] **Judge APBiology before the other nine books.** The run reads it alone and stops at
  its end. The hygiene pass is automated now -- `ml-stack-ingest wait && ml-stack-ingest
  tidy --out ~/.ml-stack/shelf.ladybug --model Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf`
  merges names by case/spacing/plural, folds inverse pairs, flags clause-shaped labels, and
  has the model judge the names a spelling apart (from knowledge, then from the passages
  re-read out of the book), applying what it decides and keeping every verdict with its
  reason in the store's `tidy:decisions`; a run started with the new code does this itself
  at each book's end. Run 2026-09-03 13:18-13:31 over APBiology + Biology2e ch. 2: 377
  pairs judged -- 348 different (RNA polymerase vs RNA Polymerase II, DNA vs RNA
  polymerase III), 29 the same (bisphosphate/biphosphate, Vertebrata/vertebrates,
  mucous/Mucus, T cell/T-cells) -- reading the passages once; the store at 9,989 nodes
  and 26,794 edges. The verb-conflict, definition and suspect steps recorded nothing on
  that run; a second run is checking whether they judge at all. Then `show --book apbiology
  --sample 20` and a few questions over it (`graph.ask`) to see whether a book-scale graph
  is worth four more days of GPU. Nothing in `docs/` yet says what reading a shelf taught
  (question banks skipped, the relations-array loop capped, one slot only, the server-gone
  stop, the tightened dedupe); it belongs beside `docs/architectures/`.
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
- [ ] **Reconcile on the way in, everywhere.** Adam: "the same dedupe mechanism should be
  used whenever the model is reading a new thing or learning something new and saving to
  an existing graph. it will for sure re-encounter the same concepts as it learns more."
  `graph.tidy.absorb(store, graph, judge=)` maps an incoming graph's nodes onto the
  store's by same name and plural, and puts close spellings to the judge with the
  incoming passage in hand; the ingest's fold calls it before every upsert (the run's
  model as judge, the unit text as source). Still to wire: ai_ceo's pipeline merge
  (`merge.py`, with `data/aliases.json` as `written`), `graph.thread` when an answer's
  entities are kept, and the world simulator's emit.

Store integrity
- [ ] **`tidy` and `fold` should run `check` at their end** and refuse to report success
  over a store that does not read back by id. Found the hard way: on ladybug 0.18.2 a
  single `DETACH DELETE` on a ~10k-node store blanked other nodes' id/attrs/data (one
  drop: 1,983 edges unfindable by their ends; 300: 3,360; the same in one transaction
  and with the label index removed), so the judged pass wrecked the shelf twice; 0.20.2
  deletes cleanly but doubles a node written twice; 0.19.1 passes both, and the pin moved
  there (7b6fba6). The two damaged stores are kept at `~/.ml-stack/shelf.ladybug.corrupt-*`
  for a report upstream; `tests/test_graph_store_scale.py` carries the two probes (a delete
  at scale leaves strings intact; a write twice updates) so a future bump is measured, not
  trusted. ai_ceo's venv is on 0.19.1 too.
- [ ] **ladybug 0.20.2 segfaults on the store's second `write()`** (Adam asked for the
  upgrade, 2026-09-03 afternoon; measured, not taken). Release notes 0.20.0-0.20.2 are
  about a cached-physical-plan fast path for re-executed parameterized queries and three
  rounds of fixes to it ("SIGSEGV re-executing a parameterized write query string",
  "stale rows on the cached-plan fast path"); it is still not right for our pattern.
  Reproduction, two lines, `PYTHONFAULTHANDLER=1`: `store.write(GRAPH); store.write(GRAPH)`
  from `tests/test_graph_store.py` -- Fatal Python error: Segmentation fault in
  `ladybug/connection.py execute`, from `store.py write()`'s transaction (MERGE with
  parameters, re-executed). Four store tests crash the interpreter and the FTS search
  returns a duplicate row. A minimal MERGE-twice on a bare table does not crash, so the
  trigger is our write sequence (transaction, MERGE node, MERGE edge, put_doc, index); the
  bisect and the upstream issue are the next step (Adam's go-ahead before posting).
  Until fixed upstream the pin stays `>=0.19,<0.20`; the scale probes gate any bump.

## Library

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
