# Models policy

Read the sections relevant to your task. [Working contract](../../CLAUDE.md) applies to every task.

## One thing on the GPU at a time

Never put two pieces of work on the GPU at once -- not a question beside a reading, not two
benchmark rows, not a smoke test while a long run is going. Serve one slot and let the second
request wait.

Two at once is more than twice as slow, and it takes the meaning out of every number either one
produces: a row measured under load cannot be compared with a row measured alone, and neither can
be trusted afterwards. Measured 2026-09-09, one machine, Qwen3.8-Flash-Next: a one-line reply
asked on a second slot while an extraction ran took 81s, against a second or two alone, and the
extraction was slowed too. So: `--parallel 1` unless something genuinely needs concurrent
conversations, and a program that reads and answers over the same model does both through the
same server, one after the other. `ml-stack-serve status` says how many slots a server has; check
it before starting a run that will take hours.

## Driving a model on this machine

Never point `ml-stack-claude`, `ml-stack-agent` or `ml-stack-chat` at a checkout you are editing.
An agent with file access edits the files it finds, and a small model will happily rewrite
`CLAUDE.md` because it was asked to say hello. Drive them in a scratch directory.

Never `git add -A` when anything else may be writing to the tree — another agent, a running
ingest, a model you just drove. Add the files you changed, by name. (2026-09-04: a 0.8B model
driven in the primary checkout deleted two paragraphs of this file and changed a heading;
`git add -A` swept it into an unrelated commit.)

**Never start a model server by hand.** `ml-stack-serve up MODEL` is a lease from the broker (it
admits against memory, queues, picks the port, applies the measured profile, and is held until
`ml-stack-serve down MODEL`); `ml-stack-claude`, `ml-stack-agent` and `agent start` take a lease
too. This paragraph is the explanation, not the enforcement: a `Lease` cannot be made outside the
broker's grant (`ml_stack.serve.grant`), `tests/test_serve_no_bypass.py` and the hard
`server-starts` budget gate fail on a new spawn site, and the bash guard refuses `llama-server`
by hand and `up` flags that would skip the lease. The owner's default 27B quant is
`Qwen3.8-27B-UD-Q4_K_XL.gguf` (16.7 GB), not Q4_K_M.

## Which models to test with, and the defaults the owner wants

Owner's standing choices (2026-10-03); do not ask again.

- **Live tests and demos use the newest Qwen family** (Qwen3.8 at the time of writing; look at
  what `ml-stack-models list` and the Hugging Face cache actually hold and name the exact id in
  the report) For large-model tests use the dense Qwen3.8-27B (`Qwen3.8-27B-UD-Q4_K_XL.gguf`); Flash-Next holds too much memory on this machine. For small and day-to-day tests prefer a smaller Qwen3.8 model. Do not use gpt-oss: it is
  too old. Old results stay as history, not as a matrix row.
- **MTP (multi-token prediction) draft heads are on by default** whenever the served model has a
  matching trusted head and the managed llama.cpp build supports it; there is a documented
  opt-out, and paths that cannot use it (one-token decisions, logprob scoring if incompatible)
  turn it off themselves. See `docs/serving.md`.
- **llama.cpp tracks the bleeding edge** through the managed head builds
  (`docs/llama-cpp-tracking.md`), not a pinned stable release.
- **Decision models should be as easy to call locally as hosted ones.** The target is the shape
  people know from Jev-style decision models: give a state and closed questions, get typed
  answers with probabilities, no text generation. The owner's decider is
  `StrandsAgents/strands-decider-2B-hobson-v19` (Qwen3.5-2B base). Treat a gap in
  `docs/decision-models.md` against that bar as a defect worth an issue; the gap analysis lives
  in the issue tracker, not in a private note.

- **Default decider: Strands 2B** (`StrandsAgents/strands-decider-2B-hobson-v19`), chosen for its small size; revisit only against measured results (JevBench and our own sets). **Fine-tuned deciders are never committed**: datasets and weights stay in caches, the repo keeps recipes and metrics only.

- **Persistent, relational state defaults to the graph** (`ml_stack.graph.GraphStore`, `docs/graph.md`): agent memory, knowledge about models, builds and tasks, ingested documents. Facts link to the things they are about, so recall can follow relations and use the hybrid search. A flat JSON file needs a stated reason (pure configuration, a tiny single-purpose cache); integrity sealing and tamper checks sit on top of the graph, not instead of it.

