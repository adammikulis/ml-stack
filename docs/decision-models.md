# Decision models

A decision model answers one question: which of these named options? It returns a
probability for every option, so a caller can act on a confident answer and ask a person
about an unsure one. It does not write text. Use one where a program needs a category (route
this message, is this tool call destructive, which of these files matters) and an LLM where it
needs a sentence or a plan.

    from ml_stack.decide import router
    got = router.decide("Which team?", "payouts have failed for 3 days",
                        {"billing": "payments, invoices", "sales": "pricing", "tech": "outages"},
                        abstain_below=0.7)
    got.choice, got.confidence, got.scores, got.abstained, got.latency_ms

`ml-stack-decide ask|train|eval|calibrate|bench|list|make-cases|export-cases|check-cases|fetch`
is the command; `POST /decide` on the daemon (same bearer token as every route, 256 KB body
limit) and the `decide` MCP tool call the same router.

## Backends

| backend | what runs | needs | options |
|---|---|---|---|
| `pointer` | a LoRA on a 2B model with its output head removed, and a ~1M-parameter pointer head, one forward pass | `pip install 'ml-stack[decide-pointer]'`, 4.7 GB of pinned files fetched on request | any number |
| `logprob` | the first-token distribution of any chat model a server reports log-probabilities for, read over the option letters | a running llama-server (or vLLM, OpenAI) | at most 26 |
| `embed` | sentence embeddings, then cosine ranking or a head trained from a few dozen cases | `ml-stack[decide]` and an embedding server | any number (`pairwise` head), a fixed set (`classes` head) |
| `rules` | patterns, words and allowed values; `ScopeDecider` checks paths and hosts | nothing | any number |

`router.decide(..., config=Config(backend="auto"))` takes the first of `pointer`, `logprob`,
`embed`, `rules` that is available and takes the option count. `Layered(rules, learned)`
lets a matching hard rule overrule a learned score.

## What was measured

2026-10-02, one Apple M4 Max with 128 GB. The cases are the built-in guard set at this
commit (412 hand-written tool calls, three questions). The set is split by whole groups
(a call and its reworded requests are one group) into a dev half and a test half, seed 0;
every temperature below is fitted on dev and every figure is scored on the 207 test cases
(96 destructive, 51 grounded, 60 scope). Brier is the sum over options of the squared
error (0 to 2, lower is better); ECE uses ten bins. "cal" is after the fitted temperature.
Latency is the median over the test cases with the model warm, one request at a time.
Raw rows: `ml-stack-decide bench guards --split test --tag QUESTION --backend NAME`; the
temperature fit is `ml-stack-decide calibrate --split dev`. The embed heads are trained on
a 70% slice of dev and calibrated on the other 30%.

| backend and model | question | accuracy | Brier raw / cal | ECE raw / cal | p50 ms |
|---|---|---|---|---|---|
| logprob, Qwen3-VL-8B-Instruct Q4_K_M | destructive | 0.969 | 0.041 / 0.063 | 0.028 / 0.032 | 141 |
| | grounded | 0.882 | 0.228 / 0.207 | 0.112 / 0.057 | 199 |
| | scope | 0.933 | 0.097 / 0.091 | 0.045 / 0.105 | 136 |
| logprob, Qwen3-4B-Instruct-2507 Q4_K_M | destructive | 0.917 | 0.156 / 0.122 | 0.076 / 0.032 | 82 |
| | grounded | 0.784 | 0.427 / 0.297 | 0.213 / 0.114 | 84 |
| | scope | 0.833 | 0.341 / 0.271 | 0.174 / 0.107 | 83 |
| logprob, Qwen3.5-2B Q4_K_M | destructive | 0.698 | 0.352 / 0.336 | 0.184 / 0.161 | 88 |
| | grounded | 0.667 | 0.459 / 0.419 | 0.190 / 0.120 | 82 |
| | scope | 0.867 | 0.366 / 0.196 | 0.285 / 0.136 | 72 |
| logprob, granite-4.0-h-350m Q4_K_M | destructive | 0.333 | 1.199 / 0.666 | 0.606 / 0.043 | 56 |
| | grounded | 0.569 | 0.842 / 0.494 | 0.417 / 0.014 | 55 |
| | scope | 0.617 | 0.484 / 0.493 | 0.102 / 0.105 | 51 |
| pointer, strands-decider-2B-hobson-v19 (MPS, bf16) | destructive | 0.927 | 0.208 / 0.117 | 0.239 / 0.043 | 263 |
| | grounded | 0.824 | 0.302 / 0.272 | 0.136 / 0.059 | 451 |
| | scope | 0.933 | 0.243 / 0.162 | 0.245 / 0.119 | 207 |
| embed, embeddinggemma-300M + classes head | destructive | 0.844 | 0.262 / 0.259 | 0.075 / 0.108 | 9 |
| | grounded | 0.725 | 0.445 / 0.372 | 0.193 / 0.076 | 10 |
| | scope | 0.650 | 0.599 / 0.449 | 0.278 / 0.132 | 9 |
| embed, embeddinggemma-300M + pairwise head | destructive | 0.802 | 0.389 / 0.295 | 0.270 / 0.167 | 9 |
| | grounded | 0.745 | 0.387 / 0.358 | 0.124 / 0.065 | 10 |
| | scope | 0.433 | 0.552 / 0.531 | 0.171 / 0.150 | 10 |
| embed, embeddinggemma-300M, no head | destructive | 0.365 | 0.749 / 0.643 | 0.347 / 0.225 | 10 |
| rules, `ScopeDecider` | scope | 1.000 | 0 / 0 | 0 / 0 | under 1 |

Reading it:

- The first-token readout of the 8B instruct model beats the released pointer checkpoint on
  accuracy and latency on this set; the 4B is 1 to 10 points below it on accuracy and still
  faster. The pointer model needs no letter labels, so it handles more than 26 options, and
  once a temperature is fitted it is calibrated about as well as the 8B.
- The pointer checkpoint's latency here (200 to 450 ms median, 1.4 s at p95) is higher than
  the 153 ms its authors report for an M3, because the guard prompts run 300 to 600 tokens
  and 18 of the 24 layers are Gated DeltaNet, which transformers runs in a PyTorch reference
  loop on MPS. An 86-token prompt takes 117 ms. On CPU in float32 the same prompt takes 9 s.
- Raw confidences are badly calibrated for the small models; a temperature fitted on dev
  fixes ECE for most rows (granite 350M: 0.61 to 0.04) without making them accurate.
- The embed backend answers in about 10 ms. Trained on 144 cases it reaches 0.84 on the
  destructive question; it is the right tool when a decision repeats thousands of times and
  a few dozen labelled examples exist.
- `rules` scores 1.000 on the scope question because the cases were labelled with the same
  path and host definition; it shows the rule covers every case in the set, not that it
  generalises.
- A Qwen3.5-4B Q4_K_M server on this llama.cpp build returned non-finite logits (the
  `logprob` backend reports the garbled first token and stops); it is not in the table.
- Memory was not measured per process. The pointer model held 3.7 GB on the MPS device.

### Fine-tuning

`ml-stack-train-decider guards --init strands --steps 120 --lr 5e-5 --out DIR` continued the
released checkpoint on 248 guard cases (61 for calibration, 103 held-out test cases from
whole groups, all three questions pooled) in 364 s: accuracy 0.903 to 0.951, Brier 0.230 to
0.069, ECE 0.216 to 0.035. The cases are one author's templates, so the test cases resemble
the training cases more than a new project's calls would; this is a pipeline check with
a direction, not a quality claim. A fresh adapter on Qwen3.5-0.8B-Base for 30 steps reached
0.398 (the three questions pooled; a floor), in 87 s. The model card each run writes says
which kind of run it was.

## Train your own

The released model is `StrandsAgents/strands-decider-2B-hobson-v19` from Strands Labs, the
open-source lab of the Strands Agents project at AWS (announcement:
<https://strandsagents.com/blog/introducing-strands-decider/>; code:
github.com/strands-labs/strands-decider). It is a rank-16 LoRA and a pointer head on
Qwen3.5-2B, scored on JevBench, and its training data, scripts and weights are released for
fine-tuning. Its three question kinds are `noul` (yes/no), `choice` and `score`, and it is
calibrated by a temperature per kind. `ml-stack-decide train` is this package's own
re-implementation of that recipe for a few hundred of your own cases; it does not run the
released scripts (the reference recipe is about 11 hours on one RTX 3090 for the full corpus).

    ml-stack-decide train --data tickets.jsonl --name tickets --dry-run
    ml-stack-decide train --data tickets.jsonl --eval held-out.jsonl --name tickets \
        --base qwen3.5-2b-base --baseline strands --steps 300
    ml-stack-decide eval held-out.jsonl --decider tickets --decider strands

**Data.** One JSON object per line (the format `export-cases` writes):

    {"id": "t1", "question": "Which team?", "state": "payouts failing for 3 days",
     "options": {"billing": "payments", "tech": "outages"}, "label": "billing",
     "group": "customer-17", "kind": "choice"}

`question`, `state`, `options` (names, or names with descriptions) and `label` (one of the
option names) are required. `group` keeps related cases on one side of every split (cases
without one are grouped with the cases that say the same thing); `kind` is `noul`, `choice`
or `score` and defaults to `noul` for two options, `choice` otherwise. `--data guards` is the
built-in guard set.

**What `train` does, in order**
1. Checks the file: every line parses and its label is one of its options; fewer than 40 cases,
   a single label, or the same case under two labels is an error; fewer than 500 cases, one
   label over 90%, labels under 5 cases, repeated cases and repeated ids are warnings
   (`--dry-run` prints these and the split sizes, and stops).
2. Refuses a bad `--name` (lower-case letters, digits, `.`, `_`, `-`), a name already
   registered (see `--replace`), an `--out` that has files in it, and an `--out` inside a git
   work tree (`--allow-in-repo`): trained weights and data are never committed. The default
   `--out` is under the state directory.
3. Splits by whole group into train, calibration (15%) and test (25%), the same way for the
   same `--seed`. With `--eval FILE` the test cases are that file instead, and the run stops if
   any of its cases (same question, state and options, ignoring case and spacing) or groups
   also appear in `--data`.
4. Holds the GPU: a claim named `gpu-training` at the Broker (visible in `ml-stack-serve
   status`). A model server held by another process, or another training run, refuses the
   hold; `--wait SECONDS` waits for the claim. A `--device cpu` run holds nothing.
5. Scores the baseline on the test cases: `auto` is the released checkpoint when `--init
   strands` and otherwise the training-set majority label; also `strands`, `majority`, `none`
   or the name of a registered decider.
6. Trains the LoRA and the head (options shuffled every pass), then fits a temperature on the
   calibration cases: one over all of them and one per kind that has 20 or more. A fit on 50 or
   more cases minimises ECE (as the upstream recipe does, because an NLL fit can leave a binary
   head badly calibrated); a smaller one minimises NLL. A trained decider applies the
   temperature of the kind its question had in training.
7. Writes `decider.json` (hashes, temperatures), `manifest.json`, `model_card.md` and the
   safetensors files, scores the test cases (accuracy, Brier, ECE, accuracy and rate of
   abstaining at confidence 0.8, `--floor` to change it) and puts the baseline's numbers beside
   them in the card.
8. Refuses to register a decider whose accuracy is below, or Brier above, the baseline's on the
   test cases (the directory is still written, `manifest.json` says `registered: false`);
   `--allow-worse` registers it anyway. Otherwise it registers the name and pins the weights
   and config in the sentinel (source `trained:NAME`), so a changed file is a finding.

`--replace` re-registers a name that is taken; the old entry and its directory stay, under
`NAME.prev`. `ml-stack-decide eval FILE --decider NAME` scores a registered decider (or
`strands`, or a directory) on any labelled file with the same metrics, repeatable to compare.
The same metrics are in `ml_stack.decide.metrics` for other runners.

**Untrusted data.** Labels, ids, groups and text are only ever data: nothing from the file
becomes a path (the name and `--out` come from the command line), a command or a pickle
(weights are safetensors; a directory with a `.bin`, `.pt` or `.pkl` file is not loaded), and
the card prints only a reduced character set. Never commit the file you trained on.

**Honest limits**
- Time and memory are not measured for this command. The measured runs are above: 30 steps on
  Qwen3.5-0.8B-Base in 87 s, and 120 steps continuing the released 2B checkpoint in 364 s,
  both on one Apple-silicon GPU. The 2B base needs about 4.7 GB of weights plus activations;
  nothing here checks that it fits next to a served model, and a held server refuses the run.
- A few hundred cases check the pipeline; they do not support a quality claim, and the card
  says so. A model trained on your cases is calibrated on cases like them and on nothing else.
- The baseline check compares point estimates on the test cases; with 60 test cases a gap of
  a few points is noise.
- It trains the head and one rank-16 LoRA; it does not resume, shard or train on several
  GPUs, and a case over 1024 tokens is refused.
- A temperature cannot change which option wins, only how sure the decider says it is.
- The real-model path has been run here only on a random 2-layer model on CPU (the tests); the
  same code on Qwen3.5 was last run by the earlier `ml-stack-train-decider` measurements above.

## Licences

| part | licence | how it is used |
|---|---|---|
| strands-decider code, github.com/strands-labs/strands-decider | Apache-2.0 (LICENSE; GitHub reports the same) | read for the architecture; none of it is copied |
| checkpoint `StrandsAgents/strands-decider-2B-hobson-v19` | Apache-2.0 (card metadata and LICENSE.md) | downloaded on request, pinned by commit, size and SHA-256; not redistributed |
| base `Qwen/Qwen3.5-2B-Base` | Apache-2.0 (card and LICENSE) | same; the release is a LoRA on this model (the post's base is Qwen3.5-2B, not Qwen2.5) |
| base `Qwen/Qwen3.5-0.8B-Base` | Apache-2.0 | the default base for `ml-stack-train-decider` |
| training data of the checkpoint | 29 public Hub datasets plus two author releases and teacher outputs from Qwen3.5-4B, each under its own licence; its `data/sources.md` lists them | not read or redistributed here; read that file before using the checkpoint where the data's terms matter |
| this package | Apache-2.0 | a re-implementation of the described architecture and prompt layout; see `NOTICE` |

## A guard for an embedding application

    from ml_stack.decide import router
    from ml_stack.decide.guard import Policy, ToolCallGuard
    from ml_stack.interventions import Call, Context, guard_tool_call

    guard = ToolCallGuard(router.build("logprob", router.shared("logprob", "http://127.0.0.1:8080")),
                          project_dir="/work/app", allowed_hosts=["api.internal.test"],
                          policy=Policy(abstain_below=0.8, scope="deny"))
    ctx = Context(task=user_request, messages=conversation)
    result = await guard_tool_call(Call(name, arguments), ctx, run_tool, [guard], confirm=ask_person)

A destructive answer or an answer under `abstain_below` becomes a Confirm, a call that follows
instructions found in tool output becomes a Deny, a path or host outside the project becomes
a Confirm or (`scope="deny"`) a Deny, and a decider that fails asks a person. On the whole guard set an early, uncalibrated run of the Qwen3-4B-Instruct logprob decider
labelled 43% of the injected calls as injected: put `Layered` hard rules and a person in front
of anything irreversible.

## Limits

- A question is read less than a document: with the state and options fixed, rewording the
  question often gives the same answer. Phrase it so the obvious reading is the one meant.
- Long multi-step documents are the weak spot of the pointer model; the prompt limit here is
  4096 tokens (1024 when training).
- Calibration is fitted on cases like the ones you will see. A temperature fitted on the
  guard set says nothing about another task.
- The state text is untrusted input to the decider. A tool result that tells the decider how
  to answer can move its answer; the guard's `grounded` check is itself a model.
- The guard set is 412 cases from one author. Labels were checked by hand, and the scope
  labels agree with the deterministic path and host check on every case.
- `make-cases` labels cases by rule from tool names and descriptions; read a sample.
- Only PyTorch is wired for training. The released checkpoint runs on CUDA, MPS and CPU
  through transformers; there is no MLX path and no prefix cache across questions about one
  state.
