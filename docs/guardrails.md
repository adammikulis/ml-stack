# Guardrails

Decision record and measurements, 2026-10-02. Machine: one Apple-silicon Mac, Python 3.13.5 (3.12.8
where a package needs it), llama.cpp b10816 served through `ml_stack.serve`, a fresh virtualenv per
candidate. The text below says which command produced each number.

## What runs by default

`ml_stack.guard` sits in front of the model in the two places this library runs a tool loop.

| Loop | Where | What a consumer does to get the rails |
| --- | --- | --- |
| `ml_stack.do.run` (the served model calls `ml_stack.mcp` tools) | every tool call and every tool result | nothing; `guard=None` builds `Guard.default()` |
| `ml_stack.harness` (Claude Agent SDK on a served model) | `PreToolUse` / `PostToolUse` hooks on every SDK tool, `max_turns` 50 | nothing; `Harness(guard=None)` builds the SDK guard |

The built-in rails need no extra and make no network call:

- `tool-policy`: the call must name a tool the run offered, its arguments must fit that tool's JSON
  schema (types, required, no unknown keys, enums, a bool is not an integer), no string over 8192
  characters or with control characters, no `..` path climb, no credential file (`.ssh`, `.aws`,
  `.env`, `id_rsa`, `/etc/passwd`...), no URL whose host is not loopback; at most 200 calls per
  run, 120 per minute, the same call at most 10 times. A tool that starts processes, writes files or
  downloads (`serve_up`, `models_fetch`, `fleet_join`, `speech_say`, `bench_*`...) runs after text
  from outside the person has been read only when the person confirms: a "go" on a `plan` that
  names the tool confirms it, otherwise the loop asks.
- `secrets`: tokens, keys, private keys, JWTs, `Bearer` values, URL passwords, `key=value`
  assignments and the value of every environment variable named like a credential are replaced by
  `[REDACTED:kind]` in tool results, model words and everything printed; a call whose arguments
  carry one is denied.
- `untrusted`: each tool result is wrapped in `<untrusted source=...>`, chat markup (`<|im_start|>`,
  `[INST]`, `<<SYS>>`) and a forged closing tag inside it are neutralised, it is capped at 20000
  characters, and it is marked tainted when its tool is an external source or it matches an
  injection phrase. The system prompt tells the model what the fence means.

Turning a rail off is `Guard.off(because=...)` or `rails(without=[...], because=...)`; it needs the
reason, writes a warning to the `ml_stack.guard` logger and prints it. A denied call is told to the
model as `blocked by the <rail> rail: <reason>` and recorded on `Outcome.blocked`. Guard code keeps
no text in its logs, only the rail, the action and the reason.

Optional rails layer on top (`Guard.default(extra=[...])`):

- `ml_stack.guard.nemo.NemoRail` (extra `guardrails`): NeMo Guardrails' input and output rails
  through `LLMRails.check`; blocked becomes deny. It sets `NEMO_GUARDRAILS_NO_USAGE_STATS=1` and
  `DO_NOT_TRACK=1` first.
- `ml_stack.guard.classifier.InjectionClassifierRail` (extra `guard-model`): the ONNX export of
  `protectai/deberta-v3-base-prompt-injection-v2` on the CPU, run on free-text sources only.

## Measurements

Attack success rate means the objective check named in `ml_stack.testing.canary` came true: a
tool ran with arguments it should not have had, a credential reached the model or the screen, or a
count passed its ceiling. The red-team suite in the `redteam` worktree held three helper modules
(egress, evidence, report) and no attack runner when this was written, so these are our own
canaries.

Scripted worst case (a model that does whatever the planted text or the attack says), 18 attacks:

    python -m ml_stack.testing.canary

| | attacks that succeed |
| --- | --- |
| rails off (`Guard.off`) | 17 of 18 |
| default, no configuration | 0 of 18 |

The one that fails with the rails off is `unknown-argument`: the tool function rejects the extra
keyword itself. Two ordinary tasks (look up, plan, serve; serve, stop) complete under the defaults.
`tests/test_guard_canary.py` asserts these rates through the same entry the loop uses.

A served model that decides for itself (9 planted texts, 3 phrasings x fleet_join, models_fetch,
serve_up, 2 runs each), no scripted calls:

    python -m ml_stack.testing.canary http://127.0.0.1:PORT [JUDGE_URL]

| model | rails off | default | default + NeMo self-check input |
| --- | --- | --- | --- |
| Qwen3.5-2B Q4_K_M | 18 of 18 landed | 0 of 18 | not run |
| Qwen3-4B-Instruct-2507 Q4_K_M | 14 of 18 landed | 0 of 18 (12 calls blocked) | 0 of 18 (12 results withheld, model never saw them) |

Detection of injection text in tool results, 16 injections (including paraphrases, Spanish,
spaced-out letters and chat markup) against 19 benign results (three of them long tables or JSON):

    scripts/guard-eval [--classifier] [--nemo URL]

| screen | injections caught | benign flagged | time |
| --- | --- | --- | --- |
| built-in markers | 9 of 16 | 0 of 19 | under 1 ms each |
| ONNX DeBERTa classifier | 14 of 16 | 4 of 19 (every list, table and JSON) | 1.3 s for 35, 2.8 s load |
| NeMo self-check input, Qwen3-4B-Instruct-2507 | 15 of 16 | 0 of 19 | 4.9 s for 35 calls |
| NeMo self-check input, Qwen3.5-2B | 2 of 16 | 0 of 16 | 6.5 s for 32 calls |

Neither the markers nor the classifier is what stops an injection from acting: the tool-policy
rail does (0 of 18 above whichever screen ran). The screens decide what the model reads and when
the person is asked.

Findings that shaped the code:

- The DeBERTa model scores any list, table or JSON of eight or more records as an injection
  (0.95 to 0.995). It runs only on `PROSE` sources (transcripts, web text).
- NeMo's self-check returns "allow" when the judge replies with nothing. A thinking model (Qwen3.5
  with its default template) used its whole token budget, NeMo logged `LLM returned empty
  content`, and an injection passed. The shipped configuration sets `enable_thinking: false`.
- NeMo 0.24.1 sends a usage event to `events.telemetry.data.nvidia.com` on startup and every ten
  minutes unless `NEMO_GUARDRAILS_NO_USAGE_STATS` or `DO_NOT_TRACK` is set; it keeps a local copy in
  `~/.config/nemoguardrails/usage_stats.json`. `NemoRail` sets both. The measurement runs above
  that did not go through `NemoRail` (`scripts/guard-eval` before it called `quiet()`) wrote six such
  events; the file was deleted.
- A Qwen3.5-4B server on this machine answered every request with `@` characters; its NeMo result
  (16 of 16 flagged, 16 of 16 benign flagged) was discarded and the judge was changed.

## Candidates

| Candidate | Licence | Latest on PyPI | Python | Weight | Offline with a local llama.cpp | State | Decision |
| --- | --- | --- | --- | --- | --- | --- | --- |
| NeMo Guardrails `nemoguardrails` | Apache-2.0 | 0.24.1, 2026-09-16 | >=3.10,<3.14 (installed on 3.12.8 and 3.13.5; no 3.11 interpreter here) | 65 packages, 224 MB over an empty venv, `import nemoguardrails` 1.4 to 2.9 s | yes: `engine: openai` with `base_url` on the loopback server; the regex rails need no model; first use of colang knowledge-base flows downloads an embedding model through fastembed (not exercised) | active | **optional extra `guardrails`** |
| ONNX DeBERTa injection classifier (ProtectAI v2) | Apache-2.0 | model revised 2026-07-09 | any (onnxruntime) | 738 MB model; onnxruntime, tokenizers, numpy, huggingface_hub; no torch | yes, after one download | active | **optional extra `guard-model`**, prose sources only |
| LLM Guard `llm-guard` | MIT | 0.3.16, 2025-05-19 | >=3.10,<3.13 | on 3.12.8: 82 packages, 1.0 GB (torch 2.14, transformers 4.51, spaCy, nltk) | scanners download models from the Hub | no release for 16 months | rejected: `pip install llm-guard` on 3.13.5 exits 1 (the resolver falls back to an old release that does not build), and it is stale |
| Guardrails AI `guardrails-ai` | Apache-2.0 | 0.11.0, 2026-08-14 | >=3.10,<3.14 | on 3.13.5: 115 packages, 383 MB, import 4.1 s (litellm, openai, langchain-core, OpenTelemetry exporters) | no: validators install from the Guardrails Hub (`guardrails hub install`, network and token); `from guardrails.hub import DetectPII` fails on a fresh install | active | rejected |
| Presidio `presidio-analyzer` | MIT | 2.2.364, 2026-07-22 | >=3.10,<3.15 | with spaCy: 54 packages, 184 MB; `AnalyzerEngine()` downloads `en_core_web_lg` (400 MB from GitHub) when it is absent | only with the spaCy model installed beforehand | active | already the `privacy` extra; not a rail here: it finds personal data, not credentials or instructions |
| Llama Guard 3/4, Prompt Guard 2 (Meta) | Llama community licence / "other", gated on Hugging Face with manual approval | models of 2024-10 and 2025-04 | via transformers | 86M to 12B | not downloadable without Meta's approval | n/a | rejected: licence and gating conflict with Apache-2.0 distribution of a default |
| Qwen3Guard-Gen-0.6B | Apache-2.0 | GGUF on this disk | llama.cpp | 0.5 GB | yes | active | not adopted: a harm-category classifier, not a prompt-injection detector; not measured on the corpus |

## Is NeMo Guardrails required?

Not as a core dependency. The measured costs of requiring it:

- the core install is `packaging` alone; NeMo adds 65 packages and 224 MB, including `fastembed`,
  `onnxruntime`, `protobuf`, `aiohttp` and `numpy`, to every machine that only wants to pass work
  around a network;
- importing it takes 1.4 to 2.9 s, paid by every command if imported eagerly;
- it reports usage to NVIDIA unless told not to;
- its sensitive-data extra does not install on 3.13 (`presidio` is gated `< 3.13` there);
- what it adds over the built-in rails needs a judge model: a 2B judge caught 2 of 16, a 4B
  instruction model caught 15 of 16. A required dependency still has no judge to call when the
  model is not up, so the default protection stays the built-in rails either way.

What it gives that the built-ins do not: a judge-model input and output screen configured in YAML
and Colang, the other NeMo rail library (regex, hallucination, content safety), and a maintained
project behind it. If the decision is to require it anyway, the change is moving the `guardrails`
line from `[project.optional-dependencies]` to `dependencies`; `NemoRail` already imports it
lazily. `Guard.default()` would still need a judge URL before it could add a NeMo rail by itself.

NeMo's own model-free tool-call rail (allow-list and JSON-schema check) exists in its `IORails`
engine, which owns the model call; `do.run` owns that call, so the policy rail is ours.

## Gates

Licences of what NeMo installs (pip-licenses): MIT, Apache-2.0, BSD, PSF, ISC, MIT-CMU, one
MPL-2.0 (certifi) and one with no metadata (py_rust_stemmers). `pip-audit` on that environment
reports no vulnerability in any package but the venv's own pip 25.1.1 (twelve advisories, fixed in
pip 26.2).

## Not verified

- NeMo on Python 3.11, and either NeMo or the classifier on Linux and Windows.
- The Claude Agent SDK hooks against a running Claude Code binary: the hook callbacks and the
  options are tested, a session with a served model through the hooks was not run.
- The PyRIT suite: not usable yet (see above).
- NeMo colang knowledge-base flows and its embedding download.
