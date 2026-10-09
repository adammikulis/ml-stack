# Integrating decision models: plan

Decision models pick among options you name and return calibrated probabilities. They do not write text. Early uses reported for this class of model: model routing, tool selection, evaluations, guardrails, memory, context management and policy classification; hybrid agents where an LLM takes the hardest decisions and a decider the rote ones (less cost and latency); decider models combined with fixed workflow languages. Sources: [Introducing Strands Decider](https://strandsagents.com/blog/introducing-strands-decider/).

This plan has two halves: the platform in Poolhouse, and pcb-engine as the first real consumer. The default decider is the Strands 2B (`StrandsAgents/strands-decider-2B-hobson-v19`, Qwen3.5-2B base) because it is small; revisit only against measured results. Fine-tuned deciders are never committed: datasets and weights stay in caches, the repository keeps recipes and metrics.

## Principles

1. **Deciders tighten, never widen.** A decider may route, rank, flag, confirm or deny. It never grants a permission, approves a human-only action or lifts a guard. Confirmation, autopilot grants and quarantine release stay rules in code.
2. **Abstain means escalate.** Below the calibrated floor the decision goes to an LLM, and if that is not allowed, to a person. A failure denies.
3. **Everything a decider reads is untrusted.** All state text passes one sanitiser on every backend, the pointer backend included.
4. **Every decision is auditable:** question, option, probability, model, build and a prompt hash, in the hash-chained sentinel log.
5. **Evidence gates adoption.** An integration replaces a regex, an LLM call or a hand rule only if it beats that baseline on a labelled set taken from our own traces.

## Phase 0: platform (in progress)

- Typed `decide(state, questions)` with `noul`, `choice` and `score` questions (merged).
- Training from the released Strands scripts and data format, with `train` and `eval`, calibration, and a refusal to register a decider worse than the baseline.
- A JevBench runner for the public set (opt-in; it downloads and uses the GPU).
- MTP on by default where it applies; off for one-token decisions.

## Phase 1: finish the platform, in order

1. Sanitise the pointer prompt (a confirmed injection gap) and add the `answer` tag to the shared sanitiser.
2. Run deciders under the Broker lease, for memory admission and queueing.
3. Batching and a shared-state prefix cache: one state, many questions, one pass.
4. Audit records in the sentinel log.
5. First-use pull of the default decider, with pin and approval flow, reusing a Hugging Face cache copy by hash.
6. A hybrid router helper: decider first, escalate to a named LLM on abstain, reporting the escalation rate.
7. Optional research: a hidden-state head that runs through llama.cpp. Until then the logprob backend with an instruct GGUF is the fast path.

## Phase 2: pcb-engine integrations

| Use | Where | Replaces | Gate |
|---|---|---|---|
| Tool selection | Copilot with 47 tools: show a narrowed set per turn | All tools shown to small local models | Live-run pass rate and call count on the 5 tasks |
| Policy classification | Is an op destructive, bulk, or outside the user's intent; feeds confirmation | Hand-written op-kind rules | Only ever adds a confirmation |
| Guardrails | Second layer on tool-result text for injection | Regex-only scrub in `agent/guard.py` | Red-team suite, no case weakened |
| Hybrid agent | "Act or ask the user?" and "is the task done?" | The model's own judgment (live runs stopped early or asked instead of acting) | The evaluation tasks |
| Evaluations | Advisory: is this footprint an equivalent part? | Nothing; the deterministic grader stays authoritative | Agreement with the grader on equivalence classes |
| Domain triage | Classify nets, triage DRC findings, rank library candidates | Name heuristics, manual picking | Held-out synthetic boards (private boards are never committed) |
| Memory and context | Which history entries and facts survive a long chat | Naive truncation | Task success at a fixed context size |

## Phase 3: later

A decider plus a fixed workflow for the board pipeline (import, place, route, DRC, export): the decider chooses the next branch, code owns the steps. Games and maze-like search are out of scope for now.

## Measurement

- Labelled sets come from recorded traces: evaluation reports, red-team cases, the guard corpus.
- Metrics: accuracy, Brier score, ECE, abstain rate and escalation rate. For hybrid agents also LLM calls saved and end-to-end latency.
- Benchmarks: JevBench for general quality, our own sets for each integration.
