# Using all the compute for coding, with cloud models as the other half

Goal (owner): organise and employ as much of this machine (and any peers) as possible for coding,
together with cloud model usage. Nothing below is built as one piece; this is the ordered plan.

1. **Measure utilisation first.** One page (and `poolhouse-serve status`) showing, per resource:
   memory held vs wired limit, GPU/CPU busy %, tokens/s per lease, queue depth and wait time,
   idle minutes per day; cloud usage (per harness, per window) against its limit. Without this
   every other step is a guess. Profiles: write the missing measured profile for the coding
   model (`poolhouse-bench report --profile`) so the broker sizes slots, context and speculative
   settings from numbers.
2. **A task queue with routing, not just a board.** Work items typed by difficulty and risk
   (bulk edit, test fix, refactor, design, review). Route cheap bulk work to local models
   (Qwen3.8-27B for quality, a mixture-of-experts model for throughput), hard planning and the
   second-model review to a cloud model, with escalation when a local attempt fails twice.
   Routing is usage-aware: it reads each cloud harness's remaining allowance and shifts work to
   local capacity as a limit nears (the lead's own weekly allowance is the scarcest resource).
3. **Keep the machine busy.** An idle-time worker takes the next target from the
   self-improvement loop (`docs/notes/self-improvement-loop.md`) when no person-requested work is
   queued, yields instantly to a person's job (lease priority), and respects memory admission.
4. **Make leases share work.** Several agents on one model: slots with continuous batching, one
   cached prefix per project (system prompt + repo map) reused across agents (slot save/restore,
   cache-reuse), MTP/speculative decoding on, thinking off by default. Measure cached-token ratio.
5. **Use the fleet.** Remote leases through the same broker: place a model on the peer with the
   memory, run draft models or the review model on a second machine, and let the queue see all
   capacity. Admission, queueing and ports stay the broker's job on every machine.
6. **Shorten the feedback loop.** The test tier (feat/test-speed, scheduler), quick gate in
   minutes, worktree pool kept warm, so each round of work is minutes.
7. **Quality gates scale with the compute**: second-model review, mutation check, the judge the
   loop cannot edit, and a person's batch review of the staging branch.

Order: 1 (measure), profile the coding model, 2 (router with usage awareness), 3 (idle worker),
4, 5, 6 in parallel with the test-speed work already running.
