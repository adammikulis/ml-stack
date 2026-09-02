# Handoff

**Every item here is a task.** A finished task is deleted, not marked done — what exists and
why is in `README.md`, the code and `git log`. Each carries the context to pick it up cold.
Rules: invented names only, everywhere (`tests/known-fixtures.txt`); tests build their own
fixtures and never read `~/.ml-stack`; a measurement is estimated before it runs and
smoked before it is paid for; nothing is pushed without Adam's go-ahead (a push cuts a
release). The app that drives this library is `~/ai_ceo`; its `HANDOFF.md` holds what is
Slack-specific.

## From 2026-09-02, ranked by the time each would have saved that day

Seven of ten landed (fakes, history, the store's check, the head chooser, the hook's
shapes, the doctor, the split into `graph/bench/`); the registrations are in.

- [ ] **A `Run` object instead of twenty keyword arguments.** `served()` forwards most of what
  it takes; what is about the serving, the asking and the client is implicit, which is how
  `tight` and `rich` leaked. Build one typed object from argv with three sections and pass
  it; the boundary becomes a type.

## Conversations of any length (asked 2026-09-02)

Landed in ml-stack (`thread.recall`, `summarise`, `WINDOW = 10`, `converse(summary=,
recalled=)`); the app has to pass them and add them to the answer cache's context.


## The store on ladybug 0.20

- [ ] **ladybug 0.20.2 returns nothing from a fresh store's scans** (CI on Linux, 2026-09-02: ten
  store tests read a count of zero where one was written; this machine runs 0.18.2 and
  passes). The store extra is pinned `<0.19` for now. Characterise it with
  `ml-stack-store check` on a scratch store under 0.20 (scan by key vs by scan, nodes and
  docs), find the release note, and either adapt the queries or keep the pin with the reason.

## Measuring across the fleet (asked 2026-09-02)

- [ ] **Place users across the fleet by what fits** (asked 2026-09-02). Adam: "it lets us
  load balance, where if we have more users we can sacrifice quality and use a smaller model
  with more simultaneous kv caches ... some users will get the bigger model and some smaller".
  Input: `ml-stack-serve fit` (measured per-token and per-sequence cache bytes per model, in
  `src/ml_stack/data/fit.json`) and each peer's room from discovery. Build `ml-stack-fleet
  plan --users N --context C [--prefer MODEL]`: for every peer, the largest ranked model
  (docs/model-ranking.md) whose weights plus N_i users' caches fit, N_i summed to N, best
  models to the most users; print the plan and, with `--apply`, serve it (`ml-stack-serve up`
  through each daemon). Then a router in front of the fleet that sends a new session to the
  peer with a free slot on the best model. Tests on fakes: three peers of different room, one
  demand, the plan; a demand no fleet fits says so and by how much.
- [ ] **`ml-stack-models layout MODEL`** (2026-09-02): the attention layout read off the GGUF
  header -- which layers hold a cache, which are recurrent (`full_attention_interval`),
  sliding (`sliding_window`, pattern, `key_length_swa`), shared (`shared_kv_layers`), and
  any compress ratio / indexer -- as one paragraph. Read by hand with a throwaway script
  today to understand Flash-Next; `fit --measure` gives the real bytes, this says why.
  `preflight._recurrent_layers` / `_sliding_layers` already compute most of it.
- [ ] **Run it for real across two machines.** Everything landed 2026-09-02 (`ml-stack-fleet
  join|status|leave`, the daemon's bench jobs, `sweep --fleet`, host on every run, the app's
  cluster view, `ml-stack-mcp`), every branch tested against fakes and loopback peers; none of
  it has crossed a real network or a real Windows box. First: `ml-stack-fleet join --persist`
  on the Windows machine (the README's Windows paragraph lists what to run and what each
  prints), `ml-stack-fleet status` here, then `ml-stack-bench sweep --fleet --serve <two small
  models> --sample 6` and `show` with a `host` column.
- [ ] **`only_one(wait=False)` truncates the holder's pid when refused** (found 2026-09-02):
  the `finally` runs on the refused attempt too, so the next asker sees "held by somebody".
  Fix in `lock.py` with a test that a refused attempt leaves the holder's record intact.

- [ ] **Run it for real across two machines.** Everything landed 2026-09-02 (`ml-stack-fleet
  join|status|leave`, the daemon's bench jobs, `sweep --fleet`, host on every run, the app's
  cluster view, `ml-stack-mcp`), every branch tested against fakes and loopback peers; none of
  it has crossed a real network or a real Windows box. First: `ml-stack-fleet join --persist`
  on the Windows machine (the README's Windows paragraph lists what to run and what each
  prints), `ml-stack-fleet status` here, then `ml-stack-bench sweep --fleet --serve <two small
  models> --sample 6` and `show` with a `host` column.
- [ ] **`only_one(wait=False)` truncates the holder's pid when refused** (found 2026-09-02):
  the `finally` runs on the refused attempt too, so the next asker sees "held by somebody".
  Fix in `lock.py` with a test that a refused attempt leaves the holder's record intact.


## Verifying

```bash
python3 -m pytest tests -q > /tmp/out.txt; echo $?   # ~2100 in ~80 s on every core; never pipe into tail
ml-stack-setup                                        # the machine
```
