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


## Measuring across the fleet (asked 2026-09-02)

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
