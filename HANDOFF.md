# Handoff

**Every item here is a task.** A finished task is deleted, not marked done — what exists and
why is in `README.md`, the code and `git log`. Each carries the context to pick it up cold.
Rules: invented names only, everywhere (`tests/known-fixtures.txt`); tests build their own
fixtures and never read `~/.ml-stack`; a measurement is estimated before it runs and
smoked before it is paid for; nothing is pushed without Adam's go-ahead (a push cuts a
release). The app that drives this library is `~/ai_ceo`; its `HANDOFF.md` holds what is
Slack-specific.

## From 2026-09-02, ranked by the time each would have saved that day

- [ ] **Fakes with real signatures.** A `--also tight` flag reached `Client.__init__` and took
  an 87G load down because the variant's test faked a client with `**kwargs`. One
  `ml_stack.testing` module holds the fakes the suite shares — client, `serve()` context,
  preflight report, a scripted model — each mirroring the real signature, and a test
  diffs every fake's signature against the real one so they cannot drift.
- [ ] **A `Run` object instead of twenty keyword arguments.** `served()` forwards most of what
  it takes; what is about the serving, the asking and the client is implicit, which is how
  `tight` and `rich` leaked. Build one typed object from argv with three sections and pass
  it; the boundary becomes a type.
- [ ] **`ml-stack-bench history`.** `measuring.json` knows the current run; nothing records the
  day. From the logs directory: every command, when, how long, on which commit, its exit,
  the estimate beside the actual. "How much time did you waste" answered by one command.
- [ ] **The store checks itself.** `docs()` reads by key since a full scan returned empty
  values; every other scan is a suspect until ladybug's trigger is known. A round-trip
  check on every write, and `ml-stack-store check PATH` reading every doc by key and by
  scan and reporting disagreement.
- [ ] **Split `bench.py`.** Past 2,500 lines — serving, measuring, scoring, the table, the
  ranking, detach and locking — so every agent is told "touch only this region". Five
  modules behind the same CLI: `bench/serve.py`, `measure.py`, `score.py`, `show.py`,
  `run.py`.
- [ ] **One head chooser that knows the binary.** `--draft auto` in the bench picks a head
  the build cannot load; the gate lives in `hub.draft_for(borrows=)` and `serve up` uses
  it. One resolver, taking the binary, for `serve up`, the bench and the app.
- [ ] **Prompt-cache accounting per turn.** `cached` is per run; per turn it would show when a
  change to the asking breaks the prefix — the cheapest speed lever there is, invisible.
- [ ] **The name hook's shapes as data.** Places, roles, events, reserved domains, uuids: each
  learned by exception in code. A data file of shape rules, and the hook printing which
  rule fired, so the next exception is a data change.
- [ ] **Estimates in the runner.** Before any measuring run, seconds per question from the
  store × questions × ways × models printed, and a refusal over a ceiling without `--yes`.
- [ ] **`ml-stack doctor`.** `ml-stack-setup` checks the machine; nothing checks the
  repositories: hooks installed, editable install pointing at the checkout, a stale
  worktree pin, a store with empty docs, a log with no matching run. One command at the
  start of a session.

## Verifying

```bash
python3 -m pytest tests -q > /tmp/out.txt; echo $?   # ~2100 in ~80 s on every core; never pipe into tail
ml-stack-setup                                        # the machine
```
