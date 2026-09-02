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
- [ ] **Prompt-cache accounting per turn.** `cached` is per run; per turn it would show when a
  change to the asking breaks the prefix — the cheapest speed lever there is, invisible.
- [ ] **Estimates in the runner.** Before any measuring run, seconds per question from the
  store × questions × ways × models printed, and a refusal over a ceiling without `--yes`.

## Conversations of any length (asked 2026-09-02)

Landed in ml-stack (`thread.recall`, `summarise`, `WINDOW = 10`, `converse(summary=,
recalled=)`); the app has to pass them and add them to the answer cache's context.


## Measuring across the fleet (asked 2026-09-02)

- [ ] **`ml-stack-bench sweep --fleet`.** The bench side landed 2026-09-02 (host and commit on
  every run, the host rule in the ranking, the `--fleet` flag calling `fleet.bench`'s
  `plan`/`dispatch`/`wait`/`gather`); the fleet side, the Windows daemon and the one-command
  join are in progress. Ready before that: `--on name=http://peer:port` measures a
  server anywhere, `ml-stack-serve up --root` announces one and `ml-stack-peers` finds it,
  and the community, the question sets and worlds are in the package or made from a seed.
  Missing: (1) dispatch — assign each model to the peer whose reported memory fits it, start
  the run there detached (the training daemon already runs jobs over the fleet; a bench
  job is the same shape), know when it ended; (2) gather — `show --export` exists, an
  `import` does not; runs come home into the one store with `server.host` recorded;
  (3) comparability — a `host` column in the table, and `ranking` never composes one host's
  accuracy with another's cost; (4) the pin — refuse a peer whose commit differs from the
  dispatcher's (`history` records commits now); (5) a peer's lock is consulted before work
  is sent. The Windows and Linux build paths are unit-tested against fakes only.

## Verifying

```bash
python3 -m pytest tests -q > /tmp/out.txt; echo $?   # ~2100 in ~80 s on every core; never pipe into tail
ml-stack-setup                                        # the machine
```
