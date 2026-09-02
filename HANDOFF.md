# Handoff

**Every item here is a task.** A finished task is deleted, not marked done — what exists and
why is in `README.md`, the code and `git log`. Each carries the context to pick it up cold.
Rules: invented names only, everywhere (`tests/known-fixtures.txt`); tests build their own
fixtures and never read `~/.ml-stack`; a measurement is estimated before it runs and
smoked before it is paid for; nothing is pushed without Adam's go-ahead (a push cuts a
release). The app that drives this library is `~/ai_ceo`; its `HANDOFF.md` holds what is
Slack-specific.

## From 2026-09-02, ranked by the time each would have saved that day

Six of ten landed that day (fakes, history, the store's check, the head chooser, the
hook's shapes, the doctor); `ml-stack-bench history` still needs its one-line registration
in `bench.py` and the detach header should write `commit:` (see `bench_history.py`'s
docstring); the bench's `--serve-draft auto` should call `hub.choose_head(model,
binary=args.binary)`.

- [ ] **A `Run` object instead of twenty keyword arguments.** `served()` forwards most of what
  it takes; what is about the serving, the asking and the client is implicit, which is how
  `tight` and `rich` leaked. Build one typed object from argv with three sections and pass
  it; the boundary becomes a type.
- [ ] **Split `bench.py`.** Past 2,500 lines — serving, measuring, scoring, the table, the
  ranking, detach and locking — so every agent is told "touch only this region". Five
  modules behind the same CLI: `bench/serve.py`, `measure.py`, `score.py`, `show.py`,
  `run.py`.
- [ ] **Prompt-cache accounting per turn.** `cached` is per run; per turn it would show when a
  change to the asking breaks the prefix — the cheapest speed lever there is, invisible.
- [ ] **Estimates in the runner.** Before any measuring run, seconds per question from the
  store × questions × ways × models printed, and a refusal over a ceiling without `--yes`.

## Verifying

```bash
python3 -m pytest tests -q > /tmp/out.txt; echo $?   # ~2100 in ~80 s on every core; never pipe into tail
ml-stack-setup                                        # the machine
```
