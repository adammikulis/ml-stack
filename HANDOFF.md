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

## Conversations of any length (asked 2026-09-02)

- [ ] **A conversation that never loses the thread.** Today the ask path re-sends the last six
  turns and the graph supplies the facts; older turns leave the model's view. Build in
  `graph.thread` and the ask path: (1) `recall(thread, question)` — the thread store's word
  index and vectors over turns pick the two or three earlier turns (and what they drew on)
  that match the new question, sent after the window; (2) a rolling `summary` turn, written
  by the small model every K turns (what is established, what the asker wants, what is
  open, the ids it rests on), always sent first and changed rarely so it stays inside the
  cached prefix; (3) facts stated in conversation become nodes and edges through the
  change-request path so the tools find them next time. Window = summary + recalled + last
  N + question. Test: a fact from turn one answered right at turn two hundred, on the fake
  model; measure the `cached` share per turn with `ml-stack-bench concurrent` so a summary
  that changes too often is caught.

## Verifying

```bash
python3 -m pytest tests -q > /tmp/out.txt; echo $?   # ~2100 in ~80 s on every core; never pipe into tail
ml-stack-setup                                        # the machine
```
