# Landing queue

Agents hand finished work to landing through the pool board, and a watcher lands every ready
entry as one batch with `scripts/land`. The batch mechanics (plan, merge, one combined
verification, ejection, finish) exist; this document is the design of the board side and the exact
cut between the two.

## Cut line

Built: `scripts/land plan|run|finish` and the `scripts/land_*.py` modules (docs/test-execution.md,
"Landing a batch"). They take branch names and a repository path, know nothing about the board, and
leave a JSON summary on the last line plus `<git-common-dir>/land/state.json`.

Not built, in this order, each its own reviewable branch:

1. `src/ml_stack/workspace/landing.py` (pure): the handoff record, entry readiness and queue order
   below, over a `Workspace` object. No git side effects beyond reading refs.
2. `scripts/land submit [BRANCH]`: builds the record from the current checkout and posts it.
3. Task hook: an accepted task with a reviewed tip enters the queue from `reviewed()` in
   `src/ml_stack/workspace/task_integration.py`; completion and rework go through its existing
   `finish` and `blocked` events.
4. `scripts/land watch [--once]` and the SessionStart suggestion line.

`scripts/land run` takes one extension for step 4: accept `--entries FILE` (JSON list of
`{branch, tip}`) and refuse an entry whose branch tip differs from `tip`, so a stale submission
cannot be merged.

## Handoff record

One record per `(branch, tip)`, written under the submitting agent's own identity (its token; never
a delegate and never a person). `submit` is idempotent: the record id is
`land:<sha256(repo-id, branch, tip)[:16]>`, and posting it again returns the existing record.

| field | source |
| --- | --- |
| `branch`, `tip`, `base` | `git rev-parse` in the worktree; `base` is the merge-base with the target |
| `target` | `0.2dev` unless named |
| `worktree` | absolute path |
| `checks` | the selectors run and their results, with the `activity/gate.py` evidence ids and the tree hash (`evidence(tree, tier)`) |
| `budgets` | metric deltas from `scripts/budgets` |
| `claims` | the agent's live claims (`ws.claims.listing`) |
| `review` | reviewer id and state when a task review exists, else `none` |
| `task` | the task id when the branch belongs to a task |

An accepted task needs no `submit`: `reviewed()` already proves acceptance, the exact proposal and
review hashes, and a released execution lease, and the tip is the worktree's `source_commit`. The
queue builds the same record from the task so the two entry paths look alike. The task lifecycle
is not forked: landing marks the task `completed` through the existing `finish` path
(`state='completed'`, `landed_commit`, claim release, worktree cleanup), and an ejection records a
`blocked` event on the task with the ejection evidence, which returns it to its worker.

## Queue

A deterministic view, computed from records and git only:

- ready: the tip still equals the submitted `tip`; `checks` are recorded and passing; no live claim
  of another owner covers the branch, its worktree or a touched file; the worktree is clean.
- order: oldest `ts` first, except an entry that other entries stack on (their unique patches
  contain it, from `land_plan.minimal_cover`) comes first.
- not ready entries stay listed with the reason (`stale-tip`, `dirty`, `claim-held`,
  `checks-missing`) so the submitter sees it.

## Watcher

`scripts/land watch` loops; `--once` does one iteration for a hook, cron or monitor. Each
iteration:

1. Return when no entry is ready, or when `other_full_run` or a live batch claim shows a gate in
   flight.
2. Take the batch id `sha256(sorted(entry ids))`, claim it (`ws.claim(token, 'area',
   'land-batch:<id>')`, live-owner check as for any claim) and stop if the claim is held.
3. Post one board line: `landing N entries: <branches>`.
4. Run `scripts/land run --entries ...`, then `scripts/land finish --apply`. The fast-forward is a
   compare-and-set on the target ref (`git update-ref refs/heads/<target> <new> <old>`), so two
   watchers cannot both land: the loser sees the moved ref, releases its claim and replans.
5. Post one board line: `landed N, ejected M: <branch>: <check>: <evidence>`; send each
   submitter, and the task's subscribers (`task-subscribe`), a message under the watcher's own
   identity. An ejection also writes the `blocked` task event.

The watcher is a role of the derived coordinator (docs/mesh-board.md, section 6); today that is the
main session that started it, as a background task running `scripts/land watch`. It does not
push. It fast-forwards the local target and says "push left to the lead" unless the session is
allowed to push the development branch (AGENTS.md, "Landing"; `scripts/hooks/pre-push`). It
installs no launchd or systemd unit: that is a machine setting and stays with a person.

The SessionStart hook prints one line when the queue is non-empty and no watcher holds a live
claim: `landing queue: N ready; run scripts/land watch`.

## Tests

Real temporary git repositories and a real temporary workspace, no board mock: submit
idempotency; two entries batched once; a stale tip rejected; a conflicting pair where one is
ejected and its submitter notified; two watchers racing on one batch; task accepted, queued,
completed; ejected task returned as blocked with evidence.
