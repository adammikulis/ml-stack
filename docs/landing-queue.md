# Landing queue

Agents hand finished work to landing through the pool board, and a watcher lands every ready
entry as one batch with `scripts/land`. The batch mechanics (plan, merge, one combined
verification, ejection, finish) exist; this document is the design of the board side and the exact
cut between the two.

## What is implemented (phase 2)

The queue lives on the board and a runner lands from it. Phase 1 (`scripts/land plan|run|finish`)
still knows nothing of the board; `scripts/land_board.py` is the only bridge.

- `src/ml_stack/workspace/landing.py`: one hash-chained log, `landing.jsonl`, in the workspace.
  Rows are stamped with the authenticated identity; the queue is a fold of the log. Requests,
  independent reviews, cancels, pause and resume, runner states and progress beats are all rows,
  and each also writes an audit row.
- Commands (`src/ml_stack/workspace/landing_cli.py`): `land-request BRANCH SHA --test SELECTOR ...
  --replaces TEXT`, `land-review REQUEST SHA --verdict accept|reject`, `land-cancel`, `land-pause`,
  `land-resume`, `land-queue`. `digest --status` also prints the queue, the runner holder and the
  age and text of the last gate step.
- Runner: `scripts/land serve [--once] [--interval S] [--stall-minutes 20] [--remote origin]`, run by
  the coordinator (or any lead) on any device that has the checkout. It is a process, not a unit:
  the runner takes the `branch 0.2dev` claim (the claim task integration also takes, so the two
  never land at once), and a second runner is told who holds it. An always-on form may use the
  autostart prepare flow later; none is installed.
- `scripts/land run --entries FILE` (JSON list of `{branch, tip}`, `scripts/land_entries.py`) refuses
  a branch whose tip is not the requested SHA, that does not exist, or that shares no history with
  `origin/<target>`.

### One pass of the runner

1. Claim the runner role; stop when paused.
2. For each waiting request, re-derive standing from the registry (below) and re-check the tip
   against the requested SHA. A request that fails is settled (`refused`) or left waiting
   (`needs-review`) with the reason, and its requester gets a message.
3. Batch up to eight remaining requests, oldest first, and run `scripts/land run --entries`: one
   integration worktree cut from fetched `origin/0.2dev`, merges with the recorded policy (a
   conflict ejects that request as `needs-human`; `HANDOFF.md` and log conflicts keep both sides),
   one combined gate through `scripts/test`. A failing check is re-run on a clean checkout of the
   base and bisected: a failure also red on plain `0.2dev` is reported as such and does not block;
   a failure only on the batch ejects the branch that introduced it (`failed`, naming the test
   files and the baseline-red checks) and the rest are re-verified.
4. On a green batch, `scripts/land finish --apply` fast-forwards the local development branch and
   removes the worktrees and branches of landed requests when `git cherry` shows nothing unique
   and nothing is dirty. Then a normal push of `0.2dev:refs/heads/0.2dev`. It refuses any
   protected target, never forces, never pushes `main`, tags or deletions, and goes through the
   repository's `pre-push` hook unchanged. A rejected push leaves the requests `landed-unpushed`
   and an announced `blocked`.
5. Messages: each requester gets a direct message per state (`handoff` for failure and
   `needs-human`, `status` otherwise); an announcement marks a landing (`milestone`), a batch not
   landed or a stuck gate (`blocked`).

### Eligibility, review and brakes

- Only a landing-level identity may request: a person, a lead, or a top-level (not delegated) agent
  with a live token, the right to send and a verified model above the lowest tier (the coordinator
  bar). A delegated helper, a lowest-tier model and an unverified model are refused. `main` and
  `master` are never a branch or target.
- A request needs an independent accept recorded at its exact SHA by another landing-level identity
  that is neither the requester nor a delegate or parent of it. Without it the request is
  `needs-review` and the runner will not land it. The runner re-derives requester and reviewer
  standing at landing time (a revoked reviewer no longer counts; any standing reject blocks).
- Pause and resume: a person, a lead or the runner. Cancel: the requester or those. A cancel of a
  running request stops the gate and queues the others again. A new request for the same branch
  supersedes the older one (a new SHA means a new review).
- Stuck gate: no output from `scripts/land run` for `--stall-minutes` (default 20) announces
  `blocked`, SIGTERMs the gate's process group (SIGKILL only after a minute of ignoring it), sets
  the batch's requests to `needs-human` and moves on to the next pass. Nothing is pushed.

### Not built

See the HANDOFF entry "Landing queue phase 2 leftovers": the task-lifecycle entry path, the
SessionStart suggestion line, the async `scripts/test submit` path for the gate, requests from a
device that does not hold the branch, and the blocked-finish retry.

## Original design (the parts above supersede where they differ)

### Cut line

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
