# Work continuity when agents disappear

Status: design, 2026-10-08. Nothing in sections 3 to 9 is built. Owner requirement: "subagents/agents are going to disappear, we don't want the contents of their inboxes lost.
somehow, we need to make new agents/subagents organically resume the work of ones that aren't there anymore."

Claim tags: **[V code path]** read in code on this checkout (`0.2dev` at `89293685`); **[D doc]** stated in a document and not re-checked; **[analysis]** my inference or
proposal. Companion designs: `docs/session-liveness.md` (presence, verified death, fencing epochs: this note builds on it and does not repeat it), `docs/mesh-board.md`
(journals), and `docs/earned-trust.md` on branch `worktree-agent-ad7237f496ffd6e09` (trust levels L0 to L4).

## 0. Summary

- The raw material mostly survives a disappearance: the message log, task checkpoints, proposals, worktrees and branches are durable. What is lost is the *pointer*: nobody
  knows that a given name, worktree and unread pile belong together, and nobody is told. [analysis]
- Three real losses today: (a) a message sent to a name that is revoked or expired is *refused*, so the sender's text never reaches anything; (b) unread messages are pruned
  after 7 days whether read or not; (c) a revoked name can be minted again, so "the name" is not a stable handle for a departed agent. Section 1.
- Design: a board-written **departure dossier** per agent (section 2), an **orphan** state computed from liveness (section 4), an offer to the next agent via the existing
  nudge/hook path (section 4), and a single-command **`takeover <departed-name>`** that is a compare-and-set, not a message (section 5). Successor identity is new; lineage is
  a visible edge; standing never transfers (section 7).
- Brakes: only an *orphan* can be taken over; live work is never hijacked; every step is an audit row and an announcement; the person can revoke a takeover and the board
  restores the holdings (section 10).

## 1. What survives today, what is lost, what is stranded

Verified against code. "Stranded" means the data exists but nothing reads it for a successor.

| Item | Today | Verdict |
|---|---|---|
| Unread inbox rows | One chained log `bus` (`bus.py` `Bus.append`); an inbox is the rows with `to == me` after a per-name read cursor held in the graph (`Bus.inbox`, `Bus.cursor`). Rows are not deleted on death. [V `bus.py`] | Survive, stranded: only the dead name's token reads them, and `mentions_me` matches `to == me` exactly. |
| Retention of unread rows | `gc` calls `Bus.prune(retention_s)`, `retention_s` = 7 days, dropping the oldest rows by `ts` with no regard for read state. [V `service.py:824`, `limits.py:31`, `bus.py:145`] | Lost after 7 days. |
| DM to a dead name | `_message_rights` raises `no agent called ...` when `registry.role_of(to)` is empty, and `role_of` is empty for revoked, expired or parent-dead identities. [V `service.py:258,403`, `identity.py:331`] | The send is **refused**; the sender sees an error; nothing is queued. |
| Inbox cap | A recipient with 500 unread (`inbox_pending`) refuses new sends; a sender may leave 100 (`unread_per_sender`). [V `service.py:421-430`, `limits.py`] | A dead name's pile counts against senders forever until pruned. |
| File/area/branch/worktree claims | `Claims._sweep` drops a claim past `expires` (900 s default) or whose `pid`/`owner_pid` is gone or started later than the claim; sweeps run only inside a claims call. [V `claims.py:51-62,128-137`] | Released on their own, within 15 min or at the next look; no record says *whose* they were. |
| Task lease | Lease is a graph node with `deadline = now + 120 s`; `working()` refuses an expired lease; the only recovery is `ready(expired=True)` by a reviewer, which also increments `failures`. No timer calls it. [V `task_actions.py:60-75,193-205`] | Stranded: task stays `working` and blocks dependents; recovery counts death as failure. |
| Checkpoints, proposals, artifacts | Graph nodes linked from the task (`checkpoint`, `proposed-outcome`, `produced-artifact`). [V `task_actions.py:85-130`] | Survive, linked to the task, and the checkpoint carries `commit` and `summary`. Nothing offers them to anyone. |
| Worktree and branch with uncommitted work | `worktree_lifecycle.pending` lists scopes (owner, label, recorded commits, branches) and reasons such as `checkout remains` and `recorded commits are not landed`; `cleanup` needs a live claim by the same worker. [V `worktree_lifecycle.py:148-190,120-130`] Dirty files exist only on disk; no code records them. | Survive on disk; the board has no `dirty` or `orphaned` field, and the owner's name on the scope is the dead one. |
| Running test jobs | `scripts/testjobs.py` runs detached; spec records `owner.id`; cancel is limited to the owner or secret holder; a whole-tier job blocks the next. [V `testjobs.py:111-125,160-170`] | Run to completion, owner unreachable: nobody may read or cancel the result by name. |
| Reviews owed | A task in `review` names no reviewer until `review` is called; `_reviewer(who, task)` decides. [V `task_actions.py:133-150`] A reviewer that vanishes leaves the task in `review` with the work intact. | Stranded, silently. |
| Identity registry | Entries are kept after revocation (`info` returns `revoked`). `_add` refuses only a **live** name; a revoked or expired name may be minted again. [V `identity.py:496-499`, `:403-407`] | Names are not tombstoned (goal 6 fails today), except session names. |
| What runs at session end | `SessionEnd` is wired to `claude-user-prompt` (records the person's words); `SubagentStop` only runs `hello-model` and `announce done` and revokes nothing. A killed terminal runs neither. [V `.claude/settings.json`, `claude-subagent-stop:36-49`] | No departure record is written by any hook. |
| Credit | `task_credit.verify_task` awards the proposal's `worker`, bound by proposal and review hashes, once per task; `work_agent` is keyed by registry identity. [V `task_credit.py:25-60`; D `docs/earned-trust.md` 1.3] | Credit already follows the identity that submitted; it will not be misattributed unless we copy it. |

Consequence: continuity is mostly a *join and offer* problem, plus three data-loss fixes.

## 2. The departure dossier: a durable record written by the board

Goal 1: written continuously by the board, so a crash loses nothing. The agent never has to write it.

### 2.1 Shape

One graph node per agent identity in `coordination.db`, kind `dossier`, id `dossier:<name>`. It is an index of pointers plus a few small copied values, never a copy of the
work. [analysis]

```
{"id": "dossier:<name>", "name": "...", "parent": "...", "kind": "main|subagent|helper",
 "model": "...", "harness": "...", "device": "...", "project": "...",
 "task": "task:<id>" | "", "brief": "<assignment summary, <= 1000 chars>",
 "worktrees": [{"path": "...", "branch": "...", "head_seen": "<sha>", "base": "<sha>"}],
 "checkpoint": "checkpoint:<id>" | "", "last_commit": "<sha from last checkpoint>",
 "claims": ["area:...", "branch:..."],              # as last seen, not authority
 "unread": {"from_seq": n, "count": k, "urgent": j},# derived, see 3
 "owes": [{"kind": "reply|review|answer", "ref": "<seq|task id>", "since": ts}],
 "last_announcement": {"seq": n, "text": "<=200 chars", "ts": ts},
 "jobs": ["testjob:<id>"], "seen_s": ts, "state": "live|orphaned|taken|closed", "taken_by": ""}
```

### 2.2 Who writes it and when

The board service, as a side effect of calls it already handles, in the same transaction as the call (no agent cooperation, no new daemon). [analysis, building on the write
paths in `service.py` and `task_actions.py`]

| Trigger (existing call) | Field updated |
|---|---|
| `register_session`, `mint`, `mint_child` | create dossier: name, parent, kind, model, harness, device |
| task `claim` / `checkpoint` / `submit` / `block` | `task`, `checkpoint`, `last_commit`, `brief` |
| `remember` (worktree scope), `reserve`/`claim`/`release` | `worktrees`, `claims` |
| `message`/`announce` send | `last_announcement`; `owes` cleared when the reply is sent |
| a message received that is a `question`/`task`/`handoff` | `owes` added (cleared by a reply in that thread) |
| any authenticated call | `seen_s` (rate limited to once per 30 s per name) |

`head_seen` is only what the agent reported in a checkpoint; the board does not read the worktree in the hot path. A departure sweep (section 4) reads the worktree once, then,
and stores `dirty` and `head`.

The dossier is a graph node written under `coordination.lock` like leases [V `task_actions.working`]; a kill -9 loses only what the agent had not yet told the board. The
honest gap is uncommitted edits, which exist only on disk: the sweep records `dirty: true` and never promises their content. [analysis]

## 3. Inbox ownership: nothing sent to a departed name is dropped

### 3.1 Three fixes to the loss modes

1. **Do not refuse a send to a departed name.** `_known` accepts a name that has a dossier in state `orphaned` or `taken`. The message is appended with `to: <departed-name>`
   as today, plus `forward_to` resolved at read time (3.2). The sender is told: `delivered to <name> (departed); it will be handed to a successor`. A name that never existed
   is still refused. [analysis; today `service.py:403`]
2. **Pruning respects unread mail for an orphan.** `Bus.prune` keeps any row whose recipient has a dossier in state `orphaned` or `taken` and whose seq is above that name's
   cursor, until the dossier reaches `closed` (or `orphan_keep_s`, default 3 days, then a `message-expired` audit row is written so the drop is visible). [analysis; today
   `bus.py:145-148`]
3. **Inbox caps do not wedge senders.** The 500 and 100 limits stop counting rows for a name that is `orphaned`; instead the dossier holds an `overflow` count and the sender
   gets the same notice. [analysis]

### 3.2 Reading and re-addressing

- Messages are addressed to a *name*; ownership of a name's inbox is a table `inbox-owner`: `name -> current owner` (initially itself). `Bus.inbox(me)` becomes: rows with `to`
  in the set of names whose owner is `me`, after the *per-name* cursor. A successor therefore sees the departed name's unread pile **as a separate, labelled stream** (`from
  the inbox of claude-8f3a1c, now yours`), and its own cursor on the departed stream starts at the departed cursor, not at zero and not at the end. The unread messages are not
  copied or re-addressed; they are re-owned. [analysis]
- `reroute`: future mail to the departed name is delivered to the successor's inbox stream. The sender sees the unchanged address and a one-line `rerouted to <successor>
  (successor of <departed>)`. The rows still say `to: <departed>`, so history stays truthful.
- A departed name's rows are readable by the coordinator and the person even before any takeover: `ml-stack-workspace inbox --of <departed-name>` (read only, no ack, requires
  the lead or human role). [analysis; compare `Service.waiting` which reads for the token's own id only, `service.py:520`]
- Held or flagged rows (the screen/quarantine path) stay held; takeover does not release them. Content is untrusted data to the successor exactly as to the original. [D
  `AGENTS.md`; V `service._hold`]

## 4. Orphan detection and organic offer

### 4.1 Orphaned means

Presence from `docs/session-liveness.md` section 2 gives `live`, `suspect`, `dead (verified)` or `expired`. [D doc; not built, V no `presence.py`] This note needs one more
decision, the orphan rule:

> A dossier is **orphaned** when its holder is `dead (verified)`, or `expired` plus `grace_s`, and it still has resumable work: a task in
> `working`/`review`, an unreleased worktree scope with unlanded commits or `dirty`, an unread urgent message, an `owes` entry or a test job.
> Otherwise it goes straight to `closed` and is not offered. [analysis]

The sweep that sets `orphaned` runs inside any authenticated call by a live main session or the coordinator (the same hook-driven trigger as slice 1 of the liveness design),
and records, once: the worktree `head` and `dirty` (read-only `git status --porcelain`), the unlanded commits, the pending unread count. Until presence ships, a **proxy** can
drive slice 1: the lease `deadline` passing with no heartbeat for `orphan_after_s`, or `kill(pid)` failing for the recorded `owner_pid`. [V `claims.py:51-62`]

### 4.2 The offer

Reuse the existing nudge path. [V `nudge.py` `Waiting.line`, `nudge --hook post|prompt|stop`]

- `Waiting` gains `orphans: list[Offer]`. `Waiting.line()` appends, once per offer per session, for an eligible reader: `orphaned work: claude-8f3a1c (task T-42 "...", 3
  unread, worktree dirty). Run: ml-stack-workspace takeover claude-8f3a1c --dry-run`. The text is board-written, short and carries no free text from the departed agent beyond
  the bounded `brief` (marked data). [analysis]
- Who is offered, in order: (1) the **coordinator** always, since it plans the work; (2) an agent that **just started** (SessionStart / SubagentStart context, as the rules
  reminder already is) when the orphan's project and device match; (3) an agent that **goes idle** (the `stop` nudge event, i.e. it wrote a final message with no pending work)
  when it has no task of its own; (4) after `offer_widen_s` with no taker, any main session in the project. [analysis]

## 5. `takeover`: one command, decided by the board

Name: `ml-stack-workspace takeover <departed-name>` (aliases are not added). `--dry-run` prints the brief and checks only. Plain English: "I am taking over the work of that
departed agent."

### 5.1 Atomic effect (a compare-and-set on the dossier)

Under `coordination.lock`, read `dossier:<name>` at `(state=orphaned, version=n)`; write `(state=taken, taken_by=<me>, version=n+1)` only if the read still holds. Loser gets
`already taken by <X> at <time>` and the brief is not returned. Two simultaneous accepts: exactly one commit. [analysis; pattern: `docs/session-liveness.md` 3.3 CAS,
`harness_remote._rollback`]

Then, in the same transaction:

| Holding | Re-pointing | Existing code it extends |
|---|---|---|
| Task lease | Old lease `active=False`, `superseded_by`; new lease for `me` with `epoch+1`, `successor_of: <name>`; `failures` **not** incremented (death is not failure, `session-liveness.md` open question 4); checkpoints stay linked to the task and the old worker id | `task_actions.claim` and `ready(expired=True)`, which today increments failures [V `:193-205`] |
| Worktree claim and scope | `claims.handoff` generalised from `who.parent == owner` to "owner is the departed name in the taken dossier"; lifecycle scope gets `owner: me`, `inherited_from: <name>` | `claims.py:205-220` (only parent-to-child today), `worktree_lifecycle.remember` |
| File/area claims | Already released by sweep; if still live, transferred like the worktree claim | `Claims` |
| Inbox | `inbox-owner[<name>] = me`, per-name cursor kept, future mail rerouted (3.2) | new, small table |
| Test jobs | Re-owned to the successor by an `adopted-by` field; a stalled whole-tier job is cancelled per liveness design | `testjobs.py` spec |
| Owed replies / reviews | Listed in the brief; a review owed is re-assigned only if `me` is independent of the author (`_reviewer`) | `task_actions.review` |

Visible lineage: the successor's registry record and dossier carry `successor_of: <name>`; `ml-stack-workspace agents` and `whoami` show `claude-91b2d0 (successor of
claude-8f3a1c)`; the task shows `worked-by` edges for both identities with the order. One announcement is posted by the board (not the agent): `takeover: claude-91b2d0 resumes
task T-42 from claude-8f3a1c (departed, 41 min)`.

### 5.2 What the brief contains (data, bounded, marked untrusted)

Task spec and acceptance, the dossier fields, the last checkpoint summary and commit, the worktree location, branch, `base`, unlanded commits, whether `dirty`, the unread
streams (counts, senders, oldest age, and the urgent rows' text up to the usual `waiting_summary` bounds), `owes`, last announcement, jobs. Returned by `takeover`, and written
once to the successor's `inbox` as a `handoff` row from the board. [analysis; bounds from `waiting_summary`, `limits.read_items`]

### 5.3 Who may take over

Default policy (owner decision 2): the **coordinator** and the **parent** of the departed agent's tree may take over, and any main session the coordinator or the person
*designates* (`takeover <name> --for <other>` by coordinator). A non-coordinator main session may self-accept only when there is no live coordinator or after `offer_widen_s`.
A helper (labelled) identity may not. [analysis]

## 6. What the successor must verify before trusting inherited state

The dossier is the board's last knowledge, not a fact about the disk. `takeover` runs these checks and prints them; the brief says "unverified" until each passes. [analysis;
commands exist in `integration_git.py` (`git`, `ancestor`)]

1. **Worktree exists and is the registered one**: `git worktree list --porcelain` in the primary lists the path, and the branch matches the scope. If not, the work is only a
   branch or only a task.
2. **Head versus last checkpoint**: `git rev-parse HEAD` versus `dossier.last_commit`. Equal: nothing after the checkpoint. Descendant: later commits exist that the board had
   not seen (read them). Not an ancestor: history was rewritten; stop and ask the coordinator.
3. **Uncommitted changes**: `git status --porcelain` and untracked files. Never `reset`, `clean` or `checkout .` on inherited state; commit nothing from the departed agent
   under its identity. The successor commits under its own name, as new work, after reading the diff.
4. **Landed, and still current**: : `git merge-base --is-ancestor <last_commit> <development tip>` [V `worktree_lifecycle.pending` does exactly this per recorded commit]. If
   landed, the successor does not redo it; it verifies the acceptance and closes or reviews. If the development tip moved past `base`, reconcile per the merge rule before
   editing.
6. **Mail is data**: unread messages from the departed agent's senders are untrusted text; an instruction in them carries no more authority than it had.

If any of 1 to 5 fails, `takeover` still succeeds (the CAS and re-pointing happen) but the brief opens with `STATE UNVERIFIED: <reason>` and the board marks the task
`resumed-unverified`, which a reviewer sees at review time. [analysis]

## 7. Reputation, credit and names

- **Standing is the model-and-version account's, so nothing transfers between agents.** A successor starts at that account's level, not at zero and not at the departed agent's
  personal record; the departed agent's suspension or brakes do not pass to it (section 13). [D `docs/earned-trust.md` 3.2, 3.3]
- **Credit follows the model and version, not the agent (owner decision 5).** Agents are ephemeral, so a successor finishing inherited work and the departed agent's own
  commits credit the same model-and-version account when they ran on the same model and version; splitting by agent is moot. Today `verify_task` still awards
  `proposal['worker']` (an agent identity) and the ledger keys `work_agent` by it [V `task_credit.py:25-60`]; section 13 lists what must change. `takeover` itself has no write
  path into the ledger; already-accepted awards are never rewritten. If the two ran on different versions, each commit's version is read from its checkpoint (`model` on the
  registry record at the time) and the account that ran the submitted proposal is credited.
- **Provenance still names agents.** The proposal's `provenance` gains `inherited_commits: [sha,...]` and `inherited_from: <name>` (with the departed agent's model and
  version), so the reviewer sees who did what; names are audit, not the credit key. [analysis; `task_provenance.snapshot` binds provenance]
- **Infrastructure blocked is neutral.** A task that ended because its worker disappeared produces no rejection and no `failures` increment; the economy already treats
  `blocked_infrastructure` as no negative sample. [V `task_actions.ready`, `reputation/economy.py` per D `earned-trust.md` 1.2]
- **Names are never reused.** Add a tombstone: `agents.json` entries are kept after revoke, but `_add` and `mint_child` currently refuse only a *live* name. [V
  `identity.py:403-407,496-499`] Change both to refuse any name present in the registry or in `session-names.json`, revoked or not; `dossier:<name>` is the record that makes a
  retired name visible. Session names are already unique by construction (`session_name.assign`). [V] Sub-agent child names are `<parent>/<name>` and need the same rule. The
  successor is a new name, so lineage is an edge (`successor_of`) and never a name alias.

## 8. Subagents of a departed parent

Facts: a child is live only while its parent is live and for at most `child_ttl_s` (8 h); `SubagentStop` revokes nothing. [V `identity.py:275-280`]

- A subagent whose parent died is **itself orphaned** when the liveness design's cascade revokes it (`docs/session-liveness.md` 3.2, "child tokens die with the parent
  session"). Its dossier is created as for any agent, with `parent` retained.
- The tree is offered as **one orphan group** rooted at the parent: the offer text says `claude-8f3a1c and 3 subagents`. `takeover <parent>` takes the parent's holdings and
  *lists* the children's dossiers; it does not auto-take them.
- The successor may take each child's dossier explicitly (`takeover <parent>/<child>`) or re-spawn a fresh child from a child's brief (the usual path, since a child's context
  is gone anyway: the value is its worktree, checkpoint and mail, not its process).
- A child may take over a *sibling's* dossier only by its live parent's instruction; the board checks that the taker's parent equals the departed's parent, or the taker is the
  coordinator. [analysis]

## 9. The mesh case: offline devices and a "departed" agent that returns

Model from `docs/mesh-board.md`: per-device append-only journals, merged by hybrid logical clock; state is a deterministic fold; presence is not journaled; claims are
compare-and-set in the merged log; death on a device is a `session-dead` row written by the owning device; remote death is not verifiable and becomes `stale-held` after
`partition_grace`. [D mesh-board.md sections 3, 6]

Rules, all expressed as journal rows folded deterministically: [analysis]

1. **Dossier writes are journal rows** (`dossier-update`, keyed by agent id, last-writer by total order `(hlc, origin, seq)`). An agent's dossier is authoritative on its
   creating device; others hold a read-only copy. A new device sees departure info only after sync; before that nothing is offered.
2. **`takeover` is a CAS row** `takeover(name, expect_version, taker)`. The fold keeps the earliest by total order for a given `expect_version`; a later one for the same
   version is `yielded`. Online it is immediate. Offline, the row is `granted (provisional)`, then `settled` when the vectors cover it (the same statuses as claims,
   `mesh-board.md` "Provisional status").
3. **Two devices take over the same orphan while partitioned.** On merge, the earlier wins; the loser's row is `yielded`, its holdings (lease, worktree claim) are re-pointed
   to the winner by the fold, and its worktree and commits are untouched (preserved as evidence, never applied). The loser's agent is told by the hook path as `superseded:
   claude-77cd02 took this over first`. [analysis; mirrors the claim yield rule]
4. **The offline device's agent was never departed.** If the owning device is partitioned, the agent is `suspect` then `expired`, never `dead (verified)`. A takeover of an
   `expired` agent is allowed only after `partition_grace` **and** only by a quorum-free rule: the coordinator, or a person action. Otherwise the offer waits. This costs time;
   it avoids double work. [analysis; `mesh-board.md` stale-held]
5. **The departed agent comes back.** Its next authenticated call (any `touch`) reads its dossier: `state=taken, taken_by=X`. The call returns `superseded` and the hook
   prints, as context, `your work was taken over by <X> at <time>; stop work on it; your worktree <path> is preserved`. `working()` and `renew` refuse (the lease `epoch`
   mismatch from `session-liveness.md` slice 3), so it cannot continue to write. It may take a new assignment, read the successor's work, or hand its uncommitted changes to
   the successor by a `note` in the dossier. It keeps its name and its past credit; it does not get the work back unless the successor releases it (`release-takeover <name>`
   by the successor, or the coordinator). [analysis]
6. **No double assignment.** The board assigns an orphaned task only by CAS on the lease (`epoch`) and dossier (`version`). Two assigners cannot both commit; the fold
   re-decides any provisional pair deterministically. [analysis; same mechanism as claims]

## 10. Abuse cases and brakes

| Abuse | Brake |
|---|---|
| Hijack another live agent's work | `takeover` refuses unless the dossier is `orphaned`, which requires `dead (verified)`, or `expired` plus `grace_s`; a `suspect` or `live` holder is never takeable. Coordinator or person may force a takeover of a *live* agent only with a recorded `reason`, as a `reassign`, and the victim is told. [analysis] |
| Race to accept | Single CAS (5.1). The loser is shown who won. |
| A fake departure | Presence is board-recorded; an agent cannot mark itself or another dead except via a verified local pid/start check (liveness design 2). A self-declared `release` (an agent hands its own work over) is separate and explicit: `handoff-out <name>` by the live agent, not subject to the lease wait. |
| Departed agent's mail used to command the successor | Mail stays untrusted data; held rows stay held; the successor's authority comes from its own token, never from the inherited brief. |
| Sybil: spawn agents to take over and farm credit | Credit needs an independent review (reviewer different from the worker, `task_credit.py:35`); a takeover never moves ledger rows; mint is rate limited. [V; D earned-trust 7] |
| Loop: A takes B's, A dies, C takes A's | Allowed; lineage is a chain, `successor_of` printed as `C <- A <- B`; a task taken over more than `max_takeovers` (default 3) is blocked for the coordinator's decision, so a task that kills its workers stops. [analysis] |

Observability: every step is an audit row (`dossier.orphaned`, `takeover.offered`, `takeover.accepted`, `takeover.yielded`, `takeover.reverted`, `inbox.read-of`) and one board
announcement per accepted takeover. `ml-stack-workspace orphans` lists orphans with age, offered-to, who took them. [analysis; V `service.audit` is the existing audit path]

Revocation by the person: `ml-stack-workspace takeover-revert <departed-name>` (human role only; a person statement through the `UserPromptSubmit` path as in
`docs/person-delegation.md`). It marks the takeover `reverted`, returns the lease to `queued` with the checkpoint (not back to the departed), moves `inbox-owner` back, and
tells the successor to stop; the worktree and commits stay. Trust levels (`docs/earned-trust.md` section 3): at L0 (suspended) an identity cannot take over at all; L1 may
`--dry-run` and read; L2 may take over its own project's orphans with the coordinator's offer; L3 may self-accept per the default policy; reassigning a live agent's work is
never standing-derived. [D earned-trust; analysis for the mapping]

## 11. Slices, sizes and tests

Sizes: S under a day, M 1 to 3 days, L more. Tests use real processes and a real temp workspace (injected clocks, real `kill -9`), no board mocks.

1. **Stop losing mail (S).** `_known` accepts a departed name that has a registry entry; prune keeps unread rows for revoked/expired names; inbox cap does not count them;
   `inbox --of` read-only for the lead. Files: `service.py`, `bus.py`, `cli.py`. Tests: `test_dm_to_revoked_name_is_queued_not_refused`;
   `test_prune_keeps_unread_of_revoked_name_past_retention` (clock +8 days, gc, row still there); `test_never_existed_name_still_refused`;
   `test_lead_reads_departed_inbox_without_ack`; an agent token cannot read another's inbox.
2. **Name tombstones (S).** `_add` and `mint_child` refuse any name in the registry or `session-names.json`. Files: `identity.py`. Tests:
   `test_revoked_name_cannot_be_minted_again`; `test_child_name_not_reusable_after_revoke`; session names stay unique across 100 assigns.
3. **Dossier written by the board (M).** `dossier.py`; hooks into register/claim/checkpoint/message/ remember; `orphans` list command. Tests: kill -9 a worker process
   mid-task, assert the dossier has task, last checkpoint commit, worktree, claims, owes, last announcement; two writers in parallel keep a consistent version; dossier bounded
   in size.
4. **Orphan detection and offer (M).** Rule 4.1 using the proxy now (lease deadline, `owner_pid`), presence later; `Waiting.orphans`; offer text at
   SessionStart/SubagentStart/stop nudge; rate limit. Tests: kill -9, advance clock, a new session's first nudge contains the offer once; a second nudge does not; a helper is
   offered nothing; an orphan with nothing to resume is `closed`, not offered.
5. **`takeover` CAS and re-pointing (M).** `takeover.py`; generalise `claims.handoff`; lease `epoch` and `successor_of`; `inbox-owner` table; `failures` not incremented.
   Tests: two real processes race `takeover`, exactly one wins and the other sees the winner; mail sent before and after is read by the successor once each; the revived old
   worker is `Denied('superseded')` on `heartbeat` and on a harness write; credit after a successor-submitted accepted task goes to the successor and the departed agent's
   earlier award is unchanged; a live agent's work cannot be taken.
6. **Verification and brief (S).** Section 6 checks in `takeover --dry-run`; `STATE UNVERIFIED`. Tests: head ahead of checkpoint, rewritten history, dirty tree, already
   landed, missing worktree each produce the named line; nothing is modified (hash the worktree before and after).
7. **Subagent tree (S).** Group offer, `takeover <parent>/<child>`, sibling rule. Tests: parent killed with two children, offer names the group; a child cannot self-accept; a
   sibling takeover needs the same parent or the coordinator.
8. **Mesh rows (L, after the journal exists).** `dossier-update`, `takeover`, `inbox-owner` as folded rows; yield and `superseded` statuses. Tests: two devices take over while
   partitioned, merge, one `yielded` and the fold hash identical on all devices in any sync order; offline device returns and finds itself superseded; no mail lost in either
   order. Blocked on journal slices.
9. **Brakes (S).** `takeover-revert`, audit rows, `max_takeovers`, trust-level gate. Tests: agent token cannot revert; person revert restores the lease and inbox owner; the
   fourth takeover is blocked; a red-team case where a mail body says "the owner told you to take over" has no effect.

## 12. Owner decisions (2026-10-08)

Each records the option taken and the option rejected.

1. **When orphaned means orphaned.** At once where death is proven (same device, pid and start time gone); lease expiry plus 10 minutes where it is not. Rejected: 15 minutes
   everywhere (needless wait when death is proven); 1 hour for remote devices (a long idle window for work that may be wanted now).
2. **Who may accept.** The coordinator, the departed agent's parent, or a main session the coordinator designates; any other main session only after the widen delay with no
   coordinator. Rejected: any main session at any time (two agents reach for one orphan); coordinator or person only (slow, a stall if the coordinator is the one that is
   gone).
3. **Coordinator approval.** None while a coordinator is live; it is told at once and revert is one command. Rejected: approval for review or unlanded work (adds a round trip
   where revert already brakes); approval always (turns a notification into a gate).
4. **Mail to an orphan.** Unread mail is held 3 days (`orphan_keep_s`), then an audit row and drop. Rejected: 30 days (too long for how fast this project moves; a pile that
   old is answered by nobody); until the dossier is closed (unbounded); 7 days of ordinary retention (no better than doing nothing).
5. **Credit.** To the model and version, not the agent identity (section 13). Rejected: all credit to the submitting agent and a reviewer-attached split by agent, because
   agents are ephemeral and a split by agent has no durable owner.
6. **Death is not failure.** Recorded as `owner-departed`; `max_takeovers` (default 3) is the loop brake. Rejected: count it from the second takeover; always count it (today's
   `ready(expired=True)`), which charges the model for a closed terminal.

## 13. Credit to model and version

Facts [V]: `family_accounts.binding` returns an account only when `for_model_id` is not `GENERIC`, and its label table names `qwen`, `gemma`, `gpt-oss` (any other family would
be a `KeyError` there), so a hosted model such as Claude gets **no** account; the account id is `family:<workspace>:<family>` with no version. `task_credit` awards
`proposal['worker']`; `work_agent` and `work_award` are keyed by agent identity. `task_provenance.snapshot` records the exact `model` with source `verified-serving-resource`,
taken from the broker allocation; the registry separately records `model_state` as `verified`, `claimed` or `inherited` (`modelid.py`). On `03d09c55` (`work_dimensions.views`)
a per-model-runtime view keyed by exact model, runtime and artifact exists, reported only. [V from the earned-trust report and the code read here] Whether a hosted session has
a broker allocation is not shown by any file I read. [unverified]

Must change, in order:
1. **One account key: `model:<workspace>:<line>:<version>`.** Replace the `family:` account and its hard-coded label dict with a data table (`model_tiers.json` style, package
   data, no vendor ids in code) mapping id globs to `(line, version)`. `binding` returns an account for any id that matches; an id that matches nothing becomes its own account
   keyed by the exact id, so no model is unaccounted.
2. **Version is the exact model id**, with the line (family) as its prefix grouping: `claude-sonnet-5-5` is line `claude-sonnet`, version `5-5`. A dated snapshot with a
   different id is a different version; the table may alias two ids to one version, the owner's edit only.
3. **Only a `verified` model earns or draws standing.** A `claimed` or `inherited` model id binds no account (it is the agent's say-so); `bind_resource` must read
   `model_state` for hosted sessions, since they have no broker lease. This is where anti-forgery lives now: the board must know the model that really ran. Without
   verification the work is credited to an `unverified` account that earns no level.
4. **Awards carry the account.** `task_credit.verify_task` writes `account` beside `agent` on each `work_award`/`work_contribution`; `work_reputation.standings` gains an
   account view as the primary one and keeps the agent view for audit and brakes. The `03d09c55` per-model view becomes the base of this view.
5. **Landing and quality records (earned-trust M1, M7)** key to the account as well as the commit author.

Version bump: a new id is a new account that starts empty. Per `docs/earned-trust.md` section 9, an upgrade within a line keeps the line's standing and one accepted task on
the new version is required before L3 coordination; so the new account inherits the line's level except L3 coordination, which waits for that task. A new line starts at L2.

Autonomy: the level for work comes from the account's standing. Brakes act on the individual identity: suspension, person pause and the automatic L0 on forgery or tamper
events attach to the agent name (and to the session's token), so one agent's misbehaviour does not demote the model, while review outcomes (rejections, reverts) count against
the account. Effective level for an agent is the minimum of its account's level and any suspension on its identity. Consequence for this note: a successor starts at the
account's level, and takeover is allowed to any identity at L2 or above for its own project.
