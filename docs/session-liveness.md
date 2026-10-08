# Session liveness, recovery and coordinator promotion

Design note. Nothing here is implemented. Line numbers are against the worktree this note was
written on (`feat/test-reuse` base).

Problem: a closed terminal, a crashed machine or a sleeping laptop leaves claims, task leases,
child tokens and a coordinator that no longer exist. Nothing notices, and nothing hands the work
on.

## 1. What happens today

Facts from code. "Blocks" means what a stale entry stops another agent doing.

### Identity versus session

- A local agent name is stable across terminals. The hooks register `--agent claude`
  (`scripts/hooks/claude-session-start:29-40`), the token file persists
  (`tokens.load`), and the minted token lives `TOKEN_S` = 30 days (`onboard.py:71`). A new terminal
  therefore *is* the old identity locally. It silently owns the dead session's claims (`claims.py:139`
  conflicts only when `other['owner'] != who.id`) and can renew them.
- Only the shared-project path derives a per-session routing identity: `project_session.name()`
  gives `agent-<32 hex of sha256(harness NUL session_id)>` from `ML_STACK_SESSION_ID`
  (`project_session.py:36-39`), exported by SessionStart (`workspace_hook.py:63-90`). There a
  new terminal is a new identity and the old one is orphaned, which is the case the owner sees.
- `ensure_presentation` assigns a stable ordinal to a top-level agent record once
  (`identity.py:186-197`); `register_session` sets `presentation.kind = 'main'`
  (`identity.py:199-218`). Neither is ever cleared. No hook runs at session end: only
  SessionStart, SubagentStart and SubagentStop exist under `scripts/hooks/`.
- There is no per-session presence record. The registry stores `expires` (token lifetime), not
  last-seen (`identity.py:275-280`, `:584`).

### (a) File, area, branch, worktree claims (`claims.py`)

- TTL `claim_ttl_s` = 900 s (`limits.py:33`). `MAX_RENEW_S` 3600, `MAX_LIFETIME_S` 8 h (`claims.py:25-27`).
- `_sweep` (`claims.py:115-124`) drops a claim when `expires <= now` or `pid` is non-zero and
  `os.kill(pid, 0)` says the pid is gone (`alive`, `:38-48`; EPERM counts as alive).
  Sweeping runs only inside a claim/reserve/release/renew/listing call; there is no timer.
- The harness path `reserve` (`claims.py:154-190`) writes `pid: 0` (`:178`). The hook's
  `owner_pid`/`owner_started` are stored (`:180-181`) but `_sweep` never reads them. So for
  every native mutation claim, death is detected only by TTL expiry, and a pid that was reused is
  never an issue because the pid is never checked.
- Refresh: `renew`/`heartbeat` (`claims.py:294-315`) are called only by the explicit
  `heartbeat` command (`harness_remote.py:231-235`). Nothing renews automatically, so a live agent
  that never runs `heartbeat` also loses its claims after 900 s, and a dead one holds them for
  up to 900 s from its last renew (at most 8 h from `since`).
- Blocks: a conflicting claim blocks `claim`/`reserve` with `Conflict` naming the owner and
  expiry (`:140-142`); `inactive_worktree` refuses recovery while any matching file, area or
  install claim lives (`:253-259`). Worst case a dead agent blocks its area for 15 minutes; a
  *renewing* wedged agent blocks it for 8 h.
- `physical_owner` namespaces owner as `canonical:<hash>:<agent>` (`harness_remote.py:43-46`); the
  board-side (`native_reserve`) and physical stores are two separate stores, released
  separately (`release_revoked`, `:173-182`, only on self-revocation).

### (b) Task leases and assignments (`task_actions.py`, `task_scheduler.py`)

- A lease is a graph node: `heartbeat_at`, `deadline = now + HEARTBEAT_S` (120 s)
  (`task_actions.py:19`, `:34-35`). `heartbeat` extends it (`:75-79`). Every worker action goes
  through `working()`, which refuses with "the task lease expired; it must be recovered" once
  `deadline` passes (`:63-66`). There is no epoch or fencing token; the check is time only, and
  the same worker id can continue a lease until someone recovers it.
- Recovery is manual: `ready(expired=True)` (`:192-205`) needs a reviewer (`board._reviewer`),
  refuses a live lease, increments `failures`, deactivates the lease, writes a `recover`
  checkpoint and requeues. Nothing calls it on a schedule. A task whose worker died stays
  `working` indefinitely and blocks its dependents.
- `assign_next` (`task_scheduler.py:19-45`) only renews a *live* child when under a quarter of
  `child_ttl_s`; it does not look at whether the child process exists. Assignment goes to any
  registry-live worker.
- Integration attempts are the one place that checks process liveness properly: it stores
  `owner_pid` + `owner_started` and compares `pid_exists` and `started_at`
  (`task_scheduler.py` `integrate_completed`, the `live` expression) and marks a dead owner's
  attempt `blocked` with `recovery_required`. It does not retry or hand over.

### (c) Child and subagent identities (`identity.py`, `child_renewal.py`)

- A child's liveness is `_live`: not revoked, `expires` not passed, and its parent live
  (`identity.py:275-280`); `authenticate` repeats it (`:584-588`). `child_ttl_s` = 8 h
  (`limits.py:39`), capped by the parent's own expiry (`:371-376`).
- When the parent session dies but its *token* has not expired (30 days), the children are still
  "live" for up to 8 h and keep their tokens. `child_renewal.renew` extends only on a live
  top-level parent's call (`child_renewal.py:10-25`).
- SubagentStop only announces `done` and records the model
  (`scripts/hooks/claude-subagent-stop:36-49`); it revokes nothing. A killed terminal never
  runs it.

### (d) Background jobs

- Test jobs (`scripts/testjobs.py` on `feat/test-reuse`): detached runner,
  `start_new_session=True` (`:130`), status stores `pid` and `started` (`:131`), `state()`
  compares pid + process start before trusting "running" (`:88-95`). The owner is recorded in
  `spec.json`; `cancel` is limited to the owner or holder of a secret (`:161-175`). A whole-tier
  job blocks any other whole-tier job (`Busy`, `:119-123`). The runner outlives the terminal, so
  a job whose owner died keeps running and keeps its slot until it ends.
- `scripts/testslots.py` leases carry `pid` (`:113-116`, `:271-272`) and drop on a dead pid.
- Builds and runtime installs: no `deploy.lock` exists in `src/` or `scripts/` at this commit,
  so installed-runtime ownership is whatever `install` claims (path claims, 900 s) provide.
- The model/GPU broker has its own leases and is out of scope; it already checks its own holders.

### (e) Main-session registration and `coordinator_eligible`

- `agent_display.metadata` computes eligible = no parent, `presentation.kind == 'main'`, role
  `AGENT` (live), and full `CAPS` (`agent_display.py:42-43`); a helper label forces false (`:44-48`).
- Nothing in it reads last-seen. A session closed yesterday whose token is still valid is
  `coordinator_eligible: true` forever (up to 30 days). `kind = 'main'` is never reset.

### (f) The board service

- The "coordinator" in `coordinator*.py`/`coordination.py` is the board RPC *service*: a device
  whose `coordinator.json` has `mode: host` answers `/workspace/v1/*` on the Fleet peer server
  (`coordinator.py:28-34`), others are `remote`. `workspace_id` is the stable workspace node
  (`coordination.py:8-19`). It is configured by the person, not elected, and has no failover:
  remote devices fail explicitly with no local fallback (docs/workspace.md, "Explicit shared
  coordinator across devices").
- This document keeps the two apart: **board host** = the process/device serving the board and
  graph; **coordinator role** = the main session that plans, reviews and lands. They may run on
  different machines. Nothing in code today implements the second.
- `nudge` hook (`nudge.py`, `cli.py:487-510`) runs after tool calls; it reads (`ws.waiting_summary`)
  and writes nothing to the board, and is throttled by a *user-wide* stamp file
  `ml-stack-nudge.<user>` at `POST_EVERY_S` = 20 s (`nudge.py:21`, `:128-131`), not per session.

### Summary of what a stale entry blocks

| Holding | Dead owner detected by | Delay | Blocks |
|---|---|---|---|
| native file/area/worktree/branch claim | TTL only (pid 0) | up to 900 s past last renew, 8 h cap | same area/branch/worktree |
| legacy claim with pid | `kill(pid,0)` at next store call | immediate on next look | same |
| task lease | nobody | forever until a reviewer runs `ready(expired)` | task and its dependents |
| child token | parent token expiry or 8 h child TTL | hours to 30 days | identity count limits (`within`, 64 live) |
| test job | `state()` on query | runner survives owner | whole-tier slot |
| `coordinator_eligible` | nobody | token lifetime | nothing yet, but any election would pick a ghost |
| board host | nobody | until restarted | all remote devices |

## 2. Presence

### Record

One record per session, in the coordinator workspace graph (the source of truth already used for
leases, `coordination.db`), node kind `presence`, id `presence:<session_key>`:

```json
{
  "agent": "claude",                  // registry id (routing identity)
  "session_key": "sha256(harness NUL native_session_id)[:32]",
  "model": "claude-sonnet-5-5", "harness": "claude-code",
  "kind": "main|subagent|helper",
  "parent": "",                       // authenticated parent id; subagents inherit device (labelled inherited)
  "device": {"device_id": "...", "source": "local-observation|enrollment", "confidence": "..."},
  "pid": 41233, "pid_started": "<started_at(pid) string>",
  "host_clock_s": 1790000000.1,       // device clock at write
  "seen_s": 1790000000.0,             // board-host clock at receipt; the only time used to compare
  "interval_s": 30, "state": "active|ending",
  "epoch_seen": 7                     // latest coordinator epoch this session has observed
}
```

`pid` is the long-lived harness process (the hook's parent, the same value `reserve` already stores
as `owner_pid`/`owner_started`, `claims.py:180-181`), not the short hook process.

### Writers (no new daemon)

- SessionStart registers the record (`state: active`) after `main-session`.
- `nudge --hook post|prompt|stop` also does `presence.touch` in the same process, after reading the
  inbox (`cli.py:_hook`). The write is rate limited per session to once per `interval_s`
  (default 30 s, a file stamp keyed by `session_key`, replacing the user-wide `ml-stack-nudge` stamp so
  two sessions no longer suppress each other). At most 2 writes a minute per session. A
  `touch` carries `claims.renew` for that session's claims and `task heartbeat` for its lease, so
  an active agent needs no manual `heartbeat`.
- Stop and a new SessionEnd hook write `state: ending` (best effort; death never runs it).
- Subagents: SubagentStart creates a `presence` record with `parent`; the parent's touch refreshes
  its live children (`child_renewal.renew`, `child_ttl_s`) instead of each running a loop.
- A touch is a normal authenticated call; failure warns and never blocks the tool call
  (workspace_hook already behaves this way).

### What "alive" means

Evaluated by the board host (the single clock):

1. Same device as the host and `pid`+`pid_started` both match a live process: **alive**.
2. Same device, pid absent or start time differs (reuse): **dead, verified**.
3. Remote device: **alive** if `seen_s` within `suspect_s` (default 3x interval = 90 s);
   **suspect** until `dead_after_s` (default 15 min); then **expired** (unverified: it may be
   asleep or partitioned).
4. Only states *dead, verified* and *expired* permit action, and they are different: verified death
   releases at once, expiry waits the grace period (section 3).

Remote devices cannot prove death. A device's own agent (`ml-stack-workspace` on that machine)
may post `presence verified-dead` for local records it can verify, as authenticated per-device
evidence; the host trusts it only for records with that device id.

### New terminal, old session

A resumed or fresh terminal is a **new session**: new `session_key`, new presence record. For
the locally stable name `claude` that would still be the same registry id, so the design requires
the per-session routing identity (`project_session.name`) for all native sessions, not only
the shared-project path. The new session records `predecessor_of: [dead session_keys]` only
when the same person/project/device and the old one is verified dead or expired. This is a link
for the board and for handoff offers, never inheritance: the new session gets no claim, lease,
token or eligibility of the old one. It must claim again, or be offered an explicit
task-scoped handoff (section 3.3). Revoke the old token on declared death.

## 3. Recovery of a dead session's holdings

Principles from AGENTS.md: verified or expired, never inferred from a pid; never steal a live claim;
keep checkpoints and exact task/resource identity; never delete unique work.

### 3.1 Who performs it

- The **coordinator** runs `recover` on every sweep (every presence touch it makes, and a
  60 s timer inside its `watch` loop, `task_scheduler.watch`).
- **Self-service path** (no live coordinator): any live *main* session may run
  `ml-stack-workspace recover --dry-run|--apply` for holdings that are already *dead, verified* (not
  merely expired). It acts under its own token, takes the `recovery` claim (below), and is
  limited to releasing claims and requeuing leases. It cannot reassign worktrees or land work.
  A claim that merely expired is already released by the sweep with no actor.
- Subagents and helpers never run recovery (they are not coordinators).

### 3.2 Per holding

State machine for a presence record: `active -> suspect -> expired` (or `-> dead` directly when
verified), `ending -> gone`. A revived record returns to `active` only via a touch that presents
the current `epoch` of each holding (fencing).

- **Claims.** Each claim gets `holder_session` and `holder_epoch` fields (`reserve`, `claims.py:178-184`).
  `_sweep` additionally reads `presence`: a claim whose `holder_session` is *dead* is released
  immediately; *expired* after the grace period. `owner_pid`/`owner_started` finally take part:
  `_sweep` checks `alive(owner_pid)` and start time when `pid == 0` and `owner_pid` is set.
  A claim whose holder is merely *suspect* is never released.
- **Task leases.** Add `epoch` (integer, incremented on every claim/recovery) to the lease
  (`task_actions.claim`, `:34-35`). `working()` requires the caller's presented `epoch` equal to
  the stored one in addition to the deadline check (`:63-66`), so a revived worker whose lease
  was recovered gets `Denied('lease superseded')` and must stop. Recovery for a dead holder is
  the existing `ready(expired=True)` logic made callable by the coordinator without a reviewer
  token, with: `failures` not incremented (death is not task failure), the previous worker and
  `reason: owner-dead` recorded on the checkpoint, the checkpoint and evidence nodes left linked,
  task `queued` with `previous_owner` and `resumable_from: checkpoint:<id>`. Assignment prefers
  a successor given the checkpoint.
- **Child identities.** When a parent session is *dead* or *expired*, the host revokes its
  descendants (`Registry.revoke_tree` semantics, `identity.py:391-400`, run as the system, not as
  the parent): "child tokens die with the parent session". Today they would outlive it up to 8 h.
  Their presence records go `gone`; their claims and leases are recovered as above.
- **Worktrees and branches.** A claim on a worktree/branch is released like any claim, but the
  checkout is never touched. `worktree_lifecycle.pending` already lists scopes with reasons
  (`worktree_lifecycle.py:153-190`). Add a `reasons` entry `owner dead` and a state `orphaned` on
  the lifecycle scope with `orphaned_from` (session), `dirty` (porcelain), `unlanded` commits.
  The coordinator posts one announcement per orphan and offers it to a successor through a
  task-scoped handoff (`claims.handoff`, `claims.py:192-206`, extended so the old owner may be a
  dead session recorded in the scope instead of the caller's parent). A handoff names the exact
  task assignment; absent one the orphan waits, owned by nobody, and `cleanup` refuses it
  (`worktree_lifecycle.cleanup` requires a live claim by the same worker, `:127-130`).
  `require_clean` on the successor does not see the orphan. Unique commits, dirty and ignored state
  are preserved.
- **Background jobs.** By kind, in the job `spec.json` as `owner_session`: test jobs are
  *adopted* by default (the run is useful and holds a slot): the job is re-owned by `system:<epoch>`, still
  cancellable by any live main session, and ends on its own; a whole-tier job whose owner is
  dead and which has not progressed (log mtime) for `stall_s` is cancelled. Builds and
  runtime installs are *cancelled* only when their target path claim is released
  and the install has no verified completed marker; a half-written environment is renamed to
  `*.orphaned`, never deleted. The background full suite (AGENTS.md: one at a time) is adopted.
- **Main-session registration.** On dead/expired the host sets `presence.kind` effective state to
  `gone` and `coordinator_eligible` false in `agent_display.metadata` (it reads presence). The
  registry record is left (audit), and ordinals are not reused.

### 3.3 Grace and fencing

- `grace_s` after *expired* (remote only; default 10 min on top of `dead_after_s`) before any
  recovery. After *dead, verified* there is no grace for claims and none for leases.
- Every recovery action is conditional on the exact record it read: it writes with a
  compare-and-set on `(holder_session, epoch)` under `coordination.lock` (as `_rollback` does for claims,
  `harness_remote.py:87-94`). If the holder touched in between, the action fails and nothing
  changes.
- A revived session (laptop wakes) calls `touch`, which returns the list of its holdings that were
  recovered. The hook prints it to the model as additional context (data, "your lease for task X was
  recovered; stop work on it"), and `working()`, `reserve` and `renew` refuse with `superseded`.
  Mutations through the harness check the claim before approving (`harness_remote.conflict`), so
  the revived agent cannot silently write into an area now owned by another.
- Announcements: one line per recovery batch (`announce milestone "recovered 3 claims, 1 lease from
  claude-8f3a1c (dead)"`), by the coordinator, within the existing 6/10 min limit
  (`limits.py:56`); detail lands in a note linked by number.

## 4. Coordinator role

### Separate from the board host

The board host (`coordinator.json` `mode: host`, `coordinator.py`) is infrastructure the person
configures. The coordinator **role** is a lease node in its graph:

```json
{"id": "coordinator-lease", "kind": "coordinator-lease", "project": "...",
 "holder": "agent-id", "session_key": "...", "model": "...",
 "epoch": 7, "acquired_s": 0, "renewed_s": 0, "deadline_s": 0,
 "origin": "elected|designated|handback|spawned", "nominated_by": ""}
```

`epoch` increases by exactly 1 on every change of holder (CAS under `coordination.lock`). All
coordinator-only writes (landing, recovery, assignment, reviews of batches) carry the epoch and are
refused on mismatch: this is the fencing token. `deadline_s` = last renew + `lease_s` (default 3x
presence interval = 90 s). The holder renews on each presence touch, so renewal costs no extra call.

### Eligibility (all must hold, evaluated at the board host)

1. `kind == main` and no parent; role `AGENT` with full `CAPS` (the existing `agent_display.py:42-43` test).
2. Presence *alive* (section 2 rule 1 or 3 and not suspect).
3. Registered model and device profile present (`model_state` recorded, device not unknown).
4. Model tier is not the lowest tier of its vendor family and the model id matches the tier
   table (section 5). Fail closed.
5. No `revoked` coordinator designation by the person (section 4.3).
6. Not a subagent, helper label or transient (`label_models`, `agent_display.py:44-48`).

`coordinator_eligible` in `agent_display.metadata` becomes this computed value (it takes the
presence and tier table as inputs), replacing today's static `kind == 'main'`. Label and parent
checks stay.

### Promotion rule (deterministic)

Triggered when the lease is absent, `deadline_s` passed, or the holder's presence is *dead, verified*.
Any eligible main session may run it; it runs inside its next presence touch, so no election
traffic. A promotion is one CAS: read lease at epoch `n`, write epoch `n+1`, fail if `n` moved.

Winner order among eligible candidates (each candidate computes it from the same graph rows, so
all agree):

1. A person designation (section 4.3), if that session is eligible.
2. A nomination recorded by the outgoing coordinator before it ended (`handoff` record).
3. Highest tier rank in the tier table.
4. Longest continuous presence in this workspace (earliest `registered_s`).
5. Lowest lexicographic `session_key`.

Only the winner may CAS; a candidate that is not the winner does nothing (it recomputes next touch,
so if the winner's presence goes suspect before it acts, the next candidate wins at the next tick).
Two simultaneous CAS attempts: one succeeds at `n+1`, the other fails and observes the new holder.
Announcement: `announce milestone "coordinator: <display> promoted, epoch 8 (previous <display> dead)"`.

- **Zero eligible sessions:** the lease stays vacant. Recovery of verified-dead holdings still
  runs through the self-service path, claims keep expiring, nothing else is promoted. When only
  lowest-tier mains exist, section 5.3 applies. The board lists `coordinator: none` in `status`.
- **Partition:** the board host is the single authority. A device that cannot reach it has no lease
  to read, takes no promotion, and its sessions keep working in "open mode" (below) but their
  coordinator-only calls fail. If the board host itself is partitioned from the rest, sessions on
  its side form the only electorate; the other side never elects, so two coordinators never exist.
  The lease and the board have one writer by construction; what is not designed here is failover
  of the board host (open question 1).
- **Eligible session joins while another is coordinating:** no preemption, except for tier
  demotion (section 5.4). A joining peer becomes a candidate for the next vacancy.

### Person designation and revocation

The person acts only through chat statements recorded by the `UserPromptSubmit` hook
(`docs/person-delegation.md` sections 3-4). A statement such as "make the Mac session the coordinator"
is interpreted into an authorization of kind `coordinator-designate` naming a session by its
display name (or "this session"); `consume` (person-delegation section 4) verifies the attestation. The
designation is a record the winner order reads (rule 1) and may force an immediate CAS if the named
session is eligible. "Stop X being coordinator" records `coordinator-revoke`, which makes the
session ineligible (rule 5) and, if it holds the lease, ends it with `origin: revoked`. Promotion
itself is an ordinary agent action under the agent's own token: no human credential is used, no
agent can write a `person-attestation` (person-delegation section 6), and an agent's own message
claiming the owner said so has no effect. A designation of an ineligible (lowest-tier or unknown) model
is recorded but not honoured and the board says why; it is the owner's table to edit (section 5.1).

### Reconstructing state

The new coordinator needs nothing from the dead one's context:
- the task graph, leases, checkpoints, proposals, reviews (`coordination.db`);
- claims (`claims.json`) and lifecycle scopes (`worktree-lifecycle.db`);
- the board, HANDOFF.md and `git worktree list`/branches in the primary checkout;
- the previous coordinator's in-flight record `landing:<id>` (section 6).
On promotion it runs `recover` once, reads `inbox`, posts one summary announcement, then resumes
`watch`.

### What works with no coordinator

Reading and posting; `inbox`/`wait`; `claim`/`reserve` and renew; task claim in open mode (a
worker with an allocation may claim a queued task); working-lease heartbeats; the board service.
What needs the coordinator: assigning, reviewing batches, landing, `recover` beyond verified death,
worktree handoff offers, retention of the lease.

## 5. Model tiers and eligibility

AGENTS.md rule (`0.2dev`): the lowest tier of a vendor (Haiku, Luna and the like) never coordinates; a model
absent from the table is not eligible; with only lowest-tier agents, one spawns a higher-level
subagent to coordinate; it hands back when a higher-tier main session joins. Tier is read from the
exact model id in the registry through a maintained table, never from a display name or label.

### 5.1 The table

Data file `src/ml_stack/workspace/model_tiers.json` (package data, in the wheel), edited by the owner and
changed only through a normal commit; there is no runtime write path an agent can reach.

```json
{"version": 1,
 "families": [
  {"family": "<vendor family name>",
   "tiers": [
     {"rank": 1, "lowest": true,  "patterns": ["<exact-id glob>", "..."]},
     {"rank": 2, "lowest": false, "patterns": ["..."]},
     {"rank": 3, "lowest": false, "patterns": ["..."]}]}]}
```

Rules: a model id matches a tier when it equals or globs (`fnmatch`, no regex) one pattern after
`modelid.clean_model` normalisation; the first family and tier that match win; a pattern may not
appear in two tiers (loader refuses the file). `lowest` is declared per family on exactly one tier.
Higher `rank` is higher tier. No vendor name or model id is written in code; tests load a fixture
table with invented family names. The loader is `workspace/model_tiers.py` (`tier_of(model_id) ->
Tier | None`, `Tier(family, rank, lowest)`), consulted by `agent_display.metadata` and by promotion.

### 5.2 Unknown and unverified models

`tier_of` returning `None` (no pattern, empty model, family missing) means ineligible. Fail
closed. The registry stores `model_state` as `verified` or `claimed` (`modelid.py`); a claimed model id is
what the agent *said* and is therefore also not eligible for the lease. Eligibility requires the
model be `verified`, or `claimed` and the person has designated that session (the person's
statement is the verification). This keeps "the model is a label, never a right"
(docs/workspace.md, "Which model is it"): a tier grants no permission, it only removes
candidacy. Inherited model labels on subagents never make them eligible (rule 6).

### 5.3 All main sessions are lowest tier

1. Presence shows no eligible main session and at least one live lowest-tier main session `M`.
2. After `grace_s`, the board posts the vacancy. `M` (the session running the nudge hook, the
   lowest `session_key` of the lowest-tier mains if several) spawns a subagent through its harness
   (Agent tool) at a model `tier_of(...)` accepts, with a brief that names the coordination role,
   the parent, the epoch and the fencing rule. Who spawns: the lowest-tier main only; the hook
   adds an `additionalContext` instruction `coordinator vacancy: spawn a coordinating subagent`.
   Others do nothing; a spawn is announced and recorded as `coordinator-spawn` with `spawned_by`.
3. The subagent registers as the lowest tier's delegate, not as a main session: SubagentStart
   gives it a normal child identity with `parent = M`. It does not elect itself. `M` posts a
   `coordinator-nominate` record (an authorised nomination by its parent, naming the child
   session and the epoch) and the subagent's lease CAS is allowed only if that nomination exists,
   is by its authenticated parent, and the child's verified model passes the tier check.
   The lease row carries `origin: spawned`, `nominated_by: M`, `lease_epoch`.
4. This is the only case in which a non-main identity holds the lease. Its authority is
   the lease, not its parent's rights. `M` cannot reuse the nomination for another child; one
   spawned coordinator per vacancy.
5. Hand-back: when an eligible higher-tier main session `H` has presence *alive*, the spawned
   coordinator's lease is demoted (section 5.4); it finishes its current atomic step, writes a
   `handback` summary note, and releases. A spawned coordinator that is *dead* (parent's terminal
   closed) goes through the same promotion: the parent is dead too, and its child tokens died with it.
6. If `M` dies, the spawned coordinator dies with it and the cycle restarts.

### 5.4 Demotion mid-flight

A higher-tier eligible session `H` appearing does not preempt a legitimately elected (rank-ordered,
non-spawned) coordinator of equal or higher tier. It demotes a *spawned* one, or a coordinator whose tier is
no longer eligible (the table was edited). Mechanics:

- `H`'s next touch finds `lease.origin == spawned` and `H` eligible; it CASes a `demote_request`
  field (not the epoch). The holder sees it on its next touch (its section 4 renew) and
  must stop starting new batches, finish or checkpoint the current landing step (section 6), post
  the summary and release; the vacancy then promotes `H` by the normal rule (the nominee ranking
  has `H` first on tier). A holder that does not release within `demote_grace_s` (default
  5 min, longer than a landing step) is treated as wedged (section 6).
- Promotion in flight: if a candidate computed itself winner and the table changes, the CAS
  recomputes eligibility inside the transaction and fails if the winner is no longer eligible.
- A coordinator whose model is later edited to the lowest tier in the table is demoted the same way
  at its next touch; the table is read per touch (cached by mtime).

## 6. Failure modes and tests

Tests are real processes against a real temp workspace (`ml-stack-workspace` with `limits.root()`
pointed at a tmp dir, a real `Claims`, `GraphStore`, hook scripts as child processes, no mocking of
the board). Children exported `PYTHON_KEYRING_BACKEND=onboard_support.FileKeyring` and
`ML_STACK_NO_REAL_KEYSTORE=1` as the repository requires. Clocks: `Claims(clock=...)` and
`Registry(clock=...)` already accept injected clocks (`claims.py:99`, `identity.py:95`); tests
advance them instead of sleeping, but the process kill is real.

| Failure | Behaviour | Test (must go red if broken) |
|---|---|---|
| Split brain | Single CAS writer; epoch fences coordinator-only writes | two real processes race `promote`; exactly one gets epoch n+1; the loser's landing call with epoch n is `Denied` |
| Two simultaneous promotions | CAS on `(lease, n)` | N=8 candidates in threads/processes; final lease epoch exactly +1 and one `announce` |
| Coordinator alive but wedged | presence touch comes from the hook thread, not the work. Add a `progress_s` field updated only by coordinator actions (a landing step, review, assign, recover). Lease renew requires `progress_s` within `wedge_s` (default 30 min) *or* an explicit `idle` note; otherwise the holder is *suspect*, announced, and after `demote_grace_s` a candidate may promote | start coordinator child, SIGSTOP its work loop but keep hooks (a child that only touches); assert promotion after `wedge_s` and old epoch fenced |
| Clock skew across devices | only `seen_s` (board-host clock) is compared; device `host_clock_s` is recorded but never used for decisions; skew > 60 s is reported in `status` | a remote record with device clock +1 h: liveness uses receipt time; `status` flags skew |
| Pid reuse | process start time must match; `pid==0` claims check `owner_started` | spawn a process, record pid/start, kill it, start another with the same pid via a stub (or compare against a different start string); the claim is dead, not alive |
| Laptop asleep an hour | remote *suspect* → *expired* → grace; on wake `touch` returns `superseded`, work stops | write presence aged 61 min, run a recovery sweep, wake and call `heartbeat` on the old epoch: `Denied('lease superseded')`; its mutation attempt through the harness claim check is refused |
| Mass death (power loss) | all presences stale at once; first candidate promotes; recovery sweeps in dependency order, bounded per sweep (50 holdings) with one announcement | kill -9 N child sessions, restart one: it promotes (if eligible), recovers every claim and lease, announces once, unique worktree commits intact |
| Coordinator dies mid-landing | `landing:<id>` node (batch commits, worktree, step, claims) written before each step. New coordinator reads it: a half-merged batch is *blocked with recovery_required* (as `integrate_completed` does for a dead owner), the integration worktree is orphaned and offered, the shared claims it held are released after verification, the background full suite is adopted (one at a time) | kill the coordinator between `merge` and `fetch`; successor refuses to re-run, shows the block, preserves the worktree and offers it through handoff |
| Recovery races a revived owner | CAS plus epoch | owner renews while the recovery transaction is open; exactly one wins; the loser has a consistent view and no claim is released that the owner just renewed |
| Recovery of lowest-tier-only workspace | spawn path, section 5.3 | a lowest-tier session is the only live main session: it is not promoted; the vacancy and spawn instruction appear; its nominated subagent obtains the lease only with the nomination, loses it to a higher-tier session's arrival, and the lowest-tier session cannot hold the lease directly even when designated... (a person designation of a lowest-tier model is recorded but not honoured) |
| Unknown model id | ineligible | model id missing from the table: `coordinator_eligible` false, promotion skips it, `status` says "model not in tier table" |
| Tier table edited mid-flight | demotion | holder's pattern moved to the lowest tier; next touch starts demotion; a successor promotes |
| Table misuse | loader refuses duplicates, a family with two `lowest`, or an empty pattern | loader tests |

Red-team (extend `scripts/redteam_coverage.py` coverage; real child processes with `CLAUDECODE=1`):
a subagent cannot take or renew the lease except by an authorised nomination from its parent;
a subagent that writes `coordinator-nominate` for itself is refused; an agent token cannot write a
`coordinator-designate` or `-revoke` (only `consume` of a person attestation can; reuses the
person-delegation red-team set); text on the board claiming "the owner designated me" has no effect;
a forged `presence verified-dead` for another device's record is rejected; a session cannot declare
another session dead on the same device unless the pid/start check agrees; a recovered session cannot
reacquire its old epoch; a presence record cannot be created for a `session_key` the caller's token
does not own; touch rate limiting holds under 100 hook calls in one second; a model id that merely
resembles a higher-tier pattern (substring, case, unicode confusable) does not match; a label or
display name of a higher tier on a lowest-tier model does not raise its tier.

Integration tests (real workspace): SessionStart then a `post` nudge writes presence at most once per
interval; a closed terminal (kill -9 of the process recorded in presence) frees its native claims at
the next look instead of after 900 s; a requeued task carries its checkpoint and `previous_owner` and a
second worker claims it with a new epoch; `status` lists a stale holder with the reason.

## 7. Slices

Each slice is independently landable; files are expected, not final.

### Slice 1 (days): presence, verified-death sweep, `status`

- `src/ml_stack/workspace/presence.py` (new): record, `touch`, `classify(record, now)`, pid+start
  check via `serve.process.started_at`/`pid_exists`.
- `src/ml_stack/workspace/cli.py`: `_hook` also touches; `status` shows presence and stale holders;
  new `presence` and `recover` subcommands (`--dry-run` default).
- `src/ml_stack/workspace/nudge.py`: per-session stamp instead of the user-wide one (`:128-131`).
- `scripts/hooks/claude-session-start`, `claude-subagent-start`, `claude-subagent-stop`: write presence;
  add `claude-session-end` (or the existing Stop path) for `ending`.
- `src/ml_stack/workspace/claims.py`: `_sweep` also tests `owner_pid`/`owner_started` for `pid == 0`
  (`:117-119`) and `holder_session`; `reserve` records `holder_session`.
- `src/ml_stack/workspace/task_actions.py`: `recover_dead(board, ws, lease)` requeues with checkpoint,
  no failure increment; recording `previous_owner`.
- `src/ml_stack/workspace/agent_display.py`: `coordinator_eligible` also requires live presence.
- `src/ml_stack/workspace/harness_remote.py`: pass `holder_session`; use presence in `conflict`.
- Docs: `docs/workspace.md` section on session liveness; tests as in section 6 for claims, leases,
  presence, hooks.
- Does not need a coordinator: a verified-dead sweep runs from any session's touch.

### Slice 2: automatic renewal and child cascade

Touch also renews claims/leases/children; revoke descendants of a dead parent. Files: `presence.py`,
`claims.py`, `child_renewal.py`, `identity.py` (system-actor revoke tree), `task_actions.py`.

### Slice 3: fencing epochs on leases and claims

`epoch` on leases and claims, `superseded` result in `working()`, `reserve`, `renew`; the touch
returns recovered holdings to the hook. Files: `task_actions.py`, `claims.py`, `harness_remote.py`,
`scripts/hooks/workspace_hook.py`, `nudge.py` output.

### Slice 4: tier table and eligibility

`model_tiers.json`, `model_tiers.py`, `agent_display.py` (`coordinator_eligible`), `docs/workspace.md`,
loader and pattern tests, unknown/fail-closed tests. No lease yet.

### Slice 5: coordinator lease, promotion, designation

`coordinator_lease.py` (CAS, epochs, promote), `person_*` authorization kinds
`coordinator-designate/-revoke` (after the person-delegation `consume` slice), `cli.py`
(`coordinator who|promote|status`; keep the existing board-host `coordinator` command under a name that does not clash:
it is `coordinator status|list` today, so the lease commands go under `lead`), coordinator-only calls take
the epoch (`task_scheduler.py`, `task_integration.py`, `task_authority.py`). Fix the naming clash in docs
(board host vs coordinator role).

### Slice 6: orphan worktrees, background jobs, landing record

`worktree_lifecycle.py` (`orphaned` state), `claims.handoff` for a dead owner, `scripts/testjobs.py`
(`owner_session`, adopt/cancel by kind), `task_scheduler.py` (`landing:<id>` node), spawned-coordinator
path and demotion (section 5.3-5.4), hooks adding the vacancy instruction.

## 8. Open questions for the owner

1. Board host failover: if the machine hosting `coordinator.json` dies, everything stops. Should
   another enrolled device be able to take the host role (needs a replicated graph and a person
   action, since it is a system setting, human-only), or is a down board host acceptable?
2. Local identity: should every native session get its own routing identity (`project_session.name`)
   on the local board too, ending the "same agent id after reopening" behaviour? Required by section 2;
   it changes every `--agent claude` call site.
3. Default timings (`interval_s` 30, `lease_s` 90, `dead_after_s` 15 min, `grace_s` 10 min,
   `wedge_s` 30 min): are these acceptable for a laptop that sleeps overnight, or should an idle
   session's expiry be longer than a working one's?
4. Should a task whose worker died count against `max_retries`? The design says no (death is not
   task failure); cost: a task that kills its worker's machine repeatedly never exhausts retries.
5. A claimed (unverified) model with a person designation: honour the designation as verification
   (section 5.2), or require a launcher-verified model id always?
6. Test jobs of a dead owner: adopt (design) or cancel? Adoption keeps a possibly wanted result and
   the whole-tier slot held until it finishes.
7. Tier-table membership of the models actually in use today; the table's content (which ids are
   in which tier, which family's lowest tier) is the owner's, and no id is proposed here.
