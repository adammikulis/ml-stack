# Board host failover, offline operation and the two coordinators

Design note; nothing is implemented. Line numbers are against base `feat/test-reuse`. It extends
`docs/session-liveness.md` (§4 lease, open question 1) and uses the channel of `docs/person-delegation.md`.
Owner decisions: another registered device can take over hosting with a person's approval; a device that loses
its network keeps working from its own copy of the board and resyncs; there are two coordinator scopes.

## 0. Where the code does not support this

- One writer by construction: the process answering `/workspace/v1/call` while `coordinator.json` says
  `mode: host` (`coordinator.py:24-34`). Other devices have no local copy; `mode: remote` fails explicitly with
  no local fallback (`docs/workspace.md`, "Explicit shared coordinator across devices";
  `project_connection.py:33`, `:189`). Offline-first changes that for one purpose: a local replica plus a
  journal merged later. It is not a second authority.
- Missing, each a slice below: an epoch anywhere (`coordinator_config.py:9-18`); a whole-board export
  (`board_graph_merge.export` covers only the `bus` stream, omits memberships, is person-only `boardapi.py:65-80`,
  bounded to 10000 events / 16 MiB `board_graph.py:20-21`); a hybrid logical clock; a per-device journal; a host
  lease; a change log for `coordination.db`; keyed chains (workspace chains are unkeyed, `chain.py:57-60`).
- A split-brain door today: an agent token can set `mode: host` on a device with no config
  (`coordinator_routes.py:48-53`, `coordinator_bootstrap.py:11-26`). Slice 1B closes it.
- `person_record`/`consume` (chat authorization) is not in the tree; until it lands approval is a terminal
  `HumanGrant` (`sentinel/human.py:35-66`). The model tier table (`session-liveness.md` §5) is not in the tree
  either; the tier rule falls back to "main session without a helper label" until it lands.
- Not verified: whether ladybug store files move between macOS, Windows and WSL builds; the bundle is a
  logical dump (`GraphStore.read()` `store.py:296`, `write()` `:143`). Windows and WSL need a test there.

## 1. Two coordinator scopes

### What session-liveness.md conflated

§4 defines one lease node, `coordinator-lease`, in the workspace graph, and §3 gives that coordinator every
recovery duty. The lease lives in the board host's graph, so a device that cannot reach the host has none to
read; the partition paragraph (lines 335-340) leaves its sessions in "open mode" with coordinator-only calls
failing, including recovery of a closed terminal's claims and worktree, which needs no other device. §3.1 runs
`recover` for everything: a pid-verified claim on this device (one machine can decide) and a task lease held by
a remote worker (needs one writer) are one duty there.

### Why a lease exists at all

The host serializes *writes to storage* (`coordinator-rpc.lock`, `coordinator_calls.py:68`). It does not
decide *which decision to make*; agent sessions do. Two sessions each acting as coordinator produce: a
task assigned to two workers, two landings racing the shared development branch, recovery requeueing a
lease whose worker is alive, two promotions of one standby. A lease with an epoch makes the second
writer's coordinator-only calls refusable. That is its only job.

### Device coordinator

A role every device has at all times, with no network.

- Holder: the device's best eligible main session (`session-liveness.md` §4 eligibility; then highest tier,
  earliest registration, lowest `session_key`).
- Acquisition: lock file `device-coordinator.lock` in the device state root holding `{session_key, pid,
  pid_started, model, tier, device_epoch, acquired_s}`. A session's presence touch takes it under an
  exclusive file lock (`chain.held`, `chain.py:41-54`) when it is absent or its holder fails the check
  `pid exists and started_at(pid) == pid_started` (`serve.process.started_at`, the check of
  `session-liveness.md` §2 rules 1-2). Taking it adds 1 to `device_epoch`. Nothing is elected or sent.
- Tier rule, both scopes: the lowest tier of a family never holds it. A device whose only main sessions are
  lowest tier spawns a coordinating subagent (AGENTS.md; `session-liveness.md` §5.3: nomination by the
  parent, verified model passes the tier check, one per vacancy).
- Authority, device-local resources only: local worktrees and lifecycle scopes, ports, test slots and
  jobs, local agents' in-progress tasks, sweeping stale local claims by the pid+start check (`claims.py`
  `_sweep` ignores `owner_pid`/`owner_started` for `pid == 0` today, `session-liveness.md` §1(a)), handing
  a dead local session's worktree to a successor on this device, revoking a dead local session's children
  on this device's registry copy (provisional).
- It may act on shared resources while partitioned; those decisions are provisional (section 3). When
  connected it is subordinate: shared decisions go to the global coordinator, and it executes that
  coordinator's instructions about this device.

### Global coordinator

The `coordinator-lease` of `session-liveness.md` §4 (CAS, epoch +1 per holder change, winner order),
restricted to decisions needing one writer: the shared task queue and assignment, shared branch and area
claims, landing and pushing the shared development branch, cross-device recovery, promotions, relaying
token mint and revoke requests.

- Held by one eligible main session on the side holding the board host. A partitioned minority never
  elects (no host to write the lease to). An offline device never elects one.
- `lease_epoch` and `board_epoch` (section 5) are separate counters. A takeover raises `board_epoch`, vacates
  the lease and continues `lease_epoch` from the replicated value plus one, so no old holder's epoch validates.
- Lost mid-landing: `landing:<id>` records (`session-liveness.md` §6) travel in the cut; after a takeover an
  unfinished landing is `recovery_required`, never resumed (its checkout may live on the dead host).

## 2. What the board consists of

Observed on this device on 2026-10-08 (`ls ~/.ml-stack/workspace`, `ls -la`; read-only): `board.db`
10.4 MB, `coordination.db` 8.8 MB.

| Part | Where | Replication |
|---|---|---|
| Host process | daemon peer server (`ml-stack-headless --port 8770`), route `/workspace/v1/*`; answers only with `mode: host` | none; the standby runs its own daemon |
| Host selection | `coordinator.json` | per device; rewritten by takeover |
| Workspace identity | node `workspace:<hex>` in `coordination.db` (`coordination.py:8-19`); clients refuse a changed ID (`coordinator_client.py:35`) | cut; this continuity is what a takeover keeps |
| Tasks, leases, checkpoints, reviews, call outcomes | `coordination.db`; `coordinator-call` nodes (`coordinator_calls.py:67-90`) | cut |
| Messages, boards, memberships, subscriptions, cursors | `board.db`; immutable events, per-origin chains (`board_evidence.py:43-68`), origin = `board-scope.replica` (`board_graph.py:42-62`) | cut plus origin-chain tail |
| Audit, notes, quarantine | `audit.jsonl`, `notes.jsonl`, `quarantine.jsonl` (`ChainLog`, `base` row supported `chain.py:82-83`) | cut plus tail by `seq` (`ChainLog.after` `:152`) |
| Token registry, device bindings, invites | `agents.json`: `sha256(secret)` per agent (`identity.py:86-88`, `:570-589`); `device-sessions.db` (`device_sessions.py:26-58`); `invites.json` | cut; a copy authenticates every existing token unchanged |
| Claims, lifecycle scopes | `claims.json`, `worktree-lifecycle.db`; `pid` is meaningful only on the writing device | cut, rebased on takeover |
| Posted files | `files` store, encrypted under a keystore key (`filestore.py:1-30`) | **not replicable**: ciphertext without the key (open question 5) |
| TLS identity | per-device self-signed certificate pinned through pairing or signed beacon (`tls.py` docstring; `coordinator_client.py:84-91`) | **never**; clients already pin the standby's certificate |
| Pairing ledger | `~/.ml-stack/onboard/devices.json`, `peers.json` (pairwise `device_secret`, `signing_key`; `Device.mine` `requests.py:109-125`) | **never** (secrets); precondition for a standby |
| Project boards (shape B) | `shared-workspaces/<project>`, `projects.json` `board_host`, checkout authority metadata | later slice |
| Client side | `coordinator.json` `mode: remote`, `remote-sessions.db`, `tokens/` | per device |

Shape A (one workspace per cluster) is designed here. Shape B (one board per project, `remote.py:24-61`)
needs the extra work listed in slice 6.

### Replicate how

Log shipping exists for append-only parts only. Ladybug files are single-owner stores with an internal
WAL; cloning the main file alone copies a stub (`snapshots.py:1-34`); no page or WAL shipping is used in
this repository. So:

- **Cut**: a bundle of logical dumps (`GraphStore.read()` per graph) plus the JSON files and `ChainLog`
  files, taken store by store, each under its own lock, in one fixed order this note defines:
  `coordination.lock`, `board-graph.lock`, `agents.lock`, `claims.lock`, then log guards (no global order
  is documented today; slice 1B adds a deadlock test). Each store is internally consistent; cross-store
  skew is tolerated (an agent registered after the registry was cut, posting to a board cut later, fails
  auth on the standby and re-runs `ensure`). The manifest lists each part's SHA-256, row counts, schema
  and ml-stack versions, `board_epoch`, `generation` and the host's Ed25519 signature. About 20 MB per cut
  today; cut only when `generation` (bumped by every mutating call and local write) moved, at most every
  60 s, at least every 5 min while changing.
- **Tail**: chain rows after the standby's head, verified link by link (`_Seen.take` `chain.py:73-92`,
  `verified` `board_evidence.py:43`), at the same cadence. Bounds message loss to the interval.
- **Apply**: into a staging directory, `GraphStore.check()` (`store.py:305`) and chain verification, then one
  rename to switch generation; a failed cut is kept aside and the previous generation stays.
- **Staleness tolerated**: messages and notes, minutes. Tasks, leases, claims are rebased with a grace window on
  takeover, so seconds to minutes costs a re-heartbeat. A revocation newer than the cut is lost on the standby;
  the person re-applies it (the audit tail names the gap).
- **Ladybug**: one read-write opener per file; readers use `read_only=True` under the shared `board-graph.lock`
  (`board_graph.py:97-110`); import writes a fresh store with `write()`; `docs()` costs a query per document
  (`store.py:280-294`), paid at the cut interval.

### Chains, the keyed HMAC and key placement

No chain key travels: a key on a standby lets it forge the host's history.

- Each device seals its own journal and origin chain with its own local key (`sentinel/sealed.py:1-60`
  pattern; the unified keyed chain of `person-delegation.md` §10a when it lands).
- Cross-device integrity is hash links plus an Ed25519 head signature by the device's signing key, checked
  against `signing_key` in the peer book (`onboard/trusted.py:pinned_keys`). Signing heads and grants is a new
  use of a key that signs file manifests today; it needs review.
- A merged log is re-sealed by the receiving host; events keep their per-origin chains, so who wrote what stays
  provable. A takeover starts a segment: a `ChainLog` `base` row at the replicated head, then a
  `board.takeover` row (epoch, host fingerprints, head hash, approval id).
- The imported `board-scope.replica` is the old host's origin; the new host writes a new uuid and re-seals
  (`board_evidence.seal`/`validate` `:71-108`). Appending under the old id forks the old chain
  ("combined origin history forks", `board_graph_merge.py:251-252`).

## 3. Offline operation

### The local replica

Each device keeps `board-replica/` per followed workspace: the last cut (read mirror, written only by the
sync process, imported like a standby's) and a `journal`, a device-origin chain of everything its agents
did since the last sync. It holds no authority. When the host is reachable `coordinator_client.client`
routes as today; when not, the client layer answers from the replica instead of raising
(`coordinator_client.py:28-37`; routing at `cli.py:101`, `:747-823`).

- Reads (the `READS` set, `coordinator_calls.py:12-14`) come from mirror plus own journal, with `as_of`
  and `offline: true` in structured output and one line in human output. No error, no delay.
- Writes append to the journal and return at once. The journal is the existing `outbox` made durable,
  keyed by `request_id` (32 hex, `coordinator_calls.py:17`) with a status the agent can read.
- Time: hybrid logical clock `(wall_ms, counter, device)` on every journal row and merged event. Host
  projection order stays arrival order (`board_graph.py:_insert` counter) so read cursors, which merge by
  max (`_cursor` `:329-341`), stay monotone; HLC is the displayed "sent at" and the conflict tiebreak.

### Every operation, offline

Classes: **S** offline-safe (append-only, commutative). **P** provisional (queued, visibly pending, applied
only if its base is unchanged). **L** device-local (device coordinator decides). **R** refused. `WRITES` is
`coordinator_calls.py:15-16`.

| Operation | Class | Offline behaviour; what the agent sees |
|---|---|---|
| `send`, `announce`, `ack`, `notes-add`, `hello-model`, `main-session`; board post, DM, board create/join/leave, subscribe | S | journaled, ok at once, status `queued`; announce keeps its 6 per 10 min limit locally and on merge; two creators of one board name: second renamed, both kept |
| `task-checkpoint`, `task-heartbeat` on a lease this device holds | S | journaled; lease held for the partition grace (section 6) |
| `task-submit` on a held lease | P | applied on merge if the lease epoch is unchanged, else stored as evidence on the task; the worker is told "superseded; your work is preserved at <ref>" |
| `claim`/`reserve` of a branch or area; `release`, `heartbeat` of one | P | local provisional claim (`provisional: true`, holder, epoch); work proceeds |
| worktree, port, slot claims | L | device coordinator decides; never conflict across devices (`coordinator_calls.py:62-63` already limits remote claims to branch/area) |
| `task-create` | P | local draft id; the host assigns the real id and maps it |
| `task-claim`, assignment, `task-credit`, `task-review`, landing, push | R | "needs the global coordinator; not available offline. Request kept as <id>; it applies only if you reconnect and are still the right worker." Review and credit need the host's evidence, so they are not silently queued |
| `mint`, `invite`, `join`, `revoke`, authority, coordinator selection, takeover, standby changes | R | person at a terminal |
| global promotion | R | the device coordinator is automatic; no global lease is created offline |

An authority-dependent operation is never shown as committed. Row status goes `queued` → `applied` |
`superseded` | `rejected(reason)`; the hook prints each transition as context (the path of
`session-liveness.md` §3.3).

### Reconnect: merge

The device pushes its journal in chunks of at most 1000 events (below `MAX_EVENTS`,
`board_graph.py:20`), then pulls the newest cut and tail. On the host:

1. Authenticate the device (paired-device MAC, `fleet/api.py:228-236`); require it active in the host's
   ledger (a revoked device's journal is quarantined, section 3 identity).
2. Verify the origin chain from the host's checkpoint for that origin (`combine` rejects a base differing
   from local history, `board_graph_merge.py:203-216`; a never-merged device starts at genesis), the head
   signature, and each row's schema (`valid_event` `:62-140`). `combine` is person-only and bus-only
   today; slice 4 generalises it to the `boards` stream, callable by a device's paired identity for its
   own origin only.
3. Class S rows insert idempotently by event id (`_insert` raises only on a different body for one id,
   `board_graph.py:201-209`).
4. Class P rows apply under compare-and-set against the `base_version` of the resource they carry.
   Deterministic rules:
   - two provisional area/branch claims: a host-accepted claim wins; between provisionals the lower
     `(hlc, device fingerprint)` wins; the loser is told, its claim becomes a note and a handoff offer,
     and its commits, worktree and checkpoints are untouched;
   - two checkpoints from one lease: both kept in HLC order, the later `resumable_from` the earlier;
   - lease expired offline and task reassigned (`lease_epoch` moved): the stale submission is attached as
     evidence; the new holder is told it exists; nothing is deleted;
   - claim on a resource since reassigned: refused, naming the new holder.
5. Every applied or refused row is audited with epoch, device and rule.

Idempotency: events use immutable ids; call-style rows use `coordinator-call:<actor>:<request_id>`
(`coordinator_calls.py:67-78`). Replay returns the recorded outcome; a different body under one id is
refused. The outcome cache expires after a day (`OUTCOME_TTL_S` `:20`, `:93-100`), so a longer queue would
be refused; provisional rows keep their call node for the partition grace.

### Identity and credentials offline

- Local checks use the replica's `agents.json` (hash, `expires`, `revoked`, parent liveness,
  `identity.py:570-589`) against the device clock. A token valid at the last cut works until `offline_cap_s`
  (7 days proposed) after the last sync; later journal rows are marked `unverified`.
- Revocation exists only on the host. At merge it compares each row's HLC with the revocation time and
  quarantines later rows (a view; not applied, not deleted), widening the margin by the measured skew (host
  clock minus device clock, carried in each sync; 60 s floor).
- Stolen offline device: the person revokes the device (`Devices.revoke` `requests.py:212`) and its registry
  binding; the host refuses journals from that fingerprint and quarantines rows since its last good sync. The
  thief reads the replica's messages; at-rest encryption is open question 5.
- A device clock orders rows only inside its own chain (by sequence); across devices HLC is a tiebreak and a
  display value, never a decision input.

## 4. Standbys

### Enrollment

A standby is a paired device with the board role `standby`, chosen by the person.

- Command (person at a terminal; the existing `coordinator` verb group, since `board` is the message
  group): `ml-stack-workspace coordinator standby add|remove|list DEVICE`; or the UI under **Board →
  Shared coordinator**, person session only (the rules of `coordinator_routes.py:61-89`: loopback, browser
  session, no access-token session).
- Checked at add: the device is `active` and `mine`, runs a compatible ml-stack version, and is paired
  with every device holding an agent binding in `device-sessions.db`. A standby's pairing ledger is its
  own and is never replicated, so a client not paired with it cannot reach it after takeover; `add`
  names missing pairings and the person runs the existing pairing (`ml-stack-cluster listen --for 10m`,
  `pair`).
- The host writes `standby:<fingerprint>` nodes into `coordination.db` (they travel in the cut). The
  standby saves `coordinator.json` as `{mode: standby, workspace, host, board_epoch}`. Clients receive the
  pre-registered successor list (`name, endpoint, cert, signing_key`) from `/workspace/v1/info` and store
  it after the verification `_device_peer` already does.
- Credential: replication routes `/workspace/v1/mirror/{manifest,part,tail}` accept only a device in the standby
  set, identified by the device MAC (`device_auth.identify`); no agent token or person credential is involved.

### While the host is alive, and its liveness view

The standby pulls cuts and tails, verifies and applies; it serves nothing on `/workspace/v1/call`
(`coordinator.py:25-27`); its agents are ordinary clients with the section 3 journal; the mirror is written
only by the sync process. `session-liveness.md` §2 judges presence at the host; a standby's view is never
authoritative: the host renews a lease node `{board_epoch, host_fp, renewed_seq}` every 30 s in the tail, and
the standby ages it on its monotonic clock (`current` < 90 s, `suspect` < 5 min, `unreachable`). It never
calls a host dead (remote devices cannot prove death); only the person asserts a machine is off.

## 5. Takeover protocol

### Host down or I am offline

`coordinator takeover --dry-run` (read-only, agent-runnable, repeatable) collects:

1. Self-network: signed beacons or an HTTP answer (a 503 from a non-host daemon counts) from at least one
   witness, an enrolled device that is neither host nor this standby.
2. Host reachability: failed probes of the host's `/workspace/v1/info` (the shape of
   `coordinator_client.discover` `:137-153`) for the whole `unreachable` window, plus a signed `reach`
   question to each witness ("can you reach H?").
3. Mirror age and entries behind.

Verdicts: `host-reachable`, `i-am-offline` (no witness answered), `partition-suspected` (a witness reaches
the host), `host-unreachable-by-all-witnesses`, `no-witness` (a two-device cluster: no evidence exists).
Only the last two allow an approval request, and `no-witness` additionally needs the typed confirmation that
the host is powered off or gone. An offline standby sees no witness, no host and no person: it cannot
promote.

### Approval

No automatic promotion exists. The approval names this exact standby and this `board_epoch`.

- Now: `ml-stack-workspace coordinator takeover --apply` at the person's terminal on the standby. It mints
  `HumanGrant("board.takeover", "<host name> epoch <n> to <this device>")` with the subject typed back
  (`sentinel/human.py:54-66`: refused without terminals, with an agent marker, or on a mismatch).
- After `person-delegation.md` slice 1a: authorization kind `board-takeover` in its closed registry (§7),
  target derived as (workspace, old epoch, this device), single use, 15 minutes. The person answers "yes,
  take over the board on this machine" to the agent's dry-run evidence; the hook attests; `consume`
  verifies. The agent never receives a credential and never writes the attestation (§6 hard rule). No
  service is installed or edited, so the chat form does not touch the human-only floor; `--apply` still
  requires the evidence block.
- The grant: the approving device signs `{workspace, from_epoch, to_epoch = n+1, new_host: {fp, name,
  endpoint, cert}, approval_id, expires}` with its signing key. Clients accept a grant only from a device
  in the pre-registered standby set or a `mine` device verified through `peers.json`.

### Applying

On the standby under `coordinator-selection.lock` (`coordinator_client.py:95`): re-verify evidence and mirror
(refuse one older than `max_stale_s` without the typed override); import the verified generation with a new
`board-scope.replica`, the `board.takeover` row and a host lease at `n+1`; vacate `coordinator-lease`; rebase
claims and task leases to `deadline = now + takeover_grace_s` (120 s = `HEARTBEAT_S`, `task_actions.py:19`) and
clear remote claims' `pid`/`owner_pid`; rewrite `coordinator.json` to `{mode: host, workspace, board_epoch:
n+1, grant}`.

### Epoch and fencing

- `board_epoch` is in `coordinator.json`, `/workspace/v1/info`, every response and a client request header.
  A client keeps the highest epoch per workspace and refuses a lower one (the identity check at
  `coordinator_client.py:35` is where the compare belongs).
- A host that receives a valid grant (signature checked against the peer book, signer `mine`,
  `from_epoch` equal to its own, `new_host` not itself) demotes: `mode: standby`, stops serving, mirrors the
  new host. A bare claim of a higher epoch does nothing.
- A revived host asks its known peers for grants before answering. With none reachable it serves, and a
  later grant demotes it. Nothing prevents a partitioned old host accepting writes from clients still
  talking to it; hence the person gate with evidence, and the orphan tail.

### Two standbys

A grant names one new host. Two grants for one `to_epoch`: clients and hosts adopt the first seen and refuse the
second as `conflicting authority` (as `project_connection._find_authority`, `:148-150`); a standby that applied
the losing takeover demotes (lower `(approved_at_hlc, fingerprint)` wins) and its accepted client writes enter
its orphan tail. The person is told.

### Clients rediscover the new host

- A client with a non-empty `standbys` list whose host is unreachable probes each standby's
  `/workspace/v1/info` over the pinned device channel. An answer holding a grant for a higher `to_epoch` from an
  accepted signer, after the client's own probe of the old host fails, makes it switch `endpoint`, `cert`,
  `name`, `board_epoch`. No person step on a client.
- TLS: the standby's certificate is already pinned by pairing; `_device_peer` needs a unique active paired row
  with that certificate (`:84-91`). No certificate moves. The rotation branch (`:117-133`) stays.
- Tokens are unchanged: the new host's `agents.json` verifies the stored secret (`identity.py:576-589`). An
  agent registered after the cut gets `Denied` and re-runs `ensure` (`coordinator_client.py:48-72`).
- A device with no pairing to the standby stays on the old host; `status` says "coordinator lost; ask the
  person to connect" (`coordinator connect --replace`, person only, `coordinator_routes.py:54-57`). A device
  that followed workspace W refuses `host` for another board without a grant (the `mode == 'remote'` refusal
  extended to `standby`).

### In-flight writes, leases, claims, jobs

- Lost response: retry with the same `request_id`. A call that reached the cut or tail returns its cached result
  (`finished`) or "original outcome is uncertain" (`started`, `coordinator_calls.py:76-77`); one that did not
  executes once. An acknowledged write that was lost is recorded as `lost_after: <seq>` in the `board.takeover`
  row and printed to affected agents.
- Landing in flight: `recovery_required`, worktree orphaned and offered. Leases and claims are rebased; a holder
  re-heartbeating within the grace keeps them. Test jobs run on their own device; their owner may re-register.
- Every deadline is rewritten from the new host's clock; skew enters no later decision.

### Rejoin of the old host

On a higher-epoch grant (startup fence, client request or its standby role) the old host demotes, then:
1. pushes its tail since the last cut the new host holds: class S events as a journal under its old origin,
   merged by section 3 (it is treated as an offline device);
2. exports class P and authority state (claims, task transitions, registry edits, minted tokens) as an
   `orphaned/<epoch>` bundle on the new host, listed in `status`, never applied automatically; unique work is
   preserved either way;
3. becomes a standby with a fresh replica id once the person confirms; its old `coordinator.json` is archived.

## 6. Changes to docs/session-liveness.md

1. Vocabulary: board host (storage, failover per this note), device coordinator, global coordinator; replace
   "the coordinator" in §3.1 and §4 by scope.
2. §2: add state `unreachable` for a remote device whose device-level heartbeat stopped, distinct from one
   session going quiet; `suspect`/`expired` (15 min, 10 min grace) apply to reachable devices. An unreachable
   device's claims, leases and child tokens are held for `partition_grace_s` (12 h proposed), then
   `stale-held`: reassignable by the global coordinator only, under a fencing epoch, its provisional work
   preserved.
3. §3.2-3.3: pid-verified same-device recovery moves to the device coordinator and runs with no network;
   cross-device recovery stays global; add `device_epoch` and `lease_epoch` next to `epoch`.
4. §4: the promotion rule is for the global lease; add the device lock; replace the partition paragraph
   (335-340). §5: the tier rule covers both scopes. §6: add section 8 below. §7 slice 1 gains the device
   coordinator. §8 Q1 is answered here.

## 7. Resync cost and limits

Journal pushes are chunked at 1000 events and resumable from the origin checkpoint; no row is dropped. After a
long partition the device pulls one cut (about 20 MB by `ls`, not a timed run; a timed cut, transfer and apply
on two real devices is owed in a dated document). Resync checks chains, schemas, `check()`, manifest hashes,
head signatures and counts; a failed part keeps the previous generation and raises `ChainBroken` (`chain.py:26`).

## 8. Failure modes

| Failure | Behaviour |
|---|---|
| Host power loss; partition with host alive | `unreachable`; dry-run evidence; a witness reaching the host gives `partition-suspected` and needs the typed override |
| Both hosts up; revived host writes | first-seen grant wins at clients; writes accepted before the old host sees a grant enter the orphan tail, after it `superseded` |
| Standby behind N; replication corruption | age and N shown, `lost_after` recorded, `max_stale_s` blocks; a bad part keeps the previous generation, a tampered cut fails its signature |
| Takeover during a landing; clock skew | `recovery_required`, worktree orphaned and offered; deadlines re-based, no cross-device decision uses a device clock |
| Two device coordinators; one dies offline | one flock holder (same pid, different `started_at` is dead); the next eligible touch takes the lock, `device_epoch + 1` |
| Offline forgery; replayed queue | journal must chain from the host's checkpoint with the device's head signature; immutable ids make replay a no-op |
| Authority op shown as committed | status stays `queued` until the host's `applied`; a test fails any offline output saying applied |

## 9. Tests

Real temp workspaces, stores and processes, no mocks of the board: two or three real daemons on loopback with
separate state roots and real pairing (as `tests/test_workspace_coordinator.py`), `kill -9` for deaths, clocks
moved only through `Claims(clock=)`/`Registry(clock=)`, children with `ML_STACK_NO_REAL_KEYSTORE=1`.

- Offline: socket-level partition; two devices post offline then merge; conflicting offline area claims; two
  checkpoints; lease expired offline then submit; offline revocation; replayed queue; 10k-event journal resumed
  after a kill mid-push; one corrupted row.
- Standby and takeover: refuses an unpaired or non-`mine` device; bad mirror hash; mirror read-only to an agent
  token; host killed mid-write; `--apply` refused without a terminal or with an agent marker; signed grant moves
  clients; restarted old host demotes on a valid grant, ignores a forged one; two standbys approved; old host
  tail merged; standby clock an hour off.
- Device coordinator: `kill -9` frees claims with the network blocked; 8 processes race the lock; a
  lowest-tier-only device spawns and nominates.
- Red-team (`CLAUDECODE=1`): forged takeover (no grant, self-signed, non-standby signer, old-epoch replay,
  another workspace); agent writing `mode: host`; standby promoting itself; higher epoch without a grant;
  journal from a revoked fingerprint or naming another origin; approval replayed after use; a board post
  saying "the owner approved".

## 10. What stays human-only

Standby add/remove, `takeover --apply`, rollback, `coordinator connect --replace`, device revocation after
theft, orphan-bundle decisions, `partition_grace_s`/`offline_cap_s`, the keystore key for posted files, and the
floor in `person-delegation.md` §7. Nothing here installs, edits or restarts a launchd/systemd/boot-time unit;
the standby's daemon already runs under whatever the person set up. Agents may run `--dry-run` and read status.

## 11. Slices

- **1A (days; closed terminals with no network): device coordinator.** `workspace/device_coordinator.py` (lock,
  pid+start check, `device_epoch`), `claims.py` (`_sweep` reads `owner_pid`/`owner_started` for `pid == 0`),
  `presence.py`, `cli.py`, `agent_display.py` (tier gate), `docs/workspace.md`.
- **1B (days, parallel): epoch, fencing, cut, dry-run.** `coordinator_config.py` (`board_epoch`, `standbys`, mode
  `standby`), `coordinator.py` and `coordinator_client.py` (epoch in info, responses and compare),
  `coordinator_routes.py` (`host` refused for followers), `workspace/board_cut.py` (manifest, ordered dump, hash
  and signature, staged import with `check()`), `cli.py` (`coordinator takeover --dry-run`). No takeover.
- **2:** standby enrollment, mirror routes, pull loop, tails. **3:** grants, client switch, demotion, orphan export, `takeover --apply` (terminal grant first).
- **4:** offline replica and journal (`board_replica.py`, `journal.py` with HLC), client fallback, `combine`
  generalised, the class table, call-node retention.
- **5:** liveness-note changes, conflict rules, quarantine view. **6:** shape B: `board_host` transfer (`projects.py:264-285`), `project_connection.bind` refusals (`:60`,
  `:76-77`), `_find_authority`.

## 12. Open questions for the owner

1. Is a two-device cluster (host plus one standby, no witness) acceptable with the typed "the host is powered
   off" confirmation as the only evidence, or must takeover require a third device?
2. Standbys limited to `mine` devices (this design), or may another person's device be one? The replica holds
   token hashes and every message.
3. `partition_grace_s` (12 h proposed) and `offline_cap_s` (7 days): acceptable?
4. Does a takeover keep the workspace ID (this design) or fork a new one with a reselect on every device?
5. Posted files are encrypted under the host's keystore key: leave host-only, or move the key to a
   person-provisioned secret that standbys unwrap? (Same question for encrypting the offline replica at rest, 6.)
6. Land slices 1A and 1B in parallel as written, or one first?
