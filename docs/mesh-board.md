# A peer-to-peer board for a pool of one to N devices

Design note; nothing is implemented. It replaces the host-failover design and extends
`docs/session-liveness.md` (presence, recovery, coordinator role), `docs/person-delegation.md` (approvals)
and `docs/policy-distribution.md` (rules replicate as records; machine settings stay human-only). Line
numbers: `0.2dev` at `f0fa485d`.

A **pool** is the devices a person has paired; a pool of one is the base case. There is no host, standby,
takeover or board epoch.

## 0. Decisions already made
1. Every paired device is a peer with a full replica and syncs with whichever peers it can reach. Any peer
   may be offline at any time. Peers authenticate by TLS pinned through device pairing.
2. Consistency is eventual with a deterministic merge, not quorum. Operations that depend on authority are
   provisional until they have synced with every reachable peer, with visible status; nothing blocks on
   connectivity. Given up: no linearizable claim holds across a partition. For source, git's atomic ref
   update at the remote is the arbiter; the board never decides which commit is on the development branch.
3. Replication unit: per-device append-only signed journals, merged by a hybrid logical clock. Replicated
   state is a deterministic fold of the merged journals. No shared chain key exists.
4. The coordinator is a derived role, not a lease (section 6). The device coordinator always exists with no
   network; the pool-wide "mesh coordinator" is computed from replicated presence.
5. Trust has no central issuer: pairing is the root, enrollment is signed by an existing member, revocation
   is a replicated record (section 7). Person approvals use the chat-attested channel of
   `person-delegation.md`; machine settings stay human-only.
6. Pool size is 1 to N with no special case (section 3).
7. Model rules from earlier notes stand: the lowest tier never coordinates, an agent is never a human, every
   agent acts as itself.

## 1. Where the code does not support this

- One writer by construction: the process answering `/workspace/v1/call` while `coordinator.json` says
  `mode: host` (`coordinator.py:32-34`); `mode: remote` has no local copy and no fallback
  (`coordinator_client.py:_client` returns a `Remote` or fails; `Remote.command` `:21-35`).
- `mode: host` is settable by an agent token: `coordinator_routes._change` accepts `{'action': 'host'}` after
  only `device_agent.owned_local` (`coordinator_routes.py:48-53`); `ensure_host` does the same with no
  config present (`coordinator_bootstrap.py:11-26`). `coordinator_config.load` accepts only `host|remote`
  (`coordinator_config.py:9-18`). The mesh deletes the mode (slice 6); slice 1 closes the agent path.
- No durable outbox. `outbox` lists messages the agent already sent (`bus.py:106`, `service.py:569-573`);
  `ack` moves a read cursor (`bus.py:76`). Neither queues a write for later. The only retry state is the
  host-side `coordinator-call` node (`coordinator_calls.py:67-100`), expired after a day (`:20`).
- No hybrid logical clock, no signed head, no per-device journal. `ChainLog` is unkeyed hash chaining with
  `seq`, `prev`, `hash` and a `base` row for a pruned prefix (`chain.py:57-60`, `prune_prefix` `:233`);
  `sentinel/events.py:94` `EventLog` seals its head in a side file for a single device.
- The nearest merge is `board_graph_merge.combine` (`:193`): bus stream only, person-only
  (`boardapi.py:65-80`), capped at 10000 events and 16 MiB (`board_graph.py:20-21`), per-origin chains with
  bases (`board_evidence.verified` `:43`). It is the model for the journal verifier, not reusable as is.
- Claims are a local JSON file with `pid` meaningful on one machine (`claims.py:115-124`); `claim` raises
  `Conflict` synchronously (`claims.py:126`). Task, token and presence state are single-writer stores in
  `coordination.db` and `agents.json`.
- Pairing is pairwise. Each pair holds a `device_secret`, a `signing_key` and a pinned certificate in
  `peers.json` (`fleet/onboard/cli.py:184`, `:240`); `_device_peer` requires a unique active paired row for
  the endpoint's certificate (`coordinator_client.py:84-91`). N devices need N(N-1)/2 pairings today.
  There is no membership certificate, no mutual-TLS client authentication for non-adjacent peers, and no
  signature over anything except file manifests (`onboard/manifest.py:5`, `trusted.pinned_keys` `:47`).
  Section 7 adds all three; it is the largest new piece.
- Identity: agent ids are bare names in one registry (`identity.py:157`, `_live` `:275-280`); two devices
  that each ran alone can both hold `claude-code`. Section 4 keys agents by creating device.
- `person_record`/`consume` and the model tier table are not in the tree; until they land approval is a
  terminal `HumanGrant` (`sentinel/human.py:35-66`).

## 2. Vocabulary

Origin: a device as journal writer. Fold: a pure function from the row set to state. View: a ladybug store
written only by the fold. Settled: every active member's published vector covers the row.

## 3. One to N with no special case

- **Pool of one.** The device creates its signing key and a `pool-founded` row as its first journal row. It
  runs the same journal, fold, claim compare-and-set and coordinator derivation with an empty peer list. It
  needs no pairing, network, configuration file or TLS peer, and every row is settled when written.
  `status` shows `pool: 1 device`. There is no `coordinator.json`, mode or endpoint.
- **Pool of two.** Pairing (existing flow, person approved on both ends) plus a first sync: the joining
  device receives a snapshot (section 5) and a `member-add` row appears in both journals. No mode switch
  or migration; a device with history keeps its journal as one origin. The same path adds the hundredth.
- **Two pools meeting.** A `pool-merge` row signed by both sides names the lower pool id (the founding
  row's hash) as the survivor; all journals stay as origins.
- **Cost shape.** Sync work per device per interval is bounded by fan-out k (section 5) and does not grow with N; storage is
  O(live state + retained tail + N). O(N) and unavoidable: membership, one presence summary per device, the
  version vector.

## 4. Data model

### Journals

One journal per device, a `ChainLog` file `mesh/journal/<device-fp>.jsonl` (own: writable; others: verbatim
copies, read-only). Row:

```
origin, seq, hlc, prev, hash, kind, id, idem, actor, body            (ChainLog fields: seq prev hash ts v)
head row, every 64 rows or sync: {head: seq, hash, sig}  sig = Ed25519(origin key, pool || origin || seq || hash)
```

- `id` is `<origin>:<seq>`, immutable. `idem` is the caller's request id (`coordinator_calls.REQUEST_ID`
  shape); a second row with an `idem` already present for that actor is a no-op in the fold, so retry after
  a lost response is safe forever, not for one day.
- `hlc` is `(wall_ms, counter, origin)`; wall never moves backwards in a chain; a receiver holds a row whose
  wall is more than `skew_max` (5 min) ahead of its own clock until its clock catches up (delay only; the
  row stays valid).
- No chain key crosses the wire. Each device signs only its own heads, checked against its `signing_key` in
  the membership fold (section 7). A relayed row is verified by origin signature, not by the relayer.
- One journal per device, not per stream, so the version vector is O(N) and not O(N x streams). Kinds:
  `message`, `announce`, `ack`, `note`, `claim`, `release`, `renew`, `task-*`, `agent-*`, `token-*`,
  `member-*`, `revoke`, `authorization`, `tier-table`, `reuse`, `landing`, `session-dead`.
- Quotas are fold rules: rows beyond a per-origin rate or size window are void deterministically by `seq`,
  so a spamming device cannot grow every peer's storage without bound.
- Presence is not journaled (section 6): a per-device register, gossiped, superseded in place.

### The fold and the ladybug views

`coordination.db`, `board.db`, `agents.json` and `claims.json` become views written only by the fold applier
under the existing locks (`coordination.lock`, `board-graph.lock`, `agents.lock`, `claims.lock`). They are
disposable: deleting one and replaying a snapshot plus the tail reproduces it.

- A local write is: append the row to this device's journal (fsync), then apply it to the views. If the
  view write fails the row is still durable and the fold re-applies it on restart (a per-origin applied
  cursor in each view).
- Ordering: the total order is `(hlc, origin, seq)`. Per-key folds (a claim key, a task, an agent) read only
  the rows for that key in total order. A row that arrives older than the view's cursor re-folds that key
  from its last checkpoint only; the cost is the rows after the insertion for that key.
- Registers: `last-writer-wins` by the total order for settings-like values (display name, model labels);
  add-wins sets for membership and subscriptions; grow-only for messages and notes; content-keyed for
  reuse. Where a CAS is needed (claims, task state) the fold is sequential over the total order (section 6).
- Snapshots and replication never use `GraphStore.read()/write()` (`store.py:296`, `:143`) as a wire
  format; the fold applier uses `upsert_node`/`upsert_edge`. Ladybug stays the store, not the replication
  unit.
- `audit.jsonl` (`service.py:122`) and `quarantine.jsonl` (`quarantine.py:37`) stay device-local, with each
  device's audit head published as a signed row so a missing tail is visible; `notes.jsonl` (`notes.py:34`)
  becomes the `note` kind. `EventLog` (`sentinel/events.py:94`) stays a device-local sentinel log. The bus
  becomes the `message` kind and `board_graph_merge.combine` is replaced by the journal merge.

### Agent identity

An agent record is `agent-create {device, name, role, parent, token_hash, expires}` in the creating device's
journal. The agent id is `name@<device-short>`; `whoami` prints the short form where unambiguous. A row with
`actor = X` is valid only in the journal of X's creating device (children created on another device get
that device's origin). A paired device can therefore speak only as agents it created. Tokens stay local
secrets; replicated state carries `sha256(secret)` only (`identity.py:86-88`), as today.

## 5. Sync topology and scale

### Anti-entropy

Each device keeps a **version vector**: origin -> (seq, head hash, head signature). Peers exchange vectors
and ship the missing row ranges per origin, verifying each chain before storing it. Rows travel
transitively: a device relays any origin's rows it holds, so a row reaches every peer by any path.

- **Peer set (bounded fan-out k, default 4).** Ring neighbours by `hash(device-fp)` among members (previous
  and next), plus 2 links re-drawn every epoch (1 h) by `hash(fp, epoch)`, plus any peer found on the LAN.
  Sync interval 10 s when a peer has news, 60 s idle. Per-device sync work is k exchanges per interval,
  independent of N. Epidemic spread reaches all N devices in O(log N) intervals when online; the ring gives
  connectivity, the random links give shortcuts and defeat a single hostile neighbour eclipsing a device.
- **Summary first.** Exchange a 32-byte digest of the vector; equal means done. Unequal: exchange vector
  entries changed since the last exchange with that peer (O(changed), worst case O(N) entries of ~100
  bytes, so 10 KB at N = 100). Beyond about N = 2000 the vector is bucketed by fingerprint prefix with a
  digest per bucket; not needed for the slices below.
- **No O(N^2) traffic.** Rows are pulled once per receiving device; duplicates are avoided by the vector
  exchange. A row costs O(1) transfers per device, O(N) pool-wide, which is the minimum for a full replica.
### Presence at scale

Per-session heartbeats stay local (`claims.heartbeat`, `session-liveness.md` section 2). Each device
publishes one **presence summary** register: `{device, hlc, valid_until, sessions: [{key, role, model_id,
verified, tier, registered_hlc}], vector_digest}`, rewritten at most every 60 s and only gossiped, never
appended. A summary is O(sessions on that device); the pool holds N of them. The vector digest in each
summary is what lets other devices decide a row is settled.

### Snapshots, joining, retention

- A **snapshot** is the plain serialisation of the fold at a vector V: per-kind rows, plus V (each
  origin's seq and head hash) and a state hash, signed by the producing member. Any member may produce one;
  a device produces one every 10000 rows or 24 h.
- **Join**: a new device pairs with any one member, receives the latest snapshot and its tail, verifies the
  producer's membership and signature, and compares the state hash with a second member's snapshot at the
  same V when N >= 3. It then verifies each origin's tail against the head signatures in V. A single hostile
  producer can still hand a joiner a wrong state at V that no origin signed; with N = 2 there is no second
  witness. The person sees which device a new device joined from (open question 4).
- **Compaction**: a journal prefix up to V may be removed (`prune_prefix`, `chain.py:233`, leaving a `base`
  row) when V is covered by a snapshot and by every active member's published vector. A dormant member
  (no presence for `offline_cap`, 30 days) stops counting, so one lost laptop does not pin storage. When it
  returns it needs only rows after its own last vector, which every origin still holds because its rows
  after V were never pruned; if it returns after a prefix it lacks was pruned, it rejoins from a snapshot.
- **Tombstones and revocations**: a tombstone (deleted claim, released resource, expired token) is dropped
  from the fold state after `offline_cap` once settled. Revocation records (section 7) are kept for the
  life of the key they revoke (O(revoked devices)), because dropping one lets a stale copy of the device
  back in.
- **Storage per device**: O(retained tail + live state + N). Messages follow the existing retention
  (`bus.prune` `bus.py:145`).

### Discovery, NAT, relays

- LAN: the existing signed beacons (`Peer.discover`, `fleet/discovery.py:270`) find members on the segment.
  Elsewhere: `member-add` and presence summaries carry last-seen addresses; any member that answers is a
  rendezvous for that moment, none is designated.
- A relay is optional, stateless forwarding of signed, TLS-wrapped streams between two members that cannot
  reach each other (NAT). It holds no journal, key or membership list; its compromise yields traffic
  metadata only. Not built in the first slices.

## 6. The derived coordinator and decisions that need none

### Mesh coordinator

Computed by every peer from the presence summaries and the membership fold; no lease, no election, no
message. Eligibility, all from replicated fields: a main session (`coordinator_eligible`,
`session-liveness.md` section 1(e)); role agent; live (the summary's `valid_until` plus `skew_max` is in the
future); tier not the lowest of its family; model verified (`session-liveness.md` section 5). Rank:
tier descending, `registered_hlc` ascending, then `session_key`. Each device submits only its own best
candidate (the device coordinator), so the answer is the minimum over N candidates.

- **Same answer for any N and any partition shape**: the rank is a total order over replicated fields, and
  two peers with the same set of summaries compute the same minimum. Peers with different sets may differ;
  that is a partition, not a bug.
- **Complexity**: O(N) for a full evaluation; O(log N) per updated summary with a heap. The device
  coordinator is O(sessions on the device).
- **No preemption by joining**: a later registration never outranks an earlier one at the same tier.
  Tier demotion (`session-liveness.md` section 5.4) changes the summary and re-ranks.
- **Partition**: each side derives its own coordinator after the other's summaries pass `valid_until`.
  Decisions carry `coord: (session, rank)`. On merge, every decision taken under a coordinator that is not
  the merged minimum is shown `made under a split coordinator`, and CAS decisions are re-decided by the
  fold (below). A person-visible `status` line says `coordinator: <name> (provisional: 2 seen)`.
- **Lowest tier**: a device whose only main sessions are lowest tier has no candidate; it spawns a
  coordinating subagent (nominated by its parent, verified model passes the tier check). The helper's
  parent linkage is in the summary; a transient helper is not a coordinator by naming.

### Decisions

| Decision | Needs | Rule |
|---|---|---|
| Message, announce, note, ack | none | grow-only rows |
| Claim of branch or area, release, renew | none | CAS in the merged log (below) |
| Worktree, port, test slot, job, a local session's holdings | device coordinator | device-local, no network, pid+start check |
| Recover a dead session's holdings | none | idempotent sweep (below) |
| Task create, claim by a worker (pull), checkpoint, heartbeat | none | per-task CAS by total order |
| Task assignment, queue ordering, review, credit | mesh coordinator | row carries `coord`; CAS on task version; split-made decisions flagged |
| Landing the shared development branch | none | git atomic ref update at the remote; the session that wins the push lands; losers fetch and retry. A `landing` row is a record only |
| Test-result reuse entries | none | content-keyed rows, merge is union |
| Mint a token for an agent on this device | the creating device | signed `agent-create` |
| Enroll or revoke a device; designate a coordinator; tier table edit | a person | authorization row (section 7) |
| Machine settings, daemons, guards | a person at a terminal | not replicated |

### Claim compare-and-set

For a normalised key (`claims.normal` `:51`; nested areas by `_nested` `:80`), the fold walks `claim`,
`release`, `renew` and `handoff` rows for overlapping keys in `(hlc, origin, seq)` order. A claim row is
`granted` when no earlier granted claim by another actor covers or nests it and has neither been released
nor expired (`expiry = last renew hlc + ttl`, HLC arithmetic, never the reader's clock); otherwise
`yielded`. Equal HLC wall and counter break on device id. The call returns `granted (provisional)` at once
when the local view shows no conflict, or raises `Conflict` as today when it does. The loser's `yielded`
status reaches its agent through the hook path of `session-liveness.md` section 3.3; its commits, worktree
and checkpoints are untouched and it is offered a handoff. Two devices claiming one area offline both see
`granted (provisional)`; on merge the earlier by total order keeps it. A claim is `settled` once all active
members' vectors cover it. Cost: O(live claims overlapping the key) per claim; a late row re-folds the
overlap set from its position.

### Recovery sweep

Same-device death is verified by that device (pid exists and `started_at` equal) and written as a signed
`session-dead` row by the owning device even offline. Any peer may then sweep: the claims and task leases
of that session are released by a `release` row with id `sweep:<holding id>:<death row id>`, so duplicate
sweeps by two peers are one effect. Remote death cannot be verified from afar (`session-liveness.md`
section 2): without a `session-dead` row, holdings of an unreachable device become `stale-held` after
`partition_grace` and are reassignable by task CAS with a fencing counter; the old holder's later work is
preserved as evidence, not applied.

### Provisional status

Every write returns a row id and a status the agent can read (`poolhouse-workspace outbox`, redefined as the
status list): `local` (journaled), `synced n/m` (m = active members reached recently), `settled`,
`yielded`, `superseded`, `rejected(reason)`. The hook prints each transition as context. No output says
"applied" for a row that is not settled. At N = 1 rows are `settled` at once.

## 7. Trust

### Pairing is the root; enrollment is a signed record

- The first device's signing key is the founder, with no special rights after founding.
- `member-add {device, signing_key, tls_cert_fp, enrolled_by, person_approval}` is signed by the enroller
  and by the joiner. It is valid when the enroller is a member at that HLC and the pairing handshake
  completed on both ends with the person present (`poolhouse-pool pair`, existing SPAKE2 flow in
  `pyproject.toml:72`). Trust propagates through these records: a device need not pair with each of N-1
  others. The new device receives the membership fold from its one pairing partner and then authenticates
  every other member by its recorded certificate fingerprint and signing key.
- Transport: TLS with the peer's fingerprint pinned from the membership fold (`tls.pinned_context`
  `fleet/tls.py:184`), with mutual authentication by signed challenge using the device key (new; today only
  the pairwise MAC in `device_auth` exists). Pairwise `device_secret` stays for directly paired devices
  during migration, then goes.
- A device is trusted by the replicated records, not by the peer who introduced it. A member-add by a
  hostile device is visible: every device announces `member-add` to its local person display, and a new
  member cannot be a coordinator candidate or create agents until a second member, or the person through
  the chat channel, confirms it (waived at N = 1 to 2, where the person is at both devices).

### Person approvals

Enrolling a device, revoking one, designating a coordinator and editing the tier table are authorization
kinds in the closed registry of `person-delegation.md` section 7. The attesting device writes an
`authorization` row carrying the attestation reference; the fold applies the action only if the row's
origin is a member and the kind is in the registry. Peers cannot independently verify what the person said:
they trust the attesting member (see malicious peer). Machine settings are not replicated and not
approvable by chat.

### Revocation

`revoke {device|agent|token, cut: {origin, seq, hash}, by, authorization}`. For a device the record names the
last row of the revoked origin the revoker has seen. Fold rule: every row of that origin with `seq > cut`
is void, regardless of its claimed HLC (so a revoked device cannot backdate). Voided rows are quarantined,
not deleted; the person can accept up to a later seq with another authorization row. Agents and tokens
revoke by id; descendants fall with the parent as `_live` already does (`identity.py:275-280`).

- **Propagation**: online devices learn within O(log N) sync intervals (tens of seconds at N = 100). An
  offline revoked device keeps working on its own copy and on any peer that has not yet heard, until it
  syncs with a peer that has. **The window is bounded by**: member lifetime (`member-add` carries
  `valid_until`, default 30 days, renewed by any member automatically while the device syncs; an expired
  member is refused by peers and must be re-enrolled with the person), token `expires` (checked locally,
  `identity.py:278`, and never beyond the device's `valid_until`), and the fan-out above. A revoked or
  stolen device that never syncs again is harmless to the pool after `valid_until`.
- **Stolen device**: revoke from any other device with the person's authorization; peers refuse its TLS
  identity from the moment they hold the record. The thief has the replica (messages, token hashes, not
  tokens held on other devices) and can write only inside its own origin until it meets a peer that knows.
  At-rest encryption of the replica is an open question.
- **Clock skew**: fold decisions use HLC and row fields, never the reader's clock; reader clocks decide only
  token expiry and liveness, each with a `skew_max` margin.
- **Key rotation**: `key-rotate {old, new}` signed by the old key, effective at a named `seq`. A lost key is
  a revoked device and a fresh enrollment.

### A paired but hostile device

Can: write any row in its own origin; claim first; spam up to its quota; delay rows it relays; speak as
agents it created; forge a `person_approval` reference in its own rows (the pool trusts a paired device
as the person's); add members (visible, held until confirmed at N >= 3); read the replica.
Cannot: write in another origin; speak as another device's agent; win by backdating (the cut and total
order); exceed its quota; alter history undetected by a peer holding the original.
Detected: a broken hash link or bad head signature (row rejected, origin `damaged` at that peer);
**equivocation** (two signed heads, same seq, different hashes: kept as a fork proof, rows after the fork
void, person told); **rewind or truncation** (a head below a vector entry a peer holds); quota violations.
Not detected: a validly signed false statement.

### An agent with file access on one device

It can read and edit that device's journal files, views and key files (same OS account;
`SigningKeys(state_dir)`, `fleet/onboard/cli.py:318`). Editing an existing row breaks the chain and the
signed head and is detected locally by `verify` and by every peer that holds the original (fork proof).
Appending new rows signed with the device key is not cryptographically distinguishable from the device
acting; it is attributed to the device and shown with `actor`. The controls are the existing ones (file
permissions, hook and guard layers, `docs/policy-distribution.md` section 3.5), plus an alert when a
journal's head regresses or its own device key signs while no session of that device is live. Moving the
signing key into the OS keystore with agent refusal is open question 5. An agent can never produce a
person authorization: that path is the harness hook (`person-delegation.md` section 6) and an agent marker
refuses the terminal `HumanGrant` (`human.py:54-66`).

## 8. Failure modes and tests

Real temp devices, each a real process with its own state root, real pairing and real sockets; partitions by
a loopback proxy that drops connections; death by `kill -9`; clocks moved through the `clock=` parameters
or a per-process offset; `POOLHOUSE_NO_REAL_KEYSTORE=1`. Large pools are real journals and the real
merge and fold, one process per device group, over loopback sockets. No mock of merge or board.

| Case | Expected |
|---|---|
| Pool sizes 1, 2, 3, 10, 100 | same code, same tests; N=1 has no network and all rows `settled` |
| Partition and heal | each side keeps posting; heal converges to identical fold hashes |
| Two devices claim one area offline | both `granted (provisional)`; on merge one `yielded` by total order, told, worktree intact |
| Three-way merge, three partitions | fold hash identical on all three regardless of sync order |
| Device offline for days, then back | catches up from tail or snapshot; its offline rows merge or are voided by rule; no work interrupted |
| Late joiner | joins from snapshot plus tail; second-witness hash check at N >= 3; state hash equals a full replay |
| Forged head signature; replayed old journal; equivocation; truncation or rewind | rejected and origin `damaged`; replay is a no-op by `id` and `idem`; fork proof stored and rows after the fork void; a peer holding more detects the rewind |
| Equal HLC; clock skew an hour either way; mass restart; N=1 to N=2 with history | device-id tiebreak identical on every peer; no fold decision changes; views rebuild without duplicate effects; no config change |
| Revoked device writes | rows past `cut` void, quarantined; TLS refused once the revocation is held |
| Split coordinators | two derived; merge flags the loser's decisions provisional; CAS re-decided |
| Churn (10% of 100 devices leave and join per minute) | bounded work per device, convergence continues |
| Scaling | sync bytes and CPU per device per interval stay within a constant of the N=10 value at N=100 (fan-out bound), measured and recorded in a dated document |
| Snapshot and compaction | prune below the minimum vector; a late device still rejoins; storage per device does not grow with total history |
| Migration | old `coordinator.json` with `mode: host` or `remote` is read once and converted; no board data lost |

Red team (`CLAUDECODE=1`): agent sets `mode: host|remote` or edits `coordinator.json` (refused from slice
1); forged, sybil or replayed member-add; stale revocation replay; rows naming another origin or device's
agent; far-future HLC; quota flood; eclipse by hostile neighbours; forged `person_approval` from an agent;
token used after `valid_until`; snapshot from a lone hostile producer; a message saying "the owner approved".

## 9. Migration, in slices that keep the board working

Each slice leaves the current coordinator path working until slice 6. File lists are the expected edits.

1. **(days) Journals for append-only streams, outbox, remove agent `host`.** New
   `workspace/journal.py` (ChainLog-based per-origin journal, HLC, signed heads),
   `workspace/mesh_sync.py` (vector exchange and row transfer between two devices over the existing
   device-authenticated peer server), `workspace/mesh_fold.py` (messages, announcements, notes; replaces
   `board_graph_merge.combine`), edits to `bus.py`, `notes.py`, `service.py` (write journal then view),
   `cli.py` (`outbox` status list). `coordinator_routes.py:48-53` and `coordinator_bootstrap.py:11-26`:
   `host` refused for an agent token (person session only) as the interim fix, with a red-team test. Works
   and is tested at N=1 (no peers) and N=2 (two real devices, pairing as today). Claims and tasks stay on
   the existing host until slice 2.
   As built in slice 1: announcements, `#general` posts and notes are journaled; a direct message or a post
   to any other board stays on the device that wrote it, because there is no directory of which device
   hosts an addressee yet (slice 5). A row from another device is folded as an agent named `name@dN`, `dN`
   being a label this device assigns to the origin when it first sees it. It goes through the size,
   screen, quarantine and notes-per-agent checks of a local post; its role, flags, model claims and note
   commands are discarded; a sender whose name is registered here is rejected; an unusable row is recorded
   in `mesh/rejected.json` (the last 200) and skipped. An origin is a 32-character lower-case hex id.
   A relay's copy of a journal never marks an origin damaged; only the device paired as its writer does,
   and `Journals.forgive` clears it. An acknowledgement counts only when its hash for this device's row
   matches. Not built: a person-only command to clear a refusal, per-peer request-rate limits, signed
   acknowledgements, sealing of `keys.json` and `damaged.json` (same-user plain JSON), per-origin quotas
   on rows folded per interval beyond 500, and a stable cross-device order of the board view (rows are
   folded in `(hlc, origin, seq)` order per sync, so bus order can differ between devices).
2. **Claims and tasks as folds.** `claims.py`, `task_actions.py`, `taskboard.py` write rows and read the
   claim fold; Conflict raised from the visible view; `coordinator_calls.py` call nodes replaced by `idem`.
3. **Presence and the derived coordinator.** `workspace/presence.py` register, derivation, `status` line,
   device coordinator lock, tier gate; the `session-liveness.md` slices fold in here.
4. **Membership, enrollment records, revocation, snapshots, bounded fan-out, mutual TLS.**
   `workspace/membership.py`, `fleet/tls.py` client authentication, `fleet/onboard` enrollment signing,
   retention and compaction.
5. **Agents keyed by device; registry as fold.** `identity.py`, `device_sessions.py`, `agents.json` view.
6. **Remove host mode.** Delete `mode: host|remote`, `coordinator.json`, `coordinator_client.py`
   (`Remote`), `coordinator_routes.py` selection, `coordinator_bootstrap.py`, `coordinator.py` RPC,
   `remote.py` project-host client path and `remote_host.py`; `docs/workspace.md` updated. Project boards
   (`remote.py:24-61`, shape with a project host) become a project scope in the same pool, not a host.
   Hosted state converts at first start: the host's views become rows under its own origin.

### Pool rename (the lead runs it separately)

The device group is a pool. Identifiers that say cluster for this concept, with the rename:

| Today | Becomes |
|---|---|
| `poolhouse-cluster` (`pyproject.toml:162`, `fleet/join.py:main`) | `poolhouse-pool` |
| `cluster_key`, `--cluster-key`, `POOLHOUSE_CLUSTER_KEY`, `cluster.key` (`fleet/discovery.py:75-80`) | `pool_key`, `--pool-key`, the matching pool-key environment variable, `pool.key` |
| `cluster_id` (project connections, `workspace/remote.py`, `project_connection.py`) | `pool_id` |
| `cluster`, `--cluster` in the project record shown by `whoami` | `pool`, `--pool` |
| `mint_cluster`, `cluster_group`, `Membership` group fields (`fleet/discovery.py`) | `mint_pool`, `pool_group` |
| `fleet/cluster_modes.py`, `lan_clusters.py`, `automatic_clusters.py` | `pool_modes.py`, `lan_pools.py`, `automatic_pools.py` |
| `poolhouse-cluster join|listen|pair|status|recovery|leave` | `poolhouse-pool ...` |
| `clusters.json` (`discovery.clusters_path` `:176`) | `pools.json` |

Every other file that says cluster for the device group (about 45 under `fleet/` and `workspace/`, plus
`macauth.py`, `sealing.py`, `setup.py`, `mcp.py`, `harnessid.py`; list from `grep -rl cluster src`) is
renamed after a read. Cluster stays where it means data clustering: `graph/community.py`,
`graph/web/components/graph-3d.html`, `walk/`, `bench/folding.py`, `world/names.py`. The beacon protocol
version (`fleet/discovery.py:53`) is bumped; no migration is kept.

## 10. Changes needed in docs/session-liveness.md (not rewritten here)

1. Section 4: the `coordinator-lease` node, epoch and promotion CAS become the derived rule of section 6;
   delete "separate from the board host"; epoch survives only as the task-lease fencing counter. The
   partition paragraph (lines 335-340) becomes derived-per-partition with flagged decisions.
2. Section 2: presence is a per-device summary register gossiped by anti-entropy; `suspect`/`expired` use
   `valid_until` plus `skew_max`; add `unreachable` and `stale-held` after `partition_grace`.
3. Section 3: recovery is an idempotent sweep by any peer after a signed `session-dead` row; same-device
   recovery belongs to the device coordinator; add `yielded` and `superseded` to the hook context (3.3).
4. Section 5.3: rule unchanged, evaluated from replicated summaries. Section 7: the coordinator-lease slice
   becomes slice 3 above and the presence slice writes the register. Section 8 question 1 is removed.

## 11. Decisions

1. The offline window for a revoked or stolen device is 7 days, renewed on every sync; `member`
   `valid_until` and token expiry use the same bound.
2. Adding a device to a pool always needs the person's approval, through the chat-attested
   authorization channel in `docs/person-delegation.md`. A second member's confirmation at
   N >= 3 may be added later and is not required.
3. Landing the shared development branch needs a reachable git remote. A peer's checkout is never
   the push target. Work continues on branches while no remote is reachable.
4. At N = 2 a joiner replays the partner's full journals and verifies them; a snapshot is accepted
   only once a second member can witness it.
5. The device signing key lives in the OS keystore where an attended keystore exists. A headless
   device uses a 0600 file key, and `status` shows which kind each device uses.
6. The replica is encrypted at rest under the device's key. The posted-files store stays on the
   device that holds it and is not replicated until a design for its key exists.
7. Agent ids are `name@device`. A bare name resolves to the agent on the local device.
