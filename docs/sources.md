# Sources: recording hostile addresses so blocking can be automatic

Status: design, 2026-10-09; companion to `docs/sentinel-open-join.md` and `docs/flags.md`. Owner:
"record hostile sources and any other relevant info so blocking can be automatic." Folds in GitHub issue
#42 (open, milestone 0.2): "Reputation for sources (IP, host, peer, repo, hash), long-term and
short-term, divergence alert". Tags: **[V path]** read in code (Python at `ca4eb9b1`; Rust node on
`worktree-agent-afbbf438d36e5c356`); **[D doc]**; **[analysis]** proposal. Nothing in sections 2 to 7 is
built.

## 1. What exists today

Most of issue #42 is **already built for one user's machine**; what is missing is the network side and
sharing. [V `src/ml_stack/reputation/`, `docs/reputation.md`]

| #42 asks for | State |
|---|---|
| per-user encrypted store of sources (IP, host, peer, URL, hash) with events | built: sealed `graph.enc`, key in the OS keystore, 1000 sources, 20 events each [V] |
| long-term reputation (weeks) | built: 30-day half-life, floored at 0.5 (watch) or 1.0 (bad) [V `model.py LONG_HALF_LIFE_S`] |
| short-term reputation (hours) | built: sum of event weights in 6 h [V `SHORT_WINDOW_S`] |
| divergence alert | built: an established source (10 clean runs, 3 days) with short term >= 1 steps to watch at once; one notice through the single-flight dialog; unknown misbehaving source is just bad [V `model.py`, `notice.py`, `hooks.py`] |
| recovery needs clean runs, not waiting | built: 10 clean runs for watch, 3x for bad; `forget` is human-only [V] |
| only tightens; only observations move a score | built: `net.Policy.admit` raises `Distrusted`; no score function takes text [D `docs/reputation.md`] |
| CLI | built: `ml-stack-reputation list|show|forget|export|stats`, person only [V] |
| observation points | built for `net.Pipeline`, `net.download`, `PeerWatch.note` (addresses), `web.read` [D] |
| signed sharing across the user's own machines, as hearsay | **missing** |
| the Rust node as an observer or an enforcer | **missing**: `accept_loop` takes every TCP connection and spawns a thread before TLS, with no source check, rate or cap; beacon intake has none either [V `net.rs` lines 73 to 103, `beacon.rs`] |
| MAC, mDNS name, certificate fingerprint, banner, ASN, interface, expiry rule, blocklist, allowlist | **missing** (traits: cert/key, IP range, redirect, hash, shape are recorded for web sources; not for LAN peers) [V `docs/reputation.md`] |

So this design **extends the existing ledger** rather than adding a second one, and moves the part that
must run before TLS into the node. [analysis]

## 2. The `source` record

A `source` is a board entry kind on the `pool` board, like `flag` (flags.md section 2): signed, hash-
chained, synced to every member, so one device's finding protects all. Local observations stay in the
Python ledger; the entry is the **published, summarised** form. [analysis]

| Field | Meaning |
|---|---|
| `subject` | the key: an IP address or a CIDR (v4 /32 /24 or smaller; v6 /128 or /64), or a certificate fingerprint (64 hex), or a device key |
| `traits` | optional, each bounded: `mac` (hardware address, only as seen on the same segment), `mdns` / `dns` name (cut to 80, control characters removed), `cert_fp`, `device_key`, `banner` (protocol banner or user-agent, 120), `asn` and `owner` (only when a local database already has them; never looked up by a network call), `iface` and `segment` (e.g. `en0 192.168.1.0/24`) |
| `first_seen`, `last_seen` | wall ms |
| `counts` | by event kind, last 24 h and total: `failed_handshake`, `malformed_frame`, `beacon_flood`, `forged_entry`, `port_scan`, `join_attempt`, `rate_hit`, `clean` |
| `raised_by_detector`, `evidence` | detector code and references: entry ids, hashes, counters, never frames or payloads |
| `severity` | `info`, `low`, `high`, `proven` (proven = a signature or hash settled it) |
| `short`, `long` | the two scores of reputation.md (6 h sum, 30-day half-life) so a reader sees both |
| `state` | `watching`, `throttled`, `blocked`, `cleared` |
| `expires` | ms; see 2.1 |
| `set_by` | `detector`, `person`, `coordinator` or `peer` (a hearsay import), with the device and detector |
| `scope` | `local` or `pool`; a `local` row is never synced; the CLI and panel show it in a column |

Transitions: `watching -> throttled -> blocked -> cleared`, any state to `cleared` by a person, and
`blocked` reverts to `throttled` at expiry, then to `watching`, then ages out. Divergence (issue #42):
a source with good long term and bad short term moves to `watching` at once and raises one notice; it
is **not** blocked on divergence alone. [analysis]

### 2.1 Decay and expiry

- `watching`: expires after 24 h without a new event (renewed by each event).
- `throttled`: 1 h, then drops to `watching`.
- `blocked` by a heuristic: 1 h, doubling on each repeat within 7 days up to 24 h (tunable; section 9).
- `blocked` by proof: 7 days, doubling to a 30-day ceiling.
- A person's block has no expiry unless they give one. A person's `allow` has none.
- Records are removed 30 days after `last_seen` (retention, 5.3).
Waiting never moves a source to `cleared` from `blocked` before the expiry above; clean runs shorten
`throttled` and `watching` as the ledger already does for recovery. [analysis; V recovery rule]

## 3. Enforcement in the Rust node

The node enforces; the Python sentinel decides and records (open-join section 4). [analysis]

| Where | Check | Cost |
|---|---|---|
| `accept_loop`, **before TLS** and before a thread is spawned | peer IP against the allowlist, then the blocklist (a sorted CIDR table, binary search), then a per-source and a global cap on concurrent and per-minute connections; a blocked source is closed with no reply and counted | O(log n); no handshake |
| beacon intake, before signature check | source IP against the blocklist and a per-source token bucket (default 10 a minute, burst 20); over it: drop and count | a hash lookup; saves the Ed25519 verify |
| after TLS | the client's certificate fingerprint against blocked fingerprints and device keys | at handshake |
| peer frame loop | a malformed or over-limit frame closes the connection and counts against the source and fingerprint | existing parse path |
| `join_open` | per-source and per-segment attempt cap (open-join 3.4 grade 2) | counter |

**What the node writes.** Counters per source in memory (a bounded LRU of 4096), flushed as a `source`
row when a threshold the sentinel set is crossed, and always for a block. Blocks are `source` rows in the
board's log, so a restart reloads them from disk; a block is never memory-only. [analysis; V board log
is crash-safe, node.md]

**Who may write the blocklist.** The node accepts `blocked` rows from (a) the person's grant, (b) the
sentinel token, which may only add `watching`, `throttled`, `blocked` rows (narrowing), (c) a pool peer
as hearsay: a peer's `blocked` row is imported as `throttled` and becomes `blocked` here only when this
device has its own matching evidence or the peer row is `proven` with evidence this device can verify
(a signature or hash). One device cannot block the pool on its word alone. [analysis]

### 3.1 Never block the owner (safe mode)

An attacker must not be able to make the node block the owner's own devices. Rules, in order of
precedence: [analysis]

1. **Never blocked, at any layer:** loopback; link-local; the node's own addresses; every address and
   fingerprint of an `active` member in good standing; every entry on the owner's allowlist. A member
   whose address is also a hostile source's is handled by **flags** (probation, restrict, suspend), never
   by an IP block, so a spoofed source address cannot cut off a real member. A member is judged by its
   key, not its address.
2. **Shared NAT and wide ranges.** A CIDR wider than /24 (v4) or /64 (v6) is never auto-blocked; a block
   on a single address that other members' traffic also came from in the last 24 h is downgraded to
   `throttled` and notified.
3. **Spoofed sources.** TCP connections and TLS-authenticated frames cannot be spoofed past the
   handshake, so counts that block come from them. **UDP beacons can be spoofed**: a beacon source
   address alone never gets a `blocked`; at most `throttled` (drop beacons from it) with a short expiry.
   A block needs a completed TCP connection (a three-way handshake the sender had to answer).
4. **Lockout rate limit.** No more than 8 new auto-blocks an hour and 64 active auto-blocks; past
   that, new blocks are `throttled` and the person is told ("blocking is at its limit"). A flood of
   spoofed TCP sources cannot fill the table with the owner's neighbours.
5. **Safe mode, one verb.** `sources safe-mode on` (person) stops all automatic blocking (detectors still
   record and notify); it takes effect at the next accept. The node enters safe mode itself when the
   sentinel is dead, when the block table's size or churn exceeds its limit, or when it blocks the
   address a local session came from. In safe mode only `throttled` applies.
6. **A person is never locked out**: the local socket does not depend on source checks; `sources
   allow` and `sources unblock` work from the socket only.

## 4. Detectors that feed it

Each is a counter in the node (sections 3 and 4 of open-join) read by the sentinel, with a `source`
subject. Thresholds are starting values. [analysis]

| Code | Signal | `throttled` | `blocked` |
|---|---|---|---|
| `failed_handshake` | TLS failures, unknown-cert connections that never speak, pin mismatches, per source | 10 in 5 min | 50 in 5 min (heuristic) |
| `malformed_frame` | bad length, unknown op/field, frame that does not parse, over the cap | 3 in 5 min | 10 in 5 min; any frame cut off after a 4 MiB length prefix, 5 times |
| `beacon_flood` | beacons per source above the bucket, oversized or malformed beacons | 30 in a minute | 300 in a minute: `throttled` only (UDP, 3.1 rule 3) |
| `forged_entry` | a signed row or head that fails its hash or signature, from the owner of the origin | n/a | `proven`, 7 days |
| `port_scan` | connections to the listener or a decoy port that close without a handshake, from one source across 5 or more distinct local ports (needs a decoy or a counted refused-connection source) | 5 ports in a minute | 20 ports in a minute |
| `join_attempt` | repeated `join_open` or pairing attempts, wrong pairing codes | 5 in 10 min | 15 in 10 min; a third wrong code in one pairing window: `blocked` for that window |
| `rate_hit` | quota or rate replies on the peer API from a non-member | 10 in a minute | 60 in a minute |
| `cert_change` | the same source or name with a new certificate or key (ledger event `cert_or_key_change`) | note | divergence notice only |
| `successor` | a new fingerprint appearing right after a revocation from the same address | note | never auto-blocks; goes to a flag (open-join 3.3) |

`PeerWatch` today maps outcomes `bad_sig`, `replay`, `clock`, `locked`, `oversize`, `unsigned` to the
same ledger; those become the Python side of these codes, so one vocabulary is kept. [V `rates.py`]

Hearsay from a pool peer is marked in `set_by` and in the CLI; it never moves a score by itself (only
this device's observations do, as reputation.md requires). [D `docs/reputation.md`; issue #42 "hearsay"]

## 5. Privacy and retention

Addresses, MAC addresses and names are personal-ish data. [analysis]

1. **What is stored:** the subject, the optional traits in 2, counts and timestamps, detector codes and
   evidence references. Not stored: payloads, frames, packet captures, URLs a user visited, user names,
   anything a source sent beyond a bounded banner.
2. **Where:** local ledger rows stay in the sealed per-user store (encrypted, no plaintext names on disk,
   canary-grep tested per #42 acceptance 3). Pool rows live in the board's log, which is not encrypted
   from the member devices that hold it; they go only to members of the one pool.
3. **Retention:** 30 days after `last_seen` for a `watching` or `cleared` row; the blocklist keeps a row
   until its expiry plus 30 days; the 1000-source cap evicts the oldest first. A row whose address is
   a private-range address of this segment is removed at 7 days. An export is `ml-stack-reputation
   export` (person only); `forget --all` removes the local store; a pool row is removed by a
   `cleared` entry that carries no subject after retention (a redaction row), so history does not hold
   addresses forever. [analysis]
4. **Not shared outside the pool.** No public feed in and none out; no network lookup for ASN or owner.
5. **MAC addresses** are recorded only for a source on this device's own segment and only from the
   frame the node already received; they are never routed or synced beyond the pool. A person may turn
   MAC recording off (`sources record-mac off`).

## 6. CLI and UI

```
ml-stack sources list [--state S] [--scope local|pool] [--json]
ml-stack sources show SUBJECT
ml-stack sources block SUBJECT [--for DURATION] --reason TEXT
ml-stack sources unblock SUBJECT --reason TEXT
ml-stack sources allow SUBJECT --reason TEXT      # also: allow --remove
ml-stack sources safe-mode on|off
```

- `list` and `show` are not privileged (as `ml-stack-security review --list`); `block`, `unblock`,
  `allow`, `safe-mode` need the person's `HumanGrant`, and refuse under any agent marker. `show` prints
  both scores, traits, counts, evidence ids, who set it and `scope` (local or pool). [V pattern
  `sentinel/human.py`]
- `reputation list/show` stay for web sources; `sources` is the network face of the same ledger and
  prints a source from either store, tagged. `ml-stack-reputation forget` and `sources unblock` clear the
  same record.
- UI: one component on the Fleet page next to the flags panel: a table (subject, state, scope, short and
  long score, last seen, set by) with Block, Unblock, Allow, each one confirmation; `digest --status`
  gains "3 sources blocked, 1 allowed; safe mode off". New divergence and block events use the existing
  single heads-up dialog. [analysis; V `explain.py` pattern]

## 7. Implemented vs missing, with slices

| Piece | State |
|---|---|
| local ledger with short/long scores, divergence, recovery, notice, `ml-stack-reputation` | implemented [V] |
| `PeerWatch` for HTTP peers | implemented [V] |
| sealed per-user store | implemented [V] |
| `source` entry kind, traits, expiry, scope | missing |
| node: accept-time and beacon-time source checks, caps, blocklist/allowlist, LRU counters | missing |
| safe mode and the lockout limits | missing |
| node detectors in 4 | missing |
| `sources` CLI and panel | missing |
| hearsay import and the evidence-based promotion | missing |
| signed sharing across the user's machines (issue #42 "optional later") | missing; the board sync is that mechanism |
| retention and redaction rows | missing |

| # | Slice | Size | Tests | Depends on |
|---|---|---|---|---|
| S1 | Node: connection admission before TLS and a thread cap: allowlist, blocklist table, per-source and global caps, close without reply | M | 10k connections from a blocked source spawn no threads; an allowed member is never refused; loopback never blocked | node network branch |
| S2 | Node: beacon intake bucket and blocklist before verify | S | flood from one source: verify count bounded; a member's beacon in the same minute is processed | S1 |
| S3 | `source` row kind, persistence in the board log, expiry and decay, reload after SIGKILL | M | a block survives a kill and restart; expiry lifts it; retention removes rows | F1 (flags.md), node |
| S4 | Safe-mode rules (3.1): never-block set, wide CIDR and NAT downgrade, UDP-only throttle, lockout limits, node-entered safe mode | M | a spoofed UDP flood from a member's address blocks nothing; 100 spoofed-then-answered sources hit the limit and the person is told | S1, S3 |
| S5 | Node detectors and counters (4), exposed on the socket | M | each detector at threshold and one below; honest sync traffic raises nothing | S1, S2 |
| S6 | Sentinel `sources.py`: reads counters, writes ledger events and rows, divergence for LAN peers (certificate/key change trait), notices | M | the divergence cases of #42 acceptance 1 on a real node: long-clean peer with a new cert goes to `watching`, one dialog | S3, S5 |
| S7 | CLI `sources`, panel, `digest --status`, explain sentences | M | `HumanGrant` and agent-marker refusals; block takes effect on the next accept | S3 |
| S8 | Hearsay import and promotion by local evidence; scope column | S | a peer's `blocked` row is `throttled` here; with a verifiable `proven` row, blocked | S3, S6 |
| S9 | Privacy: retention job, redaction rows, MAC toggle, canary-grep for the pool log | S | a row older than 30 days is gone; a grep of the local store finds no address | S3 |
| S10 | Red-team: spoof the owner's address to get it blocked, fill the table, scan the ports, race a restart against a block, split an attack across many addresses, name-based evasion (new key per attempt) | M | each leaves the owner's devices connected and produces the evidence line | S1 to S7 |

Order: S1, S2, S3, S4 (nothing auto-blocks before S4), S5, S6, S7, S8, S9, S10.

## 8. Relation to flags and probation

A flag is about a **subject that is a known identity** (an agent, a device in the pool). A source is about
**where a connection came from**, before any identity exists. They meet at three points: [analysis]
- A device's `suspend` flag adds its current address as `throttled`, not `blocked`, to avoid locking a
  shared address.
- A blocked source that later enrols under `open` is refused at accept time, so it never reaches probation.
- A source row that belongs to a revoked fingerprint lets the sentinel link a successor (open-join 3.3).

## 9. Owner decisions

1. **Heuristic block length.** (a) 1 hour, doubling to 24 h on repeats [recommended]; (b) 15 minutes;
   (c) 24 hours at once.
2. **May a pool peer's evidence block here?** (a) Only if this device can verify it or has its own
   matching events [recommended]; (b) a peer's block applies after a quorum of two; (c) never, local only.
3. **Are MAC addresses recorded at all?** (a) Yes, on the same segment, with a toggle [recommended];
   (b) no.
4. **Retention.** (a) 30 days after last seen [recommended]; (b) 7 days; (c) 90 days.
5. **Auto-block cap.** (a) 8 new an hour, 64 active [recommended]; (b) 2 an hour; (c) no cap.

Location, ASN and cluster alerts, and privacy and jurisdiction, are in `docs/sources-location.md`.

## 10. Not verified

Nothing here was run. I did not read `net.Pipeline`'s observation code, only reputation.md's list; I did
not check whether the Rust accept loop has any per-IP limit elsewhere (it has none in `accept_loop`); and
I took the 1000-source cap and the scores from `docs/reputation.md` and the `model.py` constants, not
from running the ledger. Whether "ASN or owner" can be known without a lookup depends on a local database
that does not exist today.
