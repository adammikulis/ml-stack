# The sentinel and automatic connection: making join policy `open` safe

Status: design, 2026-10-09. Nothing in sections 3 to 6 is built. Owner statement this serves: "feel
free to buff up the sentinel ... it's how we can make devices auto-connect while still being safe."
This file sits beside `docs/sentinel.md` (the built sentinel: its design, tests and measurements) and
does not replace it; that file is 514 lines and the brief's 400-line cap would have meant deleting
half of it.

Tags on every claim: **[V path]** read in that code (Python on `0.2dev` at `ca4eb9b1`; the Rust node on
branch `worktree-agent-afbbf438d36e5c356`, network side, and `worktree-agent-a3e2f7ab3d22cc15b`,
leases); **[D doc]** stated in a repo document, not re-checked; **[analysis]** my inference or a
proposal. Words: the board, one kind of pool, join policy `open | secure`.

## 0. Summary

- Today the sentinel protects one node's agent loop, model files, servers and its peers' signed
  requests. It has no notion of a device joining, and nothing in it runs in the Rust node. [V]
- The node's `open` policy enrols any device that speaks `join_open` from the same segment, as a full
  `active` member, with no cap on how many. [V `membership.rs enrol`, `peer.rs join_open`] That is
  the whole risk: one step takes a stranger from "on the wifi" to "holds the pool's boards".
- Proposal: a `probation` standing between unknown and active, enforced in the node (a state machine,
  caps, quotas, frame limits), and judged by the Python sentinel (detectors, graded responses,
  promotion, notices) over the node socket. Every decision is a sealed audit entry on the `pool`
  board. The person has a brake that bites on the next request. [analysis]
- Safety comes from what a probationary device cannot do, not from guessing it is friendly. A hostile
  device that gets in can read and write only boards it is invited to, and holds no GPU, no exclusive
  lease, no coordinator role, no link and no source. [analysis]
- Seven owner questions are in section 7.

## 1. What the sentinel is and does today

### 1.1 Parts

| Part | What it does | Where |
|---|---|---|
| Event bus and chained log | typed events, HMAC-chained JSON lines, `verify`, optional off-box anchor | `events.py`, `redaction.py` [V] |
| Store and state machine | `clear -> watch -> quarantined -> released`, one record per `(kind, key)`, fails closed if the seal breaks | `store.py`, `sealed.py` [V] |
| Observers and standing | request outcomes and denials become observations against a source (`observers.observe("peer", addr, "denial")`), which feed the source-risk ledger's good/watch/bad state | `observers.py`, `rates.py` [V]; ledger [D `docs/reputation.md`] |
| Detectors | `PeerWatch` (bad signature, replay, clock, lockout, oversize, rate, flapping, version or binary mismatch), `ToolMix`, `Abuse`, `RailWatch`, honeytokens, canaries, integrity pins | `rates.py`, `rails.py`, `honey.py`, `canary.py`, `integrity.py` [V] |
| Policy | modes `observe`, `guarded` (default), `enforce`, `off`; `guarded` acts only on high-confidence signals | `policy.py` [V] |
| HumanGrant | release, purge and mode changes need a grant minted at a terminal (or a dialog button), 120 s, one action on one subject, refused under any agent marker | `human.py` [V] |
| Heartbeat and scan threads | `Scanner` is a daemon thread in the Broker daemon and the fleet daemon: every pin and decoy at start, then every `ML_STACK_SENTINEL_SCAN` s (300), deep hash every 12th round, canaries after each scan; a sealed `scanner.json` heartbeat says it is alive | `watch.py` [V] |
| Heads-up | one dialog, never a stack; single-flight lock; fixed sentences | `heads_up.py`, `explain.py` [V] |

### 1.2 What it protects

A model file that changes after its pin; a server binary that changes; a peer address that forges or
replays signed requests to the fleet daemon (the `Authenticator` is wrapped by `watch_authenticator`;
an address sentinel holds is refused); an agent that touches a decoy or reaches a sentinel verb; held
text that reappears. Releasing anything is human-only. [V; D `docs/sentinel.md` "What is armed by
default"]

### 1.3 What it does not protect

- Its subject for a peer is an **address** (`("peer", addr)`), not a certificate. A device with a new
  address and a new key starts clean. [V `rates.py PeerWatch.note(peer, ...)`]
- It sees the Python fleet daemon's requests only. The Rust node has no hook into it. [V by absence;
  node.md "Nothing in Python calls it yet"]
- It does not look at who is joining, only at requests after a peer is already trusted. There is no
  enrolment detector, no probation, and `Roster.enrol` takes whoever the join flow authenticated. [V
  `fleet/membership.py enrol`]
- A process running as the owner can read the sentinel key and rewrite its state; the defence is
  against agents and peers. [D `docs/sentinel.md` "Self-protection"]
- State replaced by an older validly sealed copy is not detected. [D same]
- It does not stop an unmanaged process or delete anything. [D same]

## 2. Threat model for automatic connection

Attacker positions: **N** hostile device on the same network segment; **L** hostile process on the
same machine; **M** a member that was honest and is now compromised. `open` = a device is enrolled on
first contact from the segment, fingerprint recorded, revocation sticky, beacons signed and carrying a
pool id. [V node.md "Pairing", `beacon.rs`]

What the node already does, so it is not repeated as a gap: beacons are signed, at most 1024 bytes and
valid 60 s; a bad signature, stale time, loopback or multicast address is dropped; a non-member beacon
is ignored unless the policy is `open`; the dial is pinned to the beacon's fingerprint; peer frames are
at most 4 MiB with a 30 s IO timeout; a revoked certificate cannot be enrolled again; revoked wins every
merge; sync is bounded to 400 rows, 2 MiB and 16 origins a request; per-board quotas hold (64 origins,
200,000 rows, 128 KiB a row); foreign rows 5 minutes ahead are held back. [V `beacon.rs`, `wire.rs`,
`peer.rs`, `membership.rs`, node.md "Quotas"]

| # | Attack | Position | What happens today [V unless noted] | Gap |
|---|---|---|---|---|
| 1 | Spoofed beacons from an address that is not the sender | N | UDP source is not authenticated; the beacon carries its own `addr`, signed by the key it carries, which anyone can generate. A member's beacon must use the key in its certificate, so a non-member cannot move a member. | A forged beacon for a non-member costs one dial and one `hello` each. No cap on dials per second. |
| 2 | Beacon flood | N | Decode is cheap (signature check), but every verified non-member beacon under `open` may trigger a dial and a TLS handshake. | No per-source, per-segment or global rate on beacons or dials. |
| 3 | A beacon naming the real pool id | N | The pool id is public (broadcast by every member). Naming it only makes the receiver dial; the dial is pinned to the beacon's fingerprint, so the attacker must then complete `join_open` with its own certificate. Under `open` that succeeds. | The pool id proves nothing; `open` itself is the gap. [analysis] |
| 4 | One fingerprint from several addresses | N | The record is keyed by fingerprint; an address updates only for a member whose beacon key matches. | Address flapping is not observed, and nothing records how many addresses one fingerprint used. |
| 5 | Replay of captured beacons | N | Valid for 60 s by timestamp; a replay inside the window re-announces a real member at its real address. A replay cannot enrol anyone, since the replayer lacks the key. | A replayed beacon wastes a dial; at most 60 s of stale address. Low. [analysis] |
| 6 | Enrol, then misbehave | N | An enrolled device is a full `active` member: it syncs every board both hold, can acquire leases, `link`s are per session. | The core gap. Nothing separates "just walked in" from "trusted for months". |
| 7 | Resource exhaustion by many enrolments | N | `enrol` has no cap; each adds a row, a save of `pool.json`, and an entry in every peer's membership exchange (`MOST_ROWS` 4096 is the Python limit; Rust unchecked here). [V `membership.py MOST_ROWS`] | Unbounded membership growth; revoking each requires a person action. |
| 8 | A compromised member | M | Full member authority; revocation is a record that spreads by `members`. | Needs the same containment as a probationer: caps and quarantine reachable from detectors, not only by a person. |
| 9 | Clock skew games | N, M | Beacons are bounded to +-60 s; foreign rows to +5 min; the Python authenticator has a 120 s window. A member can set its HLC ahead to order itself first (first claim in the total order wins a target). | Skew is held back at 5 min but not counted or reported; an HLC far in the past is not examined. |
| 10 | Malformed or oversized frames | N, L | Local frames 1 MiB, peer frames 4 MiB, beacons 1 KiB; unknown fields do not parse; length prefix read once. | Not yet fuzzed; a length prefix of 4 MiB sent and then withheld ties a thread for 30 s. [analysis] |
| 11 | Slow-loris | N | 30 s read timeout per frame; one thread per connection by the accept loop. | No cap on concurrent unauthenticated connections or per-source concurrency. [analysis, `net.rs` accept loop reads as thread-per-conn] |
| 12 | Impersonating a revoked device with a new key | N | A new key is a new fingerprint; `enrol` accepts it under `open`. The revoked name is just a label (name is control-char-stripped, 80 chars, not unique). | Under `open`, revocation is defeated by re-joining. The sentinel must link successors (same name, same address, same host key hints) and carry the revocation reason forward. |
| 13 | Rogue DHCP or mDNS name | N | Nothing is matched by name; the dial is pinned to a fingerprint. A rogue DHCP server can route a member's address to the attacker, but the TLS pin fails. [V `beacon.rs`: "never matched by name"] | Only denial: the attacker can make a member's address unreachable. Report address flapping; do not trust it. |
| 14 | Hostile local process reaching the node socket | L | Socket is 0600 in a 0700 dir; `SO_PEERCRED` / `getpeereid` refuses another uid; tokens are stored as SHA-256; the Grants stub lets any registered session act. [V node.md "Local API"] | Same-uid processes (any agent or tool the user runs) are inside the line. Membership methods (`set_join_policy`, `member_revoke`, `pair_start`) need the owner-only grant, which is a stub today. This is the item that must land before `open` ships. |
| 15 | Fake `hello` claiming a policy to bait a joiner | N | A joiner reads `policy` from the peer's `hello` and then offers its certificate. A rogue pool at the same segment can collect certificates (public data) and learn this device's name. | Low impact; but a lone device joining the first `open` pool it hears is a hijack: the rule is "alone and the beacon's pool id sorts before its own". [V `net.rs` comment, node.md] An attacker picks a pool id that sorts first, so a **new device with `open` can be captured into a hostile pool**. |

Item 15 matters most for a user: a fresh laptop joins the strangers' pool in the coffee shop. `open`
therefore needs a statement of what a lone device may adopt (section 3.7).

What `open` must never allow, whatever else is decided: [analysis]

1. A device enrolled by `open` holding an exclusive lease, the GPU, or the coordinator role.
2. Such a device making a link, adding a source or a project, or setting policy.
3. Enrolment without a durable, sealed audit entry naming fingerprint, address, time and by-whom.
4. A revoked fingerprint, or a device the sentinel linked to one, returning silently.
5. A lone device adopting a pool without the person being told which pool and which devices.
6. Any decision the person cannot see afterwards, or cannot undo with one action.
7. Enrolment while the sentinel is off or dead (fail closed: no heartbeat, no new `open` enrolments).

## 3. The design that makes `open` safe

### 3.1 Standing: a probation state in the membership record

Add `probation` to the existing `active | revoked`. A device enrolled by `open` starts as
`probation` with `since`, `until` (earliest automatic promotion), `by: open`, the first address, the
beacon id seen and the segment (subnet) it joined from. Tokens, `pool.json` rows and the Python
`Roster` read it the same way: `Device.read` ignores unknown fields today in Python, so the field is
additive. [V `membership.py Device.read`; analysis] `quarantined` is a flag on a row, not a status
(connected but isolated), so that `revoked` stays terminal and is never undone.

| Capability | Unknown | Probation | Active | Quarantined |
|---|---|---|---|---|
| `hello`, `join_open`, pairing frames | yes | no | no | no |
| `members` (read the roster) | no | read, own row only | yes | no |
| sync boards it is **invited** to | no | yes | all boards both hold | no |
| sync other boards | no | no | yes | no |
| acquire exclusive lease, GPU, model slot | no | no (counted resources only, small) | yes | no |
| coordinator role, tiers, leases for others | no | no | per trust ledger | no |
| `link`, `project_add`, `source_add`, grants, policy | no | no | owner-only grant | no |
| revoke another device, merge membership rows | no | no | active only | no |
| rows per minute, bytes per hour | n/a | capped (below) | pool quota | 0 |

Invitation is a board entry (`kind: audit`, `event: invite`, `subject: fingerprint`, `board`) written
by an active member's session. A probationer on a board with no invite sees nothing of it. Fresh
`open` enrolment therefore reveals no project data: the pool's value is in the boards. [analysis]

Caps for probation (starting values; ranked options in section 7): 20 rows a minute, 5 MiB a day
inbound and outbound per probationer, 4 boards, 2 concurrent connections, and 8 probationers pool-wide
at once. Past the pool-wide cap, `open` answers `join_open` with `busy` and records the refusal. [analysis]

### 3.2 Promotion

To `active` when all hold: `now >= until` (default 24 h, section 7), no detector above `notice` in the
window, at least N clean syncs (default 5) and one clean heartbeat from the sentinel since enrolment.
Instantly when a person confirms ("Let it in", one button in the heads-up dialog or one CLI verb that
needs a `HumanGrant` for `promote <fingerprint>`) or a standing-grant covers it (section 3.6). The
promotion is a membership change and a sealed audit entry; time alone never promotes while the sentinel
is not running. [analysis]

### 3.3 Detectors (sentinel, Python)

Each emits a `Finding` through the existing path (`Sentinel.handle` -> policy -> store), with a subject
`("device", fingerprint)` or `("segment", subnet)`. Window sizes are `Windows` (60 s default). [analysis]

| Detector | Signal | Class |
|---|---|---|
| beacon rate and shape | beacons per source per window; size near the limit; unknown fields; repeated fingerprints with changing `addr` | heuristic |
| enrolment rate per segment | `join_open` completions per hour per subnet and per pool | reliable as a count; threshold heuristic |
| duplicate fingerprint | one fingerprint announced from more than one address in a window, or an address announcing more than one fingerprint | reliable as a fact; intent heuristic |
| address flapping | `peer.flapping` generalised from address to fingerprint | heuristic |
| failed handshakes | TLS failures, `pin` mismatches, unknown-cert connections per source | heuristic |
| protocol violations | unknown op, unknown field, frame over the cap, frame that does not parse, op outside its standing | reliable (no honest node sends it) |
| quota hits | `quota` replies, per-board row caps, 16-origin and 400-row bounds hit | reliable as a count |
| forged entries | a signed row whose hash or head fails; `damaged` from the owning device | reliable (forgery), high confidence |
| replay | an old head or row presented as new; a beacon older than its high-water mark for that key | reliable |
| clock skew | HLC beyond +5 min, or beacon ts beyond +-60 s, repeatedly | reliable as a fact; usually a clock, counted low (as `PeerWatch` "clock" is) |
| successor of a revoked device | new fingerprint with the same name, address or host within the window after a revocation | heuristic |
| lone adopter | a device that was alone adopting a pool | reliable as an event; always notified |

The node counts and rate-limits; it does not decide. It exposes these as counters and an event stream
on the socket (section 4) so the sentinel needs no packet access. [analysis]

### 3.4 Graded responses

| Grade | Trigger | Effect | Reversal |
|---|---|---|---|
| 0 note | single clock or malformed beacon | event only | none needed |
| 1 throttle | rate or quota hit, flapping | node halves the device's caps for an hour | decays |
| 2 pause intake | enrolment rate over the segment threshold | node stops accepting `join_open` for that segment; existing members unaffected | timer, or the person |
| 3 quarantine | protocol violation repeated, duplicate fingerprint, successor of a revoked device, a probationer's quota hit twice | device stays connected but receives and gives nothing; standing `quarantined`; its rows are held, not merged | person releases (HumanGrant) |
| 4 auto-revoke | proven forgery (bad head or hash from the owner of the origin), proven replay of a signed frame, a second device forging the same fingerprint's signature | `revoked`, sticky, spreads by `members` | none (new certificate only) |
| 5 close | the pool-wide count or rate is in a flood | policy flips to `secure` and the person is told | the person flips back |

Auto-revoke is for **proof**, not suspicion: only conditions a signature or a hash settles. A
heuristic can reach quarantine and no further. `guarded` mode keeps its present meaning (act only on
high-confidence signals); `enforce` also acts on the heuristics up to grade 3. No grade is silent: each
posts a board entry and a notice (3.5). [analysis]

### 3.5 Audit and notice

- Every enrolment, promotion, throttle, quarantine, revocation, policy change and brake use is a `pool`
  board entry of kind `audit`, written by the node (not by a session), signed in the device log, with
  evidence as ids, counts and digests, no frame bytes. It syncs to the members that exist, so a
  compromised device cannot erase the record on its peers. [analysis; entry shape V node.md]
- The sentinel mirrors the entry into its own chained log (`events.log`), so a tail removed on one side
  shows on the other. [analysis]
- A notice goes to the person through the existing heads-up (single-flight dialog, fixed sentences),
  extended with `device_joined`, `device_quarantined`, `device_revoked`, `open_paused`. `explain.WHY`
  gets one sentence per kind, and `tests/test_sentinel_explain.py` already fails if one is missing.
  [V `explain.py`/test; analysis]
- A new device is told once, on the first enrolment of the day, in one line ("<name> joined from
  <address>, on probation until <time>"); a flood is one dialog, not a stack. [analysis]

### 3.6 The brake, and the trust ledger

The brake is three verbs, each one action by the person (CLI with `HumanGrant`, Fleet page button),
effective at the node's **next request** because the node reads the state under its lock on every
request (as it already does for revocation: "refused at its next request on the same connection"):
[V node.md; analysis]

1. `pause-open`: no new enrolment; probationers stay probationers.
2. `revoke-probation`: every device still in probation becomes `revoked`, one audit entry listing them.
3. `policy secure`: `set_join_policy`, newest change wins.

Brake use is not a demotion of anyone (earned-trust 3.4 item 5). [D `docs/earned-trust.md`]

Trust ledger: a device is an account too. [D `docs/earned-trust.md` section 1.3 names the device
account; `work_agent` is per identity.] Proposal, tagged analysis: the probation/active boundary uses
the same vocabulary as L1/L2. A probationer is L0-L1 for device-class resources. Clean syncs and
verified signatures accumulate as evidence against its fingerprint in the source-risk ledger
(kind `peer`, key `device:<fingerprint>`, 6 h short term, 30 d half-life). A device earns L2 (active
for all boards) by evidence, and keeps it the way an agent does: a hard fault (3.4 grades 3 and 4)
drops it to L0 at once. L3 (coordinator) for a device needs the ledger's accepted-work evidence plus
`coordinator_eligible`; never from time alone. L4 never. A person's "Let it in" is a standing-grant
on one fingerprint, scoped, logged and revocable (graduated autonomy).

### 3.7 What `open` is allowed to do when the device is alone

The capture case (attack 15): a lone device adopting the first `open` pool it hears. The owner
decided (section 7, decision 4: (c), auto-adopt) that a lone device does adopt one, limited to a
pool of the same project. [owner decision]

- **Project.** A pool belongs to one project, a 16-hex-digit key (`projectid.rs`) taken from the
  repository that uses the node: the normalised `origin` remote (`github.com/owner/repo`, the same
  from every clone and spelling), else `local/<name of the main working tree>`. The owner can name
  a project instead (`--project NAME`). The key is a hash, so a remote URL is never put on the
  network. For this repository the project is ml-stack itself.
- **Start.** A node with a project and a beacon takes the project for its pool of one, stays quiet
  for the settle window (4 s) and joins the first `open` pool of the same project it hears. If it
  hears none, it makes its pool `open` and beacons it. No key file and no code is involved.
- **What is heard.** A beacon carries the project and whether the sender is alone. A pool of
  another project, a pool with no project and a `secure` pool are ignored (the joiner also checks
  the project and policy in the pool's own `hello`, so a beacon cannot lie about either).
- **Convergence.** Two lone devices keep the id ordering (the one with the larger pool id joins the
  smaller). A lone device joins any pool whose sender is not alone, whatever its id, so a device
  starting beside an established pool always joins it. Two devices that start at once both make a
  pool after the window and meet by the id ordering.
- **What this costs.** The residual risk is attack 15 for one project: anyone on the segment who
  can claim the project's key can host the pool a lone device adopts. The key is derived from a
  public remote, so it is a label and not a secret. The defence is what follows adoption: the
  device is in probation (3.1) and the detectors (3.3) watch it, and the person can set `secure`.

### 3.8 Sentinel crash and absence

If the sentinel's heartbeat (`scanner.json`) is older than three intervals, the node refuses new
`open` enrolments and holds probationers' promotions; existing members and sync are unaffected.
Enrolment fails closed, running does not. [analysis; heartbeat shape V `watch.scanner_state`]

## 4. Where each piece lives

| Piece | Where | Why |
|---|---|---|
| membership states, transitions, sticky revocation | Rust node `membership.rs` | the authority; checked per request |
| per-state capability table, invitations | Rust `peer.rs`, `rules.rs`/`grants.rs` | one place a request is allowed |
| rate caps, quotas, frame limits, timeouts, connection caps | Rust `wire.rs`, `net.rs`, `beacon.rs` | must hold with Python dead |
| counters, event stream, `device_note` / `device_state` methods | Rust `api.rs` | the sentinel's sensors and actuators |
| detectors, windows, thresholds, successor linking | Python `sentinel/devices.py` (new) | analysis changes often; reuses `Windows`, `Finding`, `policy.decide` |
| decisions: throttle, quarantine, revoke, pause, promote | Python via node calls | node verifies standing allows the call |
| notices, dialog, `explain.WHY` | Python `heads_up.py`, `explain.py` | already single-flight |
| audit entry | written by the node, mirrored by the sentinel | tail removal visible on both |
| brake | node verbs; CLI (Python) and Fleet page call them | works if either side is down |

Node-side rule: the sentinel's calls are ordinary socket requests under a sentinel token, not a
backdoor. The node enforces that the sentinel may only **narrow** (throttle, quarantine, revoke, pause)
by itself; **widening** (promotion, release) needs a person's grant, a time rule the node itself checks,
or a standing-grant. A compromised sentinel can therefore deny service but not admit anyone. [analysis]

The one-daemon refactor: the node replaces the Broker and fleet daemon (the lease and board work in the
two branches). The sentinel outlives it as a **client of the node socket**, like any session: the
`Scanner` thread becomes a small process (or thread in the node's Python companion) that polls
`pool_status` and the event stream every few seconds and writes its heartbeat where the node reads it.
The present in-daemon scan loop (integrity, canaries, decoys) keeps running in whatever hosts the model
servers. Nothing in section 1 moves; the peer observers move from "wrap the Python `Authenticator`" to
"read the node's peer outcomes", with the same `PeerWatch.OUTCOMES` vocabulary plus the new ones.
`sentinel/adapters.py` is the one seam; the Python fleet daemon keeps `watch_authenticator` until it
is removed. [V adapters seam; analysis]

Python `fleet/membership.py` and `pool_roster.py` stay as the reader of the same record while both
exist; the new `probation` and `quarantined` fields are additive and ignored by old readers, which
would treat the device as `active`. That is unsafe: ship the Python reader's understanding of
`probation` before the node writes it. [analysis]

## 5. Red-team plan

Scored by evidence a script reads (a counter, a log line, a status field, a refused call), never by a
model. Two layers.

**5.1 PyRIT, extending `src/ml_stack/redteam`.** `pyrit_bridge.fire(...)` sends a prompt through
deterministic converters and scores objective evidence; targets wrap any `Responder`. [V
`pyrit_bridge.py`, `targets.py`] The network attacks are not prompts, so PyRIT is used where text is
the attack: (a) device names and labels (control characters, bidi, very long, a name that reads as an
instruction) through converters `base64`, `confusables`, `zero_width`, enrolled via `join_open` into a
real node; evidence is that the notice shown to the person carries the escaped, cut name, and that no
held name reaches a model context unscreened. (b) a hostile board entry (note, message) from a
probationer, evidence is it is foreign-labelled and held by the screen. New scenario file
`scenarios/devices.py`, with a `Responder` that drives a local node over its socket. [analysis]

**5.2 Protocol fuzzing against the Rust node.** `cargo-fuzz` targets: `fuzz_frame` (length prefix and
body for local and peer framing, with the 30 s timeout replaced by an injected clock), `fuzz_beacon`
(`beacon::decode`), `fuzz_row` (`row::parse` / board entry decoding), `fuzz_members` (the
`members` merge), `fuzz_pair` (SPAKE2 message and confirm). Invariants: never panic, never allocate
past the cap, never produce an entry that fails re-encode round-trip, never accept a row whose hash
differs. Seed corpora from the node's own tests. [analysis]

| Scenario | Evidence it must produce |
|---|---|
| beacon flood from one source (10k/s) | node dial rate stays under its cap; `beacon_dropped` counter rises; a finding `beacon.rate`; member sync keeps working (a member pushes a row in the same minute) |
| spoofed beacon naming real pool id, attacker key | a dial is made once per window; `join_open` under `secure` is refused; under `open` the device is `probation`, not `active` |
| beacon replay inside 60 s then at 61 s | first: address unchanged for a member; second: dropped as stale; `replay` count |
| one fingerprint from 3 addresses | `duplicate_fingerprint` finding; grade 3; address of the member not changed |
| 200 enrolments from one subnet | after the cap: `busy`; `open_paused` audit entry; membership row count never exceeds cap; notice raised once |
| enrol then flood posts | throttled at the caps; rows over cap refused `quota`; no board data outside invites readable; grade 1 then 3 |
| enrol then try `link`, `source_add`, lease, GPU, coordinator | each `denied: probation`, with the audit entry; none changes state |
| probationer forges a row for another origin | `damaged` marks only when the owner sends it; relay copies never mark; sentinel finding on the sender; auto-revoke only for the owner |
| revoked device returns with a new key and same name | `successor` finding, grade 3, not silent; the old fingerprint still refused |
| compromised member (active) floods | grade 1-3 reach it identically; caps in the node bound the damage |
| clock skew +10 min HLC | rows held back from the view; `clock_skew` count; no ordering change in the claim table |
| frame length prefix 4 MiB then silence, x500 connections | per-source and global concurrency caps; threads bounded; timeout closes; `slow_connection` count |
| malformed or oversized frames (fuzz) | no panic, no cap breach, `invalid` replies counted |
| local socket from another uid | one `denied` frame and close (existing test); and from same uid without grant, `denied` on membership methods |
| sentinel killed during an `open` flood | node refuses new enrolments after three intervals (3.8); evidence in the node's status; members keep syncing |
| brake: pause, revoke-probation, switch to `secure` mid-flood | next request after the verb is refused; the three audit entries exist; timing measured in requests, not seconds |
| lone device offered a hostile pool that sorts first | no adoption without the person's confirmation; a `lone_adopter` notice |
| sealed audit tamper: delete a pool entry on one device | the other device's copy and the sentinel mirror disagree; `verify` names the gap |

Each scenario becomes a test that runs against real sockets on loopback (the node's own tests inject
addresses and a free UDP port for the beacon). The existing `tests/test_redteam_*` pattern (corpus plus
report in `redteam/report.py`) is reused for reporting. [V `tests/test_redteam_*`; analysis]

## 6. Slices

Sizes: S under 150 lines of code plus tests, M 150 to 500, L over 500.

| # | Slice | Size | Tests to write | Depends on |
|---|---|---|---|---|
| 1 | Owner-only grants in the node (replace the `Grants` stub for membership methods) | M | a same-uid session without the grant is `denied` for `set_join_policy`, `member_revoke`, `pair_start`, `link`, `project_add`; with it, allowed | node network branch |
| 2 | Probation status and capability table (Rust): membership row, per-state checks in `peer.rs`, invitations as board entries | L | each cell of 3.1; `merge` keeps `revoked` terminal; round trip through `pool.json`; old Python reader still parses | 1 |
| 3 | Python reader understands `probation`/`quarantined` (`fleet/membership.py`, `pool_roster.py`) | S | a roster with a probationer is not "active" anywhere callers check; `Device.read` round trip | none (do before 2 ships) |
| 4 | Node limits: beacon and dial rate, enrolment cap per segment and pool-wide, concurrent connections, per-device caps, `busy` reply | M | flood tests on loopback; caps never exceeded under 8 threads | 2 |
| 5 | Node counters and event stream (`device_events`, `device_state`) and the narrowing-only sentinel token | M | the sentinel token can narrow, cannot promote; a counter per detector input | 2 |
| 6 | Sentinel `devices.py`: detectors in 3.3 on the existing `Windows`, `Finding` and policy path; subject `device`/`segment`; `explain.WHY` and heads-up kinds | M | each detector on a synthetic clock (as `rates.py` is tested); false-alarm corpus: 500 honest enrolments and syncs produce nothing above `notice` | 5 |
| 7 | Graded responses and promotion rules, audit entries written and mirrored | M | each grade; `guarded` never exceeds grade 2 on heuristics; promotion only with a live heartbeat | 2, 5, 6 |
| 8 | The brake: CLI verbs (HumanGrant), Fleet page buttons, next-request effect | M | verb then the very next request refused; an agent marker refuses; page button calls the same route | 2 |
| 9 | Lone-adopter confirmation | S | a lone device does not adopt without confirmation; confirmation by id | 2 |
| 10 | Trust-ledger link: device account evidence, promotion by evidence, standing-grant | M | clean syncs raise the ledger state; a fault drops it; the grant is scoped and revocable | 7, earned-trust slices |
| 11 | Fuzz targets in `app/poolside-node/fuzz` | M | the five targets, run for a fixed iteration count in CI, long runs by hand | node network branch |
| 12 | Red-team scenarios (`redteam/scenarios/devices.py`) and the scenario table above as tests | L | each row of 5.2 | 2 to 8 |
| 13 | Sentinel as a node-socket client and heartbeat check by the node | M | kill the sentinel: enrolment closes in three intervals; restart: it reopens | 5 |

The order that makes `open` shippable: 1, 3, 2, 4, 5, 13, 6, 7, 8, 9, then 11 and 12 as the gate.
Slices 10 is the generosity that follows. `open` is not enabled by default until 12 is green. [analysis]

## 7. Owner decisions

Each is one question, ranked options, my recommendation first.

1. **Probation length.** Options: (a) 24 hours and 5 clean syncs, or sooner by the person [recommended:
   long enough to see misbehaviour, short enough to feel automatic]; (b) 1 hour; (c) 7 days; (d) no
   timer, promotion only by evidence or the person.
2. **Which actions auto-revoke?** (a) Only proofs: a forged hash or head from the owner of an origin,
   a replayed signed frame, a signature by one fingerprint's key from two origins [recommended];
   (b) also a quarantined device that stays quarantined 24 h; (c) nothing automatic, quarantine only.
3. **Must `open` also need the person's confirmation on the first device beyond the second?**
   (a) No: devices 3 and later are automatic, with probation and one notice [recommended; the point
   of `open` is no clicks]; (b) yes for the third device only; (c) yes for each, i.e. `open` becomes
   `secure` with a button.
4. **May a lone device adopt an `open` pool it hears?** Decided by the owner: (c), yes, auto-adopt,
   limited to a pool for the same project (3.7). Options were: (a) only with a one-time
   confirmation of pool id and device list; (b) only when it was given the pool id at install;
   (c) yes.
5. **Default for probation caps.** (a) 20 rows a minute, 5 MiB a day, 4 boards [recommended, tunable
   per pool]; (b) stricter (5 rows a minute, 1 MiB a day); (c) no caps beyond the pool quotas.
6. **When the sentinel is down, what do new `open` enrolments do?** (a) Refuse them after three
   intervals [recommended]; (b) enrol but hold in probation without promotion; (c) carry on.
7. **Does the pool-wide flood switch `open` to `secure` by itself?** (a) Pause intake first
   (grade 2) and ask the person before changing policy [recommended]; (b) switch automatically and
   notify; (c) never switch automatically.

## 8. What I did not verify

- The node's accept loop and thread model beyond `net.rs` comments; slow-loris numbers are an
  inference from the 30 s IO timeout.
- Whether Python `Roster.enrol` has an in-code member cap beyond `MOST_ROWS` rows read from a peer.
- The Fleet page route list; the brake's page route is a proposal.
- That PyRIT's current converters apply to device-name payloads; the converter table is read from
  `pyrit_bridge.CONVERTERS`.
- Nothing here was run.
