# The node

One `poolside-node` runs per device. It holds the boards that device takes part in, gives every
session one unique name per board, stamps every write from the session's token, and answers a
local API on a Unix socket (a named pipe on Windows). It is the Rust crate `app/poolside-node` (lib and bin) in the cargo
workspace `app/`; the Tauri app (`app/src-tauri`) is the other member and depends on the lib.

This is slices A and B1 of the node refactor: the crate (boards, identity, local socket, in-process
sync) and its network side (peer transport, membership, board sync over it, discovery, pairing),
built and tested alone. The shipping section below puts it in every runtime and keeps it running; no Python client calls its board methods yet (see "What is left").

## Boards, projects, links

- **One pool of devices, many boards.** A board belongs to one project. Every entry carries its
  board id, and the id is part of the hash, so an entry cannot be moved to another board.
- **Hosting.** The node hosts any number of boards, each in its own directory
  `<state>/boards/<board>/` with its own logs, its own origin id and its own names. A session
  registers on its project's board; names, claims, reads and posts are all per board. A device
  only syncs boards it holds.
- **Projects.** The node keeps a project registry (`projects.json`): project id (= board id) and
  its sources, each `git`, `folder`, `gdrive` or `onedrive` with a local path (the sync folder for
  the last two). A client resolves its board with `project_resolve {path}`: the deepest source
  containing the path wins; a linked git worktree resolves through the repository's common
  directory, so every worktree of a repo is the same project; a directory that matches nothing
  gets `denied: this directory is not part of a project`, never another project's board. A git
  source is stored as the repository's main working tree. One place cannot belong to two projects.
  `.poolside/project.toml` in a repo (`board = "id"`) is an optional hint only
  (`project::propose_id`, `project::init_project`); it is never required.
  `--board ID` still names a board directly, and is checked against the session's registration.
- **Links.** Nothing crosses between boards except through a link: a session of board `from`
  makes `link {to, channels, mode}` and the sessions of `to` may then use the named channels (none
  named = the whole board) of `from`, read-only (`ro`) or read-write (`rw`). Channels are
  `#general`, `#announcements` (messages by their `to`), `#notes` (notes and their verifications),
  `#leases` (claims), `#identity`, `#audit`. Messages to one session are on `#dm`, which a link never
  shares (see "Direct messages"). A linked session writes as `name@its-board`. `unlink {id}` revokes it at once. Making
  and revoking a link writes an audit entry on both boards. Links are node-local for now, and any
  session of the sharing board may make one; the owner-only grant arrives with the trust ledger.
- Unknown board, a token of another board without a link, and a write over a read-only link are all
  `denied`.

## Board entries

Each device appends to one signed log per origin per board: `<state>/boards/<board>/log/<origin>.jsonl`,
one JSON row per line, mode 0600 in 0700 directories. Only this device's own origin file is ever
appended to; the others are verified copies of other devices' logs.

A row: `v` (schema version, 1), `board`, `origin` (32 lower-case hex), `seq`, `prev` (hash of the
row before), `hlc` (wall ms, counter, origin), `kind`, `actor`, `idem`, `body`, `hash`
(SHA-256 of `prev` plus the canonical row without its hash). Unknown row fields do not parse.
Bodies hold integers, text, booleans, arrays and objects; fractions are refused so the hash never
depends on a float format.

Kinds: `message`, `note`, `verify`, `identity`, `lease` and `audit` are folded into the view now; `task`,
`landing_request` and `reputation_event` are in the schema so a later log still parses, and a
foreign entry of those kinds is not shown yet. `head` is a signature row, never shown.

- **Crash safety.** Append is one write plus `fsync`; it returns when the row is on disk. On open, a
  last line without its newline was never acknowledged and is cut off (a torn write from `kill -9`);
  damage anywhere else is an error, never a silent drop.
- **Heads.** A head row signs (Ed25519, device key, `device.key` 0600) the row directly before it.
  Only rows through the last verified head are given to peers or stored from peers. A pool of one
  never needs to sign.
- **Merge.** All rows sort by (wall, counter, origin, seq): one total order for any delivery order,
  with a row delivered twice kept once and a repeated `idem` key of one actor on one origin kept at
  its first row. Foreign rows dated more than five minutes ahead are held back from the view.
- **Foreign rows** become entries only through explicit field allow-lists per kind, each field
  checked for type and size; the sender is the log's actor (a `from`/`author`/`name` that differs is
  refused), must be a usable name and must not be a local session's name, and is shown as
  `actor@dN`. Posts target `#general` or `#announcements` (with announcement types only). Notes per
  foreign sender are capped. Refusals are recorded (the latest 200) and shown in `status`.
- **Quotas.** 64 origins per board, 200,000 rows per log, 128 KiB per row, 64 KiB per body text.
- **Damage.** A forged, forked or broken copy is `damaged`. It marks the origin damaged only when
  it came from the device that owns the origin (`authoritative`); a relay's bad copy never does.
  Oversized or malformed requests are `invalid`, never evidence against an origin.
- **Acks** are keyed by device fingerprint (SHA-256 of the device public key) and count only for a
  row whose hash matches; a row is `synced` once every device in the roster holds it.
- The key and `origin` ids of a device are bound the first time seen; a device that later presents
  a different origin is refused.

## Identity

`register {model, harness, session}` on a board returns `{name, token}`. The name is the model
family (`claude`, `chatgpt`, `qwen`, else `agent`), a dash, and the first six hex of
SHA-256(`harness NUL session`), extended by two hex per collision, exactly the algorithm of
`src/ml_stack/workspace/session_name.py`. The same session always gets the same name back. A
`register` carrying the token of a session of the same board makes the new session its subagent
(`parent` recorded in its identity entry); the parent is never a request field.

The token is 32 random bytes as hex. The node stores only its SHA-256 in `tokens.json` (0600). A
write's sender is the name the token resolves to; a request carrying `sender`, `name`, `parent`,
`label`, `from`, `author`, `actor`, `identity` or `as` anywhere in its envelope, params or `fields`
is refused `denied`. At most eight tokens per name; 10,000 sessions per board.

## Local API

Frames are a 4-byte big-endian length (at most 1 MiB) and JSON. The socket is
`<state>/node.sock`, mode 0600, in a 0700 directory; the peer's uid is read with `SO_PEERCRED`
(Linux) or `getpeereid` (macOS) and a different uid gets one `denied` frame and is closed. One node
per state directory: `node.lock` is held with `flock`; a stale socket from a killed node is replaced
by the next one.

**On Windows** the same frames and methods run over a named pipe, `\\.\pipe\poolside-node-<key>`, where `<key>` is the
first 32 hex characters of the SHA-256 of the state directory's absolute path (backslashes, lower case, no `\\?\`
prefix, no trailing separator; `sys::key_of` in Rust and `node_health.key_of` in Python, one test vector in both).
The pipe is created with a protected DACL that grants the current user's SID alone (`D:P(A;;GA;;;<sid>)`),
`PIPE_REJECT_REMOTE_CLIENTS`, and `FILE_FLAG_FIRST_PIPE_INSTANCE` so a second node (or any process squatting the name)
cannot share it; `node.lock` is still taken first, with `LockFileEx` on the byte the Python `only_one` locks. The server
reads the client's SID by impersonating it for one call (`ImpersonateNamedPipeClient`), so a different user gets the
same `denied` frame as on Unix. The Rust client opens the pipe at identification level and checks that the process
serving it runs as the same SID before it sends a token. The pipe is overlapped, so a read has the same 30 s timeout and
the listener polls like the non-blocking socket. A pipe vanishes with its node: there is no stale file to replace.
`socket_path(state)` and the `socket` field of `hello`/`status` hold the pipe name.

Request `{"v":1, "id":…, "method":…, "board":…, "token":…, "params":{…}}`; reply
`{"v":1,"id":…,"ok":true,"result":…}` or `{"ok":false,"error":{"code","message"}}` with codes
`denied`, `invalid`, `quota`, `damaged`, `gap`, `io`. Unknown envelope fields, params and methods are
`invalid`.

| method | needs | params | result |
|---|---|---|---|
| `hello` | | | `{node, version, pid, fingerprint}` |
| `register` | board | `model, harness, session` (token = parent) | `{name, token, created, parent, origin}` |
| `whoami` | board, token | `model, harness` (claim what you run on) | `{name, board, target, identity}` |
| `session_lookup` | board | `harness, session` | `{name}` (denied when none is registered) |
| `agents` | board, token | `retired` | `{agents: [{name, parent, family, model, model_state, harness, retired, seen_ms}]}` |
| `retire` | board, token | `target` (empty = yourself) | `{name, retired, by, tokens_revoked, leases_released}` |
| `post` | board, token | `kind` (message or note), `idem`, `fields` | `{id, seq, sender, status}` |
| `read` | board, token | `since` (cursor map), `kind`, `channel`, `by` (sender), `limit`, `inbox`, `with` | `{entries, cursor}` |
| `claim`, `release`, `claims` | board, token | `kind, key, ttl_s, pid` / `kind, key` / `kind` | `{changed, claim}` / `{kind, key, released}` / `{claims}` |
| `notes` | board, token | `ref, query, kind, all, limit` | `{notes}` |
| `note_verify` | board, token | `note, exit, out_sha` | `{notes: [the note]}` |
| `link`, `unlink`, `links` | board, token | `to, channels, mode` / `id` | the link |
| `project_add` | | `id, kind, path` | the project |
| `source_add` | token of that project | `id, kind, path` | the project |
| `project_list`, `project_resolve` | | `path` | projects / `{board}` |
| `status` | (board, token for detail) | | counts; with a token, the board's detail |
| `shutdown` | board, token | | `{stopping}` |
| `pool_status` | | | the pool id, join policy, this device, `listen`, `pairing_open`, and each member with `status`, `by`, `connected`, `addr`, `last_seen_ms`, `last_sync_ms`, `last_error` |
| `member_revoke` | token with the grant | `fingerprint` | `{fingerprint, status}` |
| `set_join_policy` | token with the grant | `policy` (`open` or `secure`) | `{policy, changed}` |
| `pair_accept` | token with the grant | `passphrase` (else a six-digit code is made), `ttl_s` | `{code, expires_in_s, port, fingerprint}` |
| `pair_start` | token with the grant | `host, port, passphrase`; without a passphrase `fingerprint` (join under `open`) | `{pool, peer}` |
| `sync_now` | token with the grant | | `{reached}` |

| `lease_acquire`, `lease_renew`, `lease_release`, `lease_list`, `lease_wait` | board, token | see "Leases" | a lease |

The grant check is `grants::Grants` (the standing-grant system comes later); the stub lets any
registered local session act, and a node can be given a stricter one. The network methods
(`pair_accept`, `pair_start`, `sync_now`) run without the node's lock held, so a slow peer blocks no
other request.

`read` cursors are a map origin to last seq returned, so a read returns each entry once even when a
sync later brings entries with older clocks, and a `limit` never skips anything. A cursor belongs to
the question it came from (seq counts every kind), so a client keeps one per question, such as its
inbox. `inbox` returns only messages sent to the caller; `with` the conversation between the caller
and one session.

**Direct messages.** A message whose `to` is a session name is a direct message. The recipient must
be a live session of the board (here or announced by an identity entry from another device); the post
is refused otherwise. `read` returns it only to its sender and its recipient, on every path, including
a linked board that shares everything. Peers replicate it like any row (every member of the pool holds
the log), so this is privacy between sessions, not encryption between devices. A sender shown as
`name@dN` (written on another device) is matched by recipient only.

**Sessions.** `register` records the model id and harness (`model_state` `claimed`; a subagent that names no
model gets its parent's, `inherited`, and the name's family comes from it; the node itself may mark
one `verified`, and a claim can never lower a verified one); `whoami` with `model`/`harness`
changes the claim. Each change is an identity entry. Registering a session that has a parent without
that parent's token, or under another parent, is `denied`, and so is registering a retired one.
`retire` revokes every token of the session, ends its leases (written to the board), marks the
identity retired and writes an `audit` entry. A subagent retires itself; a parent (any ancestor)
retires a descendant; a main session never retires itself.

**Claims** are leases of one `claim` resource taken without waiting. A branch claim is held under the
board's name, so two boards may claim the same branch name; ports, servers, paths and installs are the
device's. `claims` lists the claims of the caller's board with the owner. A holder's `pid` and `ttl_s`
behave as for any lease (dead process or expiry drops it).

**Notes** carry `nkind` (decision, rule, fact, question), `title`, `body`, `source`, `tags`,
`supersedes` (references to older notes, a bare number for a note of this device or the full id; a
verified note cannot be superseded), `verify_cmd` and `ttl_days`. `notes` folds them: `trust` is
`test-verified` while the latest passing `verify` entry for the note's current command is fresh, else
`agent-claimed`; `stale` once `ttl_days` passed since the last verification (or the note). The node
never runs `verify_cmd`: a client does and records `exit` and the SHA-256 of the output with
`note_verify`, and the entry stores the command the note carries. Recording one is the grant
`note_verify`.

The Rust client (`client::Client`, `client::ensure_running`) finds a dead socket, takes
`start.lock` (single-flight, so callers queue and the second finds the first's node), spawns
`poolside-node run --state DIR` in its own process group, and polls `hello`. State is the logs on
disk, so a killed node restarts with every committed entry and every token.

## Leases

One table, `<state>/leases.json` (0600, written atomically on every change), admits everything that
used to be admitted six ways: the GPU, CPU slots, memory, named claims, served models. Code is
`src/lease/` (`types`, `policy`, `table`, `live`, `merge`, `rpc`). Config is `lease-config.json`
(optional; every field has a default): `cpu_slots` (cores), `memory_mb` (0 = no budget),
`max_wait_s` 600, `background_share_percent` 50, `unknown_background_s` 180, `queue_patience_s` 60,
`default_ttl_s` 300, `max_ttl_s` 86400.

**Resources** (JSON objects with a `type`; `device` defaults to `local`, the only device this node
decides; another device is `denied`):

| type | fields | rule |
|---|---|---|
| `gpu` | `device` | one holder at a time |
| `cpu_slots` | `device, count` | counted against `cpu_slots` |
| `memory_mb` | `device, mb` | counted against `memory_mb`; more than the budget is `quota`, less than what is left queues |
| `claim` | `kind, name` | `kind` is worktree, branch, area, port, server or install; one holder per name; worktree and area names are absolute paths and conflict with a nested path |
| `model_slot` | `device, model, context, draft, parallel` | one holder per device and model; the shape is kept and shown |

A request names one to eight resources and gets all of them at once or waits. **Order** is
`scripts/testslots_policy.py`: shortest `estimate_s` first, less the seconds already waited; a request
that has waited past `max_wait_s` goes ahead of every ordering and cap, by arrival; arrival breaks
ties (`class` `interactive` estimates 0, `background` estimates `unknown_background_s`). While a
short run holds or waits for CPU slots of a device, a not-yet-aged long run may hold at most
`background_share_percent` of them. A request never overtakes an earlier one that wants the same
resource.

**Holder.** The holder is `board/name` from the token; a request that carries `holder`, `name` and the
like is refused. Only the holder renews or releases. A holder may pass `pid`: the node records the
process start time and drops the lease when the process is gone, a zombie, or started at another
moment (a reused pid). A `remote` holder has no pid and lives by expiry. Every lease expires
`ttl_s` after the last renewal; a queued request not polled for `queue_patience_s` is dropped. The
sweep runs on every lease call, and at the first one after a restart, so a stale holder is dropped
and a live one kept.

**Takeover.** When a request is granted resources a dead or expired holder had, the node writes an
audit entry (`event lease_takeover`, `subject` the new lease, `detail` naming the old holder and why).

| method | params | result |
|---|---|---|
| `lease_acquire` | `resources, class, estimate_s, ttl_s, pid, remote, wait` | the lease, or `{state: "busy", blockers}` when `wait` is false and it is not free |
| `lease_renew` | `id, ttl_s` | the lease |
| `lease_release` | `id` | `{id, released}` (a queued request is cancelled) |
| `lease_list` | | `{leases, config}`; other boards' holders show as `other-board` |
| `lease_wait` | `id, timeout_ms` | the lease once `held`, or still `queued` at the timeout (at most 10 minutes; the node's lock is not held while it waits) |

A lease is `{id, state (queued|held), holder, resources, class, estimate_s, since_ms, granted_ms,
expires_ms, ttl_s, pid, remote, position, wait_estimate_s}`. Rust: `node.leases` (`Leases`, with
`tick`), `lease::table::Table`, `lease::rpc::call` and `serve`, `lease::live::is_alive`.

**On the board.** Every grant and end is a `lease` entry (`#leases`) written under the holder's
name: `{lease, action: acquire|release|expire|dead|abandon, resources: [lines]}`. `lease::merge::view`
reads them back in the board's total order: the earliest acquire of an exclusive resource holds it
until its own end entry; counted resources sum. Gpu, model slots and the path, port, server and
install claims are scoped to the origin that wrote them (a device's table decides its own, others only
read); branch and area claims are pool-wide, and a local request for one a peer holds on the board
queues behind it (`merge::foreign_holds`).

## Sync

`sync::exchange(a, b)` makes two copies of one board hold the same rows: per origin, trusted rows
past the peer's vector (one row early, to compare), at most 400 rows, 2 MiB and 16 origins per
request, every row through `ingest` (chain, hashes, heads, pinned key, size, origin id). Rows reach
a third device through a relay and stay verifiable. `sync::exchange_nodes` does every board both
nodes host and leaves the rest alone. Two devices converge after a partition.

A peer stores rows only through a verified head, so a log signs its own tip after 64 rows or 512 KiB
without one (`board::HEAD_EVERY_ROWS`, `HEAD_EVERY_BYTES`); otherwise a run longer than one batch
would never arrive.

## The pool and its peers

The network is off unless the node is started with `--listen ADDR` (`net::Net`, `NetConfig`). One
listener is the one network port.

- **Device identity.** Each device has one self-signed Ed25519 certificate made from its device key
  (`cert.rs`, `device.cert`, valid ten years). Its fingerprint is the SHA-256 of the DER and is the
  device's identity in the pool; the board fingerprint (hash of the public key, used for acks and
  origin binding) is read back out of the certificate a peer showed.
- **Transport** (`tls.rs`, rustls with the ring provider). TLS 1.3 only; a TLS 1.2 client is refused. Both
  sides show a certificate. A client pins the one fingerprint it expects (a changed digit is a refused
  handshake) or, when pairing, takes any certificate and lets the exchange authenticate it. A server
  completes the handshake for an unknown certificate (it may pair or join, nothing else) and refuses
  one the pool has revoked. No certificate, no connection.
- **Membership** (`membership.rs`, `pool.json`; semantics of `fleet/membership.py`). Per device: fingerprint,
  certificate, status `active` or `revoked`, who recorded it. A row whose fingerprint is not the hash of
  its certificate is dropped; revoked wins every merge and is never undone; a revoked certificate cannot
  be enrolled again. Members swap records with the `members` op; only an active member's rows are merged.
  Every request is checked against the record when it arrives, so a device revoked after it connected is
  refused at its next request on the same connection. Changes are written to the board `pool` as audit
  entries; the record is the authority.
- **Join policy**: one pool attribute, `open` or `secure` (default `secure`), newest change wins, set by
  `set_join_policy` and carried by the `members` op.
- **Peer ops** (`peer.rs`, frames up to 4 MiB): `hello`, `join_open`, `pair_exchange`, `pair_confirm` for
  a non-member; `members`, `boards`, `vector`, `pull`, `push` for a member. A device syncs only the boards
  both hold, through `sync::rows_since` and `sync::take`, so a wire row meets the checks an in-process row
  does and the same quotas (400 rows, 2 MiB, 16 origins per request). A relay's copy never marks an origin
  damaged; only the device that owns the origin can, by sending a forged copy of it.
- **Discovery** (`beacon.rs`). One UDP datagram with the pool id, the device fingerprint, an address and
  a port, signed with the device key and valid for a minute; sent to the multicast group on a configurable
  port. Never matched by name. Loopback, unspecified and multicast addresses are never advertised (a
  test on one machine passes `allow_loopback`). A member's beacon updates its address (its key must be the
  one in its certificate). A non-member's is ignored unless this pool's policy is `open`, and then only
  when it names this pool, or this device is alone and the beacon's pool id sorts before its own (so two
  lone devices never join each other at once); the dial is pinned to the fingerprint in the beacon.
- **Pairing** (`pairing.rs`). `secure`: `pair_accept` opens a window (a six-digit code or a passphrase,
  three tries, a time limit), `pair_start` on the other device runs SPAKE2 (`spake2`) with both certificate
  fingerprints as its identities and HMAC-SHA256 confirmations, as in `fleet/onboard/pake.py`; the pool
  record then arrives sealed under the exchange. `open`: the device is enrolled on its first contact from
  the same segment and the enrolment is recorded with the certificate fingerprint (`by: open`). A device
  joins only if it is alone in its pool.

`poolside-node run --state DIR --listen 127.0.0.1:0 --beacon-bind ADDR --beacon-send ADDR
--advertise IP --sync-ms N` starts the network; tests use real sockets on loopback with injected
addresses and an unused UDP port for the beacon. No default test listens beyond loopback or uses multicast
(`tests/loopback_only.rs` and `tests/test_node_join.py` enforce it): that makes macOS ask the person for Local
Network permission. The LAN tests are `#[ignore]`d / need `ML_STACK_LAN_TESTS=1` and are run by a person at the machine.

`poolside-node run --state DIR --network [--port 47321] [--beacon-port 47322]` is the real network: every interface,
the multicast group and each network's broadcast address (`netif.rs` reads the interfaces; a VPN or loopback is not
a segment), each beacon advertising the address of the interface it leaves by. Open join is accepted from, and dialled
to, an address inside a subnet of this machine only. `pool_status` adds `beacon_sent|heard|own|rejected`, the last send
error and `last_join`: a node that sends but never hears its own beacon is blocked by the system (macOS Local Network,
a firewall, WSL2 NAT). `python -m ml_stack.node_join join --policy open|secure` starts it that way after showing what
it does and getting a yes (`--yes` for a script), and `scripts/pool-join-check` walks every step on one device and prints
the fix for the first that fails.

## Poolhouse.app: the node's identity on macOS

macOS ties Local Network permission to an app's bundle id and code signature, so a bare cargo-built binary asks again for every
build. On macOS the LAN-enabled node (`--lan`, started by the supervisor) therefore runs as the executable inside
`~/.ml-stack/apps/Poolhouse.app` (outside every checkout), never as a bare binary: `node_app.bundled` rebuilds the bundle
when the verified binary's checksum differs from the one inside, so the supervisor still checks the checksum it always did.

- **Fixed id.** The bundle id is `app.poolhouse.node` and is never changed: a new id is a new app to macOS and loses every grant.
  `CFBundleName` is Poolhouse, `LSUIElement` hides the dock icon, `NSLocalNetworkUsageDescription` is the sentence the prompt
  shows, and `NSBonjourServices` lists `_poolhouse._tcp`. The node discovers by raw multicast beacon, not mDNS; the keys are
  there so the prompt reads right.
- **Stable signature.** `node_signing` makes the self-signed code-signing certificate `Poolhouse Local Signing` once in the login
  keychain (trusted for code signing in the user domain only; no Apple developer account) and signs the bundle with the designated
  requirement `identifier "app.poolhouse.node" and certificate leaf = H"<sha1>"`. It names no cdhash, so rebuilding the binary
  keeps the grant. If the certificate cannot be made, the bundle is signed ad hoc with the identifier-only requirement and a
  warning says the permission may be asked again after a rebuild. A bundle is built beside its place and moved in only once signed.
- **Asking once.** `ml-stack node permission` launches the bundle through Launch Services with `poolside-node probe`, which joins
  the beacon group, sends datagrams of its own (multicast, each network's broadcast address and the router) and listens for
  itself, up to 30 seconds. It prints GRANTED, DENIED (the sends were refused or never heard) or NOT ASKED (the bundle did not
  run). This is the only place that touches multicast on a Mac outside a node the person turned the network on for.
  `ml-stack node build [--binary PATH]` builds and signs the bundle by hand.
- **What the probe cannot see.** Hearing its own datagram is delivered inside the machine, so a refused send (`No route to host`
  on the multicast group) is the signal of a denied permission, and a run launched from a terminal borrows that terminal's
  permission; only the `open` launch the command uses measures the bundle's own.

## Shipping and keeping it up

The Python side of getting the node onto a device and keeping it there: `ml_stack.node_binary` (build, checksum,
verify; the binary is `poolside-node.exe` on Windows), `node_build` (into a runtime), `node_health` (the socket or pipe call), `node_supervise` (the restart loop) and
`node_launch` (find, start, stop, swap, smoke; also `python -m ml_stack.node_launch ensure|status|stop|swap|supervise`).

- **In the runtime.** `runtime ensure` builds the node from the same commit as the wheel (`cargo build --release
  --locked -p poolside-node`, target directory shared by all builds in `<runtimes>/cargo-target`), copies it to
  `<runtime prefix>/node/poolside-node` with `node.json` (`sha256`, `bytes`, `target`, `commit`), starts it once on a
  scratch state and asks `hello`; a build that does not answer is discarded before it is selected. The checksum is
  written again into the tree's `verified.json` (`node_sha256`) after the smoke, and `node_binary.verified` refuses a
  binary that differs from either record (and one built for another platform). The binary is as immutable as the tree.
  `packaging/build.py --node` writes the same binary to `dist/node/poolside-node-<target>` with a `.sha256` file for a
  bundle or release asset.
- **Start on demand.** `node_launch.ensure_node(state)` is what every client calls: it returns the node's `hello`
  (plus `socket` and `latency_ms`), and when nothing answers it verifies the binary, takes `start.lock` (single-flight:
  eight clients at once start one node, the rest queue and find it), starts the supervisor if none holds
  `supervisor.lock`, and waits for health. A binary that fails verification raises `NodeBinaryError` before any process starts.
- **Supervisor** (`node_supervise.supervise`, detached, one per state directory). It starts the node, waits, and restarts
  it when it exits: 0.1 s, doubling to 30 s while it keeps dying, reset after it ran 60 s. Every start re-resolves the
  binary and checks its checksum: `node-binary.json` (a pin written by a failed swap) wins, else the selected runtime's. It
  writes `node-run.json` (pid, binary, sha256, started_at, supervisor pid, previous binary) and logs to `node.log`. It
  watches the pin and `selected.json`, so selecting a new runtime moves the node by itself; it gives up after five
  consecutive starts with no verified binary. It owns no service: the optional always-on form is any unit that runs
  `python -m ml_stack.node_launch supervise` (not wired into `fleet/autostart` roles yet).
- **Stopping on Windows.** A detached process has no signal to receive and `os.kill(pid, SIGTERM)` is `TerminateProcess`,
  so the node creates a named event `Local\poolside-node-stop-<key>` and the supervisor
  `Local\mlstack-node-supervisor-stop-<key>` (both open to the current user alone, `ml_stack.win32`). `stop_node`,
  `swap` and `smoke` set them first (the node leaves with exit code 0, the supervisor stops restarting), and only
  terminate a node still running after the wait. Ctrl+C, Ctrl+Break and a console close stop a node run by hand the
  same way (`SetConsoleCtrlHandler`); the supervisor takes `SIGBREAK` and `SIGINT`. `lock.pid_alive` asks the process
  by handle on Windows, because `os.kill(pid, 0)` there is `CTRL_C_EVENT`.
- **Crash-only.** The node is killed with SIGTERM or SIGKILL (`TerminateProcess` on Windows) at any moment (`stop_node`); the next start reads the logs
  on disk, cuts a torn tail, and every acknowledged entry and token is back. A real-process test posts, `kill -9`s, and
  reads everything again after the supervisor's restart.
- **Upgrade.** One node owns a state directory (`flock`), so there is no second node answering on the same socket while
  the first runs: the swap is stop-old, start-new on the same path with state on disk. `runtime ensure` calls
  `node_launch.swap()` after it selects a runtime: `current`, `idle` (no node running), `swapped` (a node of the new
  checksum answers and is the one in `node-run.json`) or `failed`. On `failed` the previous binary is pinned, waited for,
  and `runtime status` shows `PINNED to the previous binary` until the next successful swap clears it; `ensure` reports
  the node outcome in its detail and exits non-zero.
- **Health.** `node_health.node_health(state)` is one framed `hello` (2 s timeout) returning `node, version, pid,
  fingerprint, socket, latency_ms`, or `None`; `node_launch.status` adds uptime (from the run record, only when its pid is
  the answering pid), binary, sha256, `supervised` and `pinned`, and `ml-stack runtime status [--json]` prints the line.
  `node_health.call(state, method, params, board=, token=)` is the minimal socket call for tests and tools; the
  Python board client is `ml_stack.board` (below).

## The Python client

`ml_stack.board` is the client. `client.Client` speaks the socket (starting the node through
`node_launch.ensure_node` when it is dead) and raises `NodeError` (`Denied`, `Invalid`, `Quota`) with the node's code;
`place.resolve` maps the working directory to a board (`ML_STACK_BOARD`, else the project registry; a git repository
the node does not know is added as a project, named by its folder and a few hex of its git common directory);
`credentials` keeps a private token file per board and name under `<state>/client/` and the read cursors of each
question (inbox, announcements); `session.Session` is one token's typed API (`post`, `read`, `notes`, `claim`,
`agents`, `retire` ...) returning small dataclasses, with an injectable `clock`. `session.register` /
`session.find` are what the hooks call. The workspace commands `whoami`, `announce`, `send`, `inbox`, `attention`,
`digest`, `brief`, `nudge`, `notes-*`, `claim`, `release`, `claims`, `heartbeat`, `who`, `spawn`, `retire` and `agents`
are in `workspace/board_cli.py`, `claims_cli.py` and `notes_cli.py` and go to the node only. Text from other
sessions is fenced as untrusted data on the way out and a credential in outgoing text is refused on the way in
(`screen`, still Python). Tests use `tests/node_kit.py`: a real node on a short `/tmp` root.

## Run and test

```
cd app
cargo build -p poolside-node
target/debug/poolside-node run --state ~/.poolside/node     # serve
target/debug/poolside-node status --state ~/.poolside/node
cargo test -p poolside-node                                  # about 93 tests; crash.rs starts two node processes and SIGKILLs one
cargo test -p ml-stack-app                                   # the window; needs the sidecar binary in src-tauri/binaries
```

The workspace Cargo.lock is `app/Cargo.lock`, the build output `app/target/`. Crypto is
`ed25519-dalek`, `sha2`, `hmac`, `rustls` (ring), `rcgen`, `x509-parser` and `spake2`; nothing is hand-rolled.

## Windows

What `sys` (`src/sys/{unix,win}`) does on each platform, so nothing else in the crate knows the difference:

| | Unix | Windows |
|---|---|---|
| local API | socket file, mode 0600 | named pipe, protected DACL for the user's SID, remote clients refused, first instance exclusive |
| who is calling | `SO_PEERCRED` / `getpeereid` | `ImpersonateNamedPipeClient` + token SID |
| private directory | `mkdir` 0700 | `SetNamedSecurityInfoW`: protected DACL, the user's SID only, inherited by files (`private_file` is then a no-op) |
| atomic replace | `rename` + directory fsync | `MoveFileExW(REPLACE_EXISTING \| WRITE_THROUGH)`; no directory flush exists |
| log append | `O_APPEND`, `fsync` | the log seeks to the end before each write (an append-only handle cannot be cut back), `FlushFileBuffers` |
| single instance, `start.lock` | `flock` | `LockFileEx` on byte 2^30, released when the holder dies |
| lease liveness | pid + start time (`/proc`, `proc_pidinfo`), zombies count as dead | process handle + creation time + exit code; no zombies; a process this user may not inspect has start 0 and counts as alive |
| start in the background | new process group | `DETACHED_PROCESS \| CREATE_NEW_PROCESS_GROUP` |
| graceful stop | `shutdown` request, SIGTERM | `shutdown` request, the stop event, Ctrl+C / Ctrl+Break |

`lease::types` accepts a drive path (`C:\x`, `C:/x`) or a UNC path as an absolute worktree or area claim, and compares
claims with one separator and no case on a drive. The state directory is `%USERPROFILE%\.poolside\node` by default.
Tests that need a socket path short enough for macOS use `kit::short_dir()`; `kit::kill_hard` is `kill -9` or `taskkill /F`.

## What is left

- Windows has been compiled (`cargo check --target x86_64-pc-windows-gnu --all-targets`, with a stand-in C compiler for
  `ring`'s build script) but never run by the author; the first real run is the `windows` job in `ci.yml` and the
  Windows step of `release-build.yml` (see "Windows" above). A pipe server owned by another user is refused by the Rust
  client; the Python `hello` check opens the pipe at identification level and sends nothing secret.
- Starting the node on a default port and the multicast group (`NetConfig::standard`) and a default sync
  interval; today the supervisor starts it with `run --state` only (extra flags after `--`), so the network is off.
- The always-on form: a `fleet/autostart` role for `node_launch supervise` (a person-installed unit); a wheel
  that carries a prebuilt node instead of a Rust toolchain at `runtime ensure` time.
- Python lease clients (the serve broker, `gate.py` tickets, `testslots` permits, fleet `JobRunner` slots,
  `lock.only_one` and `claims.py` calling the lease methods, and their own admission code deleted), a local
  holder yielding when the board shows a peer's earlier acquire of a pool-wide claim, leases on a remote
  device, the rest of the entry kinds (`task`, `landing_request`,
  `reputation_event`), content screening and quarantine of foreign text (still in Python), owner-only
  grants (the `Grants` stub allows every registered session; links and project sources have the same
  gap), rebuilding the graph index from entries.
- Pools of two or more members do not merge: a device that already has members refuses to join another
  pool (`Pool::adopt`).
- The head signature still uses an empty pool id as its domain (`Node.pool`); the pool id lives in
  `pool.json` and the beacon. Moving it into the head would need a re-sign when a device joins.
- Addresses of members come from the beacon, from pairing (the caller's address and announced port) and
  `peers.json`; a member that moves between pairings is found again by its beacon only.
