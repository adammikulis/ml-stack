# A home pool: every device a runner, a model host and a thing you command from a phone

Design and gap analysis, written against `0.2dev` at `c1007d3b`. Nothing here is built unless a line says so.

Owner goal, verbatim: "I want to be able to run poolside on all my devices at home, and have any of them
used for test runners, local models, etc. all commanded from an app on my phone, or through claude
code/codex/whatever harness, or the app."

Tags on each claim: **[V]** read in code at the path given, **[D]** taken from a document and not
re-checked in code. The code still says cluster and fleet where the plan says pool
(`docs/poolside-refactor-plan.md`, [D]); this note uses the commands as they are today. Timing and
performance are not claimed anywhere.

## 0. The five findings that shape the order

1. A phone app exists already, and it is native Android, not a web page. It enrolls by a QR invite, pins the
   daemon certificate and is allowed `fleet.status` and `chat` for one hour
   (`app/android/README.md` [D]; `fleet/invites.py` `_grant` [V]; `fleet/companion_routes.py` [V]).
   It cannot command anything else.
2. No person action can come from a phone today. Every person-session browser route requires the client
   address to be loopback and a Host header that is not a DNS name (`workspace/person_session.py:_validate`
   [V], `fleet/room_routes.py:_person` [V], `fleet/ui.py:host_ok` [V]). That is a deliberate floor, so the
   phone needs a new person credential kind, not a looser check (section 2.3).
3. Tailscale is already half-built as a route: read-only detection, a learned tailnet address after the
   certificate matched, and no install or config (`fleet/tailnet.py`, `fleet/onboard/routes.py` [V]). But the
   Android companion and its invite refuse tailnet IPv4, see section 2.5 [V].
4. Fleet jobs cannot run tests. The daemon refuses every argv except `ml-stack-bench` and
   `python -m ml_stack.fleet.calibration` (`fleet/commands.py:allowed` [V]). The test broker and the reuse
   store are per device (section 2.1).
5. The pool-wide board (journals, `mesh_sync`, membership, derived coordinator) is a design note with no code:
   `workspace/journal.py`, `mesh_sync.py` and `landing.py` do not exist [V]. What runs today is one coordinator
   device holding the board and the others reaching it.

## 1. What works today for several home devices

### 1.1 Pairing and joining

- A device starts its daemon with `ml-stack-traind`; it listens on loopback until it has joined a cluster or is
  given `--lan` (`fleet/daemon.py:bind_address` [V]). Once joined it binds all interfaces over TLS with a
  self-signed certificate that peers pin; loopback is plain HTTP (`Daemon.listen`, `tls.py` [V]).
- Development mode is the default: nearby devices discover one another by signed beacon and join the same
  cluster over pinned TLS without a passphrase (`docs/fleet.md` "Joining a cluster" [D]; beacon verification in
  `fleet/discovery.py` [V]).
- Person-approved pairing for a device that must be explicitly admitted:
  - on the existing device: `ml-stack-cluster listen --for 10m`; a dialog shows Decline, Accept as mine,
    Accept as someone else's, then a six digit code (`cli/reference.py` [V]);
  - on the new device: `ml-stack-cluster pair --host H [--port 8772]`, type the code. A SPAKE2 exchange hands
    over the cluster key; three wrong tries or a machine in the middle fails (`docs/onboarding.md` [D]).
  - `ml-stack-cluster accept ID --mine|--other` and `decline ID` answer from a terminal (`cli/reference.py` [V]).
- Explicit Production admission: `ml-stack-cluster join --group NAME --mode prod --persist` (`docs/fleet.md` [D]).
- Nearby and status: `ml-stack-cluster nearby`, `ml-stack-cluster status` (`cli/reference.py` [V]) and `ml-stack-peers ls`
  ([D] `docs/fleet.md`). `ml-stack-cluster devices [--learn]` shows each paired device as lan, tailnet or
  unreachable (`fleet/onboard/devices_cli.py` [V]).
- Pairing is pairwise today: N devices need N(N-1)/2 pairings, there is no membership certificate
  (`docs/mesh-board.md` section 1 [D]; `fleet/onboard/cli.py` pinned rows).

### 1.2 Signed journals

Not built. `docs/mesh-board.md` is the design (per-device append-only signed journals, hybrid logical clock,
derived coordinator, membership as signed records) and says "nothing is implemented"; `journal.py`,
`mesh_sync.py`, `mesh_fold.py`, `presence.py` and `membership.py` are absent from `src/ml_stack/workspace`
[V]. What exists is the hash-chain `ChainLog` (`workspace/chain.py` [V]) and a per-origin bus merge
(`workspace/board_graph_merge.py` [V]). Section 3 slice 6 is the first piece of the design.

What stands in for it, one coordinator device:

- On the coordinator (a person): `ml-stack-workspace coordinator host`, `ml-stack-cluster listen --for 10m`;
  elsewhere `ml-stack-cluster pair ...` then `ml-stack-workspace join CODE --coordinator FLEET_NAME --name N
  --model EXACT_MODEL --harness codex` (`docs/workspace.md` "Explicit shared coordinator" [D]; the `join` and
  `remote-agent` verbs registered in `workspace/cli.py` [V]).
- A matching Git checkout on an enrolled Development device connects with no code: the first
  `ml-stack-workspace` command registers the project and discovers its board (`docs/workspace.md` [D];
  `workspace/automatic_connection.py` [V]).
- Remote commands are bounded board operations only (`workspace/remote_protocol.py:METHODS` [V]); a lost
  response is retried with `--request-id ID` (`coordinator_calls.py` [V], one day retention [D]).
- No coordinator reachable means no fallback to a local copy (`coordinator_client.py` [D via mesh-board.md]).

### 1.3 Fleet jobs

- `ml_stack.fleet.Peer.discover()`, `run(units, peers)` places work by label, backend, VRAM, RAM, CPUs and
  schedule window, retries on a different machine, sets a machine aside after three failures (`fleet/pool.py`
  `Requires`, `fleet/work.py` `QUARANTINE_AFTER = 3` [V]).
- What a daemon will run: `ml-stack-bench` and `python -m ml_stack.fleet.calibration`, nothing else
  (`fleet/commands.py` [V]). A bench job is refused on a differing commit, a held `measuring.lock` or too little
  room (`fleet/sweeps.py`, `docs/fleet.md` [D]).
- `ml-stack-workspace remote-agent --device NAME --task ...` starts a Board-message worker running a
  downloaded Qwen on another Development device, admitted through that device's broker (`docs/workspace.md`
  [D]; `workspace/remote_workers.py` [V]).
- Uploads and downloads of files resume and are digest-verified (`docs/fleet.md` [D]).

### 1.4 Model serving leases

- One manager and one broker per machine: `ml-stack-serve up MODEL [--for 'why'] [--context 256k] [--kv q8_0]`,
  `status`, `leases`, `queue`, `history`, `down MODEL`. A lease records a reason, requester, process and
  place; a server nobody tracked is reported, never killed (`docs/serving.md` [D]; broker over a loopback
  socket in `serve/broker_wire.py` [V]).
- The broker is reachable only on `127.0.0.1` (`serve/broker_wire.py` `DEFAULT_HOST` [V]). Another device
  cannot take a lease on this one.
- Which peer serves what is announced in the daemon's device report (`serving` list) and visible in
  `ml-stack-peers ls` (`fleet/chat.py:targets` reads it [V]).
- `ml-stack-cluster plan --users N --context C [--apply]` picks a model per peer and, with `--apply`,
  calls `POST /serve` on each daemon (`docs/fleet.md` "Placing users" [D]).
- A conversation reaches a model held by another peer through `<peer>/infer/v1/chat/completions`, which the
  peer's daemon forwards to its own loopback server (`fleet/chat.py:targets`, `fleet/api.py` `/infer` [V]).

### 1.5 Test broker

- `python scripts/test fast|full|slow|all|quick [paths]` runs as a background job through a shared admission
  broker: CPU permits, heavy lane, FIFO queue, `scripts/testslots.py status` (`docs/test-execution.md` [D];
  scripts present [V]). The supervisor binds loopback.
- Passing files are reused when the file, config, platform and installed plugins match
  (`docs/test-reuse.md` [D]). The store is `~/.cache/test-reuse`, per user per device
  (`workspace/testruns.py:STORE_BASE` [V]); the key includes platform and machine, so a Mac pass is not a Linux
  pass.
- Landing a batch: `scripts/land plan|run|finish` take branch names and a repository path and know nothing
  of the board (`docs/landing-queue.md` [D]; `scripts/land_*.py` [V]).

### 1.6 Linux and WSL runner

- `scripts/test-on-linux [--single] [--rebuild] [--platform P] [pytest args]` runs the suite in a Docker
  container against the worktree (`scripts/test-on-linux` header [V]).
- Inside WSL: `scripts/setup-wsl-dev` creates a venv and refuses outside WSL (header [V]); the application
  can run in a WSL distribution (`fleet/wsl.py`, `wsl_network.py`, `wsl_startup.py` [V]); admission fixes
  for the Linux runner sit on unmerged branches (`fix/linux-admission-stdio-current`,
  `fix/linux-immutable-runner`, `docs/linux-wsl-current` [V: branches exist]; contents not reviewed here).
- A Windows or WSL device joins a pool as any other device would; whether it does today is a live-proof
  question and is not claimed.

### 1.7 Harness surface

- `ml-stack-workspace` is the CLI every harness uses; `ml-stack-mcp` exposes the same functions as MCP tools
  over stdio, including the board tools `workspace_status`, `inbox`, `send`, `thread`, `announce`, `claim`,
  `release`, `heartbeat`, `tasks`, `task` and more (`mcp.py` line 492, `workspace/tools.py` [V]). Register with
  `claude mcp add ml-stack -- ml-stack-mcp` (`mcp.py` docstring [V]).
- Each harness registers as itself; person-only operations check a terminal and no agent marker
  (`person.py:require_person`, `AGENT_MARKERS` [V]). `docs/person-delegation.md` invariants 1 and 2: every
  agent acts as itself, none can pose as a person [D].

## 2. What is missing, ranked within each goal

Rank is by how much of the goal a missing item blocks, first item first.

### 2.1 Any device as a test runner

1. **No pool-wide dispatch of tests.** `fleet/commands.py:allowed` refuses pytest [V]. Needed: a runner role
   whose job is `scripts/test ... --label` on a device checkout at a named commit, outside the bench
   allowlist, with its own admission. The target must hold the same commit; the bench path already refuses a
   commit mismatch (`docs/fleet.md` [D]), the same refusal applies. Security: this is remote code execution on
   a peer, so it needs a person grant per project and device, never an agent token (section 2.4 rule).
2. **No shared result store.** Reuse is per device and per platform [V]. Needed: evidence records
   (`activity/gate.py` evidence ids and tree hash, named in `docs/landing-queue.md` [D]) posted to the board
   by the runner under its own agent identity, so the queue reads a result from any device. Per-platform
   keys stay separate; a pass on Linux does not satisfy a macOS requirement.
3. **No capability or admission advertisement for tests.** The daemon report carries labels, backend, room,
   `measuring` and schedule windows [V] but nothing about test permit capacity, held GPU lease, branch checkouts
   present or WSL/Linux platform for tests. Needed: a `test` block in `Daemon.report()` (capacity from
   `testslots.py status`, platform key, checkout commits) so `Requires` can select on it.
4. **Board-fed landing queue unbuilt.** `workspace/landing.py`, `scripts/land submit|watch` and the
   `--entries FILE` extension to `land run` do not exist [V]; the four steps are listed in
   `docs/landing-queue.md` [D]. Runs on one device first; it does not need the mesh.
5. **Cross-device broker coordination.** Test admission and the GPU lease are separate per device. A machine
   running a model needs the test broker to see the lease (heavy lane while a model holds the GPU). Today
   `docs/test-execution.md` [D] gates heavy tests by lane, not by lease.
6. **Linux/WSL runner activation.** The runner branches above are not on `0.2dev`; an always-on Linux box
   additionally needs the daemon as a service (autostart, section 2.5(c)).

### 2.2 Any device serving local models for the pool

1. **Routing is chat-only and by announcement.** `targets()` lists peer models and sends chat to the peer's
   `/infer` [V]. Missing: routing for coding harnesses, a "run this on whichever device holds model X or has room"
   decision, and failure fallback. `ml-stack-cluster plan` computes placement but a person runs it [D].
2. **No lease broker reachable per device.** `broker_wire` is loopback [V]. A remote request to load a model
   today is `POST /serve` from `plan --apply` [D]; it should be a lease with reason and requester under the
   caller's own identity, taken through the target's broker, so the target's queue and history stay the one
   record. Open: the `/serve` route authorizes by the cluster-derived token, not by agent identity (token
   derivation in `fleet/discovery.py` [V]); a lease request needs the board identity of the asker.
3. **One thing on the GPU, pool wide.** The rule is enforced per machine by the manager (one set of settings
   per port, a lease for other settings is refused [D]). Nothing coordinates two devices, which is fine since
   each has its own GPU; the real gap is a device that is both a test runner and a model host (2.1(5)) and a
   request that would evict a model another caller leases.
4. **Wake and unload.** No Wake-on-LAN, sleep detection or idle unload exists: no WoL or idle-unload code was
   found under `serve/` or `fleet/` (searched `idle_timeout`, `idle_unload`, `unload_after`, `keep_alive`, `wake_on_lan`, `magic packet` under `serve/` and `fleet/`) [V]. `workspace/wake.py`
   is board wake-ups over named pipes, not machine power. Needed: an "asleep" state in the report and a
   person-granted wake action; unload is `ml-stack-serve down` and must go through the same lease check.
5. **Model files.** Paired devices already fetch model files from each other before the Hub
   (`ml-stack-models pull`, `fleet fetch`, `docs/onboarding.md` [D]), so a device without the model can
   obtain it without the internet. This works.

### 2.3 Commanding from a phone

What exists [V]: a native Android companion with QR enrollment, biometric unlock, one hour scope
`fleet.status` and `chat`, revocable from the owner's local Fleet page; invites are minted only from the
owner's loopback page by a person browser session (`fleet/invite_routes.py:ui_route` [V]). Physical-device
checks of QR, biometric and pinning are still outstanding (README [D]). iPhone: nothing exists.

Missing, ranked:

1. **A phone person credential.** The person session is loopback only [V]. A phone must be a
   person-credentialed *device*: a device key enrolled by a person-approved QR or passphrase on the host's
   own owner page, with scoped capabilities and expiry, revocable, and recorded as a device row (like
   `Invitations.devices`). It must never be an agent token and must never let a model hold or relay it:
   the host checks `person.require_person`-style provenance on the minting side only, and the phone's use of
   it is bounded to a named capability list. It must not widen the loopback floor for the browser session of
   the host itself. (`docs/person-delegation.md` invariants 1, 2, 4 [D].)
2. **Capabilities beyond status and chat.** Needed in order: read the board and pool status; approve or deny
   a pending approval (the structured approval question channel, `docs/person-delegation.md` sections 7 and 12
   [D]); start or stop a model lease; submit a test job; read results. Each a named capability on the
   credential. Machine settings, daemon install and guards stay terminal-only.
3. **A phone client surface.** Two ways, section 4 decides:
   - PWA served by the pool daemon. Needs HTTPS the phone trusts (section 3.2), a web app manifest and a
     service worker (neither exists: no `webmanifest` or `serviceWorker` in `src` [V]), and the UI components
     are written for the desktop owner page (`fleet/web/components/*.html` [V]); only a narrow phone view
     is needed.
   - Native app: the Android code exists; extending the protocol is the work. iOS is a new app.
4. **Push notifications.** Neither route has push. A PWA gets Web Push only over HTTPS with a service worker,
   and on iOS only when installed to the home screen. A native Android app needs FCM or a local poll; this
   introduces a third-party service and is an owner question (section 5).
5. **QR pairing UI for a web client.** QR drawing exists on the host (`fleet/invite_routes.py` uses the
   `qrcode` package [V]); the scanning end in a browser needs the camera, which needs a secure context
   (section 3.1). Pasting the invite works without the camera [V for Android; web untested].

Remote access off the home network: out of scope except that section 3 shows the existing Tailscale route
covers it for native clients and for the daemon, with no new public port.

### 2.4 Commanding from a harness

Mostly present: workspace CLI and MCP exist (section 1.7). Gaps, ranked:

1. **The MCP surface covers the board, not the pool.** Tools for the pool are `look`, `bench` and a few
   serving reads (`mcp.py` [V]); missing tools: list peers with capability, request a lease on a peer, submit
   or read a test job on a device, read the landing queue. Each takes the agent's own token (`_token()`
   in `workspace/tools.py` [V]); none takes a person credential.
2. **Remote board access needs coordinator enrollment per harness per device.** Done by invitation or
   automatic project enrollment [D]; the claim that Codex and Claude on another device both work needs a live
   proof on the owner's devices and is not asserted.
3. **Rule for every new remote action.** The action is allowed to an agent only through a capability the
   agent's own token carries, minted by a person; a job that executes code on a device (2.1(1), 2.2(2))
   requires a person-recorded grant for that device and project. No path takes a phone credential and gives it
   to a harness, or the other way. This is `docs/person-delegation.md` invariants 1 and 2 [D] applied.

### 2.5 Reachability, secure context and Tailscale

This section answers the coordinator's added request. Web-platform requirements below are browser behavior,
not repository facts; the repository facts are tagged.

**(a) Secure context.** A page loaded over plain `http://` from a LAN address is not a secure context in a
phone browser. There, service workers, install to home screen as a PWA, the camera (`getUserMedia`, needed to
scan a pairing QR in a page), WebAuthn and the async clipboard API are unavailable or degraded. What the UI
actually uses today [V, `grep` over `src/ml_stack/fleet`]: `fetch`, `localStorage`, `sessionStorage` and
`navigator.clipboard` (copy buttons in `chat-view.html`, `projects-view.html`, `cluster-actions.html`). It uses
no service worker, manifest, camera, WebAuthn or `crypto.subtle`. So the UI itself loads over plain LAN HTTP,
except the copy buttons; what breaks is everything a phone app would add. The daemon on the LAN is already
HTTPS, but with its own self-signed certificate that browsers do not trust [V `tls.py`]; a phone page there
shows a certificate warning and, even once accepted, browsers refuse to register a service worker on an
untrusted certificate. The session cookie sets `Secure` only if asked, and sign-in sends a passphrase
(`fleet/session.py:cookie_header`, `routes.py:_sign_in` [V]).

| option | HTTPS name | Phone sees | Breaks or limits |
|---|---|---|---|
| option A: the tailnet (`tailscale serve`, a certificate for the tailnet DNS name) | `host.tailnet-name.ts.net`, stable | Valid public-CA certificate, secure context | Needs the Tailscale app on the phone and every device and a Tailscale account; certificate issuance is transparent-log public (the machine name appears in CT logs); the daemon's `host_ok` refuses a Host that is a DNS name and a `serve` proxy arrives as loopback with that Host, so person routes stay refused until section 3 slice 4 changes the credential model [V `ui.py`, `person_session.py`]; the Android companion and invite refuse tailnet IPv4 (below) |
| option B: a local certificate authority the phone trusts | a name the owner picks, needs local DNS or mDNS | Secure context once the CA is installed | iOS needs the profile installed and enabled as full trust, Android 11+ cannot trust a user CA for apps that do not opt in (the native app pins its own certificate and is unaffected); we would own a CA private key, an attack target and a security surface; the CA install is a person-only device setting; renewals and DNS are ours to run |
| option C: plain LAN, no HTTPS trust | IP or `.local` | Not secure, or a warning page | No PWA install, no service worker, no web push, no camera in page, no WebAuthn, copy buttons fail; native Android app unaffected because it pins the certificate itself (`app/android` `PinnedHttps.java` [V]); the sign-in passphrase protection rests on the self-signed TLS only |

**(b) Tailscale as transport only.** Already the design for detection and routing
(`docs/onboarding.md` "Devices on other networks", `docs/security.md` [D]; `fleet/tailnet.py`,
`fleet/onboard/routes.py` [V]): a tailnet address is a route, the pinned certificate still decides, and an
address is stored only after the certificate matched there.
- Tailnet membership grants nothing: no device, agent, approval, lease or job is accepted because a peer is on
  the tailnet. Identities, journals and person credentials stay ours. The "tailnet identity of the caller" is
  explicitly not checked [D onboarding.md], and this note keeps it that way.
- A Tailscale ACL is a second fence, written by the person in the Tailscale admin console to limit which
  tailnet nodes reach the daemon port (default 8770, `fleet/daemon.py:DEFAULT_PORT` [V]); we neither write nor
  rely on it.
- Binding: a joined daemon already binds `0.0.0.0` (`bind_address` [V]), so it is reachable on its tailnet
  address with no change. The not-yet-built "listen on the tailnet interface only" is listed as not done
  [D onboarding.md]. The mesh transport (`/workspace/v1/mesh/pull|push` per `docs/mesh-board.md`) would be
  routes on that same listener, authenticated by the pinned certificate and by a row signature from the origin
  device's key, so a relayed row is verified by the origin and not by the path or the tailnet.
- Verified gap: `Invitations.mint` requires `address.is_private`, and `companion_routes.answer` requires
  `source.is_private`; for `100.64.0.0/10` Python's `ipaddress` returns `is_private == False`
  (checked with `ipaddress.ip_address('100.101.1.2').is_private`), while `fd7a:115c:a1e0::/48` returns True.
  So the Android companion cannot enroll or connect over a tailnet IPv4 address today. A one-line widening
  to `or in_tailnet(...)` (`fleet/onboard/lan.py:in_tailnet` [V]) fixes both; it is a code change and is not
  made here.

**(c) Install and `tailscale serve` are person-only system settings.** We prepare and check; the person
installs. The pattern is `ml_stack.fleet.autostart`: `prepare` stages unit files and a manifest and installs
nothing; `install`, `rollback` are for a person at a terminal (`fleet/autostart_cli.py` imports `HumanRequired`
and says "a person at a terminal only" [V]); `status` and `verify` compare installed with prepared and change
nothing [V]. A Tailscale slice follows it: `prepare` writes the exact `tailscale serve` command and a
manifest; `verify` runs only the read-only `tailscale status --json` that `READ_ONLY` already limits
the detector to [V]; the person runs the commands in a terminal. Funnel is never used (public exposure).

**(d) Detection without a dependency.** Done [V]: `detect()` finds the CLI on `PATH` or in the macOS app,
runs only `status --json`, and returns not installed, down, or up; nothing runs when no command needs the
tailnet (`docs/onboarding.md` the section on running without it [D]). LAN works unchanged without it. A phone page can
learn whether it is secure from `window.isSecureContext` and show the pairing instructions accordingly.

## 3. Recommended slice order

Sizes: S under a day of agent work, M a few days, L about a week or more. Each slice is its own reviewed
branch landed before the next starts. Dependencies name slice numbers.

| # | Slice | Size | Needs | Builds on |
|---|---|---|---|---|
| 1 | Admit tailnet IPv4 in the companion and invite checks; `ml-stack-cluster devices` shows the route per device | S | none | `fleet/onboard/lan.py:in_tailnet`, `fleet/invites.py`, `companion_routes.py`; `tests/test_onboard_tailnet.py` |
| 2 | Device capability block in `Daemon.report()` (test capacity, platform key, checkouts, awake/asleep) and a `Requires` filter on it | S-M | none | `fleet/pool.py`, `fleet/daemon.py:report`, `scripts/testslots.py status` |
| 3 | Board-fed landing queue: `workspace/landing.py`, `scripts/land submit`, `land watch --once`, `land run --entries` | M | none (single device) | `docs/landing-queue.md`; `scripts/land_*.py`; `workspace/task_integration.py` |
| 4 | Phone person credential: device-key enrollment from the owner page, named capabilities, expiry, revoke, read-only status and board | M-L | 1 | `fleet/invites.py` `devices` and `_grant`, `companion_routes.py`, `docs/person-delegation.md` |
| 5 | Remote test runner: a person-granted job kind for `scripts/test` on a named commit, result posted as evidence by the runner's own agent | L | 2, 3 | `fleet/commands.py`, `docs/test-execution.md`, `docs/test-reuse.md`, `workspace/testruns.py`; branches `docs/linux-wsl-current`, `fix/linux-admission-stdio-current`, `fix/linux-immutable-runner` to be reviewed first |
| 6 | Mesh board stage 1: journals for append-only streams (messages, notes) between two devices, outbox, remove agent-settable `host` | L | none, but ahead of any N>2 claim | `docs/mesh-board.md` slice 1; `workspace/chain.py`, `board_graph_merge.py` |
| 7 | Lease through a remote broker under the asker's identity; `plan --apply` becomes a lease with reason | M | 2, 4 | `serve/broker_wire.py`, `serve/leases.py`, `fleet/api.py` `/serve` |
| 8 | Phone capabilities: approve/deny, start/stop lease, submit test, read results | M | 4, 5, 7 | `docs/person-delegation.md` sections 7 and 12 |
| 9 | Pool MCP tools: peers, lease request, test submit/read, landing queue, each with the agent's own token | M | 3, 5, 7 | `mcp.py`, `workspace/tools.py` |
| 10 | Tailscale prepare/verify slice in the autostart pattern (commands staged, person installs) | S-M | 1 | `fleet/autostart_cli.py`, `fleet/tailnet.py`, `docs/onboarding.md` |
| 11 | PWA client for phone: manifest, service worker, phone views, QR scan | M-L | 4, 10 (HTTPS) or the owner picks native only | `fleet/web/components`, `fleet/invite_routes.py` |
| 12 | Wake and idle unload | M | 2, 7 | none exists; Wake-on-LAN needs the sleeping device's NIC settings, a person setting |

Optional, never authority: the parked NATS bundle (`docs/nats-coordination-quickstart` and the
`feat/nats-*`, `fix/nats-*` branches, listed by `git branch`) is one possible transport for slice 6; the
mesh note's pairwise signed journals do not require it, and a fix there depends on a review of those branches
that this note did not make.

Slices 1, 2, 3 and 10 have no dependency on each other and can run in parallel. The phone path is 1 then 4 then 8
(with 5 and 7 supplying the things it would command). Harness commanding is already usable for the board;
slice 9 completes it.

## 4. What the order assumes

- The pool is a set of paired devices with one coordinator device holding the board until slice 6 and the
  later mesh stages; no new always-on device is required for slices 1 to 5, 8 and 9 except the coordinator.
- A phone credential is never an agent identity, never relayed by a model, and its minting needs the
  person at the host's own owner page. Nothing in this note adds an environment variable, flag or trusted
  agent role that skips that.
- Anything that executes code on another device needs a person-recorded grant for that device and project.
- Tailscale, a local CA and plain LAN all remain supported transports; none adds authority.

## 5. Owner-only decisions

Each is one question, options with costs. None has been decided.

1. **Phone client: PWA first, native Android first, or both?**
   - PWA: one code base for every phone including iPhone, no store; needs HTTPS trust (question 2), a manifest,
     a service worker, and a phone view. Cost M-L plus the HTTPS setup.
   - Native Android first: the code exists with QR, pinning and biometric; the work is protocol capabilities
     (slices 4 and 8). No iPhone. Physical-device checks outstanding.
   - Both: the capability protocol (slice 4) is shared, the clients are separate. Cost is the sum.
   Prior question to you: which phone does the household use, Android, iPhone or both?
2. **First transport for the phone: Tailscale, a local CA, or LAN-only?**
   - Tailscale: valid HTTPS and a stable name with the least certificate machinery; costs a Tailscale account,
     the app on every device, CT-log visibility of the machine name, and slice 1 plus slice 10. Also reaches the
     phone from outside the home with no new open port, which is the only remote path the design covers.
   - Local CA: no third party and works offline; costs a CA key to protect, a per-phone profile install,
     local naming and renewal, all ours.
   - LAN-only: nothing to install; no PWA install, push or in-page camera, and the native Android app is the
     only polished phone client. Cheapest and safest to start.
3. **Remote access off the home network at all?** No: the pool stays at home, tailnet slices are optional.
   Yes via Tailscale: the same transport, the ACL is yours to write, and the phone credential's expiry becomes
   more important. Not via any public port or Funnel in any option.
4. **Which devices are always-on?** The coordinator device needs to be (it holds the board until the mesh
   exists); each always-on device needs the autostart install (a person action) and a decision on sleep. Cost of
   making a laptop the coordinator: the board vanishes when it sleeps; cost of a desk machine: its power draw.
5. **May a phone approve things, or only watch and chat?** Approve/deny (slice 8) lets a phone answer the
   structured approval question for `release-main`; the cost is that a lost, unlocked phone holds that power
   until revoked. Options: watch and chat only; approve with a biometric prompt per approval; approve for a
   short window after unlock.
6. **Push notifications through a third-party service (FCM or Web Push relays), or poll while the app is open?**
   Push costs a third party learning that events occur (not their content if we send only a tick); polling costs
   freshness.
7. **Which device may run tests for which project, and may a runner be reached by an agent without a fresh
   person grant each time?** Per-job grant is safest and slowest; a standing grant per device and project
   (expiring) is the middle; none is not acceptable for slice 5.
8. **Does the NATS bundle get reviewed as a slice 6 transport, or is the pairwise journal design the only path?**
   Review costs a read of the parked branches; skipping costs nothing now and leaves the choice open.
