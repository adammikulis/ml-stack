# What is encrypted in the pool, and the plan

Audit of the code at `c1007d3b` (`0.2dev`), read in the source, not taken from the docs. Where a doc and
the code disagree the code is reported. "Sealed" is this repository's word for application-layer
AES-256-GCM (`sealing.py`). Nothing here asks for hand-written cryptography: every primitive named is
from `cryptography` (already a dependency, `pyproject.toml:28`), the stdlib `ssl` (OpenSSL), `spake2`,
the OS keystore through `keyring`, or the Android keystore.

## 1. Findings in one paragraph

The fleet daemon (device to device, and phone to device) is in good shape: pinned TLS, signed and sealed
requests, no CA. The gaps are elsewhere. (a) The per-device journal mesh (`/workspace/v1/mesh/pull|push`)
**does not exist**; `docs/mesh-board.md` says "nothing is implemented" and `grep` finds no such route, so
today's board is a host-and-clients project board, not a replicated mesh. (b) Authentication between
devices rests on one **shared cluster key** per cluster, so there is no per-device identity on most
routes, no forward secrecy in the sealing layer, and no way to revoke one device without rekeying.
(c) Several stores at rest are plaintext: the cluster key itself, the TLS private key, job records and
logs, saved conversations, workspace tokens, the project-board graph. (d) File, model and `/infer`
streams are signed but not sealed; they rely on the transport.

## 2. The table (verified in code)

Legend: **TLS** pinned self-signed certificate (`fleet/tls.py`); **sealed** AES-256-GCM above TLS;
**signed** HMAC-SHA256 request MAC with timestamp and nonce (`macauth.py`); **plaintext** nothing.
"Mutual" means both ends are authenticated; here the server is authenticated by the pinned certificate and
the client by a MAC over the request, not by a client certificate and not bound to the TLS session.

### 2.1 In transit

| Channel | Today | Authenticated | Primitive and library | Key custody | Gap |
|---|---|---|---|---|---|
| Fleet daemon, any non-loopback listener (`fleet/daemon.py:listen`, `fleet/tls.py`) | TLS 1.3 only (changed by slice 1) | Server: pinned certificate. Client: its own certificate, which must be an active member (slice 2, section 10), plus the MAC | stdlib `ssl`; ECDSA P-256 self-signed, ten year lifetime (slice 2: an identity, not a session credential); `cryptography` makes it, else the `openssl` binary | `cert.pem`/`key.pem` 0600 in the daemon's identity dir, plaintext on disk | (the TLS-off switch is gone.) Key at rest unencrypted |
| Loopback listener (`127.0.0.1:8770`) | plaintext HTTP by design | MAC or session | n/a | n/a | In scope only for the same-machine threat (section 3) |
| Request and response bodies on `/workspace/v1/*`, `/jobs`, work dispatch, board/note/DM routes (`http.build_request`, `fleet/api.py:_unsealed`) | sealed inside TLS | Request is signed (method, target, host, body, 120 s window, nonce); answer is sealed with the request nonce and status as AAD | HMAC-SHA256, HKDF-SHA256 written with stdlib `hmac`/`hashlib` (`macauth.derive`, `keystore.hkdf`); AES-256-GCM, random 96 bit nonce (`sealing.py`) via `cryptography` | Key derived from the cluster key (`/workspace/v1/*` also accepts per-device secrets, `fleet/device_auth.py`) | No forward secrecy: a leaked cluster or device secret opens every captured body. Nonce cache is in memory (resets on restart; the 120 s window bounds it). `Sealed.open` accepts an unsealed answer when status >= 400, empty body or non-JSON, so streams and errors are not sealed |
| Fleet job dispatch (`POST /jobs`) | as above | cluster MAC | as above | cluster key | Job argv and env travel sealed; job results and logs are fetched through the same sealed JSON, but log streaming is not sealed |
| Model requests, `/infer/*` proxy (`fleet/api.py:_proxy`) | sealed request body; streamed answer is TLS only | cluster or device MAC | as above | as above | Streamed tokens are not sealed by the application (`docs/fleet.md:315`). The proxy hop to the model server is plaintext loopback, and the model server sees plaintext prompts |
| Board messages, notes, DMs (project-board RPC, `workspace/remote.py`, `coordinator.py`) | sealed; response refused unless sealed (`remote.py:75`) | per-agent token plus device MAC | as above | agent token files (plaintext, restricted) | Encrypted to the host device, not to the addressee. The board host reads every DM and note. `remote.py` accepts an `http://` LAN project host: then no TLS, only sealing, and the host is trusted only because cluster discovery matched it |
| Board listener and `/ui` routes (`workspace/boardroute.py`, `fleet/ui.py`) | `/ui` on a LAN is TLS-only (`--ui-from-lan`); loopback is plaintext | Passphrase sign-in; same-origin and Origin checks (`boardroute._checked`) | cookie session | session in memory | Passphrase travels inside TLS only; no second factor. Loopback session is exposed to a same-user process (out of scope) |
| Mesh journal exchange `/workspace/v1/mesh/pull|push` | **not implemented** | designed: pinned TLS plus signed journals (`docs/mesh-board.md:281`) | designed | designed | The whole mesh is design only. Ship it with the plan in section 5, not without |
| Remote worker controls (`workspace/remote_host.py`, `remote_workers.py`) | as the board RPC: sealed, MAC | agent token | as above | agent tokens | Same as board |
| Test job results and logs (`workspace/testruns.py`, `activity/reuse.py`) | local files; remote fetch sealed JSON | hash-chained entries (unkeyed hashing) | sha-256 hashes | none | Integrity only; contents plaintext on disk |
| Model weights and file transfer (`fleet/onboard/transfer.py`, `fleet/files.py`) | TLS to the pinned peer; ranged chunks | Manifest signed with Ed25519 (`fleet/onboard/signing.py`), each chunk sha256-checked | `cryptography` Ed25519 | signing key wrapped under the OS keystore | **Signed, not sealed**: with TLS off the weights are readable on the wire. Fine otherwise; weights are not secret except private fine-tunes (section 5, slice 6) |
| Pairing handshake (`fleet/onboard/pake.py`, `pairing.py`) | TLS to an unverified certificate, then SPAKE2 whose identities are both certificate fingerprints; confirmations HMAC-SHA256; the cluster key is sent sealed under the exchange key | Mutual, by the six digit code, three tries, 120 s | `spake2` (python-spake2), `cryptography` | Cluster key lands in `clusters.json` plaintext | Sound design. Result is a shared cluster key, not a per-device identity |
| Signed-sealed fleet capabilities ("sealed" in `workspace/remote.py`, `sealing.py`) | sealed + signed bodies | see above | AES-256-GCM | derived from cluster key | Capability confidentiality equals cluster-key custody; a guest with the key is a full member (section 7) |
| Beacons (UDP 8773) | sealed under a key derived from the cluster key | the seal is the authentication | AES-256-GCM | cluster key | The one unsealed datagram is the join-by-name request and answer. Beacon carries the certificate to pin, so a holder of the cluster key can forge a beacon: the cluster key is the root of trust |
| Phone app (`app/android/.../PinnedHttps.java`) | TLS 1.3/1.2 to a pinned certificate; `usesCleartextTraffic=false` | server pinned; client by enrolment secret | platform JSSE | enrolment sealed with AES-GCM in the Android keystore (`DeviceVault.java`) | Good. No hardware-backed attestation requested |
| Desktop app (Tauri) | opens only the local UI; tests refuse other origins | n/a | n/a | n/a | n/a |
| Tailscale and other overlays | not trusted by design: the daemon uses its own TLS and pins; `lan.py` allows 100.64.0.0/10 and fd7a:115c:a1e0::/48 as local | as the daemon | WireGuard under it | Tailscale's | Correct. Keep: encrypt inside the transport (section 6) |

### 2.2 At rest

| Store | Today | Primitive | Key custody | Gap |
|---|---|---|---|---|
| Memory store, `requests` store, `reputation` graph, activity log, workspace posted-files store | encrypted (`memory/vault.py`, `reputation/sealed.py`, `activity/log.py`, `workspace/filestore.py`) | AES-256-GCM, `cryptography` | HKDF subkey of a 32 byte master in the OS keystore (`keystore.py`); passphrase+scrypt mode when headless | Good where attended. Background processes never create the master |
| Cluster key (`~/.ml-stack/cluster.key`, `clusters.json`) | **plaintext**, 0600 (`private_file`) | none | file | Root of all fleet trust sits in a file. The recovery file and passphrase are kept in the keystore (`fleet/recovery.py`) but the key itself is not |
| Fleet TLS private key (`cert.pem`, `key.pem`) | **plaintext**, 0600, symlink and ACL checks (`tls.py`) | none | file | Same |
| Cluster signing key (Ed25519) | wrapped | AES-256-GCM under keystore subkey `fleet-signing`; headless: scrypt + ChaCha20-Poly1305 | keystore or passphrase | Good |
| Credentials | encrypted when stored with `--keychain` | keystore `wrap` | keystore | Others live in the environment or file by the user's choice |
| Board graph (`workspace/board_graph.py`, `graphlog.py`, ladybug store), conversations (`fleet/conversation_graph.py`), person log (`person_store.py`) | **no encryption found in code**; hash chain (unkeyed) or "sealed" meaning HMAC | none | none | Messages, DMs, notes and chat history are plaintext at rest on every device that holds them |
| Job records and logs (`fleet/jobs.py` writes `job.json`, `log`) | **plaintext** | none | none | Command lines and output of every job, including remote dispatch, on disk |
| Sentinel events (`sentinel/sealed.py`) | HMAC only | HMAC-SHA256 | key beside the file | Tamper-evident, not confidential |
| Workspace token files (`workspace/tokens.py`) | plaintext, restricted perms, sentinel-registered | none | file | Agent capability tokens |
| Journal replica | does not exist yet | planned in `mesh-board.md:450` | planned: device key | Specify in slice 3 |
| Backups and bundles | no backup writer found that encrypts; the recovery file (`fleet/recovery.py`) is the one export, passphrase-derived | scrypt (check KDF params in slice 1) | passphrase | Anything a person copies from `~/.ml-stack` is plaintext |

Keystore usage: `keystore.py` is the only importer of `keyring`; unattended processes never create the
master and read it only after `ml-stack-security unlock`. Consequence: a headless device holds its
secrets in a passphrase-wrapped key file (decision 8.1), the only mode section 3 counts as protection
against a stolen disk.

### 2.3 Misconfiguration fixes

None made. The three candidates were checked: `lan.py` `require_local` calls `allowed_address` (name
reads backwards, behaviour is right: it refuses public addresses); every `http://` use in `src/` is
loopback or the explicit LAN project-host opt-in; every pinned context sets `CERT_REQUIRED`, loads only
the pinned certificate and disables hostname checks on purpose. `TLSv1_2` as the minimum
(`tls.py:179,189`, `pairing.py:284`, `bootstrap.py:53`) is not a bug: a P-256 certificate only gets ECDHE
suites, and the phone sets 1.3 then 1.2. The owner has decided to raise the floor to 1.3 with mutual authentication (section 8.1).

## 3. Threat model

| Adversary | In scope? | What defends today | What the plan adds |
|---|---|---|---|
| Passive LAN or Wi-Fi observer | In | Pinned TLS (ECDHE, forward secret at the TLS layer) + sealed bodies. Failure only with `ML_STACK_FLEET_TLS=off` or an `http://` LAN project host | Remove TLS-off; seal streams (slices 1, 2b) |
| Active malicious device on the LAN, not paired | In | Pinned certificates, no first-use trust; MAC with nonce; SPAKE2 pairing; `lan.py` | Nothing structural |
| Malicious or compromised paired device (holds the cluster key) | In, partly. Today it can read and forge everything in the cluster | None: one shared key | Per-device keys, revocation, E2E to addressee (slices 2, 2b, 4, 5) |
| Network operator or overlay (Tailscale coordination server, relay) | In | Transport never trusted | Keep |
| Stolen or lost powered-off disk | In | Memory, requests, reputation, activity, files, signing key are encrypted under the keystore | Encrypt the rest (slice 3) |
| Stolen unlocked laptop or a live session | Out | Needs OS lock and disk encryption (FileVault, BitLocker, LUKS); the OS keystore releases the master to the logged-in user | State in `SECURITY.md` |
| Malicious process of the same OS user | Out as a confidentiality goal. It can read the keystore master after one prompt, ptrace, edit code, read process memory. The sentinel and sandbox reduce it and detect it; they do not prevent it | Sentinel registers sensitive paths; sandbox for agents | Document honestly; per-purpose keystore prompts where the OS offers them |
| Malicious process of another OS user on the same machine | In for files (0600/ACL) and loopback tokens; out for kernel or root | Perms | Guest tenancy (section 7) uses a separate OS user |
| The device owner against data placed by a guest | Out at the software level; see section 7 | n/a | Level 1 only |
| Quantum adversary recording today's traffic | Out (note: ECDHE P-256 is not post-quantum; OpenSSL hybrid groups come with OpenSSL 3.5+, revisit when `ssl` exposes them) | | |

## 4. Choice of mechanism

Candidates for authenticated, confidential, forward-secret device links: Noise (needs a new dependency:
no maintained, audited Python Noise library is in `pyproject.toml`; `noiseprotocol` is unmaintained) or
**TLS 1.3 with mutual authentication and pinned self-signed device certificates** using stdlib `ssl`
over OpenSSL. The codebase already has the certificate, the pinning (`pinned_context`), the server context
and the beacon that carries the certificate; the phone already pins. **Recommendation: TLS 1.3, mutual,
pinned, same `ssl` module.** No new dependency and no protocol written here.

Required changes, all configuration of `ssl`:

- `minimum_version = TLSv1_3` for pool links (not for the loopback listener or bootstrap), `CERT_REQUIRED`
  on the server context plus `load_verify_locations(cadata=<paired device certs>)`.
- Every device has **its own certificate** (it already does, per daemon) listed in a per-device membership
  record, replacing "the cluster key holder is trusted". Server accepts a client only if its certificate
  is a paired, unrevoked one.
- Channel binding: the request MAC includes the TLS exporter value (`SSLSocket` does not expose
  exporters in the stdlib; if that stays true, bind with the peer certificate fingerprint in the MAC
  input instead, which closes relay by a different paired device).
- Application sealing stays as the second layer for bodies that cross a relay (see below), and moves
  from a shared-key derivation to a per-pair key (slice 5).
- Replay: keep the 120 s window and nonce cache; persist the cache's high-water timestamp so a restart
  cannot reopen the window; the TLS 1.3 handshake itself resists replay except 0-RTT, which `ssl`
  does not enable. Do not enable early data.
- Forward secrecy: TLS 1.3 gives it per connection; sealed bodies add none (static derived key). Where
  a body must stay confidential past a device compromise (DMs, notes), use an ephemeral-static
  agreement: **HPKE (RFC 9180)**, which `cryptography` 50.0.2 (the installed version, floor `>=50`) ships as
  `cryptography.hazmat.primitives.hpke` (import checked; its API and suites are to be read in slice 4).

## 5. Ranked plan with slices

Ranked by (gap closed) per effort. Sizes: S under a day, M one to three days, L a week. Every slice has
tests against real sockets and real files, not mocks (`AGENTS.md`).

| # | Slice | Closes | Size | Depends on | Notes |
|---|---|---|---|---|---|
| 1 | **Make unencrypted impossible by default.** `ML_STACK_FLEET_TLS=off` and `http://` LAN project hosts removed (decision 8.1), loopback http only; seal `/infer` streams and file chunks (framed AES-GCM, per-chunk nonce counter, AAD = request nonce and chunk index); pairing recovery file KDF parameters reviewed | passive observer in every configuration; streamed tokens | M | none | Sealing streams costs CPU at weight-transfer rates; measure on a 20 GB file before deciding to seal weights (private fine-tunes only) |
| 2 | **Per-device keys and identities, membership record and revocation** (decision 8.1). Each device has its own identity key and a device record (device id, certificate fingerprint, status) in a signed membership log (`mesh-board.md:275`, `member-add`, revocation) or a simple host-signed list as step one; request authority moves from the cluster key to these. Cluster key stays for beacon sealing only | malicious paired device acting as another; revocation by device | L | 1 | Precedes slice 2b: a certificate cannot be pinned to an identity that does not exist yet |
| 2b | **Mutual TLS 1.3 with pinned per-device certificates** as the floor on every pool link. `server_context` requests the client certificate; request MAC covers the client certificate fingerprint; TLS 1.2 refused | MAC not tied to channel; downgrade | M | 2 | Android below API 29 is refused (decision 8.1) |
| 3 | **Encrypt the plaintext stores** under keystore subkeys with `keystore.wrap` / the memory vault pattern: board graph, conversations, person log, job records and logs, workspace tokens, cluster key, TLS key. A device with no OS keystore uses the passphrase-wrapped key file of decision 8.1; no 0600 plaintext fallback | stolen disk | M per store, L together | none; reuse `memory/vault.py` | Each store gets a migration (as `memory/migrate.py`). The cluster and TLS keys are the first two: move the file contents into a wrapped blob |
| 4 | **End-to-end DMs and notes.** Payload encrypted to the addressee device's X25519 key (HPKE as in section 4), key published in the signed membership record; sender signs with its Ed25519 key. The host stores and relays ciphertext plus metadata (from, to, time, size, board). Group posts to a board stay readable by members: encrypt to a per-board key rotated on membership change | board host and relay learn nothing of DMs; compromised host | L | 2, a per-device Ed25519 identity (exists: `fleet/onboard/signing.py`) | Search, recall and agent reading of DMs happen on the addressee device only; the host cannot index them. Federated search of DMs is lost, by design |
| 5 | **Per-pair sealing keys.** Replace "cluster key derives everything" with an X25519 key agreement between device certificates' keys (ephemeral on each connection), sealing stays AES-256-GCM, MAC key from the same agreement | forward secrecy and compartmentalisation for the sealed layer | M | 2b | With mutual TLS 1.3 this is partly redundant; do it only for traffic that crosses a relay or the board host |
| 6 | **Journal replica at rest and mesh transport** (when the mesh lands): per-device journals signed with the device Ed25519 key (integrity), replica encrypted at rest with a keystore subkey (`mesh-board.md:450`), transport mTLS from slice 2b, entries addressed to a device encrypted as in slice 4, entries all members read stay signed-only | the mesh nonexistent today; stolen disk | L, part of the mesh work | 2, 2b, 3 | Do not build the mesh transport before slices 2, 2b and 3 |
| 7 | **Rotation and revocation.** Revoking a device: signed record, peers refuse its certificate at the next handshake, per-board keys and the cluster beacon key rotated on the next membership change, device certs rotate on the existing 30 day renewal and the beacon carries the new one signed by the old. Key compromise runbook | stolen paired device | M | 2, 2b, 4 | Beacon key rotation breaks discovery for offline devices until they rejoin: accept, document |
| 8 | **Hardware-backed keys where attended**: Secure Enclave (macOS), TPM 2.0 (Windows/Linux) for the device identity key, so the key cannot be copied off the disk. Windows uses CNG, macOS `kSecAttrTokenIDSecureEnclave` through `keyring` extension or `cryptography`'s lack of support is the dependency question | key copied by a same-user process (partially) | L | 2 | No vetted Python wrapper is already a dependency; may need a native helper. Not for the first release. **Secure Enclave half: blocked until the product takes off** (a signed app with the keychain entitlement needs a paid Apple developer account, section 8.1); unblocked alternative on macOS: the keystore-wrapped key or the passphrase-wrapped key file. The TPM half (Windows/Linux) is not affected |

What stays **signed-only**, and why encryption would add nothing: beacons' public half and presence
announcements (the point is to be discovered; the seal under the cluster key is for passive observers,
and a holder of the key can read it either way); the signed release and tracked-commit signatures and
model manifests (public artefacts verifiable by anyone); per-device journal rows that every member must
read (membership changes, revocations, task state, claims, public board posts), where the signature gives
integrity and origin, and encrypting to every member costs per-recipient wrapping for no confidentiality
gain inside the pool. Open weights downloaded from the internet are likewise hash- and manifest-verified,
not secret. Confidentiality for these comes from TLS to the peer, not from payload encryption.

## 6. Transports

Encrypt inside the transport; the transport is never trusted (Tailscale/WireGuard, a relay, a hotel
network, a phone hotspot). The device certificate is pinned from the signed membership record and not
from an address; a Tailscale address is just another way to reach the same pinned TLS endpoint
(`lan.py` already allows it). A relay forwards TLS records it cannot read and signed journal rows; it
never sees plaintext and never holds a key. Do not rely on Tailscale ACLs for authorisation: they say
which machine, not which person or tenant.

## 7. Guest tenancy

Scenario: a friend joins the owner's pool and sends test jobs, model requests and files to the owner's
devices; the owner must not see that work. And in reverse, the guest must not see the owner's data.

### 7.1 What today's design gives a guest, honestly

Nothing like isolation. A joiner receives the **cluster key** (`fleet/onboard/pake.py` path), which
derives the request secret, the seal key and the beacon key. That makes a guest a full member: it reads
every sealed body on the LAN it can capture, can sign requests as any device that uses the cluster
secret, and `/workspace/v1/*` device secrets are the only per-device credential. No `guest` pool role
exists (`grep -w guest` finds only unrelated sandbox/guard uses). Board channels have no read scoping
for devices. So section 7 is a **requirement for slice 2 and 7**, and the roles below are new.

### 7.2 What software can and cannot do on a machine the owner controls

The owner, as root, can read process memory, attach a debugger, replace the sandbox launcher, read any
unencrypted scratch, change the code that "never renders" a log, or snapshot the disk. No software
mechanism on that device prevents this. A guest should therefore be told: **Level 1 protects against
accidents, shared screens, other agents, other guests and later disclosure; Level 2 is the only
protection against the owner**, and today it does not exist.

### 7.3 Level 1: no casual visibility (buildable now)

Guest payloads, logs, results and prompts are encrypted to **the guest's key and the executing
sandbox's key**, and never decrypted into anything the owner's tooling renders.

- **Identity and role.** Guest gets its own device certificate and membership record with `role: guest`
  and a `valid_until`. It never receives the cluster key; it receives a per-pair key (slice 5) for each
  device it uses. A guest capability is scoped (`workspace/task_caps.py`-style, project/tenant
  bound): submit job, read own job, read own result. No read of owner boards, channels, DMs, notes,
  conversations, memory, activity log or other guests' jobs. The board enforces that on the server by
  tenant id from the certificate, never from a field the client sends.
- **Submission.** The job spec, input files and prompt are HPKE-encrypted to the **sandbox key**: an
  ephemeral X25519 key generated by the executing device inside the job's sandbox process and
  released (public half) only to the authenticated guest. The owner's daemon sees ciphertext, size,
  time and the resource request; it hands the blob to a launcher that holds no key.
- **Execution.** A dedicated OS user or profile per tenant, created on first use (`sandbox/` has
  seatbelt and bubblewrap; add a per-tenant uid on Linux/macOS, a restricted user on Windows). No host
  mounts, no network unless the job asks and the owner grants it, scratch on an ephemeral encrypted
  volume (a per-job file or tmpfs keyed with a random AES-256-GCM key that exists only in the job
  process). Core dumps off, swap off or the scratch not swappable (`mlock` for the key), no
  `ptrace` from other users (Yama `ptrace_scope`, macOS SIP-protected hardened runtime for the
  runner). The key is never written to the job record or to the environment of another process.
- **Results.** Results and logs are encrypted to the guest's public key (HPKE, same construction as DMs,
  slice 4) before they leave the sandbox; the owner's side stores ciphertext until the guest fetches
  it, then deletes it. Scratch is destroyed (crypto-erase: the key is dropped, then the file
  overwritten and unlinked) when the job ends.
- **Never rendered.** Guest jobs get an opaque id in the owner's UI, the board, the activity log,
  `hook_diagnostics.py`, transcripts and agent context. The activity log records `guest job <id>:
  <size>, <seconds>`, not command, path, stdout or error text. Error strings from guest code are
  scrubbed before logging; a test greps every owner-visible sink for a planted marker.
- **Metadata the owner still sees:** which guest (name, certificate fingerprint), when each job
  started and ended, size in and out, CPU/GPU/memory use and exit status, which model was requested
  (model names are needed to schedule), the guest's IP address. The owner cannot hide this without
  breaking scheduling and accounting. Decision 8.1 limits what the guest sees of the owner to one shared guest channel.
- **Local-model requests.** The model server process (`llama-server`) holds the plaintext prompt and
  the generated tokens in its memory and KV cache, and the pool's `/infer` proxy sees them. Implication:
  a guest prompt is protected from the owner's UI, logs and other tenants but not from the owner as root,
  and the KV cache is shared across requests in a slot (prefix reuse), so a guest's prefix is
  recoverable by a later request if cache reuse is shared. Options: (a) run guest inference in a
  per-tenant server instance with the prompt cache disabled or wiped on completion (costs throughput and
  memory); (b) for sensitive work, **the guest brings their own device** and sends only the result of
  the pool (the recommendation for anything the guest would not email the owner); (c) wait for Level 2.
  Do not claim more than (a) offers.
- **Test jobs.** Source and tests arrive encrypted; the sandbox user unpacks to the encrypted scratch;
  results are encrypted to the guest.
- **Caches and reuse.** The test-reuse cache (`docs/test-reuse.md`, `activity/reuse.py`) is keyed by
  sha256 over file bytes and settings: anyone who can guess a file can probe the cache for a hit, and
  an entry's `agent` and `command` fields are stored. For guests: a **per-tenant namespace**, keyed
  by `HMAC-SHA256(tenant secret, lookup)` so equal inputs from two tenants never collide or reveal
  each other, entries encrypted under the tenant's key, and no cross-tenant reuse of any kind (the
  speedup is not worth the oracle). Model KV and prompt caches the same. Compiled-artifact and
  download caches shared across tenants are public-content only (hash of a public URL).

### 7.4 Level 2: resistance to a determined owner

Only attestation or confidential computing can offer it, and it needs hardware **both** the owner and
the guest can verify. What exists and what does not:

- **Apple silicon.** The Secure Enclave holds P-256 keys that never leave it and can sign or
  perform key agreement; it does **not** run arbitrary code or run models, and the Apple GPU/ANE
  memory path is not encrypted against the owner. the attestation service and device attestation prove an app and
  device to Apple's service, not a sandboxed job to a third party. Unified memory is not protected from
  root. Verdict: protects the device identity key, not the workload.
- **x86/Intel/AMD.** TPM 2.0 measured boot and key sealing prove boot state and hold keys; they do not
  encrypt running memory. AMD SEV-SNP and Intel TDX encrypt VM memory against the hypervisor and host
  root and provide remote attestation, but need a supporting server CPU, firmware, a hypervisor
  configured for it, and a verifier. NVIDIA H100/Blackwell confidential computing extends this to the
  GPU. Consumer desktops and laptops, which is what this pool is, do not have SEV-SNP/TDX or GPU CC.
  I could not verify here which of the pool's devices (not enumerated in the repository; run
  `ml-stack` device reports to list them) have TPM 2.0, and no pool device is known to have SEV-SNP,
  TDX or GPU CC. Claim none.
- **What Level 2 would protect** (on supported hardware): prompt, weights placed by the guest, code and
  intermediate results from the host owner's software and from a debugger, to the extent the TEE
  implementation has no known side-channel. **What it would not**: traffic analysis and metadata,
  denial of service by the owner, side channels, a malicious guest job against the owner (the sandbox is
  still required), or the quality of the attestation vendor chain (you trust Intel/AMD/NVIDIA).
- **Therefore:** Level 2 is not buildable in this pool today, and on Apple silicon it is not attempted (decision 8.1). The honest guarantee to a guest is Level 1
  plus "bring your own device for secrets".

### 7.5 The owner's data against the guest

Tenant isolation is the same coin: the guest gets a `guest` role certificate, per-tenant namespaces for
jobs, caches and results, no read of owner channels, DMs, notes, conversations, memory or activity log,
no cluster key, no filesystem access outside its scratch, no network by default, no access to the owner's
loopback services (the model server and broker run on loopback and are reachable by a process of the
same OS user; the guest runs as another user and in a sandbox with `net` denied except the granted one),
and rate limits and quotas for resource use (`workspace/rates.py`, `limits.py`). Revocation (slice 7)
ends the guest immediately at the next handshake and kills running jobs, discarding scratch.

### 7.6 Slices for guest tenancy

| # | Slice | Size | Depends on |
|---|---|---|---|
| G1 | `guest` role in the membership record, scoped capabilities, board-side tenant filter, no cluster key for guests; the one shared guest channel (guests see each other there, nothing else of the owner); leak tests that plant a marker in every owner-visible sink | M | 2 |
| G2 | Per-tenant sandbox user and encrypted ephemeral scratch, key only in the job process, crypto-erase at the end | L | G1, 3 |
| G3 | Encrypted submission and results (HPKE to sandbox key / guest key) | M | G1, 4 |
| G4 | Per-tenant keyed result and prompt caches; cache off for guest model requests unless a per-tenant server instance | M | G1 |
| G5 | Attestation spike: enumerate the pool's hardware, TPM quote verification for the device key (Windows/Linux only; the Apple half, app attest, is blocked until the product takes off, section 8.1) | S then decide | 8 |

## 8. Owner-only decisions

Decisions taken on 2026-10-08 are in section 8.1. No question is open.

### 8.1 Decisions (2026-10-08)

**Per-device keys and identities replace the shared cluster key, with per-device revocation.** Decided: each
device has its own key and identity, recorded in a signed membership record; request authority rests on
those, and a device can be revoked alone. The cluster key stays, if at all, for beacon discovery only.
Rejected: keeping the shared key and adding per-pair sealing only, because every paired device would still
read and forge everything in the cluster and a lost device would mean re-keying all of them; and "pools are one
person's devices, so neither", because a stolen phone or laptop is the likeliest event.

**TLS 1.3 with mutual authentication and pinned per-device certificates is the floor.** Decided: pool links
accept nothing below it, so `ML_STACK_FLEET_TLS=off` and `http://` LAN project hosts are removed (not
flagged) in slice 1. Android devices below API 29 are refused. Rejected: TLS 1.2 with ECDHE-only suites,
because it keeps a downgrade path for the benefit of old phones only. Order: the key model (slice 2) lands
before certificate pinning (slice 2b); a pinned certificate has nothing to pin to until a device has its own
identity.

**Board DMs and notes are end-to-end encrypted to the addressee's device (HPKE).** Decided, accepting that
host-side search, summaries and agent recall of DMs stop unless the addressee's device indexes them locally.
Rejected: a per-board key only (the host cannot read, but every member can, which is not a DM), and "not
now", because the board host is the likeliest compromised party and the DM store is the one that grows.

**Guest privacy: build Level 1 now, Level 2 only if attestation hardware is bought.** Decided: Level 1 (no
casual visibility to the owner, section 7.3) is built; Level 2 (section 7.4) waits for hardware both sides can
verify, and slice G5 stays a spike. Rejected: "no guests, friends bring their own pool", because Level 1 is
buildable now with its limit stated; and building Level 2 on devices that cannot attest, which would claim a
guarantee nobody can check. The stated limit of Level 1 is that a determined owner can still read a guest's job.

**What a guest sees of the owner: its own work plus one shared guest channel.** Decided: a guest sees its
own jobs and results and one shared guest channel with the owner. It sees no other owner channel, no member
list and no device inventory; guests in that channel see each other's names and messages. Rejected: own work
only, because a guest could not talk to the owner inside the pool; a member list, because it exposes the
device inventory. Consequence: slice G1's tenant filter admits exactly that channel and nothing else, and
the guest channel is the one place guest identities are visible to each other (a leak test plants a marker
in every other sink).

**A device with no OS keystore: a passphrase-wrapped key file.** Decided: such a device holds its secrets in
a passphrase-wrapped key file, unlocked at start and held in memory afterwards; a rebooted headless device
waits for the person before it joins the pool. Rejected: TPM-sealed with a 0600 fallback, and plain 0600,
because the file fallback protects nothing at rest. Consequence: slice 3 has no "headless: weaker" row, and a
device that cannot be unlocked by a person at boot is not an unattended pool member.

**Level 2 on Apple silicon is not attempted now; Level 1 only.** Decided: no attestation spike and no
hardened-helper work on Apple silicon; revisit if the owner buys confidential-computing hardware or a real
guest needs it. Rejected: an attest-only spike now, because it needs a paid Apple developer account and the
in-process inference engine is the decisive gap anyway (`docs/darkbloom-comparison.md`, section 9); the
full path, because it is months of work and the project would become a trust root.

**If Level 2 is ever built, the release authority a guest pins is the owner, with a published code-hash
allowlist.** Decided as a future direction only. Rejected: an offline separate release identity, and
reproducible builds, as more machinery than the owner wants to run for a path not being built.

**Constraint (2026-10-08): nothing on the plan depends on a paid Apple developer account until the product
takes off.** The owner has no such account and will not get one before then. Everything that needs one is
marked "blocked until the product takes off" in this note, `docs/home-pool.md`, `docs/service.md`,
`docs/darkbloom-comparison.md` and `HANDOFF.md`, with the unblocked alternative: Level 1 guest tenancy only,
keystore-wrapped or passphrase-wrapped keys on macOS instead of a Secure Enclave app, unsigned local Mac
builds on the owner's own devices, the phone as a PWA over HTTPS over Tailscale, the Android app sideloaded.

## 9. Verification this audit did not do

I did not run a packet capture; "TLS" rows come from reading `fleet/daemon.py:listen`, `fleet/tls.py`,
`fleet/discovery.py:_trusted` and `http.py`. I did not inspect the Tauri app's Rust beyond its tests, the
LLM harness hook-diagnostics sinks for guest leak paths, or whether the `openssl` binary fallback in
`tls.py` produces the same certificate profile. Slice 1 starts with a loopback capture test that proves
nothing but ciphertext crosses a non-loopback listener with a planted marker in the body and the stream.

## 10. Progress (2026-10-08)

Section 2 above is the audit of `c1007d3b`; where it describes the shared key, the TLS-off switch or TLS 1.2 it
is superseded by this section.

### The first slice (done in part)

- **Done.** `ML_STACK_FLEET_TLS` and `tls.disabled` are removed. `LimitedServer` refuses to be built on an
  address beyond this machine without a TLS context, so a plain listener on the LAN cannot exist; a plain
  connection from another address is dropped. Every pool context (`server_context`, `member_context`,
  `pinned_context`, the pairing and install clients) has `minimum_version = TLSv1_3`; the Android client
  offers 1.3 only. A beacon with no certificate is ignored unless it is on this machine. `Peer`,
  `RemoteWorkspace` and a project's `board_host` accept `https://`, or `http://` on this machine only; the
  join client refuses a plain-HTTP daemon off this machine. Tests: `tests/test_fleet_wire_floor.py` (a relay
  keeps every byte of a request with a planted marker: none of it is plain and the version is TLS 1.3; plain
  HTTP from a LAN address gets no answer; a client capped at TLS 1.2 is refused; the listener cannot be
  built without TLS; the old environment variable changes nothing; `http://` LAN hosts are refused).
- **Not done.** Sealing `/infer` streams and file chunks (framed AES-GCM) and the recovery-file KDF parameter
  review. Those are the rest of slice 1.

### The second slice (and the mutual-TLS half of 2b), done

- A device is its certificate (`fleet/tls.py`, now valid ten years so that its fingerprint is a stable
  identity; replacing it is a new pairing). A cluster's record of devices is `fleet/membership.py`
  (`Roster`: one file per cluster, named for the key's hash, 0600, locked, atomically written; a row whose
  fingerprint is not the hash of its certificate is dropped on load; `revoked` is sticky and wins every
  merge; a revoked certificate cannot be enrolled again). `fleet/pool_roster.py` (`Pool`) is the view over
  every cluster the machine is in.
- **Mutual TLS.** The daemon's TLS context (`tls.member_context`) asks every client for a certificate and
  completes the handshake only with an active member's; it is rebuilt whenever a record changes, so a revoked
  device fails its next handshake. A client context presents this machine's certificate (`tls.local`, or the
  daemon's own via `tls.present`).
- **Request authority.** `fleet/callers.py` is the per-request check: a TLS caller must show a current member's
  certificate (else 403, even on a connection opened before the revocation), and a request signed with a
  cluster's secret must come from a member of that cluster. A caller with no certificate (a phone, a browser)
  may use only a device's own enrolment secret on `/workspace/v1/*`, never a cluster secret. The cluster key
  remains the secret that signs and seals request bodies and seals beacons; it is no longer enough to be
  served. `GET /fleet/v1/self` says which device the daemon took a caller to be.
- **Everywhere a device is checked.** The TLS handshake, every request, discovery (`discover` pins and returns
  only beacons whose certificate is an active member and un-pins a revoked one; `enrolled=False` is for the
  person choosing whom to enrol), the membership exchange route, and the sync loop (it talks only to enrolled
  peers).
- **Spreading.** `POST /fleet/v1/members` merges a member's record and answers with the receiver's;
  `membership_sync` pushes at once on enrolment and revocation and every 20 s each daemon asks its peers.
- **Joining.** Pairing lists each side's certificate in the other's record (the asker's certificate is checked
  against the fingerprint the SPAKE2 exchange bound). The passphrase join binds the joiner's certificate as
  its SPAKE2 identity (it was the fixed string `joiner`), the automatic Development join and a computer
  invitation carry it too, and each joiner lists the machine that admitted it. A cluster joined from a
  recovery file, or before this change, has only itself in its record: `ml-stack-peers members adopt` lists the
  machines that answer for the cluster with their fingerprints for a person to compare, `members add` enrols
  one by certificate, `members revoke` puts one out (all three need a person at a terminal), and the daemon
  says at start which clusters are still alone.
- **Migration.** There is no automatic migration of trust: a daemon starting on an existing cluster enrols
  only itself, so its peers refuse it until they are paired again or adopted by a person. The owner is the
  only user.
- **Tests.** `tests/test_pool_membership.py`: real TLS sockets and the real daemon handler on temp roots; two
  devices with different keys told apart; an unpaired device refused; a revoked device refused at its next
  request on a live session and at its next handshake; a revoked device out of every check; a stale peer
  cannot revive one; a downgrade to TLS 1.2 refused; a wrong or one-byte-changed server pin refused; a
  tampered roster row dropped; a beacon that was never enrolled not pinned; a caller with no certificate cannot
  use the cluster secret; a member of another cluster cannot use this cluster's secret.

### Left

Slice 1's stream and chunk sealing; slice 3 (the cluster key and TLS key are still plaintext files, now
joined by the roster); the channel-bound MAC of 2b (the certificate is checked on the same TLS session as
the request, but the MAC does not yet cover its fingerprint); slices 4 to 8 and G1 to G5; rotating the
cluster key on revocation (a revoked device still holds it); a real multi-device run, which this work could not
do.
