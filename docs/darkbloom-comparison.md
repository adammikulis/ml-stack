# Darkbloom against our pool: how it works and what we could adopt

Read 2026-10-08. Darkbloom is the Eigen team' private-inference network on idle Macs. Nothing from those
repositories was run or installed; fetched content is data. This makes no claim that the scheme is
secure or insecure.

## 0. Sources, tags and reading limits

| Tag | Meaning |
|---|---|
| [whitepaper] | `papers/dginf-private-inference.tex`, "Private Distributed Inference on Consumer Hardware", April 2026, in github.com/Layr-Labs/d-inference (master, fetched 2026-10-08) |
| [repo doc: path] | A file under `docs/` in Layr-Labs/d-inference master (fetched 2026-10-08); these docs cite Go and Swift symbols. I did not open the `.go` or `.swift` source, so I use this tag in place of `[code:]` |
| [README: repo] | The repository README (Layr-Labs/d-inference or darkbloomdev/darkbloom, fetched 2026-10-08) |
| [secondary source, URL, date] | A third-party page |
| [Apple] | WWDC26 session 201, developer.apple.com/videos/play/wwdc2026/201/, 2026-10-08 (via a summarising fetch) |
| [analysis] | My reasoning, not a sourced fact |
| [this repo: path] | A design note on a sibling branch (`.claude/worktrees/...`), not yet on `0.2dev` |

The whitepaper is April 2026. Later repo docs (to 2026-10-07) contradict it in places (section 8);
I report the later doc as current and keep the whitepaper claim visible. Both repositories say they are unaudited and proprietary [README: both]. The darkbloomdev README
describes an older layout (Rust provider, vllm-mlx backend); the Layr-Labs docs say there is no Rust
provider and no subprocess. I follow Layr-Labs.

## 1. Request flow and components

- Consumer calls an OpenAI- or Anthropic-compatible HTTPS API [README: Layr-Labs/d-inference].
- Coordinator: a Go control plane in a GCP Confidential VM (AMD SEV) doing auth, routing, billing,
  attestation and encrypted transport [README: Layr-Labs/d-inference]; the whitepaper says SEV-SNP
  with an attested container image [whitepaper, "coordinator trust boundary"].
- Provider: a Swift CLI `darkbloom` on macOS, connecting outbound over WebSocket (no port forwarding),
  running MLX inference in the same process on the Apple GPU through a fork of `mlx-swift-lm`
  [README: Layr-Labs/d-inference]; [whitepaper, "In-Process Inference"]. No subprocess, local server or
  IPC; the stated reason is that localhost TCP can be captured even with SIP on, and a subprocess
  binary can be swapped [whitepaper].
- Also: MicroMDM plus SCEP enrollment, APNs for the code-identity push, an AppKit host (APNs needs a
  GUI session) [whitepaper]; [repo doc: attestation.md].
- Path of one request [repo doc: docs/architecture/security/encryption.md]:
  1. Consumer sends the body over TLS, optionally sealed to the coordinator's X25519 key.
  2. Coordinator opens it in memory (routing, cache affinity, billing), strips caller `user`,
     `metadata`, `safety_identifier`, `prompt_cache_key`, picks a provider by a cost function.
  3. Coordinator seals the body with a fresh ephemeral X25519 key and random nonce to the provider's
     attested X25519 key and sends it down the WebSocket.
  4. Provider decrypts, runs MLX, seals each response chunk back to the request's ephemeral key.
  5. Coordinator decrypts chunks and relays them (re-sealed if the consumer sealed).
- Scheduling: lowest estimated completion cost (queue, backlog, health) among providers passing
  trust, runtime-manifest, challenge-freshness and private-text gates [whitepaper, "routing gates"].

## 2. The attestation chain

### 2.1 Layers, and what each proves

Five layers in the whitepaper [whitepaper, "Five-Layer Attestation Architecture"]:

| Layer | Apple mechanism | What is attested | Limit |
|---|---|---|---|
| 1 secure enclave blob | CryptoKit `SecureEnclave.P256` key, non-exportable | A P-256 signature over a JSON blob of self-reported fields: SIP, secure boot, ARV, chip, model, OS, serial, `binaryHash`, X25519 key, timestamp | Self-signed; a software key signs identically, and the SIP value comes from `csrutil status` [whitepaper]; [repo doc: attestation.md] |
| 2 MDM SecurityInfo | MDM enrollment (MicroMDM), `SecurityInfo` command | OS-reported SIP, SecureBootLevel "full", AuthRootVolume, FileVault, recovery lock | Granted `hardware` trust only when it matches the blob; MDM access rights limited to bits 0, 4, 10 (inspect, device info, security info) [whitepaper] |
| 3 MDA over MDM | Managed Device Attestation via `DeviceInformation` / `DevicePropertiesAttestation` | Apple-signed chain to the Enterprise Attestation Root (P-384): serial, UDID, OS, SepOS and LLB versions, SIP, secure boot, kext status; `DeviceAttestationNonce` = SHA-256 of the SE public key, echoed as `FreshnessCode`, binding that SE key to the device | Needs MDM enrollment; Apple rate-limits fresh attestations to about one per device per 7 days; chain is attached only after the `hardware` grant [whitepaper]; [repo doc: attestation.md] |
| 4 Challenge-response | none (Darkbloom protocol) | Every 5 min a 32-byte nonce: SE signature over nonce and timestamp plus SIP, secure boot, RDMA status; 30 s reply window; 3 hard failures untrust; SIP or secure boot false untrusts at once | The posture bits are still self-reported, but status fields are now SE-signed [repo doc: attestation.md] |
| 5 APNs code identity | Apple's push service (APNs) | Coordinator pushes a nonce sealed to the provider's X25519 key `K` to a device token for topic `io.darkbloom.provider`; only a binary carrying that app id, the team's Developer ID signature and a provisioning profile with `aps-environment` can obtain the token; the provider returns the nonce signed by its SE key | Proves team and app id, not exact binary version; needs a logged-in GUI session; best-effort, throttled delivery [whitepaper]; [repo doc: attestation.md] |

Trust levels exposed to consumers: `none`, `self_signed`, `hardware` (MDM SecurityInfo agreement),
with `mda_verified` and `code_attested` as flags beside the level [repo doc: attestation.md]. Private
text routing also needs an X25519 key, encrypted response chunks, runtime manifest check and (once an
enforcement date passes) code attestation [repo doc: encryption.md, "Invariants"].

### 2.2 Verification and refresh

- Coordinator verifies the ECDSA signature, requires SIP, secure boot and SE availability true,
  checks `encryptionPublicKey` equals the registered key and the timestamp within about 2 minutes
  [repo doc: attestation.md, "Layer 1"].
- MDM SecurityInfo runs on a durable scheduler with retries (2 to 4 min, then 6 to 12, then 15 to 30);
  a mismatch is terminal untrust [repo doc: attestation.md, "Layer 3"].
- Reconnect caps `hardware` back to `self_signed`; evidence is reused only after a fresh signed
  challenge, within 5 min of the last proof or a 90 s offline gap (chosen because a recovery-OS round
  trip to flip SIP takes longer) [repo doc: attestation.md]; the docs call that timing premise a
  "physical qualification assumption" [repo doc: reports/2026-09-27-hybrid-provider-trust-review.md].
- APNs proofs are reusable for 30 min by resume challenge; a runtime manifest of accepted metallib
  hashes deroutes a mismatching provider without untrusting it [repo doc: attestation.md].

### 2.3 What the per-response signature proves

The whitepaper's tier table lists "Response integrity (signed): SE signature" [whitepaper, "Trust
Property Summary"]. The consumer privacy page says the opposite: "There is no per-response signature
or receipt: the `X-Provider-*` headers are the coordinator's assertion over TLS, not a provider-signed
proof" [repo doc: consumer/privacy-expectations.md, 2026-10-06]. The threat model lists "no consumer-side
output proof (output signing not implemented)" as open [repo doc: threat-model.yaml, T-007]. Reading
the later docs, no signature over a response exists. What the 5-minute SE signature proves is
possession of the registered key at that time, not that a given response came from a given model.
[analysis]

## 3. The encryption path

- Primitive on all hops: NaCl `box` (X25519 and XSalsa20-Poly1305), 24-byte random nonces
  [repo doc: encryption.md]; [whitepaper, "End-to-End Encryption"].
- Hop 1, consumer to coordinator: TLS; sealing to the coordinator key is optional, plaintext is
  also accepted [repo doc: encryption.md].
- Hop 2, coordinator to provider: mandatory, fresh ephemeral coordinator key per request. The
  provider's key `K` must equal the `encryptionPublicKey` in its SE-signed blob [repo doc: encryption.md].
- Hop 3, provider to coordinator: each chunk sealed from the provider's static key `K` to the request's
  ephemeral public key; a plaintext or wrong-key chunk untrusts the provider and fails the request [repo doc:
  encryption.md].
- **The coordinator sees plaintext.** The docs say so repeatedly: "The coordinator is therefore a
  trusted plaintext intermediary for routing, not a blind relay" [whitepaper]; "'The coordinator never
  sees plaintext' is false" [repo doc: consumer/privacy-expectations.md]; the launch post says the
  coordinator "is still part of the trusted routing layer" [secondary source,
  https://www.eigenlabs.org/blog/project-darkbloom-unlocking-idle-compute-for-ai/, 2026-04-15].
- What protects the coordinator: it runs in an AMD SEV(-SNP) Confidential VM with an attested
  container image, so the cloud operator is meant to be unable to read its memory [whitepaper];
  prompts are not logged or stored, metadata only [repo doc: encryption.md, "What the coordinator logs"].
  The coordinator key comes from an environment mnemonic, a listed exfiltration risk (T-entry
  "Coordinator encryption mnemonic exfiltration from environment") [repo doc: threat-model.yaml].
  [analysis] The consumer's trust therefore rests on the Eigen team' build and operation of that VM plus
  the AMD attestation; the docs I read do not describe a consumer-side check of the coordinator's
  attestation report, only that "consumers can verify" [whitepaper] (not shown how).
- Forward secrecy: the whitepaper says each request has a fresh coordinator ephemeral key, so
  compromise of one ephemeral key does not open other requests [whitepaper]. The later doc says
  fresh sender keys do not give forward secrecy against recipient-key compromise: `K` lasts for the
  process lifetime, and `K` plus the transmitted ephemeral public keys decrypts recorded requests from
  that lifetime [repo doc: encryption.md, 2026-10-06]. I take the later doc as correct.
- Streaming: each SSE chunk is sealed separately; token timing stays observable [repo doc:
  encryption.md]; [whitepaper, "Limitations"].
- Provider zeroing: the whitepaper says buffers are `memset_s` after each request. The encryption doc says the
  zeroing helpers "are not invoked by the inference path" and GPU buffers are not zeroed [repo doc:
  encryption.md]; [repo doc: threat-model.yaml, T-028].

## 4. The hardened runtime, and what a root owner can still do

Claimed controls [whitepaper, "access path elimination"]; [repo doc: threat-model.yaml, the memory-inspection and SIP entries]:
`PT_DENY_ATTACH` at startup; Hardened Runtime without `get-task-allow` so `task_for_pid` and
`mach_vm_read` fail; SIP on (checked at start and every 5 min); Kernel Integrity Protection; IOMMU
(DART) default-deny for DMA; Developer ID signing plus notarization; a no-interpreter Swift binary.
Argument for SIP immutability: SIP can only change from Recovery, a reboot ends every process, so a
SIP check at start holds for the process lifetime (Theorem 1, under "no unpatched kernel bug")
[whitepaper].

My assessment of a root owner, per path. [analysis]; not tested by me.

| Owner capability | Does the stated control hold? | Notes |
|---|---|---|
| Attach lldb/dtrace, `task_for_pid`, `mach_vm_read` | Holds under the assumptions: SIP on, hardened runtime, no `get-task-allow`, kernel without a bypass | Apple-platform behaviour; depends on the exact entitlements of the shipped binary, which I did not read (`provider-swift/entitlements.plist` is cited, not opened) |
| Load a kext / modify kernel | Holds while "full security" and SIP are on | Changing boot policy needs a reboot into recovery with owner auth, ending the process; a bug in the OS is outside the model |
| `DYLD_*` injection, library injection | Hardened Runtime normally ignores `DYLD_*` and enforces library validation unless entitlements disable them | Audit item: confirm neither `allow-dyld-environment-variables` nor `disable-library-validation` is present |
| Modified binary | The claim "SIP prevents execution of modified signed binaries" [whitepaper] is not how macOS works: a root user can run a re-signed or ad-hoc-signed copy | What holds is that the copy cannot obtain the genuine team's APNs token or app attest identity (team, bundle ID, code hash). So the binary-identity proof carries the weight, and SIP does not |
| Swap, hibernation, core dumps | Unclear | Hardened process denies `gcore` via task port; swap is encrypted on Apple silicon; neither is evidence that no plaintext leaves RAM. Needs an experiment |
| GPU (unified) memory | Same boundary as process memory in principle | Residual KV and prompt data are not zeroed between requests (T-028, open); behaviour of other root processes against Metal buffers untested |
| Run genuine binary inside a VM or under a hypervisor the owner controls | Not addressed in the docs I read | Hypervisor isolation was "never implemented"; the `hypervisorActive` field is retired [repo doc: threat-model.yaml, T-028] |
| Thunderbolt RDMA, DMA, physical probing | DART default-deny; RDMA needs Recovery to enable and is reported, not enforced; probing is out of scope (soldered LPDDR, as in Apple Private Cloud Compute, but here the owner has custody) | [whitepaper] |
| Kernel or SEP zero-day; side channels | Explicit assumptions 1 and 2; side channels and token timing are named, not defended | [whitepaper] |

Net: the argument is that no software observation path remains under listed assumptions; there is no
memory encryption or hardware isolation of the workload [whitepaper, comparison table]. [analysis]

## 5. Provider verification, pricing, fraud and reputation

- **Was the work done?** Weight-hash advertisement at registration and in heartbeats, with a
  challenge on model hashes; the check is fail-open when the hash is omitted (finding SEC-007, open)
  and there is no consumer-side output proof [repo doc: threat-model.yaml, T-007]. Provider-reported
  token usage and cache counts feed settlement; a malformed cache report is cleared so it cannot lower
  the bill [repo doc: billing.md]. I did not find redundant execution, canary jobs or spot checks in
  the docs I read, and did not search the source for them. [analysis]
- **Identity and untrust.** Posture failures, challenge failures, hash drift and plaintext chunks
  untrust a provider; a hard untrust writes a durable tombstone that wins races with a grant; admin can
  revoke an app attest key (propagates within 30 s) or withdraw a build [repo doc: attestation.md];
  [repo doc: reference/provider-authorization.md].
- **Reputation.** No score; health enters the routing cost and fault ejection is keyed by serial, SE
  key, then account [whitepaper]; [repo doc: identity-binding.md].
- **Pricing and payment.** Consumers prepay; platform prices per model in micro-USD, a provider may set
  a custom price; a platform fee is credited to a `platform` account with a per-user override;
  providers withdraw through Stripe Connect (legacy) or Global Payouts [repo doc: billing.md]. The launch
  post says providers keep 95% and some models are priced at about half of aggregators [secondary
  source, https://www.eigenlabs.org/blog/project-darkbloom-unlocking-idle-compute-for-ai/, 2026-04-15];
  one search summary says 0% fee during alpha, which I could not confirm [secondary source,
  https://wavect.io/blog/darkbloom-ai-private-inference-mac/, 2026-08-25, search snippet only]. Base
  rewards (a floor per epoch) now need macOS 27 and a current app attest authorization; they dedupe
  on a single machine id, which "do[es] not prove physical uniqueness" across reinstalls or new
  accounts [repo doc: reference/provider-authorization.md].

## 6. Openly stated limits and independent review

Stated by the project:
- Unaudited public alpha [README: Layr-Labs/d-inference]; coordinator and provider are plaintext
  parties, the provider because in-process native-speed inference requires it [repo doc: privacy-expectations.md].
- Kernel zero-day, timing channels, APNs needing a GUI session, hypervisor/RDMA unenforced, MDA needing
  an enrolling organisation [whitepaper, "Limitations"].
- Open threat-model items: no output proof, SEC-007, GPU residue, "inference/result integrity" and
  host-compromise resistance "need independent adversarial qualification; neither path proves them"
  [repo doc: threat-model.yaml, T-056]; platform security and SIP/boot transition behaviour "need
  adversarial evidence on supported hardware" [repo doc: provider-trust.md].
- A "patient insider on own hardware" class is named an irreducible residual of the APNs challenge
  [repo doc: threat-model.yaml, "R-snoop" entry].

Independent review: I found no third-party audit. A vendor-evaluation page by an integrator calls it
"hop-by-hop encrypted private inference", "stronger than an ordinary unmanaged peer, but not local-only
or zero-knowledge", cites no audit and relies on the project's documentation [secondary source,
https://wavect.io/blog/darkbloom-ai-private-inference-mac/, 2026-08-25; the page's structured data
shows a conflicting date of 2026-10-08].

What an independent audit would need to check (none of this is established by the documents):
1. The shipped binary's entitlements, library validation, and that the notarized build matches a
   reproducible source build.
2. Real behaviour of `task_for_pid`, `gcore`, swap, hibernation and Metal buffers for a root owner on
   current macOS, including a root process created before the provider starts.
3. Whether the provider can be run in a VM or under a debugger-capable hypervisor and still pass
   every attestation.
4. Does an app attest key or MDM state survive a SIP or Full Security downgrade and restore
   (Darkbloom lists this as still to qualify) [repo doc: design/app attest-migration.md].
5. The coordinator VM: attestation report verification by consumers, who can change the image, how
   the mnemonic is held, log and crash-dump handling.
6. Whether a response can be attributed to the attested process (no signature exists today).
7. Cross-tenant KV and prefix-cache leakage, which the project lists as a threat (cache sharing and a
   TTFT timing oracle) [repo doc: threat-model.yaml].

## 7. The macOS 27 change: MDM is being replaced by app attest

- Darkbloom's setup now skips the MDM profile on macOS 27 or later and waits for coordinator approval
  of app attest; older macOS keeps the legacy path and a notice of "upcoming" MDM deactivation with no
  date; `darkbloom unenroll` offers an app attest migration only when a fresh, unexpired
  "removal-ready" authorization exists [repo doc: reference/provider-authorization.md, 2026-10-04].
- Policy: a connection may satisfy legacy MDM/APNs verification or a qualified app attest
  authorization; neither sets the other's flags. A frozen cohort of existing machines stays on the
  legacy path; new identities need app attest [repo doc: provider-trust.md, 2026-10-06].
- Apple: app attest is supported on macOS 27 and higher "which was previously not supported", may not
  be available to all app types, available to Action and SSO extensions but not other extensions, and
  clients should gate on `isSupported`. On macOS "each generated key" gets a policy "that requires full
  security mode and System Integrity Protection"; the leaf certificate carries a key access-control
  blob (the "ACL Blob OID") describing the conditions the secure enclave enforced; the attestation is
  derived from a boot-time snapshot of hardware properties; the relying-party id is team id plus
  bundle id; assertions are local, do not call Apple, and carry a strictly increasing counter; a
  receipt gives a fraud metric (approximate count of attested keys for this app on this device over 30
  days) to be used as an investigation signal [Apple]. The session does not cover entitlements,
  signing or notarization, code-directory hashes or app types [Apple].
- Darkbloom's reading: the Apple chain, nonce, relying party, environment and mac acl are verified;
  the assertion carries a CodeDirectory measurement (Developer ID category 6, type-2 hash) when the
  provisioning profile grants a "CDhash opt-in"; builds are qualified by exact binary and CodeDirectory
  hash through an admin API; assertions are fresh for 15 min [repo doc: reference/provider-authorization.md];
  [repo doc: reports/2026-09-14-app attest-macos27-validation.md].
- Physical test, 2026-09-14, macOS 27.0 on an M5 Max: `isSupported` true in a LaunchServices-launched
  app; attestation accepted by the production-root verifier; an SSH-launched executable could not
  read its Keychain record (`keychain_error`); the captured attestation had no bundle-version or
  validation-category extension [repo doc: reports/2026-09-14-app attest-macos27-validation.md].
- Reported problems: early reports of `isSupported` false on a macOS 27.0 beta (from the lead's brief;
  I did not verify); issue #1161 on Layr-Labs/d-inference (closed 2026-09-21) reports an M5 Max on
  macOS 27.2.0 with app attest serving authorization still "pending" and no cause established in the
  report [repo doc: github.com/Layr-Labs/d-inference/issues/1161, fetched 2026-10-08]. I did not read
  how it was closed.

What MDM/MDA proves that app attest does not [analysis from the sources above]:

| Property | MDM + MDA | app attest on macOS 27 |
|---|---|---|
| Real Apple hardware, Apple-signed | Yes (leaf OIDs) | Yes (SE-held key, Apple chain) |
| Serial and UDID | Yes, Apple-signed | No; "the documented app attest payload does not supply an immutable serial" [repo doc: design/app attest-migration.md] |
| SIP / secure boot / kext state | Apple-signed OIDs at issue, plus OS-reported SecurityInfo on demand | One ACL requiring full security and SIP, enforced by the SE when the key is used [Apple]; no kext or version fields |
| OS, SepOS, LLB versions | Apple-signed OIDs | Not certified; Darkbloom's OS claim is app-reported and authenticated by the assertion and qualified executable, "Apple does not independently certify it" [repo doc: provider-authorization.md] |
| RAM, chip, GPU count | Not provided either | App-reported only |
| Which code runs | Not provided; APNs added team and app id | Team id, bundle id, and a code-directory measurement when the profile opts in |
| Continuous posture | Re-query anytime, but MDA is rate-limited | Fresh local assertion every few minutes; Apple receipt renewal separate |
| Needs an operator | Yes: MDM server, Apple push certificate, enrollment of the device | No MDM; needs a developer account, profile and a server verifier |

How Darkbloom closes or leaves the gap: code identity moves from APNs to the Apple-measured
CodeDirectory hash plus an exact-build allowlist; posture moves from MDM SecurityInfo to the key's
ACL; the endpoint is bound by putting `K`, the session and the account into the assertion's
`clientHash`; hardware facts (chip, memory, SIP value, binary hash) go into that same transcript as
app-origin claims, "not independent Apple hardware measurements"; dedup uses the single machine
and the fraud receipt. Left open: serial and RAM assurance (so reward tiers cannot rest on them),
SIP-downgrade-and-restore behaviour (to be qualified), and everything that attestation never covered
[repo doc: identity-binding.md]; [repo doc: design/app attest-migration.md].

## 8. Where the whitepaper and the later docs disagree

| Topic | Whitepaper (April 2026) | Later doc |
|---|---|---|
| app attest on macOS | `isSupported` false, so use APNs | Used on macOS 27+; MDM being retired |
| Forward secrecy | Per-request ephemeral key gives it | None against provider-key compromise (`K` lasts the process) |
| Response signature | SE signature on responses | No per-response signature |
| Buffer zeroing | `memset_s` after each request | Helpers not called on the inference path; GPU not zeroed |
| Hypervisor isolation | Validated experimentally, planned | Never implemented; field retired |
| `binaryHash` | Telemetry only | Enforced only when a hash policy is configured |

## 9. Comparison with this repository

Our side, from sibling-branch notes [this repo: docs/pool-encryption.md, docs/service.md,
docs/home-pool.md]: the pool is a mesh of the owner's own devices with pinned per-device TLS and a
shared cluster key; guest tenancy Level 1 ("no casual visibility") is buildable now; Level 2
(resistance to a determined owner) "is not buildable in this pool today" and its Apple-silicon entry
says secure enclave and device attestation prove a device and app to Apple's service, "not a sandboxed
job to a third party" [this repo: docs/pool-encryption.md, section 7.4]. The service note limits
untrusted-provider work to non-sensitive jobs without attestation and keeps any operator out of the
data path [this repo: docs/service.md, sections 3.6 and 4.1]. Models run in a separate `llama-server`
process that holds plaintext prompts and KV cache [this repo: docs/pool-encryption.md, section 7.3].

| Darkbloom mechanism | Our counterpart | Fit |
|---|---|---|
| Central coordinator: routing, billing, attestation verification | None by design; mesh, no host | Differs. Each guest must verify the host itself; no one trusted third party |
| Coordinator in a Confidential VM | n/a | Out of reach for a mesh; also removes the plaintext intermediary, which is an advantage |
| sealed-box (X25519 plus XSalsa20-Poly1305) per request to an attested key | G3: HPKE to an ephemeral "sandbox key" made inside the job, returned only to the guest | Same idea, better: the key is born in the sandbox and the guest encrypts directly, with no re-encryption hop |
| SE-held device identity key | Slice 8 hardware-backed device key (no vetted Python wrapper) | Adoptable; needs a native helper |
| MDM, MDA | None | Needs an operator running MDM and an Apple push certificate; contradicts the no-host design. Out of reach |
| APNs code identity | None | Needs a team-signed app, entitlement and a coordinator that sends pushes. Superseded by app attest on 27+ |
| app attest + CDhash qualification | None | Possible; see below |
| In-process hardened engine | Separate `llama-server` | The decisive gap: attesting a helper while the model runs in an unattested process proves nothing about the prompt |

### What we could adopt for Level 2 on the owner's Macs, with sizes

Sizes as in the pool-encryption note: S under a day, M one to three days, L a week. [analysis]

| Step | What it needs | Size |
|---|---|---|
| A. Attestation spike on one Mac | Developer Program membership, an app id with the app attest capability, a Developer ID-signed helper app with a profile, launched through LaunchServices in a GUI session (SSH launch failed in Darkbloom's test); a Swift helper calling generate, attest, assert; a Python verifier checking the chain to Apple's app attest root, nonce, relying-party hash, ACL blob, counter (`cryptography` is already a dependency; CBOR decoding would be a new small dependency or a bounded hand parser) | M |
| B. Pool binding | Guest challenge; host helper signs a transcript holding the host's pinned device-certificate fingerprint and the job's sandbox public key (G3); guest verifies before sending; refresh every few minutes | M |
| C. Release authority and allowlist | A project-owned team id; notarized helper per release; published list of accepted code-directory hashes with revocation | M, ongoing |
| D. In-process engine | Embed inference (MLX or llama.cpp) in the signed hardened helper, no get-task-allow, `PT_DENY_ATTACH`, no loopback server, no shared KV with other tenants | L or more; breaks the current patched `llama-server` model leases |

What it would protect, if all of A to D were done and the audit items in section 6 pass: a guest
could verify, without an operator, that its job key is held by a specific signed app on genuine Apple
hardware with SIP and Full Security enforced by the SE when it signed, and that the app's code hash is
one the project published. It would not protect against a kernel or SEP bug, physical probing,
side channels, a remaining plaintext path outside the helper (today's `llama-server`), a malicious
guest job against the owner (the sandbox is still needed), or metadata. [analysis]

Two structural points. First, the guest must pin a team id the owner does not control. If the owner
signs the helper with the owner's own team id, the attestation says only "some app by the owner"; so
Level 2 makes the project (or another release authority) a trust root, in place of Darkbloom's single
coordinator operator. Second, a mesh gives Apple-chain verification to each guest, but the hardest
Darkbloom component, the central policy and revocation service, would have to be rebuilt as
signed, replicated revocation records (slice 7 of the pool-encryption plan). [analysis]

Out of reach as a mesh: MDM/MDA, the Confidential-VM coordinator, serial or RAM certification,
hypervisor isolation, and any guarantee against physical or kernel-level attack.

## 10. Decisions for the owner

Decided 2026-10-08:

- **Level 2 on Apple silicon is not attempted now; Level 1 only.** Revisit if the owner buys
  confidential-computing hardware or a real guest needs it. Rejected: an attest-only spike now (step A),
  because it needs a paid Apple developer account and the in-process engine (step D) is the decisive gap
  anyway; the full path (A to D), because it is months of work and the project becomes a trust root.
- **If Level 2 is ever built, the release authority a guest pins is the owner, with a published code-hash
  allowlist.** Rejected: an offline separate release identity; reproducible builds.
- **Constraint:** the owner has no Apple developer account and will not get one unless the product takes
  off. Steps A and C (Developer Program membership, Developer ID-signed and notarized helper, app attest,
  APNs) are **blocked until the product takes off**; the unblocked alternative is Level 1 only.

Deferred until Level 2 is revisited (all three depend on it): what runs the guest's model (keep
`llama-server` for Level 1 and make no Level 2 claim for inference; the in-process engine, step D, is the
Level 2 option); what we may tell a guest (Level 1 wording only, no "owner cannot see prompts" claim); whether
we ever add an operator (MDM, push, coordinator; the mesh stays host-free meanwhile).
