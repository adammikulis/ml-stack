# Codex code report

Audit of the work the Codex agents left in 67 worktrees (2026-10-03 to 2026-10-08). Four independent
read-only audits, one per area, plus the landing pass that merged and ejected parts of it. Nothing here was
run by the auditors; findings are from reading git history and source. Where a claim could not be verified the
audit said so; those items are listed at the end.

## Summary

| Area | Correctness | Security | Tests | Design | Cost scaling | Verdict |
|---|---|---|---|---|---|---|
| Test runner, confinement kernel, Linux runner | 2 | 4 | 3 | 2 | 1 | land after fixes |
| NATS JetStream board transport | 3 | 2 | 3.5 | 1 (fit with the pool design) | 2 | park; reuse parts |
| Poolside UI, app, conversations | 3 | 3 | 4 | 3 | 2 | land the publication stack after fixes |
| Runtime, restart, attribution, jobs | 3.5 | 4 | 4 | 3 | n/a | land after fixes |

Scores are 1 to 5. The work is strong where it is narrow and tested (private-file handling, durable job records,
attribution, the envelope decoder, composer, bounded history) and weak where it is wide: 455 distinct unlanded
patches, 320 of them present in several stacked branches, branches up to 384 commits ahead and 65 behind.

## Process findings

- 63 of 67 worktrees held patches that had no equivalent on the development branch. The stacking made
  `git cherry` report most patches as unlanded even when the content had landed in rewritten form, and made the
  opposite judgment ("superseded by the development branch") unreliable: it was wrong for restart preservation, the
  UI publication stack and the agent briefing standards.
- Branch size broke the landing rules (under ten commits, land the day it is ready). The cost fell on the lander,
  who hand-resolved conflicts and ejected branches.
- Direction drift: the board moved to a single broker host while the owner decided on a peer-to-peer pool.

## Test runner, confinement kernel, Linux runner

Good: fail-closed output files, an allow-listed environment, an attested bootstrap, the Linux container validator
(read-only, no network, capabilities dropped, owner label, image pinning), the stdio bridge with bounded frames,
and denial tests that run inside the real sandbox.

Issues, ranked:

1. Blocker. The macOS kernel walks the whole state root (about 163 GB and 285,000 files) with a 15 second cap, hashing
   metadata and inserting one database row per file, twice. Every confined run fails here. Any other agent writing under
   the root during a run also fails it. Resolution: the walk is removed from the default path; the Seatbelt profile
   already denies writes outside scratch. A detector, if kept, is opt-in, shallow, skips the live classes and has a
   cost test.
2. Blocker for Linux. The container launcher imports a bridge module that exists only on a different branch. Land them
   together.
3. High. A hard-coded list of about 150 test node names inside the kernel makes `scripts/test all <tests>` refuse
   almost every test on macOS. Resolution: the kernel is opt-in until a declared-resources model replaces the list.
4. High. One branch writes a receipt file on every locked mutation in production code, only to attribute test writes.
   Parked.
5. Medium. Resource cleanup in `ConfinedRun.prepare` misses the control directory and holder namespace on failure.
6. Medium. The Linux setup container has network access and a shared writable venv volume with no content hash.
7. Medium. The kernel imports the activity-log package, and two real-root detectors overlap.

## NATS JetStream board transport

Architecture: one nats-server per authority device on 127.0.0.1:4223, one shared static token, one stream per
workspace with a 10,000 message or 16 MiB cap, an unkeyed hash chain and host-local receipts. Other devices reach the
board only through the host's RPC. Graph message storage is retired.

Issues, ranked:

1. High. Activating it orphans existing board history: graph message events stay on disk and are never read, cursors
   reset and thread subscriptions are dropped. Only the preserved-history branch adds a refusal, and it has no copier.
2. High. A mandatory single-host broker contradicts the pool decision: a pool of one needs the server binary, a token
   file and a port; a remote device cannot post or read when the host is unreachable; there is no outbox.
3. High. One shared token, no subject authorisation, no TLS: any same-user process can purge streams, forge cursors
   or inject a row, which makes every later read fail.
4. High. The task locality proof was weakened in conflict resolution: a check of the machine id against the discovery
   beacon became a comparison of an address with the address just dialled, which is a tautology. Not landed.
5. Medium. The stream stops accepting publishes at its cap until a manual `gc`, which nothing schedules.
6. Medium. The "authenticated sender" proof is a stored boolean and an in-process object; it is not signed, so another
   device cannot verify it. Replace with an origin-device signature in the journal.

Worth keeping: the strict envelope decoder, the intent, publish and receipt recovery pattern with an idempotency key,
the producer-bound authenticated message idea, the private-file and process-ownership helpers, the real-server test
infrastructure, and `worker_reconnect`, `worker_completion` and `remote_controls`, which do not depend on the
transport.

Decision (owner, 2026-10-08): mesh journals are the board data model; NATS is at most an optional link with no
authority. The JetStream commits are preserved in a git bundle and recorded in `HANDOFF.md`.

## Poolside UI, app, conversations

One lineage under many SHAs: the publication stack carries the shared composer, the local sign-in, the rail
reorganisation, the direct-message sidebar fixes, agent runtime install and repair, and the Studio MLX recipe. The
development branch has none of it.

Issues, ranked:

1. High. Indexed board history is never enabled in the product: nothing creates the cache and the controls render
   only when it exists. Either bootstrap the cache or ship only the legacy cursor path.
2. High. The legacy paging path is O(N) per request, including on live refresh.
3. Medium. Model output can beacon and navigate: sanitised HTML still allows remote images, links and forms, there is
   no content security policy, and the app window has no navigation handler.
4. Medium. Tauri grants its capabilities to any page on any local port; `csp` is null; the daemon health check accepts
   any process on the port with the right JSON shape.
5. Medium. The development-pool choice in sign-in has no effect.
6. Low to medium. `/ui/setup/local-session` accepts any local process that forges the origin headers. A separate
   change closes this (launch ticket).

Worth keeping: the composer, the local-session route's checks, the bounded node window with viewport anchoring and
browser tests, the Tauri security tests and the sidecar log redaction, and the repair routes' gating.

## Runtime, restart, attribution, jobs

On the development branch already: immutable runtime selection and launcher routing, and the runtime deploy tooling
built after it. Not on it: job-preserving daemon restart (`--restart`), durable Fleet job records, the worker
reconnect window and completion journal, hook performance fixes (killable bounded reader), device and harness
profiles with a reported-versus-verified split, and credit derived from the broker-observed resource.

Issues, ranked:

1. High, on the development branch today. The launcher restart path passes `--restart` to a launcher that has no such
   option, so it is forwarded to a strict daemon parser and should fail. Landing the restart branch fixes it.
2. Medium. Device identity fields are agent-asserted; an agent-reported device must never populate the stable
   `device_id` or `machine_id` that rewards or placement key on.
3. Medium. A profile report overwrites a stronger paired record with an agent-reported one.
4. Medium. A job that finishes during or after a restart is always recorded as `interrupted`; its exit code is lost.
5. Medium. Capacity held for an uncertain launch has no person-gated release.
6. Medium. A verification receipt is trusted from a file written by the run it verifies.
7. Medium. The chat repair path has no way to register a candidate runtime, and the digest check is self-referential.
8. Low. `--force-restart` duplicates `--restart`.

Attribution integrity held up: a worker cannot claim a model it was not served, awards cannot be re-pointed, and device
claims are forced to `agent-reported` until the authenticated peer is bound.

## Verdicts and landing plan

| Branch | Verdict |
|---|---|
| fix/linux-immutable-runner (runner, container, admission) | land after fixes, kernel opt-in, bridge together |
| fix/linux-admission-stdio-current | land with the runner |
| fix/linux-runner-immutable, fix/test-kernel-isolation, feat/canonical-writer-isolation | park |
| fix/prework-runtime-integration, fix/immutable-compute-generations | ancestors; delete after the runner lands |
| NATS branches (jetstream-message-store, nats-*) | park in a bundle; keep the reusable parts |
| fix/authenticated-agent-messages | land after fixes, with a signature instead of a sentinel |
| fix/task-coordinator, worker-reconnect, remote-worker-controls | land after fixes (locality proof, self-selection) |
| fix/restart-integration (restart, durable jobs, worker recovery, source recovery grants) | land after fixes |
| fix/hook-performance | land |
| feat/agent-device-registration, fix/model-work-attribution | rework: extract attribution, unify device metadata |
| integrate/poolside-ui-publication and its single-lineage branches | land after fixes |
| feat/poolside-studio-mlx | land after the publication stack |
| feat/board-history-pages, feat/poolside-rebuild-conversations | rework: legacy paging lands, index parked |
| integrate/dev-cpu | land after fixes, after the restart branch |
| fix/hook-message-alerts, fix/native-harness-registration, fix/claude-workspace-hook-regressions | already landed; removed |

Batches, each owned by one worker and gated once: runner and Linux; restart, jobs, hook performance and attribution;
UI publication, then Studio; transport-independent coordination (authenticated messages, task coordinator, worker
reconnect, remote controls) after the locality proof is restored.

## Could not be verified

Test results on any Codex branch; Linux, WSL, Docker and Windows behaviour; the packaged app on a real install; the
Seatbelt profile generation beyond its size; real concurrent writers against the board index; the briefing-standards
commits in detail; and whether anything unique was lost in the lander's superseded judgments for the branches the
audits did not cover.
