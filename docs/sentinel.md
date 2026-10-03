# Sentinel: detecting and containing attacks on a node and its fleet

Status: implemented in `src/ml_stack/sentinel/`. The first half of this file is the design
and threat model; "What is built and tested" and "Results" say what exists and what was
measured. Where the two differ, the last two sections are the truth.

Sentinel raises the cost of an attack and limits how far one reaches. It is not a guarantee
and it does not claim to find a model that was trained to misbehave.

## What it is made of

One event stream, a set of detectors that write to it, one quarantine store with one state
machine for every kind of subject, and a policy that decides which events act. The core uses
the standard library only. Other modules reach it through a handful of small adapters
(`sentinel/adapters.py`) so that sentinel imports none of them.

Designed against these branches (their public interfaces, no merge into this branch):

| Branch | Used for |
|---|---|
| `agent/guardrails` | the `ml_stack.guard` logger and `Guard.events` (a deny or a tainted modify is a security event); `ToolCall` and `Rail` shapes |
| `agent/hardening` | `macauth.Authenticator.check` (auth failure, replay, lockout), `httpguard` refusals, the `credentials` redaction rules (sentinel carries its own copy of the patterns), orphan sweep |
| `fix/serve-admission-control` | the Broker (canaries go through it, serially), its unmanaged-server detection |
| `agent/decide`, `agent/port-pcbe` | tool-loop and MCP surfaces that sentinel must keep away from its own verbs |
| `agent/redteam` | the attack suite whose cases the sentinel red-team tests extend |

This branch merges none of them. The adapters are tested against the real `Authenticator`,
the real `Guard` and the real process finder on a throwaway integration branch
(`agent/sentinel-integ`: hardening and guardrails merged, plus
`tests/test_sentinel_integration.py`). The admission-control branch conflicts with hardening
in `http.py` and `serve/manager.py`, so the Broker itself was not part of that merge.

## Threat model

Four places an attack lands. For each: what can be detected reliably, what only
heuristically, and what is out of scope. "Reliable" means a deterministic check with no
false positives on an unmodified install. "Heuristic" means a threshold that trades missed
attacks against false alarms; the false-alarm cost is stated.

### a. Hostile input to the agent loop

Prompt injection in tool results, web pages, documents, model output; tool abuse; secret
exfiltration; resource exhaustion (huge tool results, runaway loops).

| Signal | Class | False-positive cost |
|---|---|---|
| a guard rail denies or marks a tool call or text as tainted | reliable that the rail fired, heuristic that it was an attack | a held message or a parked tool call; a person releases it |
| repeated denies in one session within a window | heuristic | session frozen read-only; person unfreezes |
| tool-call mix far from the session's baseline (total variation distance of the latest 40 calls from the session's first 100) | heuristic, weak | watch state only, never auto-quarantine |

Measured on 200 simulated days (7 sessions of ~430 calls over 6 tools, evenly and skewed): the first settings (baseline 50, window 30) put a benign session on watch on about 1 day in 14; baseline 100 and window 40 do so on none of the 200, and a session that turns to a single tool is still noticed within one window (40 calls). The test corpus is fixed in `tests/test_sentinel_benign.py`, not read from the docs.

| text identical (normalised fingerprint) to content already held | reliable | none beyond the first hold |
| a secret pattern in model output or tool arguments | reliable pattern, heuristic intent | credential marked suspect, rotation prompt |
| tool-result size, call count, wall time per session beyond limits | reliable as a count | tool disabled until released |

Out of scope: judging whether a held message was in fact malicious (a person does that);
semantic injection that no rail flags; a model that is steered by text no detector sees.

### b. Compromised or tampered models

| Signal | Class | False-positive cost |
|---|---|---|
| sha256 of a file differs from the digest pinned when the model was pulled | reliable | a legitimate in-place re-download looks like tampering until re-pinned |
| size, inode, mtime, symlink target changed since the pin | reliable that something changed; cheap pre-check that triggers the hash | same |
| hash on load differs from the manifest | reliable | same |
| file missing, replaced by a symlink, symlink retargeted | reliable | same |
| canary answers drift from the baseline recorded at install | statistical | a false alarm costs a held model and a re-run; thresholds come from the baseline's own variance |

What this does not do. Whether a model was backdoored or poisoned in training is not
decidable from outside with the tools here, and nothing in sentinel pretends otherwise. What
is measurable is that the bytes on disk are the bytes that were pinned, and that a model
answers a fixed set of probes the way it answered at install. A model that was bad when it
was pinned passes integrity. A backdoor that only fires on a trigger no canary contains
passes canaries. Both are out of scope; the pin records where the file came from so a person
can judge that.

Canaries: known-answer probes (arithmetic, a fixed extraction, a fixed format), a refusal
probe set (requests that must be refused and requests that must not), and format probes
(JSON validity, length). They run through the Broker, serially, on a schedule and after a
model changes. At install the baseline records the pass rate of each probe over n runs at
temperature 0 and at the serving temperature. Drift is declared when the current pass rate is
below the baseline's lower Wilson bound by a margin, so a stochastic probe does not alarm on
noise. Models served at temperature 0 have deterministic outputs on a fixed build, so for those a
changed output is exact evidence of a changed model or build.

### c. Compromised or misbehaving devices and peers

| Signal | Class | False-positive cost |
|---|---|---|
| request signature fails, with the failing address | reliable per request; heuristic as an attack | a misconfigured peer is blocked until released |
| nonce replay ("already seen") | reliable | none in a correct client |
| clock outside the window | reliable as a fact; usually a clock, not an attack | counted at a lower weight |
| lockout fired by the authenticator | reliable | the lockout already costs the peer a minute |
| request rate far above that peer's own windowed baseline | heuristic | a rate limit and watch, no block |
| peer reports a different version or binary digest than the cluster pin | reliable | a peer mid-upgrade; watch only |
| peer flapping (join and leave counts in a window) | heuristic | watch only |
| payload sizes beyond limits | reliable as a count | request refused |

Out of scope: a peer that holds the cluster key and behaves within limits. The security
model already treats key holders as fully trusted; sentinel can contain a peer once its
behaviour shows, and can drop its key from this node, but a stolen key is a rotation matter.

### d. Compromised servers and processes

| Signal | Class | False-positive cost |
|---|---|---|
| llama-server or other binary digest differs from the one pinned at build or install | reliable | a rebuild looks like tampering until re-pinned |
| a listener on a port in the model-server range that ml-stack does not manage | reliable as observation; the Broker already classifies it | an unrelated program is only reported, never stopped |
| a managed server's recorded pid now has a different executable | reliable | none |
| a managed server has children or connections it should not | heuristic, platform dependent | watch only |

Out of scope: a rootkit, a compromise of the account itself, a kernel bug. The state root
and the user account are trusted (`docs/security.md`); sentinel hardens its own files against
the agent and against peers, not against the owner's account being taken over.

### Subjects other than models and devices

The quarantine store holds one record per subject, whatever the kind:

| Kind | What is held | Measurable trigger | Reversal |
|---|---|---|---|
| `message` | a chat message, tool result, web page, document, retrieved chunk, as an inert, redacted, size-capped copy; the model sees a placeholder with the id | a rail deny, a fingerprint match, an injection score | human release returns the text to a human reader; purge drops the payload |
| `tool_call` | a pending call, arguments redacted, parked unrun | a rail deny, a tainted sensitive call | approve (human) runs nothing by itself, it marks the call approvable for the loop |
| `tool` | a tool or a caller class marked disabled | repeated abuse, a tool-policy violation | release re-enables |
| `session` | a conversation frozen read-only, no tools, transcript kept | denies per window | release thaws |
| `memory` | a summary, note, KV slot or prompt cache | the content matches a held fingerprint, or its source session was frozen | rebuild from clean sources; the held copy is for review |
| `model` | a model file moved into a quarantine directory beside its root, hash and original path recorded | pin mismatch | restore moves it back byte-identically |
| `artifact` | a downloaded file or archive, LoRA adapter, dataset, installed binary | pin mismatch, failed signature | restore |
| `mcp_server` | a third-party server disconnected, its tools dropped | unexpected tools or schema change | reconnect on release |
| `config` | a plugin or config file changed since its pin | hash mismatch | restore moves the changed file back |
| `credential` | a name marked suspect, withheld from child processes | its value seen in output or a held message | rotation clears it; the value is never printed |
| `peer` | a device id and address blocked, key and lease revoked on this node | auth failures, replay | unblock on release |
| `server` | leases revoked and the managed server stopped | binary digest mismatch, unexpected children | restart through the Broker after release |

Considered and left out, with the reason: network egress filtering of the whole machine (no
portable measurable signal; the account owns the network); scanning model weights for
anomalies (research problem, no reliable threshold); kill of unmanaged processes (the rules
say unrelated processes are never touched); automatic deletion of anything (nothing is ever
auto-deleted; purge is human and confirmed).

## One state machine

`clear -> watch -> quarantined -> released`, and `released -> watch` on the next signal. A
subject is `(kind, key)`. Allowed transitions are in `store.TRANSITIONS`; anything else
raises. Only `quarantined -> released` and purge are human-only.

```
 clear --signal--> watch --evidence--> quarantined --human--> released
   ^                  |                   |                      |
   +----decay---------+                   +----human purge       +--signal--> watch
```

Every transition is an audited event with a reason and evidence references (ids and
digests, never secrets or raw content).

## Reviewing what is held

`ml-stack-security review` is one screen with every held subject, quarantined and watched,
newest first: a number, the kind, a short name (a file's name, an address, a port; never the
raw key), how long ago, why in a sentence, what it blocks ("Requests from this machine are
refused", "This model cannot be leased or started", "This session is frozen") and what to do
about it ("Watch only. Release to stop watching; nothing changes for anything else.").

```
ml-stack-security review   1 held, 1 watched

> 1  QUARANTINED peer 127.0.0.1   3 min ago
  2  WATCH       server :51089   1 h ago

  why:    This machine sent requests with a forged or replayed signature, which no honest ml-stack peer does.
  blocks: Requests from this machine are refused.
  next:   Keep it held unless you know this is a false alarm; releasing puts it back in use.

1-9 select   r release   k keep   d details   R release all watched   p purge   q quit
```

| Key | What it does |
|---|---|
| `1`-`9`, arrows | select an item |
| `r`, then `y` | release the selected item: shows what it unblocks, one confirming key, no id to retype; any other key cancels |
| `k` | keep: change nothing, go to the next |
| `d` | details: the escaped record, its history and event log, and the exact `quarantine show/release/purge` commands |
| `R`, then `y` | stop watching every item that is only watched, after one confirmation; quarantined items are never touched |
| `p` | purge: deletes what is held, and still needs the id typed in full |
| `q` | quit |

`review --list` and `review --json` print the same table anywhere, inside an agent too:
viewing is not privileged. `status` and `chip` name what is held ("2 held: peer 127.0.0.1
(forged traffic), server :51089 (unmanaged). Review them: ml-stack-security review").
Everything printed from a record (names, reasons, peers, paths, arguments) is untrusted: it
is secret-masked, control, bidi and invisible characters become visible escapes such as
`\x1b` and `‮`, and every field is length-bounded. The sentence for each finding kind
is `explain.WHY`, and `tests/test_sentinel_explain.py` reads the detectors and fails when a
kind has none.

The interactive screen needs a POSIX terminal. It refuses, before drawing anything, when
stdin or stdout is not a terminal or when `CLAUDECODE`, `ML_STACK_AGENT` or
`ML_STACK_NONINTERACTIVE` is set, and exits 2 with the reason. Elsewhere use `--list` and
`quarantine release ID`.

**One click.** `ml-stack-security review --install-launcher [--dir ~/Desktop] [--force]`
writes `Review quarantine.command` (macOS) or `Review quarantine.desktop` (Linux), mode
0700, never over an existing file without `--force`. It is a fixed template holding only the
absolute path of the installed `ml-stack-security`, run in a login shell; it changes no
directory, activates no environment and carries no held text. Double-click it to get a
Terminal outside any agent.

**A heads-up.** When something is quarantined (not merely watched) a desktop dialog appears
with two buttons, `Dismiss` and `Review…`; `Review…` opens a terminal running the screen. It
never releases anything, shows only the sentence from `explain` and a bounded escaped name,
raises at most one notice per subject per hour and folds a flood into one burst notice, and
does nothing for a hold you made by hand. `ML_STACK_SENTINEL_NOTIFY=off` turns it off and
needs `ML_STACK_SENTINEL_NOTIFY_BECAUSE`; without a reason the switch is ignored, and an off
is logged as `sentinel.notify_off`. `ML_STACK_NOTIFY=console|off` (the onboarding switch)
also silences it. Where no dialog can be shown (no desktop, Windows) nothing is raised and
`status` still names what is held.

## Policy: what acts on its own

Modes, chosen by a person:

* `observe`: events and watch states only. No action is taken, ever.
* `guarded` (default for a new install): auto-quarantine only for high-confidence signals,
  which are exactly: a pinned file whose hash no longer matches, a held-content fingerprint
  reappearing, and a peer whose signed request replays or forges a MAC more than a
  threshold within a window. Everything else is watch plus an alert.
* `enforce`: the heuristic rules also act (freeze a session at N denies, disable a tool at
  M abuse events, block a peer on sustained rate abuse).
* `off`: the kill switch. Set by `ML_STACK_SENTINEL=off` with `ML_STACK_SENTINEL_BECAUSE` at process start or by the person
  through the command line; detectors keep logging that they are off.

Dry-run: `Sentinel(dry_run=True)` writes the event that says what would have been done.

## Self-protection

* State, key, log and held text live under `<ML_STACK_HOME>/sentinel/`; files are written
  atomically and are mode 0600.
* The state file, the pin manifest, the canary baselines, the decoy list and the log head are
  each sealed with an HMAC under a key file beside them. A state file that fails its seal
  raises a critical event and is replaced by the previous sealed copy when that holds;
  otherwise the store starts in fail-closed mode: every subject reads as blocked and nothing
  can be released. The held files stay where they were moved (`.ml-stack-quarantine/<id>/`
  under the managed root), but the list of what was held is gone in that case.
* Releasing, purging, reading held text and changing the mode need a `HumanGrant`. The
  command line mints one only when stdin and stdout are terminals, when none of
  `CLAUDECODE`, `ML_STACK_AGENT` or `ML_STACK_NONINTERACTIVE` is set, and when the person
  types the record's id back. A grant covers one action on one subject for two minutes.
* `human.agent_may(tool, arguments)` is the check an agent loop makes before running any
  tool: it refuses a call whose name or arguments name a sentinel verb, the sentinel code,
  or the state directory (including the directory a running sentinel was pointed at).
  `adapters.agent_gate` joins it to the screening of decoys and frozen sessions.
* Limits. The checks are on strings and on the environment. A call that builds a forbidden
  name at run time (`"ml-stack " + "sec" + "urity"`, a base64 payload, a script written to
  disk and run) is not recognised by `agent_may`; the terminal and environment checks and the
  seal are what stop it then. Code running as the owner inside this process can build a
  grant, and a process running as the owner can read the key file and rewrite everything.
  The defence is against the agent loop and against peers, which reach sentinel only through
  tool calls and requests. A state file replaced by an older, validly sealed copy is not
  detected (there is no counter outside the file).

## Events and the audit log

One typed record: `kind`, `severity` (info, notice, warning, critical), `source`, `subject`
(`kind:key`), `ts`, `evidence` (references: digests, sizes, counts, ids). Every string passes
through `redaction.redact` (token prefixes, bearer and MAC headers, `key=value` secrets, long
mixed-case runs, private key blocks) and every value under a key-like name is masked.

The log is JSON lines, mode 0600, rotated at 1 MB and kept to four files. Each line carries
`seq`, the `prev` hash and its own `hash`, an HMAC-SHA256 under the head file's key. The head
(count, last hash, the hash before the oldest kept line) is sealed in `events.log.head`.
`ml-stack security verify` finds an edited line, a removed or reordered line, a log cut
short, a deleted log, a head that fails its seal, and a line appended without the key.
`ml-stack security verify --anchor "<count> <hash>"` also compares the head with a line
written down elsewhere; `<state>/anchor.log` appends the head after every event, and the
person can copy it somewhere the account cannot write. A process killed between writing a
line and updating the head leaves a log that verifies and is adopted by the next append.
What it does not do: stop someone with the key and write access from rewriting the whole
log and head together; only an anchor kept elsewhere catches that.

Subscribers (`bus.subscribe(fn)`) get every event; a program that embeds ml-stack subscribes there.

## Honeytokens

`ml-stack security honey plant` (also `baseline --honey`) writes decoy files into the state
root (`.env`, `credentials.toml.bak`, `cluster.key.old`, never over a file the person already
has) carrying random values shaped like a Hugging Face token and a cluster secret. Nothing
legitimate reads or sends them, so a sighting is a high-confidence event: the value, or its
base64, URL-safe base64 or hex form, in tool arguments, tool results, model input or output
or an outbound request; a decoy path named in a tool call's arguments; a call to one of the
decoy tools (`export_credentials`, `dump_all_secrets`, definitions from `Honey.schema()` to
be offered beside the real tools). The subject is the session (else the caller), which is
frozen in `guarded` mode. A decoy file's access time is a weaker signal: it is reported and
never acts alone. A listening decoy endpoint was left out: it is a new surface to defend.

## What is armed by default

A process that does nothing about sentinel gets this (tests: `tests/test_sentinel_wiring_agent.py`,
`tests/test_sentinel_wiring_serve.py`, each checked by breaking the wiring):

| Where | What runs | Turn it off |
|---|---|---|
| Every `agent.Agent` | Each tool call goes through `agent_gate` before the rails: a call naming a sentinel verb or path, a decoy, a tool or session sentinel holds is refused with the reason. A call the rails refuse is reported to sentinel, which parks it as a `tool_call` and counts the denial. Each tool result is screened with the rails' own answer: held text, a decoy value or a text the rails withhold becomes a placeholder. A frozen session ends the run (`Done("denied")`) before the next model call. | `Agent(..., sentinel=agent.unwatched(because="..."))`, or `interventions=guard.off(because="...")`. Both need a reason, log a `sentinel.opt_out` event and print a warning. `ML_STACK_SENTINEL=off` turns the whole sentinel off only together with `ML_STACK_SENTINEL_BECAUSE=<reason>`: it then logs `sentinel.opt_out` with the reason and `ml-stack security status` shows `mode: off` and `off_because`. Without a reason the switch is ignored, sentinel stays armed and `sentinel.off_refused` is logged. |
| First run of an agent, the Broker daemon or the fleet daemon | The decoy files are planted under `ML_STACK_HOME` (never in a project). | `ML_STACK_SENTINEL=off` with a reason, or `ml-stack security honey remove`, which the next run plants again. |
| Every server start (`ServerManager._launch`, the one place a process is asked for) | `serve/guarded.verify`: a model held by sentinel is refused; a pinned model must equal its pin (it is hashed in full the first time, and again whenever its size, mtime or inode differ from the last full verification, which is remembered in the sealed `verified.json`), else it is refused and, when it sits under a managed root, moved aside; a model with no pin is pinned on first use (`source=first-use`, event `model.pinned_first_use`, a warning in the log), so the next start is checked against it; a model ml-stack pulled already has its pin (below), so first use is only for files it did not bring in. After the pin check, `verify` looks the file up by name in the signed manifests this machine accepted and returns `verified by manifest serial N` (event `model.manifest_verified`), or, when a manifest names the file and lists other bytes, quarantines it and refuses (finding `integrity.manifest_mismatch`; the file is not pinned). A directory or a name that is not a file here (an MLX directory, a repository id) is not pinned. The check also runs in `ServerManager._permitted` and `Broker.lease`/`start` against the store only, so a held model is not shared from a server that is still up. | `ML_STACK_SENTINEL=off` with `ML_STACK_SENTINEL_BECAUSE`. |
| Stop hooks | `serve_hooks` is registered with the manager's lease file on first use: quarantining a `model` or a `server` stops the servers ml-stack recorded for it (`ServerManager.reclaim`). A process not in the lease file is never touched. | `ML_STACK_SENTINEL=off` with `ML_STACK_SENTINEL_BECAUSE`. |
| Broker daemon and fleet daemon | A scan loop in a daemon thread: every pin and decoy once at start (hashing every file), then every `ML_STACK_SENTINEL_SCAN` seconds (default 300), hashing everything every twelfth round. It writes a sealed heartbeat `scanner.json`; `ml-stack security status` reports `scanner: armed, pid N, every Ns` or `NOT ARMED (reason)`, and is armed when a process beat within three intervals and did not stop it. | `ML_STACK_SENTINEL_SCAN=off` together with `ML_STACK_SENTINEL_SCAN_BECAUSE="..."`. `off` without a reason is ignored (the scan runs at the default and `sentinel.scan_off_refused` is logged); with one it logs `sentinel.opt_out`. A number sets the cadence in seconds. |
| Fleet daemon | The request authenticator is wrapped by `watch_authenticator`: forged, replayed and locked-out requests count against the sender's address, and an address sentinel holds is refused. | `ML_STACK_SENTINEL=off` with `ML_STACK_SENTINEL_BECAUSE`. |
| Behavioural canaries, in the Broker daemon's and the fleet daemon's scan loop | After each scan, every served model that is due is asked six fixed probes (a product, a capital, a JSON object, an exact word, a refusal and a benign recipe) three times each at temperature 0, at most 64 tokens per answer: 18 short requests per model per round, default once an hour (`ML_STACK_SENTINEL_CANARY`, seconds). The Broker daemon asks through a lease on the running server (never starting another; a lease that would is let go), the fleet daemon over the lease file's servers without one. A server that is processing is skipped, and the requests queue behind real traffic in the machine's request gate. The first healthy round records the baseline, sealed in `canaries.json` and keyed by model path and pin, so a replaced file gets a new baseline. A fall of the pooled pass rate by 0.15, or pass-rate intervals that no longer overlap, puts the model on `watch`; intervals apart together with a fall of 0.25 (or four probes moved) is a hard drift, confirmed by a second run straight after, and quarantines the model (moved aside, its servers stopped, restored by a person). | `ML_STACK_SENTINEL_CANARY=off` together with `ML_STACK_SENTINEL_CANARY_BECAUSE="..."`; `off` without a reason is ignored (`sentinel.canary_off_refused`); with one it logs `sentinel.opt_out`. |
| Guard events, aggregated to a score per session or caller | Every denial the guard logs for a running agent (`GuardLogHandler`, attached when the first `Agent` is watched, reading the session of the agent whose context is current), a call refused because a Confirm went unanswered, every sandbox event of a run while a session has a tool in flight (`sandbox_listener`, called from `sandbox.run` through a hook that `sentinel.default()` sets, so a process that never asks for the sentinel leaves sandbox runs unwatched), every refused lease and every resource-limit hit of a Broker caller (`broker_listener`, called from `Broker.lease` and `start`) adds its weight (`sentinel/score.py`: denial 1, sandbox denial 2, refused lease 2, resource limit 3, sandbox timeout 0.5) to the subject over a 600 s window. At 3 the subject is on `watch`; at 10 it is quarantined (a frozen session, a blocked caller; `Broker.lease` refuses a blocked caller) in `guarded` and `enforce` mode, and only watched in `observe`. This is a threshold, so it can be wrong: a person releases, which also starts the count again. Listeners ml-stack did not start are reported once each (`unmanaged_findings`) and only watched. | `ML_STACK_SENTINEL=off` with a reason, or the agent opt-outs above. There is no separate switch for the score. |
| Model output, compaction summaries | The final reply of a watched agent is screened before it is stored or returned: a decoy value, held content or a secret-shaped string in it is labelled in `guarded` mode (a `[sentinel: ...]` line in front of the text, event `output.suspect`) and held as a message with a placeholder in `enforce`. A decoy value also freezes the session. A compaction summary goes through `screen_memory`: one that repeats held content or carries a decoy value is quarantined as a `memory` and replaced by nothing, so the compaction falls back to its other strategies and the summary is never fed back. Rewording defeats the held-content check. | The agent opt-outs above, or `ML_STACK_SENTINEL=off` with a reason. |
| Child processes | `scrub_env` leaves any variable named as a quarantined `credential` out of the environment of the sandbox's child (`sandbox.run`), a detached job (`jobs.detach`), a model server (`serve.backend.launch`) and an MCP server launched by `agent.confined` (the last two are wired but have no test of their own). An environment passed as `None` (inherit) is not scrubbed. | `ML_STACK_SENTINEL=off` with a reason. |
| MCP connect | `McpTools.__aenter__` asks `mcp_allowed(label)`: a server that is quarantined is not connected (`McpBlocked`). Nothing quarantines an MCP server on its own yet: a person or an embedding program does that. | `ML_STACK_SENTINEL=off` with a reason. |
| Decoy endpoint, in the Broker daemon and the fleet daemon | A listener on 127.0.0.1 and an ephemeral port, the onboarding servers' shared handler, answering any path with an obviously fake credentials document. Its address is written only to `credentials.endpoint` under `ML_STACK_HOME` (and is a decoy value, so naming it in a tool call, result or reply is caught as the other decoys are). A request to it is a high-confidence `honey.endpoint_hit`: every session with a tool call running at that moment is quarantined (the only attribution a loopback request has; two sessions running tools together are both frozen), and with none running the hit is recorded against `caller:unattributed-decoy-hit`. | `ML_STACK_SENTINEL_DECOY=off` with `ML_STACK_SENTINEL_DECOY_BECAUSE="..."`; without a reason it is ignored and `sentinel.decoy_off_refused` is logged. |

Not armed: a plain process that only calls `serve.serve()` gets the start-time checks but no
scan loop, canaries or decoy listener (they live in the daemons); an in-process `Agent` run in a
process that is not a daemon has no scan loop either, and no decoy listener, so a decoy endpoint
hit cannot happen there. The canaries judge a model only against how it answered when the
baseline was recorded: a model that was already wrong then passes, and a model that behaves
differently only on inputs outside the six probes passes. Nothing quarantines an MCP server on
its own. `taint.subscribe` has no sentinel subscriber. The score and the decoy attribution are
thresholds and a guess about which tool made a request; a person releases both.
False-alarm measurement: `tests/test_sentinel_decoy_endpoint.py` replays a corpus of 48
scripted legitimate agent runs (8 shapes of conversation times 6 tasks; 72 tool calls; plain
answers, fetches, file reads, arithmetic, parallel calls, a corrected unknown tool) through the
real Agent loop with the decoy listener, the guard tap and the output screen armed, and asserts
no decoy hit, no decoy event, no session on watch and nothing quarantined. The corpus is small
and written for this test; it bounds nothing about a real day of use. The canaries have no such
corpus: their false-alarm control is the temperature-0 floor and the confirming second run.
`honey.file_read`, `file_changed` and `file_gone` only watch.
Start-time verification skips the hash when the file's resolved path, size, mtime and inode
are those recorded at its last full verification against the same pin, and its size equals the
pin's. Residual window: a file edited in place with its size and mtime restored (same inode) is
not caught at the next start. The scan loop's deep rounds (every twelfth, and the first) and
`ml-stack security scan --deep` hash every file regardless of the cache and catch it; between
deep rounds (about an hour at the default cadence) the window is open. The window is pinned by
`test_an_edit_that_restores_size_and_mtime_passes_the_start_and_fails_the_deep_scan`.

### Supply chain: pins at pull time and signed manifests at load

Armed (tests: `tests/test_supply_chain_pins.py`, `tests/test_supply_chain_audit.py`):

* **Pinned at pull.** Every model file (`gguf`, `safetensors`) that `net.download` or `net.accept` keeps
  (so `hub.pull`, `hub.fetch`, `hub.snapshot`, a file from a paired device in peer-first, and the other
  callers of `net.download`) is recorded in the same sealed pin store the load check reads, in `net.download._pin_pulled`,
  after the size, SHA-256, format and scan checks passed and the file was promoted. The pin is
  `source=pull`, with `origin` (the URL the bytes came from, or `peer:<name>`) and `digest_from`:
  `expected` when the download was verified against a digest from another source (the Hub's listing, the release's
  digest, a signed manifest), `computed` when none was published and the pin is the digest of the bytes as they arrived. The
  digest is the one the download already computed over the staged file; nothing is hashed again. A download that fails any
  check records nothing. An event `model.pinned_at_pull` is written. The first start therefore checks a pulled model against what
  arrived; a file swapped between the pull and the start is refused.
* **Signed manifests at load.** A signed file list (`fleet/onboard/manifest.py`, signed by the owner's Ed25519 key) is
  accepted into a sealed store (`signed-lists.json` under the sentinel directory, `fleet/onboard/trusted.py`) when it is
  fetched from a peer (`PeerSession._ask`) or with the onboarding `fetch` command, and only if it verifies under a key a person
  pinned (`peers.json`, `trust.json`, this machine's own `signing.json`; the secret half lives in the OS keystore on the signer) and its
  serial is not below the highest accepted for that key: an older list is refused (`ManifestError`) and does not replace
  the newer one. At load each kept list is verified again under the keys pinned now, so a key that was un-paired or
  revoked stops counting. Optional: with no list, or none that names the file by name, nothing changes.

Not armed:

* A manifest is matched by file name. A list that names `x.gguf` applies to any local file called `x.gguf`.
* A list reaches the load check only after a fetch ran on this machine (peer-first pull or the onboarding `fetch` command);
  there is no command yet that accepts a list from a file, and no vendor-published digest list is read.
* The serial high-water mark lives in a sealed file next to its key; code running as the same user can delete both and
  reset it. Expiry is not applied at load (a list bounds how long it can be fetched, not how long held bytes are checked).
* Only `gguf` and `safetensors` files are pinned at pull. Archives (llama.cpp builds, the Python bundle) are unpacked and
  their archive deleted, so there is no stable file to pin; they are verified against their release digest at download only.
* A model whose size, mtime and inode are restored after an edit still waits for a deep scan (see above).

## What is built and tested

| Piece | Where | Status |
|---|---|---|
| event, bus, redaction, chained log, verify, anchor | `events.py`, `redaction.py` | built, tested, mutation-checked |
| sealed files, quarantine store, state machine, retention, text hold, file move and restore | `sealed.py`, `store.py`, `moves.py` | built, tested, mutation-checked |
| human grants, agent surface check | `human.py` | built, tested, mutation-checked |
| model, binary, artifact, config pins; load-time and periodic checks | `integrity.py`, `Sentinel.verify_before_load`, `scan`, `start` | built, tested |
| canaries and drift | `canary.py`, `serve/canaries.py` | built; scheduled in the daemons' scan loop, tested against real fake-llama processes whose answers change while they run; also run against two real GGUFs |
| peer outcomes, flapping, version and binary mismatch, tool mix, resource use | `rates.py` | built, tested on synthetic clocks |
| guard-rail verdicts to findings, held messages, parked calls, tainted text | `rails.py`, `Sentinel.screen` | built; tested against the real `Guard` on the integration branch |
| derived memory: summaries that repeat held text, memories of a frozen session | `Sentinel.screen_memory`, `register_derived` | built, tested |
| decoys | `honey.py`, `serve/decoy.py` | built, tested (files, tokens, tools, endpoint) |
| suspect credentials withheld from child environments | `Sentinel.scrub_env` | built, tested; applied at the four spawn helpers |
| unmanaged listeners, changed server executable | `servers.py` | built; unmanaged listeners are reported by the Broker's adoption, the changed-executable check is not scheduled |
| hooks that stop a server or a model's servers | `adapters.serve_hooks` | built, tested with real processes |
| status mark | `Sentinel.chip`, `ml-stack security chip` | built; not yet drawn by the page |
| `ml-stack security` | `cli.py` | built, tested |

Not built. Rotating a
credential is left to the person. No detector reads KV cache or prompt cache contents:
`screen_memory(key, text)` is the call that decides whether a stored summary may be reused,
and a cache slot is quarantined by key (`memory`, `kv:slot-3`) by whoever manages it. Nothing
scans model weights. A server's children and network connections are not examined. The tool
mix, flapping, rate and unmanaged-server signals only watch.

Considered and left out: filtering the machine's network egress (no portable signal);
judging a model's weights for backdoors (unsolved); stopping any process ml-stack did not
start (against the rules); deleting anything on its own (only a person's confirmed purge
deletes).

## Results

Measured on one Apple-silicon laptop, 2026-10-02. "Scripted" means a function that answers
the canary probes correctly with a set probability; "real" means a llama-server leased
through `ServerManager`, one at a time, stopped afterwards.

**Integrity.**

| Check | Result |
|---|---|
| one flipped byte at the start, middle and end of a 200 kB pinned file, deep check | 3 of 3 found |
| a change that keeps size and mtime (inode unchanged) | the quick check misses it, the deep check finds it |
| real GGUF (a 484 MB copy of Qwen3Guard-Gen-0.6B Q4_K_M), one bit flipped mid-file | refused at load, file moved aside, original untouched |
| cost of hashing that file | 0.2 to 0.5 s, 1 to 2.7 GB/s with the file in the page cache; a cold disk is slower |
| touched but unchanged file, deep and quick scans | no finding |
| symlink retargeted, file deleted, binary replaced | each found; the link or the binary is the thing moved |

**Canaries.** 12 probes (5 known answers, 1 JSON extraction, 1 format, 2 refusals, 3
compliances), baseline 10 runs each, check 5 runs each.

| Model | Result |
|---|---|
| scripted, unchanged, baseline p = 0.98 or 0.95 | 0 false alarms in 600 checks each |
| scripted, got worse to p = 0.5 or 0.2 | 300 of 300 caught each |
| scripted, p = 0.7 | caught 220 of 300 (baseline 0.98), 184 of 300 (baseline 0.95) |
| scripted, p = 0.8 | caught 53 of 300, 30 of 300 |
| scripted, p = 0.9 | 0 of 300 caught |
| scripted, stops refusing the two refusal probes | caught, both probes named |
| real Qwen3-VL-2B-Instruct Q4_K_M, temperature 0, 3 runs | baseline 36 of 36; the same model again: no drift |
| real, replaced by Qwen3Guard-Gen-0.6B | 9 of 36 pass; drift found |

So the detector is built not to cry wolf and sees a model that is much worse or different; a
slip of ten points in the pass rate is invisible to it, and so is any change the 12 probes do
not exercise. At 3 runs per probe no single probe has the power to be named; the pooled
interval does the work.

**Tool mix.** The first version (20 calls of baseline, windows of 20, distance 0.6) raised two
false alarms across seven sessions of steady random use; simulation put its rate at about 0.2%
of windows for a fresh baseline and higher for an unlucky one. The defaults are now 50, 30
and 0.5: 0 alarms in 4000 simulated windows and in 20 sessions of 600 calls, and an unseen tool
filling a window is flagged every time. It remains a watch-only signal.

**Peers.** A request replayed three times: the peer is quarantined at the third and its next
valid request is refused (against the real `Authenticator`). 35 forged signatures from one
address: quarantined; an honest address sending 50 valid requests meanwhile: untouched.
Against the peer watch alone, on a synthetic clock: 5000 valid requests at two per second
from one peer gave 0 findings, and four failures followed by 200 good requests gave no
high-confidence finding.

**Benign use.** 3000 screened documents (paragraphs of this repository's `docs/`), 3000 tool
calls, 3000 peer requests with an occasional single failure and six pinned files touched and
rewritten with the same bytes, in `enforce` mode: 0 quarantined, 0 watched. 2000 further
screens and calls with decoys planted: 0 decoy findings. Held-text overlap against 536
paragraphs of `docs/` after three injections were held: 0 false matches. Against the real guard rails (integration branch), 651 paragraphs of `docs/` shown as web
results in `enforce` mode: 3 held (0.5%), none of the session frozen. The rails mark every
result from an external tool as tainted; sentinel acts only on text whose reason says it reads
like an instruction, and five of those inside five minutes freeze the session in `enforce`.

**Held text copied into a summary.** Six verbatim copies (whole, and quoted sentence) of held
injections: 6 of 6 found. Three paraphrases: 0 of 3 found. The signal is shared six-word runs;
a model that rewords the instruction gets past it. The first PyRIT baseline put a planted
instruction in a summary 7 of 9 times, and this catches the copies among those, not the
rewordings.

**Agent surface.** 17 attack calls (a shell command or file edit naming a sentinel verb or
path, a sentinel tool name, an environment override) refused; 11 ordinary calls (including
the words "security", "release", "mode" and "sentinel" in prose) allowed. `release`, `purge`
and `mode` through the command line, run as a real subprocess: refused with no terminal,
refused under `CLAUDECODE` even on a pseudo-terminal, refused on a wrong id, accepted on a
pseudo-terminal when the id is typed.

**Crash safety.** The state writer was killed with SIGKILL at 12 different moments, the log
writer at 8: the state always loaded with its seal intact (`tampered` false) and a non-decreasing
record count, and the log always verified. A rotation interrupted after the head was updated
and before the oldest file was removed verifies.

**Mutation check.** 112 hand-written faults, one at a time, each run against the sentinel
tests: 110 killed, 2 not applicable (a fault written against code that was then removed, and
one mis-specified). The first pass killed 83; the 15 survivors named missing tests (the
sequence and link checks each hiding behind the other, an interrupted rotation, the
previous-copy fallback, probe voting, the any/complies/rate paths, protected-path
resolution, a mode of `off` reaching `handle`, the file mode of held text) and a redundant
`chmod`, which was removed. Each now fails a test.

**Not verified.** The Broker (admission-control branch) was not merged, so canaries were run
through `ServerManager` rather than through its request queue, and the fleet daemon was not
wrapped; the `Authenticator` is where the integration was tested. No pyright run was made on
the sentinel package beyond the repository's budget check. Nothing was tried on Linux or
Windows. `agent_may` was not run against a live model-driven attack: the PyRIT suite
(`agent/redteam`) was not merged, and the attack strings are hand-written.
