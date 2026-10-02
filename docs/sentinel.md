# Sentinel: detecting and containing attacks on a node and its fleet

Status: design (phase 1). The implementation is `src/ml_stack/sentinel/`; the last section
of this file is rewritten from measurements when it lands.

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

The adapters are tested against the real classes on a throwaway integration branch
(`agent/sentinel-integ`, the three branches above merged), not on this one.

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
| tool-call mix far from the session's baseline (total variation distance) | heuristic, weak | watch state only, never auto-quarantine |
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
| `config` | a plugin or config file changed since its pin | hash mismatch | restore the pinned copy kept in the store |
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

## Policy: what acts on its own

Modes, chosen by a person:

* `observe`: events and watch states only. No action is taken, ever.
* `guarded` (default for a new install): auto-quarantine only for high-confidence signals,
  which are exactly: a pinned file whose hash no longer matches, a held-content fingerprint
  reappearing, and a peer whose signed request replays or forges a MAC more than a
  threshold within a window. Everything else is watch plus an alert.
* `enforce`: the heuristic rules also act (freeze a session at N denies, disable a tool at
  M abuse events, block a peer on sustained rate abuse).
* `off`: the kill switch. Set by `ML_STACK_SENTINEL=off` at process start or by the person
  through the command line; detectors keep logging that they are off.

Dry-run: `Sentinel(dry_run=True)` writes the event that says what would have been done.

## Self-protection

* State, key, log and quarantine payloads live under `<ML_STACK_HOME>/sentinel/` with mode 0700
  and files 0600. Writes are atomic.
* The state file is sealed with an HMAC under a key file next to it. A bad seal raises a
  critical event, loads the previous sealed copy if its seal holds, and otherwise starts in
  fail-closed mode: nothing is released, held payloads stay held.
* Releasing, purging, changing the mode and turning sentinel off need a `HumanGrant`. A grant
  is minted only by the command line after it finds a terminal on stdin and stdout, finds
  none of the agent markers in the environment, and the person types the subject's id. The
  agent-facing surface (`sentinel.human.agent_may`) refuses any tool call whose name, command
  line or path touches a sentinel verb or the sentinel directory; the MCP and tool-loop
  adapters call it first.
* Limits stated plainly: code running inside this Python process as the owner can forge a
  grant, and a process running as the owner can edit the files. The defence is against the
  agent loop and peers, which reach sentinel only through tool calls and requests.

## Events

One typed record: `kind`, `severity` (info, notice, warning, critical), `source` (which
detector or adapter), `subject` (`kind:key`), `ts`, `evidence` (a mapping of references:
digests, sizes, counts, ids). The log is append-only JSON lines, bounded by size with
rotation and a count, and every string passes through the same redaction as credentials
(token prefixes, `Authorization` values, long base64 or hex runs next to key-like names).
Subscribers (`bus.subscribe(fn)`) are called with each event; pcb-engine subscribes there.

## Results

Written after the tests and measurements run. See the end of this file.
