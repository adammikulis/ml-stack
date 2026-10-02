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

## What is built and tested

| Piece | Where | Status |
|---|---|---|
| event, bus, redaction, chained log, verify, anchor | `events.py`, `redaction.py` | built, tested, mutation-checked |
| sealed files, quarantine store, state machine, retention, text hold, file move and restore | `sealed.py`, `store.py`, `moves.py` | built, tested, mutation-checked |
| human grants, agent surface check | `human.py` | built, tested, mutation-checked |
| model, binary, artifact, config pins; load-time and periodic checks | `integrity.py`, `Sentinel.verify_before_load`, `scan`, `start` | built, tested |
| canaries and drift | `canary.py` | built; run against scripted models and two real GGUFs |
| peer outcomes, flapping, version and binary mismatch, tool mix, resource use | `rates.py` | built, tested on synthetic clocks |
| guard-rail verdicts to findings, held messages, parked calls, tainted text | `rails.py`, `Sentinel.screen` | built; tested against the real `Guard` on the integration branch |
| derived memory: summaries that repeat held text, memories of a frozen session | `Sentinel.screen_memory`, `register_derived` | built, tested |
| decoys | `honey.py` | built, tested |
| suspect credentials withheld from child environments | `Sentinel.scrub_env` | built, tested; the caller must use it when it starts a child |
| unmanaged listeners, changed server executable | `servers.py` | built; takes the finder's output and a pid-to-executable function |
| hooks that stop a server or a model's servers | `adapters.serve_hooks` | built, tested with real processes |
| status mark | `Sentinel.chip`, `ml-stack security chip` | built; not yet drawn by the page |
| `ml-stack security` | `cli.py` | built, tested |

Not built. Wrapping the fleet daemon so that it calls `watch_authenticator` and refuses a
quarantined address is a line in `fleet/api.py`, which another branch owns. Rotating a
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
