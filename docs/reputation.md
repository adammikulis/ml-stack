# Reputation of sources

ml-stack keeps a record of how every source it deals with has behaved: hosts, URLs, addresses,
fleet peers, model repositories, artifact hashes and connectors. The record tightens what is
asked of a source. It never grants anything.

## Store

One sealed file per user under the state directory (`reputation/u-<id>/graph.enc`), a
`GraphStore` snapshot sealed with the memory vault's AES-256-GCM under a key of its own in the OS
keystore (profile `reputation`). The key is created on the first write; constructing the ledger or
reading an empty store never touches the keystore. Nothing about a source is on disk in plaintext
except `summary.json` beside it: counts by state and waiting notices, no names.

Nodes: `source` (kind, canonical key, scores, traits, state) and `event` (a bad event tied to its
source by an `event` edge). At most 1000 sources and 20 event nodes per source; the oldest
unknown or established sources go first.

## Scores

| | Rule |
|---|---|
| Events | `denial` 1, `scan_hit` 3, `injection_flagged` 1 (0.5 on the host), `cert_or_key_change` 3, `hash_change` 3, `redirect_change` 0.5, `ip_change` 0.5 |
| Short term | sum of event weights in the last 6 hours |
| Long term | weighted badness with a 30 day half-life, floored at 0.5 for a watched and 1.0 for a bad source |
| Established | 10 clean runs (5 minutes apart at least), first seen 3 days ago, nothing recent against it |
| Watch | short term >= 1 |
| Bad | short term >= 3 for a source with no standing, or a second event after a divergence |

A repeat of the same event inside a minute counts once. Traits (certificate or key fingerprint,
IP range, redirect targets, served hashes, size and content-type shape) are recorded; a value
outside the baseline for `cert`, `ip_range`, `redirect` or `hash` is the matching event. `shape`
is recorded and never scored.

**Divergence.** An established source that gets fresh evidence (short term >= 1) is stepped to
watch at once and one notice is queued. An unknown source that misbehaves is just bad.

**Recovery.** Waiting restores nothing. A watched source returns after `ML_STACK_REPUTATION_RECOVER`
clean runs in a row (default 10); a bad source steps to watch after three times as many. A person
clears a source with `ml-stack-reputation forget`.

## The notice

Raised through sentinel's single-flight dialog: one dialog at a time on the machine, the shared
cooldown, off when `ML_STACK_NOTIFY=off`. Buttons: Later, Keep watching, Block it. With
notifications off the notice stays queued and the counts show in `ml-stack-security status` and the
chip.

## Observation points

Only the system's own observations move a score; no function that changes a score takes text.

- `net.Pipeline.get/open`: clean runs, redirect targets, content shape; refusals as `denial`.
- `net.download`: a failed pin is `hash_change`, a scanner hit is `scan_hit` (host and artifact
  hash), another rejection is `denial`; a kept file is a clean run.
- `PeerWatch.note`: authentication failures and replays are `denial`, good requests are clean runs.
- `web.read`: a page that reads like an instruction is `injection_flagged` against the URL it was
  fetched from.

Observation points call `ml_stack.sentinel.observers`, which does nothing until
`ml_stack.reputation.hooks.install()` has run in the process (the web tools do it).

## What it changes

`net.Policy.admit` raises `Distrusted` (a `NeedsApproval`) for a host or URL that is watched or bad,
unless a person approved the host after the verdict. Reputation never admits a host the policy
refuses, and never touches a role, rule or human-only step.

## Commands

`ml-stack-reputation list | show kind:name | forget kind:name | forget --all | export | stats`.
A person at a terminal only; an agent marker in the environment refuses every one.
