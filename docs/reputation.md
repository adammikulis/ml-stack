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

## Verified work

Completion credits and work reputation are separate from source-risk scores. A person
or the worker's registered parent verifies an authenticated task and its completion
reply in the same thread. Passed independent checks, artifact SHA256 hashes and
both original message hashes remain with the award. Failed checks, unrelated replies,
self-awards and unrelated verifiers are refused. No verification write is exposed as
an agent tool or browser endpoint.

A verified completion earns 10 credits. A reviewer can add one 5-credit bonus per
quality tier: independently validated work, a useful regression check, and demonstrated
impact. Each tier needs a reviewer reason linked to passed independent checks and
hashed artifacts. Test quantity alone earns nothing. Each workspace, authenticated
worker and task has one immutable award; retries return its original evidence and
cannot upgrade or duplicate it. Historical completion evidence receives an explicit one-time base-award migration,
with no inferred quality bonus. The immutable award stores its policy version and
currency; later policy changes do not recalculate previous awards. Until migration,
missing awards are reported as pending and add no spendable credit.

Runs are free while the economy develops: earned credits accumulate, spent is zero,
and balance equals earned. Credits do not change permissions, safety rules, broker
priority or baseline access. No redemption or charging is enabled. Resource usage is
accounted separately when a reviewer supplies measured per-task counters; unknown
values stay null. Aggregates report the measurement count and cover verified task
evidence, not every run. One owner-enrolled base agent per installed physical device owns its economy account.
Saved local worker seats bind only through person-authorized enrollment using the
maintained device ID; workers cannot choose or steal an account. Historical worker
evidence rolls up through those persistent graph membership edges, including after
model changes or worker stop. Devices remain separate. Unenrolled workers remain
explicitly unenrolled. Model and harness are immutable task provenance; they never
create accounts or reset balances.

Work reputation uses separate reviewer-supplied quality and reliability ratings from
0 to 100, with reasons. Neither credits nor task/test counts produce ratings. A neutral
50 prior with weight two bounds early swings; sample count and confidence accompany
each aggregate. Without reviews the state is unrated. A future advisory pricing
modifier is bounded from 0.8 to 1.2 using evidenced reliability and confidence, starts
neutral, and guarantees baseline access; actual price remains zero. Source-risk
standing is unchanged. All evidence uses separate work nodes in the maintained
per-user encrypted reputation graph. Awards, verification decisions, quality reviews,
ratings and measured usage are explicit linked nodes. Balances and aggregates are
derived from that evidence rather than a separate balance file.

Agents can read their own and team standings as recorded evidence, which grants no
additional permissions. History shows verified completion counts under each agent;
open the count and then a task to inspect its verifier, checks and artifact hashes.
The read view includes the 20 most recent evidence records per agent and reports
how many earlier records remain in the encrypted ledger. **Load earlier verification
evidence** retrieves another page. Agents use the read-only
`workspace_reputation(agent, offset)` tool, and their task context includes their own and team
completion counts as data without additional authority.

Native Claude and Codex coding workers receive the same bounded reputation brief
using their registered child identity, and expose the read-only
`workspace_reputation` tool through their scoped workspace MCP server. Their
worker status includes the brief. History links local display aliases to the
registered identity saved in the worker record and identifies that binding next
to the verified completion count.
