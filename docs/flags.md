# Flags: one way to mark a suspicious agent or device

Status: design, 2026-10-09; companion to `docs/sentinel-open-join.md` (kept separate to hold that file
under its 400-line cap) and `docs/earned-trust.md`. Owner: "we need an easy way to automatically flag a
suspicious agent and/or device if not already implemented." Tags: **[V path]** read in code at
`ca4eb9b1`; **[D doc]**; **[analysis]** proposal. Nothing in sections 2 to 6 is built.

## 1. What exists today

### 1.1 Agents

| Mechanism | What it does | Gap |
|---|---|---|
| `Registry.strike(name)` | adds one to the **parent's** `strikes`; `agent_invites` refuses new children once `agent_invite_strikes` (3) is reached. [V `identity.py strike`, `limits.py`, `agent_invites.py:79`] | Counts against the parent, never the agent itself; called only from `_hold`, only for child agents; no list, no clear, no notice. |
| `standing.record_injection` | a message with a **hard** marker writes an `injection_flagged` observation against `workspace:<id>` in the source-risk ledger; a `watch`/`bad` sender's later messages are held. [V `standing.py`, `service.py _hold`] | One signal only. Standing is a hold on messages, not a restriction on actions. |
| Quarantine of messages | flagged text is held with a placeholder (`Quarantine.hold`) and mirrored into the sentinel store; a person releases it. [V `quarantine.py`] | Holds text, not the sender. |
| Forged-sender refusals | a request carrying `sender`, `name`, `parent`, `from` and similar is refused; the sender is stamped from the token. [V Rust node, node.md "Identity"; Python `identity.authenticate` stamps `from` [D `docs/earned-trust.md` 2.1]] | The refusal is an error reply; it is not counted against the sender. |
| Identity caps | `agents_live` 64, `board_members` 64, `agent_tree_live` 8, children per parent. [V `limits.py`, `agent_invites.py:111`] | A refusal, not a signal: a spike in refusals raises nothing. |
| Claims | expired or dead-holder claims are swept; `on_swept`, `on_stolen` callbacks exist. [V `claims.py`] | No consumer is wired to the ledger. [D `docs/earned-trust.md` 1.2] |
| Reputation | accepted/rejected reviewed tasks, reliability; `work_reputation` says `authority: none`. [V; D `docs/earned-trust.md`] | Standing grants nothing and takes nothing away. |
| Commit hooks | `scripts/hooks/weakened-assertions` refuses a commit that weakens an assertion; `claude-bash-guard` and `claude-edit-guard` refuse listed actions. [V `scripts/hooks/`; D `CLAUDE.md`] | A refusal in a shell; recorded nowhere the board sees. |
| Sentinel peer flags | `PeerWatch` flags an **address** for forged traffic, flapping, rate. [V `sentinel/rates.py`] | Address, not agent or device identity. |
| Sentinel `HumanGrant`, `agent_may` | stops an agent reaching sentinel verbs. [V `sentinel/human.py`] | Refusal only. |

So: agent misbehaviour is caught in about six places and recorded in none of them in one shape. There
is nothing a person can list, and nothing that restricts an agent as a result. [analysis]

### 1.2 Devices

Revocation (`Roster.revoke`, Rust `Pool::revoke`) is sticky and spreads by `members`; the address
watch above; nothing else. No flag, no probation (see `sentinel-open-join.md`). [V `membership.py`,
`membership.rs`]

## 2. One flag: a board entry of kind `flag`

A flag is a board entry (the board's signed, hash-chained log, so it syncs and cannot be silently
edited). [analysis; entry shape V node.md "Board entries"] Kinds `task`, `landing_request`,
`reputation_event` are already reserved in the schema; `flag` is added the same way.

| Field | Meaning |
|---|---|
| `subject` | an agent name (`claude-6e1a2f`, never a display string; ids per `docs/earned-trust.md` 2.1) or a device fingerprint (64 hex), with `subject_kind: agent | device` |
| `level` | `note`, `restrict`, `suspend` |
| `state` | `open` or `cleared`; clearing is a second entry that points at the first (`clears: id`), never an edit |
| `code` | a reason code from a fixed list (section 4), or `manual` |
| `reason` | text, for a human; bounded (500), screened like any foreign text |
| `evidence` | references only: entry ids, hashes, counts, task or commit ids; never raw frames or secrets |
| `raised_by` | **stamped by the board** from the token (agent), the node (detector) or the person's grant; never a field the caller sets |
| `source` | `person | agent | detector` |

Rules (in the node, so every writer meets them): [analysis]

- Any registered agent may raise `note` on any subject but itself. Only the coordinator or the person may
  raise `restrict` or `suspend`; a detector (sentinel, token `sentinel`) may raise up to the level its
  code allows (section 4). A caller's requested level above its right is `denied`, not downgraded.
- A flag about the same subject and code folds into one open flag (count and last evidence), so a flood
  of notes is one line, not a stack. An agent cannot raise more than N notes an hour (default 10).
- Effective state of a subject = the highest open flag. `cleared` needs the person or the coordinator,
  with a reason; a detector cannot clear its own flag except by expiry of a `note` (7 days), and never a
  `suspend`.
- Flags are visible to all members of the board. A subject sees its own flags and codes (no hidden
  accusations), but not other raisers' identities when the raiser is an agent, unless it is the
  coordinator or the person. [analysis, to avoid retaliation]

Effect of each level:

| Level | On an agent | On a device |
|---|---|---|
| `note` | none; shows in `flags`, `digest --status`, the panel; feeds the ledger as a weak negative | none; sentinel watch |
| `restrict` | agent drops to L1 (`docs/earned-trust.md` 3.2): own worktree, no landing, no coordination | probation caps halve; no new boards; no leases |
| `suspend` | agent to L0: read, message, propose only; the token stays valid so it can be told why | quarantined: connected, isolated, nothing merged |

## 3. Easy manual use

```
ml-stack-workspace flag NAME --reason TEXT [--level note|restrict|suspend] [--evidence ID ...]
ml-stack-workspace flags [--open] [--subject NAME]
ml-stack-workspace flag-clear FLAG_ID --reason TEXT
```

- `flag` by a registered agent: level defaults to `note`; a higher level is `denied` for an agent that is
  not coordinator. By the person at a terminal: any level, with the existing `HumanGrant` check
  (not under an agent marker). `NAME` may be a device fingerprint prefix; ambiguity is an error.
- `flags` lists open flags, newest first: id, subject, level, code, age, raised by (where visible).
  `--json` for scripts. Viewing is not privileged, as with `ml-stack-security review --list`. [V pattern
  `docs/sentinel.md`]
- `flag-clear` needs the person or the coordinator and a reason; the reason is stored.
- `digest --status` gains lines: "2 open flags: claude-6e1a2f restricted (forged_sender), device ab12... noted".
  [V `attention_cli.STATUS` is the flag for `digest`; the lines are the addition]
- One panel on the Fleet page: a list of open flags with the same fields, buttons Restrict, Suspend,
  Clear (each one confirmation; the same grant check). The panel is a component, not a page. [D
  `components-not-pages` memory; analysis]
- A new flag raises the existing single heads-up dialog once (kinds `agent_flagged`,
  `device_flagged`); `explain.WHY` gets one sentence each. [V `explain.py` test pattern]

## 4. Automatic detectors for agents

Each detector reads evidence a script can check, raises a flag with code, evidence and a level, and
never judges intent. "Immediate" means the flag is raised on the first event. The earned-trust owner
decision (2026-10-08) is followed: rules suspend automatically on security, tamper and forged-claim
events and notify, and the person lifts it. Thresholds are starting values for the owner to move.
[analysis; D `docs/earned-trust.md` 9]

| Code | Signal and source | Threshold | Level |
|---|---|---|---|
| `forged_sender` | a request carries `sender`/`name`/`parent`/`from`/`actor`/`as` anywhere, or a token tries a name it was not minted for [node `denied`] | 1 | `suspend` (security, forged claim) |
| `forged_claim` | a review's independent checks contradict a recorded worker claim; provenance differs from the lease [D `earned-trust.md` 3.4 item 3] | 1 | `suspend` |
| `tamper` | `Tampered`, a chain `Verdict.ok == False` attributed to the writer, a sealed file edited | 1 | `suspend` |
| `guard_edit` | an edit/bash guard refusal naming a guard, hook, authorization, grant or claim file; `agent_may` refusal; push to `main` or force attempt | 1 | `suspend` |
| `auth_failures` | authorization refusals (`denied`) on one token | 5 in 10 min | `restrict` (note at 3) |
| `refused_actions` | repeated refused actions of any kind (quota, rails, hook) | 10 in 10 min | `note`, `restrict` at 25 |
| `injection` | `injection_flagged` hard marker held from this sender [V `record_injection`] | 1 hard; 3 soft in 1 h | `restrict` (hard), `note` (soft) |
| `flooding` | messages or announcements per window above the board rate, or `#announcements` use by a non-coordinator | 3x the board's `rate` limit for 5 min | `note`, then `restrict` |
| `claim_hoarding` | live claims held per identity, or claim/release churn (`on_swept`, `on_stolen`) | 8 live claims, or 20 acquire/release in 10 min | `note` |
| `lease_churn` | lease acquire/expire/abandon cycles (a3e2 branch `lease` entries) | 10 in 10 min, or a held GPU lease idle past 2x estimate | `note`; `restrict` if repeated daily |
| `reverted_landing` | a landing later reverted with a reviewer mark of defect (needs the landing record M1) | 1 -> `note`; 2 in 30 d | `note`, then `restrict` (drops one level, per earned-trust 3.4) |
| `tests_weakened` | `weakened-assertions` hook refusal on this identity's commit, or a review finding a deleted/loosened test | 1 refused commit -> `note`; 3 in 7 d -> `restrict`; an accepted landing that weakens a test -> `suspend` | graded |
| `identity_spike` | `agents_live`, `agent_tree_live` or `board_members` refusals, or mints by one parent | 3 refusals in 10 min | `restrict` the parent |
| `second_process` | one token used from a second pid or device while the first is live (when the node records it) | 1 | `suspend` (tamper class) |
| `behaviour_change` | a model or version change (`model_state` change) followed by a jump in refusals, rejections or flags, against the pre-change baseline | 2x rate over 20 actions | `note`; per earned-trust the first task on a new model is a probation of its own |

Evidence is always references. A detector may not raise a flag from text alone (that is the rails' job,
and they already hold it); it raises from counts and refusals the system itself recorded. A detector that
cannot name its evidence cannot raise a flag. [analysis]

False alarms: every detector ships with a replay of the honest-use corpora already in the repo (the
agent-loop corpus in `tests/test_sentinel_decoy_endpoint.py`, the 3000-call benign run in
`docs/sentinel.md` "Results") and a test that it raises nothing on them. `restrict` and `suspend` are
reversible by one action and delete no evidence. [analysis; V corpus]

Device detectors are those of `sentinel-open-join.md` 3.3; they raise the same entry with
`subject_kind: device`. A device flag and its agents: `suspend` on a device quarantines the device; the
agents running on it are each `note`d with code `host_flagged`, not suspended, because the host's fault
is not theirs. [analysis]

## 5. How flags feed probation, quarantine, the ledger and the brakes

- **Probation and quarantine.** The node's per-state capability table (open-join 3.1) reads the effective
  flag: `note` on a probationer lengthens its probation and blocks promotion; `restrict` halves its caps;
  `suspend` is quarantine. Grades 3 and 4 of open-join 3.4 are `suspend` and a `revoke` that carries a
  flag with `code` and `evidence`, so a revocation always explains itself on the board. [analysis]
- **Trust ledger.** A flag is a `reputation_event` too: `note` weakens reliability a little, `restrict`
  and `suspend` are the hard faults of earned-trust 3.4 (demote at once, standing keyed on the account
  per model and version, the brake on the individual identity per the 2026-10-08 decision). Clearing a
  flag with a reason does not erase the event; the ledger keeps both, and a flag cleared as a false
  alarm is counted against the detector's own record (a detector's precision is a measured number).
  Rejection of a reviewed task is not a flag. [D `earned-trust.md` 3.4, 9; analysis]
- **Brakes.** (1) Person pause is unchanged and wins over everything; (2) the person or coordinator
  clears any flag with one action, effective at the next request, with the reason stored; (3)
  `flags --open` and the panel are the first thing shown after a pause; (4) a detector's authority is
  narrowing only: it can raise but never clear a `restrict` or `suspend`, and the sentinel token cannot
  promote anyone. A dead sentinel raises no flags; the node's own counters (open-join 3.8) still
  refuse new `open` enrolments. [analysis]

## 6. Implemented vs missing, with slices

| Piece | State |
|---|---|
| strike on the parent, injection standing, message quarantine, caps, claim sweep callbacks, address watch, weakened-assertions hook | implemented, separate, unrecorded as one thing [V] |
| `flag` entry kind, rules, folding, rate limit | missing |
| CLI `flag`, `flags`, `flag-clear`; `digest --status` lines; panel; heads-up kinds | missing |
| agent detectors in section 4 | missing; inputs exist for `forged_sender`, `injection`, `identity_spike`, `claim_hoarding`, `tests_weakened` (hook); inputs missing for `reverted_landing` (M1), `second_process`, `lease_churn` (lease branch) |
| flag-to-capability link in the node | missing |
| ledger link | missing |

| # | Slice | Size | Tests | Depends on |
|---|---|---|---|---|
| F1 | `flag` entry kind in the node: schema, rights by level, folding, clear entry, stamped `raised_by`, rate limit | M | agent cannot raise `restrict`; `raised_by` cannot be supplied; clear points at the first; fold keeps one line | node network branch |
| F2 | CLI `flag`, `flags`, `flag-clear`, `digest --status` lines | S | each verb against a real node; `HumanGrant` refusal under an agent marker | F1 |
| F3 | Effects: agent level cap (L1/L0) read by landing, coordination and lease paths; device caps in open-join slice 2 | M | a `restrict`ed agent's landing is refused with the flag id; clear restores at the next request | F1, earned-trust levels |
| F4 | Detectors with existing inputs: `forged_sender`, `guard_edit`, `auth_failures`, `refused_actions`, `injection`, `flooding`, `claim_hoarding`, `identity_spike`, `tests_weakened` | M | each on a synthetic clock; honest-use corpora raise nothing; evidence references resolve | F1; sentinel `devices.py` pattern |
| F5 | Detectors that need new inputs: `reverted_landing`, `second_process`, `lease_churn`, `tamper`, `behaviour_change` | M | per detector; `tamper` against a real edited sealed file | F1, M1 landing record, lease branch |
| F6 | Panel and heads-up kinds, `explain.WHY` | S | `test_sentinel_explain` finds a sentence per kind; panel route calls the same verbs | F2 |
| F7 | Ledger link: flag -> `reputation_event`; clear keeps both; detector precision | M | a `suspend` demotes at once; a cleared false alarm is counted | F1, earned-trust ledger slices |
| F8 | Red-team: an agent tries to raise `suspend` on a rival, to forge `raised_by`, to flood notes, to clear its own flag, to evade a detector by splitting across identities | M | each refused or folded, with the evidence line | F1 to F4 |

Order: F1, F2, F4 (inputs that exist), F3, F6, F8, then F5 and F7.

## 7. Owner decisions

1. **Who may raise `restrict`?** (a) coordinator and person, detectors per table [recommended];
   (b) person only; (c) any agent with a second agent's concurrence.
2. **Can an agent see who flagged it?** (a) sees code and reason, not an agent raiser's name [recommended];
   (b) sees everything; (c) sees nothing but the level.
3. **Which detectors may `suspend` without a human first?** (a) the five security/tamper/forged codes in
   section 4 [recommended, matches the 2026-10-08 decision]; (b) forged_sender only; (c) none, restrict
   only until the person looks.
4. **Note expiry.** (a) 7 days; (b) never, until cleared; (c) 24 hours.
5. **Do host-device flags suspend the agents on the device?** (a) note only [recommended]; (b) restrict;
   (c) suspend.

## 8. Not verified

Nothing here was run. I did not read `Quarantine.hold` beyond its header, the whole of `service.py`, or
whether `agent_invites` strikes are cleared anywhere. The 64 cap is `agents_live`/`board_members` in
`limits.py`, defaults that `limits.json` can change.
