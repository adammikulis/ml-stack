# Earned trust: delegated control from the reputation ledger

Status: design, 2026-10-08. Nothing in sections 3 to 7 is built. Owner statements this note serves:
"the whole point of having a trust/reputation ledger is so agents can be trusted", and "we're
eventually going to have to give agents more control over things as projects get larger. doing it in
a trusted, observable manner where we can hit the brakes is what's key." Added principle: give agents
the benefit of the doubt by default, and make that safe with strong anti-forgery and observability.
Generosity is bought with unforgeable evidence and fast brakes, not with waiting periods.

Tags on every claim: **[V path]** verified by reading that code on `0.2dev` at `173e52f1`, or on the
named branch; **[D doc]** stated in a repo document, not re-checked; **[analysis]** my inference or a
proposal. Branches read: `worktree-agent-a969015e912acecf3` (attribution, `work_dimensions.py`,
`03d09c55`), the `worktree-agent-a274f998cf72ffbfb` checkout (`session_name.py`, unique names, not
committed at the time of reading), `worktree-agent-a359e808db0d18e70` (`docs/service.md`).

## 0. Summary

- The ledger today measures one thing well: independently reviewed task outcomes (accepted, rejected,
  infrastructure-blocked), credited once, bound by hashes to the exact spec, proposal and review.
  [V `workspace/task_credit.py`, `reputation/work.py`]
- It measures almost nothing else about an agent: not landed commits, not reverts, not test-claim
  honesty, not claim hygiene, not guard refusals or red-team findings. The only agent-attributed
  misbehaviour event is `injection_flagged` on a sender. [V `workspace/standing.py`]
- Nothing reads the ledger to widen or narrow an agent's control. Standing is "recorded evidence, no
  authority" by design. [D `docs/reputation.md`] [V `work_reputation.standings` returns `authority: none`]
- Proposal: five levels, a new verified identity starts at L2 (land after independent review), L3 and
  the narrow relaxations are earned from the same ledger, L4 is never standing-derived, brakes are
  person pause, coordinator takeover and automatic suspension on a small set of anomalies.
## 1. What the ledger records today, per agent identity

### 1.1 Stores

- **Work ledger.** One AES-256-GCM sealed graph per user, key in the OS keystore, profile
  `reputation`. Nodes `work_evidence`, `work_contribution`, `work_award`, `work_verification`,
  `work_quality_review`, `work_rating`, `work_usage`, `work_agent`, `work_task_reference`. Capped at
  5000 evidence and 5000 contribution nodes. [V `reputation/sealed.py`, `reputation/work.py`]
- **Source-risk ledger.** The same package and sealed file family, kinds `host`, `url`, `hash`,
  `peer`, ... with events (`denial`, `scan_hit`, `injection_flagged`, `hash_change`, ...), a 6 hour
  short term, a 30 day half-life, states `good/watch/bad`. [D `docs/reputation.md`]
  A workspace sender is the kind `peer`, key `workspace:<id>`. [V `workspace/standing.py`]
- **Workspace audit and board logs.** Hash-chained JSON lines (`sha256(prev + body)`), `verify(anchor)`. [V `workspace/chain.py`, `service.audit_verify`]
- **Coordination graph.** `coordination.db` holds tasks, proposals, reviews, `work-credit-reference`
  and family/device account edges. [V `workspace/task_credit.py`, `family_accounts.py`]

### 1.2 Signals, by the list the owner named

| Signal | Recorded today? | Where, and how it is measured |
|---|---|---|
| Verified completed work | Yes, as accepted reviewed tasks, not as landed commits | 10 credits per accepted task, up to three 5-credit quality tiers, once per workspace, agent and task. [V `reputation/economy.py` `BASE_CREDITS`, `reputation/work.py` `record`] No node ties an award to a git commit, a landing, or whether the work later survived. [analysis] |
| Review outcomes | Yes | Every independently reviewed attempt is a `work_contribution` with outcome `accepted`, `rejected` or `blocked_infrastructure`; reliability is Beta(1,1) over accepted versus rejected, infrastructure blocks add no negative sample. [V `reputation/economy.py` `summary`] Quality is the mean of reviewer 0 to 100 ratings with a prior of weight two. |
| Test-pass honesty | No | A worker's claimed checks are replaced by the reviewer's checks (`proof['checks'] = task_schema.checks(review['checks'])`); the claimed value is not compared or scored. [V `task_credit.py`] A rejected review lowers reliability but does not distinguish "tests failed honestly" from "claimed a pass that was false". [analysis] Notes carry `test-verified` only when a recorded verify run exited 0 and has not rotted. [V `workspace/notes.py` `trust_of`] |
| Claim hygiene | No | Claims expire or are swept when the holder pid is dead and callbacks `on_swept` and `on_stolen` exist; a grep of `src/ml_stack` for writers into the ledger finds none wired from them. [V `workspace/claims.py`; grep of `observers.observe`] |
| Failures | Partly | Rejections count against reliability. A task carries a `failures` counter and `blocked_seconds`. [V `workspace/taskboard.py` create] Infrastructure blocks are deliberately neutral. |
| Reverted work | No | No code path in `reputation/` or `workspace/task_*.py` mentions a revert. [V grep] |
| Security findings | Barely | One event: `injection_flagged` against `workspace:<id>` when a message is held as a hard marker; it makes later messages from a `watch` or `bad` sender get held. [V `workspace/standing.py`, `workspace/service.py` line 243 to 247] Hook guard refusals, red-team findings, `auth.denied`, claim steals are audited but not attributed into the ledger. [analysis] |
| Model and runtime of the work | Yes on the attribution branch only | `task_provenance.snapshot` overwrites claimed provenance with the broker lease's model, device, runtime, `broker_runtime`; `work_dimensions.views` derives overlapping per-agent and per-model-runtime views from the same reviews, never awards twice. [V on `03d09c55`] On `0.2dev` the lease model, allocation id, device and runtime are already overwritten in `submit`, with the worker's claim kept as `claimed_provenance`. [V `workspace/task_actions.py`] |
| Resource use | Only if a reviewer measures | Worker-reported usage is not a measurement; unknown stays null. [V `reputation/economy.py` `summary`, `task_credit.py`] |

### 1.3 What the accounts group

- A **family account** is created only when the serving model id matches a known local family
  (`qwen`, `gemma`, `gpt-oss`); `GENERIC` returns no account. [V `workspace/family_accounts.py` `binding`,
  `client/families.py`] A Claude or other hosted model therefore earns evidence keyed by agent identity
  only, with no family rollup. [V by reading `binding`; the label table has only those three]
- A **device account** is person-enrolled, one per physical device. [V `workspace/device_accounts.py`;
  D `docs/reputation.md`]
- AGENTS.md says economic accounts aggregate by model family across devices and a family switch does
  not rebucket history. [D `AGENTS.md` "The main session and its agents"] The code matches that only
  for the three local families. [V above]
- Standing is therefore three overlapping things: per registry identity (`work_agent`), per family
  account (local models), and, on the attribution branch, per exact model plus runtime plus artifact
  or device. [V `work_reputation._accounts`; `work_dimensions.views` on `03d09c55`]

### 1.4 What coordinator eligibility reads today

`coordinator_eligible` is true for a main, non-helper session with the agent capabilities whose exact
model id is listed in `model_tiers.json` at a non-lowest tier **and** whose model state is `verified`;
otherwise a reason string. The ledger is not consulted and there is no tie-break beyond that.
[V `workspace/agent_display.py`, `workspace/model_tiers.py`]

## 2. What is forgeable, and what verifies it

### 2.1 Identity and attribution

- **Sender is server-stamped.** A token authenticates against a registry of SHA-256 token hashes
  with `hmac.compare_digest`; the `Identity` the server builds from the registry row is what is
  written as `from`. Message and task writers use the authenticated identity, not a body field.
  [V `workspace/identity.py` `authenticate`, `workspace/service.py` message path]
  Attribution is sound on a single coordinator. [analysis]
- **A token is a bearer secret.** Anything that holds it is that identity; the token travels in
  `ML_STACK_WORKSPACE_TOKEN`. [V `identity.py` `TOKEN_ENV`] A compromised agent process inherits
  all standing of its identity. Mitigation is revocation plus brakes, not detection. [analysis]
- **Names are chosen at mint.** `valid_name` limits shape and reserves `human`, `owner-token`,
  `system`, `all`. A person (or lead, for helpers) picks the name. [V `identity.py`]
- **Unique session names.** `session_name.py` derives `family-<6 hex>` from a hash of harness plus
  native session id and lengthens on collision. Its own docstring says "The suffix is presentation,
  never authority." [V `session_name.py` in the `a274f998` checkout] So standing must key on the
  registry id, never on a display string; two sessions can share a family word. [analysis]
- **Model id is a claim until verified.** The registry stores `model_state` in
  `verified`, `claimed`, `inherited`; only `verified` reaches a tier. Subagents inherit the parent's
  device and label it inherited. [V `workspace/modelid.py`, `model_tiers.tier_of`; D `AGENTS.md`]

### 2.2 Per signal

| Signal | Who can forge or game it | Independent verifier today | Gap |
|---|---|---|---|
| Accepted task credit | A worker can submit a proposal; it cannot review itself (`who.id == worker` refused, review must bind `proposal_hash` and `review_hash`). [V `task_credit.py`] | Reviewer authenticated live, hashes recomputed. In `open` enforcement a reviewer is any registered project member other than the worker; in `strict`, a person-granted one. [V `taskboard._reviewer`, `workspace/enforcement.py`] | Reviewer and author can be two sessions of one person's machine or one mind; the device-account check applies only when both are enrolled. [V `taskboard.py` line 76 to 78] |
| Quality bonuses | The reviewer sets them; each links to passed checks and hashed artifacts. [V `economy.assessment`] | None beyond the reviewer. | A friendly reviewer inflates. No reviewer-leniency measure. |
| Model/runtime | A worker could claim any model. | Lease-based overwrite from the broker. [V `task_actions.submit`, `task_provenance.snapshot` on `03d09c55`] | Hosted models: `artifact` is `unknown`; "verified" is "the broker lease said so". |
| Ledger contents | A same-user process could try to edit the sealed file. | AES-GCM authentication with a keystore key; `Tampered` if it fails and no earlier copy holds. [V `reputation/sealed.py`] The keystore is a person's floor. [D `AGENTS.md` "System settings and the authority registry"] | Sealed against outside edit, not against a rollback to an older valid snapshot. [analysis] |
| Audit and board logs | Same-user process rewriting the file. | Hash chain; `verify(anchor)` detects mid-chain edits. [V `chain.py`] | The digest is unkeyed SHA-256: a full rewrite from genesis is consistent unless an external anchor exists. [V `_digest`] |
| Human-only steps | An agent claiming to be the person. | Registry roles; `mint` is person-only; `HumanGrant` is 120 s and minted after a terminal confirmation; `_FORBIDDEN` keeps agent tool arguments off sentinel, authority and log state. [V `workspace/service.py` mint; D `docs/person-delegation.md` section 1] | Delegated gates pass on an environment marker (`CLAUDECODE`, `ML_STACK_AGENT`) that the model's own process sets. [D `docs/person-delegation.md` section 1] That is the weakest link for any control tied to "agent acted". |
| Test claims in commit messages and reports | Free text. | A reviewer's independent checks, the `test-verified` note trust, and `scripts/test` receipts where present. [V `notes.py`; D `AGENTS.md` "Saying that something works"] | No scored comparison of claim versus verified result. |

## 3. Proposed mapping from standing to delegated control

### 3.1 Principle

Default generous, verified evidence widens, a hard fault narrows at once. Friction (a wait, an
extra review, a count) is kept only where it catches forgery or misuse or bounds blast radius. A
check that only makes the agent wait is dropped. [analysis, owner principle]

### 3.2 Levels

| Level | Control | Who has it |
|---|---|---|
| L0 | Propose only: read, message, write notes, submit proposals; no worktree writes outside a scratch, no landing. | After a demotion or suspension; never the default. |
| L1 | Edit and test in its own claimed worktree and branch; claim; ask for review. | An identity whose model is `claimed` or `unknown`, or whose parent has not granted a project. |
| L2 | L1 plus land to the development branch after one independent review. The review is the check; no count is required. | The default for a registered agent with a `verified` model state and a project grant. |
| L3 | L2 plus coordinate others: create and assign tasks, designate reviewers, claim lead-class resources, review as an economic reviewer, and (separately) coordinator eligibility. | Granted on evidence. |
| L4 | Release-class actions: push to `main`, tags, release, force operations, key operations, system settings. | Never from standing. Each action needs the person's own words/approval. [D `docs/person-delegation.md` sections 7 and 12] |

L2 is the default because `AGENTS.md` already makes the agent land its own reviewed work, and the
review verifies the work whoever wrote it. [analysis]

### 3.3 Evidence to enter, windows, demotion

| Level | Enter | Verified by | Stay / decay | Immediate demotion |
|---|---|---|---|---|
| L1 | Registered identity with a project grant. | The person's mint and grant. [V `mint`, `enforcement`] | None. | to L0 |
| L2 | Model state `verified` and the identity is bound to a live broker allocation or a device account. | Registry plus broker lease. [V `modelid`, `resource_allocations`] | None: no count and no wait. A landing without independent review is not possible at L2. | Any item in 3.4, or a reviewer-confirmed false claim, drops to L1. |
| L3 | At least 5 accepted tasks, from at least 2 distinct reviewers, in the last 60 days, at least 1 review rated 70 or more on quality, with reliability at least 80 with its sample count shown, **and** no demotion event in 30 days. Starting values, to be tuned from data. | Reviewers independent of the worker and of each other; one of them a person, or a project-granted peer on a different device account. | Evidence older than 60 days stops counting toward entry; it does not erase standing already held unless inactivity exceeds 90 days, which returns the agent to L2 until one new accepted task. | Any item in 3.4 drops to L2 at once; two in 30 days drop to L1. |
| L4 | Not available. | Person only. | n/a | n/a |

The 60 day, 5 task and 2 reviewer figures are proposals; the tier fits the existing Beta reliability
and confidence terms, which already report sample counts. [V `economy.summary` confidence =
n / (n + 2)] They protect against one lucky review and one friendly reviewer; the L2 default carries
no similar numbers because the independent review per landing is the control. [analysis]

### 3.4 Events that demote at once

Each needs a record that a script can check, not an opinion: [analysis]

1. A **revert** of a commit the identity landed, when the revert's message or reviewer marks it
   a defect (not an unrelated rollback). Needs a landing record (section 3.6, slice 2).
2. A **security finding** attributed to the identity: a guard hook refusal for a forbidden action
   (push to main, force, `git add -A` is not security), an `injection_flagged` hard hit, an attempted
   human-only action, a claim on a path outside its grant.
3. A **forged claim**: a review whose independent checks contradict a recorded claim the worker made
   about the same checks (tests said passed, reviewer's run failed on the same selector), or a
   provenance claim that differs from the broker's lease.
4. A **tamper signal**: `Tampered`, a chain `Verdict.ok == False`, a token used from a second
   process while the first is live (when added).
5. **Brake use**: a person pause is not a demotion; it is a hold (section 3.5).

Rejection of a reviewed task is **not** a demotion event; it moves reliability, which is how a
normal failure should look. Infrastructure blocks stay neutral. [V `economy.summary`]

### 3.5 The brakes

1. **Person pause.** A single action by the person at a terminal or in the Fleet page sets an
   identity, a project or the whole workspace to L0 for a bounded time. It is a record in the
   enforcement store, not a message: `enforcement.set_mode` is the pattern (lead or person only, audited, helper
   refused). [V `workspace/enforcement.py`] Extend it from project mode to a per-identity cap.
2. **Coordinator takeover.** The coordinator (or the person) can revoke a token and its tree
   (`revoke(name, tree=True)`) and reassign a task's worker. [V `Workspace.revoke` signature in
   `service.py`; `task_authority.authorize`] Takeover is a reassignment of the claims, not a new
   power. [analysis]
3. **Automatic suspension on anomaly.** Any of section 3.4 items 2 to 4 sets the identity to L0
   pending a person or coordinator look, with a notice raised through the sentinel's single-flight
   dialog. [D `docs/reputation.md` "The notice"] Items 1 and 3 drop one level and notify.
   Suspension must be reversible by one person action and never delete evidence. [analysis]

### 3.6 Anti-forgery mechanisms behind the default generosity

Existing means [V]; missing means [analysis]. Ranked by how much generosity each missing mechanism
unlocks; the order of the M rows is the ranking: M1, M2, M3, M4, M5, M7, M6.

**Needed for L1 and L2 (the generous default)**

| Mechanism | State | Note |
|---|---|---|
| Server-stamped sender and per-identity token hash | Exists [V `identity.py`] | Unlocks everything; rests on a single coordinator. Remote coordinators: unchecked. |
| Independent review bound by proposal and review hash | Exists [V `task_credit.py`] | The verification behind L2 landing. |
| Lease-overwritten provenance | Exists on `0.2dev` for model, device, runtime [V `task_actions.py`] and richer on `03d09c55` | |
| Revocation with tree | Exists [V `service.revoke`] | The brake for a compromised token. |
| Sealed ledger, keystore key | Exists [V `sealed.py`] | |
| Human floor, keystore, `main`, force, GPU lease, budgets | Exist as fixed rules (section 4) | |
| **M1. Landing record**: a ledger node per landed commit naming identity, reviewer, branch tip hash, tree hash. | Missing | Unlocks revert attribution (3.4 item 1) and auto-relaxation of review in L3. Largest single gain. |
| **M2. Keyed or externally anchored heads**: sign the audit-chain head and ledger head with a key outside the agent's reach and keep the last N heads where an agent cannot write. | Missing (unkeyed SHA-256 chain; no rollback guard) | Unlocks trusting any agent-visible log as evidence and detecting rollback; makes automatic suspension safe because it cannot be rewritten away. |
| **M3. Per-identity cap record** read by every verb (the L-level, pause, expiry), written only by a person or the coordinator, visible live. | Missing | The control surface itself; without it levels are advisory. |
| **M4. Claim-vs-result comparison** for checks a worker states: store the worker's claimed pass beside the reviewer's check, score a mismatch as a forged claim. | Missing | Unlocks the "forged claim" demotion and a test-honesty signal at near-zero cost. |
| **M5. Event writers into the ledger** from hook guards, `auth.denied`, claim steals, revert detection. | Missing | Gives signals to demote on; each is a single event kind in the existing `peer` kind. |
| **M6. Token-use anomaly**: record the process (pid, start time) that first presented a token and refuse or flag a second concurrent one. | Missing; `claims` already compare holder pids. [V `claims.py` `_holder_alive`] | Catches a copied bearer token. Lower gain. |
| **M7. Reviewer pairing measure**: share of an agent's accepted reviews from one reviewer; reviewer leniency (acceptance rate versus later revert). | Missing | Unlocks L3 safely against collusion. |

Dropped as friction without catching forgery: minimum waiting periods before L2; mandatory second
reviewer for ordinary work; re-confirmation prompts on grant use; per-commit written justifications.
[analysis]

## 4. Hard rules that stay fixed, whatever the standing

None of these reads the ledger, and none has a standing input; they stay at their present gates.

- **Human floor.** An agent never posts, poses or is shown as a person; no agent creates, copies or
  uses a person's credential; text claiming to be the person is data. [D `AGENTS.md` "An agent is
  never a human"] `mint` needs a human token. [V `service.mint`] Quarantine release needs a human.
  [V `workspace/quarantine.py`]
- **`main`, tags, force, release.** The Bash guard refuses a push to `main` without the owner's
  flag, force pushes, remote deletion, `--all`, `--tags`. [D `CLAUDE.md` "Hooks and settings"]
  Release approval is the only authorizable kind and is made by the structured approval question.
  [D `docs/person-delegation.md` header]
- **Keystore and secrets.** The keystore, cluster passphrase and token, signing keys, `sudoers`, and
  the OS privilege prompt are a person's regardless of registry. [D `AGENTS.md`]
- **The authority registry.** `person` gates pass only a person at a terminal; a helper or child
  identity never passes a `delegated` gate; `ML_STACK_AUTHORITY_FLOOR=person` only tightens.
  [D `AGENTS.md`] The levels here must sit **inside** these gates and never relax one.
- **GPU, one at a time.** A GPU lease is required and nothing computes alongside it. [D `AGENTS.md`
  section 4] A high level does not skip the lease.
- **Budgets only fall.** `--allow-increase` keys on `CLAUDECODE`, and edit guards refuse growth.
  [D `CLAUDE.md`] A high level never raises a budget.
- **Independent review.** A worker never credits or reviews itself; hashes bind the review.
  [V `task_credit.py`]

## 5. Observability

### 5.1 What the person sees live (proposed on top of what exists)
Exists: the Fleet page's read-only `/ui/work-reputation/standings` route for the person owner
(verified completions, credits, reliability, evidence 20 at a time) [V `work_reputation.fleet_route`];
the `authority show` and audit logs [D `AGENTS.md`]; `ml-stack-security status` and chip for source
risk [D `docs/reputation.md`]. Proposed: [analysis]

- **Grants in force.** One table: identity, level, source of the level (default, earned, capped),
  expiry, last change and by whom.
- **Who used them.** Last use per capability, with the audit row id: landings, claims, task creations,
  coordinator actions. Click opens the row (and its commit once M1 exists).
- **Anomalies.** Anything in 3.4, a chain verdict, a ledger `Tampered`, reviewer concentration, a new
  identity that jumped to L3 faster than the median. A badge, and a sentinel notice for suspensions.

### 5.2 The sealed record

- Each level change is one `peer`-kind ledger event with `from`, `to`, `reason`, evidence ids, and the
  actor (rule, coordinator or person); appended, never edited. [analysis]
- Evidence stays in the existing sealed work graph. Level state is **derived** from evidence plus
  caps, as balances already are ("derived from that evidence rather than a separate balance file").
  [D `docs/reputation.md`]
- Heads are signed and anchored outside the agents' reach (M2). Until then the record is sealed against
  outsiders and not against a same-user rewrite, and this note does not claim more. [V `chain.py`, `sealed.py`]

## 6. Coordinator eligibility and tie-break from the same ledger

Today: eligibility is model tier plus verified state, with no tie-break and nothing from the ledger.
[V `agent_display.py`, `model_tiers.py`]. Proposed, one ledger: [analysis]

1. **Gate (unchanged).** Main session, agent capabilities, model verified and not lowest tier.
2. **Level gate (new).** The identity is at L3. A tier-eligible identity below L3 cannot coordinate
   while an L3 one exists.
3. **Tie-break among eligible.** Order by: higher reviewer-diverse accepted count in the last 60 days
   (distinct reviewers capped, so one friendly reviewer is worth little), then lower
   reverts-per-landing (M1), then reliability with more samples, then longer continuous liveness, then
   registry id for determinism. Show the keys on the page so the choice can be read.
4. **No bootstrap hole.** When nobody is L3, the tier rule alone decides (today's behaviour), and the
   person can name the coordinator.
5. **Takeover.** The coordinator is demoted like anyone else; a coordinator suspended to L0 hands its
   claims to the next eligible identity or the person.

Do not add a second table: use the same sealed graph.

## 7. Abuse cases

| Case | What happens today [V] | Defence in this design |
|---|---|---|
| **Sybil: many new identities to farm standing** | Minting needs a person (or a lead for helper agents); ids are unique per workspace. [V `identity.py` `MINTS`] Helper children are bound to the parent token tree. | Standing never accrues to the parent from a child's own review. Rate-limit lead mints per window and show the mint count; the L3 rule counts distinct reviewers who are not that identity's own spawn. L2 default means a fresh identity is not trying to earn anything for ordinary work. |
| **Collusion: reviewer and author** | The worker cannot review itself; same-device-account pairs are refused when enrolled; open mode allows any registered project member other than the worker. [V `taskboard.py`] | M7 reviewer concentration and leniency; L3 needs a reviewer who is a person or a project-granted peer on a different device account; reviewer rating contributions are weighted down when a reviewer's accepted work is later reverted (M1). `strict` enforcement gives person-set grants. |
| **Model upgrade resets or inherits standing** | The agent identity (`work_agent`) is unchanged by a model change and keeps its evidence. Local families bind to a family account and a "family switch does not rebucket"; hosted models have no family account; on `03d09c55` the per-model-runtime view is keyed by exact model plus runtime plus artifact/device, so an upgrade starts a new row there. [V `family_accounts.py`, `work_dimensions.py`] | Level follows the **family account where one exists, otherwise the agent identity**, consistent with AGENTS.md. A new exact model in a known family keeps the family standing but L3 requires one accepted task on the new model before L3 coordination is re-granted (a distinct-model check, not a wait); a new family starts at L2. The per-model view is reported, not used to reset. Needs a decision (Q3). |
| **Compromised agent** | A stolen token acts as the identity, up to its caps and its parent's. Revocation kills it and children. [V `identity.authenticate`, `service.revoke`] | Brakes (3.5); M6 detects a second user; M2 prevents history rewriting; the fixed rules in section 4 mean even a top-standing compromised agent cannot push `main`, read the keystore or act as the person. Landings need independent review, so the blast radius at L2 is a reviewed change. |
| **Ledger poisoning by a reviewer** | The reviewer chooses ratings and bonuses. | M7 leniency; a person can void an award with an appended correction event, not an edit. |

## 8. Gaps and slices, with size

Sizes: S under a day, M one to three days, L more. [analysis]

| # | Slice | Size | Unlocks |
|---|---|---|---|
| 1 | M3: per-identity cap record, read by a single `ws._may` path, written through `enforcement`-style person/lead verbs, shown in Fleet. | M | The levels exist; pause works. |
| 2 | M1: landing record at the integration step (`workspace/task_integration.py`, `integration_git.py` already integrate) writing identity, reviewer, tips. Also revert detection. | M | Revert signal, L3 safety, coordinator tie-break key. |
| 3 | M4: store claimed-vs-independent check results on the proposal/review; mismatch event. | S | Test honesty and forged-claim demotion. |
| 4 | M5: event writers from the Bash and edit guards, `auth.denied`, claim steals into the `peer` kind. | M | Security-finding signal. |
| 5 | Level derivation and demotion engine over the existing graph, plus the brake events and sentinel notice. | M | Automatic suspension. |
| 6 | M2: signed, anchored heads; rollback guard on the sealed ledger. | M to L (needs a key placement decision) | Safe automatic action; evidence for the commercial pool. |
| 7 | Land `work_dimensions` and attribution from `03d09c55` and decide model-upgrade policy. | S (landing) | Per-model views. |
| 8 | M7 reviewer concentration and leniency measures. | S to M | L3 safe against collusion. |
| 10 | Coordinator tie-break from the ledger in `agent_display`/coordinator selection. | S | Section 6. |

### Shared pieces with the commercial pool (`docs/service.md`)

That note builds on the same ledger: verified completion credit, quality bonuses, reliability and its
advisory 0.8 to 1.2 price modifier, the sealed store, device and family accounts, and resource
attribution, and calls out `work_dimensions` as unlanded. [D `docs/service.md` on `a359e808`, section
3.1] The shared pieces are: the signed anchored heads (M2), the landing/verified-work record shape
(M1, which for providers is a verified job result), the per-identity cap record (M3, which for a
consumer or provider is a quota and a suspension), the event writers (M5), and the overlap-view
rule that one award is never counted twice. The difference: the private pool trusts the person who
paired it, the open market trusts nobody, so M2, M6 and redundant verification are required there and
only "wanted" here. Keep one ledger; do not fork a "market reputation".

## 9. Owner decisions (2026-10-08)

Each decision records the option taken and the option rejected.

- **Start level.** A new, model-verified, project-granted agent starts at L2 (land to dev after one independent review). Rejected: L1 (own worktree only until its first accepted review), because it makes every new agent wait for a review round before it is useful.
- **L3 review.** No relaxation: independent review stays on every landing, and L3 adds coordination only. Revisit after the landing record (M1) and reviewer concentration (M7) exist. Rejected: post-hoc review of docs-only and tests-only changes, because tests can weaken guards and the checks that would catch it do not exist yet.
- **Model upgrade.** An upgrade within a family keeps standing, and one accepted task on the new model is required before L3 coordination. Rejected: keeping standing completely, because an unproven model could coordinate at once.
- **Automatic suspension.** Rules suspend to L0 on security, tamper and forged-claim events and notify; the person lifts it. Rejected: notify only, because a real forgery would keep acting until someone reacted.
- **Head anchor.** The external anchor for signed heads lives in the OS keystore under the person's gate, signed by a key agents cannot read. Headless devices follow the passphrase-wrapped key file decision in docs/pool-encryption.md. Rejected: an append-only file in a path agents cannot write, because it rests on filesystem permissions a same-user process can often bypass.
- **Coordinator eligibility.** L3 standing is preferred for the coordinator role, and a fresh pool needs no hand-named coordinator: when no eligible live agent is L3, the best eligible live main session by model tier coordinates (lowest tier of any vendor, subagents and unverified models stay ineligible), and the role passes to an L3 agent as soon as one exists. A person can still name a coordinator at any time. Rejected: requiring L3 with a hand-named first coordinator, and tier-only eligibility with the ledger as tie-break alone, because the first leaves a fresh pool without a coordinator and the second never prefers earned standing.

## 10. Proposed wording for AGENTS.md (not edited here)

Under "The main session and its agents": "Delegated control follows an identity's recorded standing and
never exceeds a fixed rule. A new verified agent starts able to land independently reviewed work;
coordination is earned from independently reviewed outcomes; a revert, a security finding or a forged
claim lowers the level at once; the person can pause any identity, project or the workspace; release-class
actions, `main`, tags, force, the keystore, the GPU lease and the budgets do not depend on standing."
