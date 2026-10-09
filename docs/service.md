# A commercial distributed-compute service on the pool

Status: plan, 2026-10-08. Nothing here is built except what section 3 marks as implemented. The
owner decided to treat the pool as a real product: strangers contribute devices (providers), others
send work (consumers), alongside private-pool use. This note is a plan and a set of questions, not
legal, tax or financial advice.

Every claim carries a tag:

- **[code]** verified in this tree (`0.2dev` at `c1007d3b`, read 2026-10-08), file named.
- **[docs]** stated in a repo document, not re-checked against code.
- **[ext]** external fact, with source and date. Web figures come from aggregator and blog pages,
  not vendor pages, and move weekly; none is a benchmark or a forecast.
- **[plan]** a proposal, not a fact.

`docs/pool-encryption.md` and `docs/home-pool.md` now sit beside this note and carry the pool's key
model and device plan; the decisions of 2026-10-08 are in section 9.1. `model_work_attribution` and
`work_dimensions` exist only on unlanded branches (commits `133dd374`, `03d09c55`;
`feat/board-history-pages` and three more) and are marked below. **[code]**

## 1. Product definition

Two modes, one system.

| | Private pool | Open market |
|---|---|---|
| Who | Devices of one person, family or team who already trust each other | Providers and consumers who have never met |
| Trust | Pairing with a shared cluster key and pinned TLS today [code] `fleet/tls.py`, [docs] `docs/fleet.md`; per-device keys and revocation replace the shared key [plan] `docs/pool-encryption.md` | None assumed; every party untrusted |
| Money | None; runs are free, credits are a record [code] `reputation/economy.py` | Priced, settled, paid out |
| Operator | Nobody; no host [docs] `docs/mesh-board.md` (design note, "nothing is implemented") | An operator for accounts, payments, abuse |
| Workloads | Anything the owner runs | Only what survives an untrusted provider (section 4) |

Roles: **provider** (contributes a device), **consumer** (sends work), **operator** (runs accounts,
payment, abuse handling, optional matching). One person can hold all three. **[plan]**

### The wedge

The product that already works is not a marketplace; it is a pool that serves local models on
the devices of people who trust each other, with work that is independently reviewed before it is
credited. **[code]** `workspace/task_credit.py` The honest wedge, in order of strength:

1. **Managed private pools for small teams** (phase 2): a team's own laptops and office machines
   serving a local model to its own members, with no data leaving, plus the invite, metering and
   isolation a team lacks. Consumer and provider are the same party, so confidentiality needs no
   attestation. This is the nearest customer and the lowest legal exposure.
2. **Invited guests** (phase 1): a trusted outsider uses spare capacity on a pool.
3. **Open market** (phase 3): last, because every hard problem (section 4) arrives with it.

Whether the open market is a niche worth entering is argued in section 2.4.

## 2. Landscape

All rows [ext], searched 2026-10-08 (standard web search). Prices are what third-party pages
reported, are rough snapshots, and disagree between sources.

### 2.1 GPU rental marketplaces

| Service | Model | Reported price points | Trust model |
|---|---|---|---|
| Vast.ai | Host-set marketplace; on-demand, interruptible, reserved, billed per second | RTX 4090 about $0.34-0.50/h; H100 about $0.90/h unverified hosts, about $1.50-1.87/h datacenter-verified | Verified-host tier versus unverified; the renter trusts the host |
| RunPod | a community tier (third-party hosts, no formal SLA) and a secure tier (operator-run datacentres, SLA) | RTX 4090 about $0.34/h community, about $0.69/h secure; H100 about $1.99/h community, $2.89-2.99/h secure | Two tiers, split on who owns the hardware |
| Lambda | Operator-owned datacentre | H100 SXM about $3.99-4.29/h | Operator is the provider |

Sources: [spheron.network comparison](https://www.spheron.network/blog/gpu-cloud-pricing-comparison-runpod-vs-vastai-2026/),
[spheron.network RunPod vs Vast.ai](https://www.spheron.network/blog/runpod-vs-vastai-2026/),
[hivenet RunPod guide](https://www.hivenet.com/post/runpod-pricing-complete-guide-to-gpu-cloud-costs),
[costbench Vast.ai](https://costbench.com/software/ai-gpu-cloud/vast-ai/), all read 2026-10-08.

What this shows: strangers' consumer GPUs already rent for well under a datacentre price, and the
market itself splits into "cheap, unverified host" and "dearer, accountable host". The cheap tier
sells raw GPU time in a container; the renter carries the trust risk.

### 2.2 Decentralised compute projects

Akash (reverse auction, providers bid on containerised workloads, Kubernetes providers), Render
(began as 3D rendering, now AI inference too), io.net (large GPU clusters), Bittensor (subnets
compete to produce outputs). One source describes a Bittensor subnet using "proof-of-compute and
hardware attestation"; another says many such systems are off-chain compute with token incentives
and reputation, "not fully trustless". I could not find sourced detail on how Akash, Render or
io.net check that a job ran correctly. Sources:
[digitalwealthinsider](https://digitalwealthinsider.beehiiv.com/p/decentralized-gpu-computing),
[blockeden forum](https://blockeden.xyz/forum/t/decentralized-compute-for-ai-akash-render-and-the-gpu-shortage-solution/316),
[hackernoon, Nodexo on Bittensor](https://hackernoon.com/nodexo-crosses-50-verified-gpus-on-bittensor),
read 2026-10-08. Treat all as unverified until the projects' own documentation is read (an open
task before phase 3).

### 2.3 Serverless GPU and managed inference

| Service | Billing | Reported price points |
|---|---|---|
| Modal | Per second of GPU, CPU and memory | H100 about $0.001097/s ($3.95/h); $30/month free credit |
| Replicate | Per second for custom hardware; per token or output for official models | H100 about $5.49/h |
| Together AI | Per million tokens serverless; hourly dedicated; batch at an introductory 50% off | Most common models $0.27-3.00 per million tokens; dedicated H100 reported $2.99-5.49/h across sources |

Sources: [morphllm Replicate vs Modal](https://www.morphllm.com/comparisons/replicate-vs-modal),
[blaxel Modal pricing](https://blaxel.ai/blog/modal-pricing-alternatives-guide),
[morphllm Together pricing](https://www.morphllm.com/together-ai-pricing),
[eesel Together guide](https://eesel.ai/blog/together-ai-pricing), read 2026-10-08. The sources
conflict on Together's dedicated rate.

### 2.4 Where a pool of home and office devices differs, and whether the niche is real

Differences this system actually has: local-model serving with admission control [docs]
`docs/serve-admission.md`; work credited only after an independent review that cannot be the worker
[code]; a mesh with no host [docs, design only]; sandboxing with deny-by-default egress [code]
`sandbox/policy.py`. What it lacks relative to every row above: tenants, prices, settlement, uptime
SLAs, any redundant execution or hardware attestation (a grep for `attestation`, `redundan`,
`spot.check` over `src/` finds nothing relevant [code]).

Honest assessment, not a forecast:

- Raw GPU hours from strangers: **crowded and price-led** (2.1, 2.2). A home pool has nothing there
  that Vast.ai or RunPod Community does not already sell, and they have the payments, KYC and abuse
  desks. Entering on price alone is not a niche.
- Token-priced inference of open models (2.3): competitors serve it from datacentres at cents per
  million tokens. Home devices rarely beat that on cost; they could on **privacy** (data stays on
  devices the customer trusts) and **locality** (a team's own machines).
- The plausible niche is therefore **private and semi-private pools** (section 1 wedge 1 and 2),
  where the competitor is "nothing, or a cloud bill the team does not want". Whether enough teams
  want this is unknown; the phase 1 and 2 gates test it before any open-market build.

The open market is the most speculative part of the plan. **[plan]**

## 3. Architecture on what exists

### 3.1 Inventory: the economy that exists

Metering, credits and reputation exist and are the base. Gaps are commercial extensions of this
system, not a second one. State per piece:

| Piece | Where | Implemented and tested | What it measures; who attests |
|---|---|---|---|
| Verified completion credit | `workspace/task_credit.py` (71 lines); `tests/test_task_credit.py` | Yes. One immutable award per workspace, worker and task; refused if the reviewer is the worker, or if proposal or review hashes mismatch | 10 base credits for an accepted task [code `reputation/economy.py` `BASE_CREDITS`]; attested by an independent reviewer: the person, the worker's task-creator parent, or a peer with a person-set grant [docs `docs/reputation.md`] |
| Quality bonuses | `reputation/economy.py` `assessment` | Yes. Up to three 5-credit tiers (`validated`, `regression`, `impact`), each linked to passed checks and hashed artifacts | Reviewer reasons; no test-count credit [code] |
| Work reputation | `reputation/economy.py` `summary`; `workspace/work_reputation.py`; `tests/test_work_reputation.py` | Yes. Quality: reviewer 0-100 ratings with a prior of weight two. Reliability: Beta(1,1) over accepted versus rejected outcomes; infrastructure blocks add no negative sample | Independent review outcomes; sample counts and confidence returned with every score [code] |
| Advisory price modifier | `economy.summary` `pricing` | Computed only. Bounded 0.8-1.2, neutral at start, `price` 0, `active_charging` False, baseline access guaranteed | Derived from reliability and confidence [code] |
| Ledger storage | `reputation/work.py` `WorkLedger`, `reputation/sealed.py` | Yes. Encrypted graph under a keystore key; capped at 5000 evidence and contribution nodes; a retry returns the original award, no duplicate | Balances are derived from evidence, no separate balance file [docs] |
| Device and family accounts | `workspace/device_accounts.py`, `workspace/family_accounts.py` | Yes. One person-enrolled account per physical device; evidence rolls up by model family across devices; workers cannot choose an account | Enrolment is by the person; family derived from the model id [code] |
| Resource attribution | `workspace/resource_allocations.py`; `tests/test_resource_allocations.py` | Yes. An allocation binds worker, task, device and a live broker grant; `verified_binding` refuses if the broker holder, enrolment or execution config changed | Broker-observed: holder pid and start time, lease id, model and runtime provenance are immutable on the task [code] |
| Measured usage | `economy.assessment` `usage` | Yes, thin. `tokens_in`, `tokens_out`, `wall_seconds`, each bounded or null | Counted only when a reviewer supplies it; worker-reported usage is not a measurement; unknown stays null [code, docs] |
| Actor and model-runtime views | `workspace/work_dimensions.py`, `tests/test_model_work_attribution.py` | **Unlanded.** On `133dd374` and `03d09c55`, not on `0.2dev` or this branch | Overlapping views over the same reviews: per actor, per exact model plus runtime plus artifact; derives, never awards twice [code on those commits] |
| Source-risk reputation | `reputation/store.py`, `docs/reputation.md` | Yes, separate from work reputation. Events, a 30-day half-life, watch/bad states | Only the system's own observations move a score [docs] |

### 3.2 What is missing for commercial use

Each is an extension of 3.1. None is built. **[code]** by absence unless stated.

| Gap | State today | Extension, in this system |
|---|---|---|
| Price and exchange rate | `price` 0, mode `free` hard-coded in `economy.summary` | Make mode and a credit-to-currency rate policy data on the award's recorded policy version (awards already store policy version and currency [docs]) |
| Settlement | Nothing moves; `spent` is 0, balance equals earned | A spend record next to awards in the same graph, so balance stays derived |
| Payout | No redemption enabled [docs] | Operator-run (section 3.6); needs a payment rail (section 9) |
| Disputes | A rejected review is final; no appeal path | A dispute node linked to the review, resolved by a second independent reviewer |
| Per-tenant accounts | No `tenant` identifier anywhere in `src/` (grep, zero hits). Accounts are per device and per model family | A tenant id on the allocation and the award, rolled up beside the device account |
| Fraud resistance | One independent reviewer per task; a reviewer that colludes with a worker is not detected | Redundant execution and spot checks (3.4); reviewer reputation |
| Redundant execution | None | 3.4 |
| Metering granularity | Tokens in and out and wall seconds only, reviewer-supplied | Broker-side metering of GPU seconds and memory per allocation, since the broker already observes the lease |
| Credit-to-resource link | Credit is per accepted task, not per resource used | A resource-weighted price that reads measured usage, never the worker's claim |

### 3.3 Tenancy and roles

Today the unit of identity is a workspace plus an agent, and the unit of an account is a device
(`workspace_id` in `workspace/coordination.py`). A reputation store lives under one user
(`reputation/u-<id>/graph.enc`) [docs]. A "tenant" for the service is a new, outer key: a tenant owns
consumers, providers and tokens, and every allocation and award carries its id. Roles map onto
the existing roles in `workspace/identity.py` (a person, parents and children) rather than a new
model; the provider is the enrolled device account, the consumer is a requesting identity. **[plan]**

### 3.4 Job verification

What exists: independent review of a task by a different identity, bound by hash [code]. What a
stranger provider needs on top, cheapest first: **[plan]**

1. **Redundant execution** of a deterministic, small sample of jobs on a second provider, compare
   outputs, and refuse credit on mismatch. Needs deterministic decoding (fixed seed, temperature
   0) and a tolerance rule, because floating point differs across devices.
2. **Spot checks**: replay a known-answer job (canary) among real jobs; a failure drops reputation
   and withholds payout (3.7).
3. **Attestation**: a hardware root of trust on the provider. Not in this repo; consumer GPUs
   mostly lack it. Without it, section 4 limits what may be sent.

Verified-work accounting stays independent of who ran it: the award already names the reviewer and
binds proposal and review hashes, not the worker's claims. **[code]**

### 3.5 Isolation and keys

Exists: encrypted-at-rest reputation graph under a keystore key [code]; a TLS certificate per daemon
pinned by peers, with discovery sealed under a key derived from the cluster key [docs/code
`docs/fleet.md`, `fleet/tls.py`]; an OS sandbox with deny-by-default files, network, programs and
environment [docs `docs/sandbox.md`]. Gaps: one cluster key means every paired member can read
every other member's traffic, so it is a *private-pool* primitive only [docs `docs/fleet.md`]; there
are no per-tenant keys, caches or model-cache separation. The per-tenant key plan from
`docs/pool-encryption.md` cannot be summarised because that file is absent. **[plan]** per-tenant
data keys, a per-tenant KV and prompt cache that is never shared across tenants, and a job
envelope encrypted to the provider's key.

### 3.6 Control plane: mesh versus operator

The design intent is a mesh with no host, peers with full replicas, a deterministic merge
[docs `docs/mesh-board.md`, marked "nothing is implemented"]. A commercial service needs an
operator. Split: **[plan]**

| Operator-run (centralised, replaceable) | Peer to peer (stays in the mesh) |
|---|---|
| Accounts, sign-up, KYC where required | Job data path: consumer to provider direct, or via relay |
| Payment collection and payout | Device pairing and TLS pinning inside a private pool |
| Matching and price discovery (optional; a private pool needs none) | Verified-work evidence replicated as signed records |
| Abuse handling, takedown, dispute arbitration | Local sandbox enforcement on the provider |
| Credit-to-currency policy | Review of work within a pool |

Rules: the operator is never in the data path of private-pool traffic, is optional, and could be
replaced by another operator that reads the same signed evidence. A pool with no operator keeps
working. A consumer's job body is never readable by the operator.

### 3.7 Scheduling, pricing and holdback

Scheduling exists: `fleet/pool.py` chooses a peer from capability and measured rates (`Rates`,
keyed on machine id); the broker leases models and admission [code, docs]. Pricing is a 0.8-1.2
advisory modifier from reliability (3.1). **[plan]** For provider reputation: a new provider holds
back a fraction of earnings until it has N verified outcomes; a failed canary or mismatched
redundant result forfeits the holdback and drops reputation; no stake or cryptographic slashing
until phase 3. This reuses `work_reputation` reliability and confidence.

## 4. Trust and abuse

### 4.1 Untrusted provider

A provider sees what its device runs. Without hardware attestation (a TEE) it can read prompts,
weights and outputs, and can return wrong answers. So, **[plan]**:

- Open-market provider: only non-sensitive work, or work that is encrypted and verified end to
  end, with the consumer informed which. No claim of confidentiality is made for stranger-run
  inference without attestation.
- Verification is required, not optional (3.4), because a provider paid per task is paid to cheat.
- A provider's local owner is also the attacker; the sandbox protects the provider *from* the
  job, and the job from the provider only with attestation.

### 4.2 Untrusted consumer

- **Sandbox escape.** All foreign work runs in `ml_stack.sandbox`, which is deny-by-default
  [docs `docs/sandbox.md`]. Backends: macOS seatbelt, bubblewrap, a Linux container runner
  [code `sandbox/`; the container runner is on `b1a12013`, check it has landed before relying on it].
- **Network egress.** Default `Net.deny()`; no consumer job gets the network unless the policy
  names ports [code `sandbox/policy.py`].
- **Crypto-mining and abuse of compute.** Wall-clock and CPU limits exist [code `Limits`]; a job
  type allowlist (inference of named models only in the open market) is the stronger control.
- **Illegal content.** Cannot be detected by the provider side if jobs are encrypted; this is an
  operator and terms-of-service problem (section 5) and a reason for narrow job types.
- **Denial of service.** Per-tenant rate limits; `workspace/rates.py` has a per-sender sliding
  window that is a starting point [code].

### 4.3 Dispute handling

Consumer disputes a result, provider disputes a rejection. Evidence already exists (hashed
proposal, hashed review, artifacts [code]); the missing part is the second-reviewer path (3.2).
Final arbitration is an operator function and a legal one.

## 5. Money and legal: questions for a lawyer

Not advice. Each is a question to put to counsel in the chosen jurisdiction.

1. **Payments.** Is a card processor (with a marketplace or connected-account product) permitted
   for paying out to individual providers, and which fees and holds apply? Is crypto settlement
   acceptable, and under what rules?
2. **Money transmission.** Does holding consumer funds until a job completes, or crediting
   provider balances, make the operator a money transmitter or require a licence? Does a
   processor's connected-account product shift that?
3. **Tax reporting.** What must be collected and reported for providers (income forms, thresholds)
   and for consumers (sales tax or VAT on compute services) across the jurisdictions involved?
4. **KYC and sanctions.** What identity checks are required for providers who are paid and
   consumers who pay? How are sanctioned persons and regions excluded?
5. **Terms of service.** What must the provider terms say about hardware wear, power cost,
   performance and prohibited use, and the consumer terms about prohibited content, indemnity and
   suspension?
6. **Liability.** Who is liable when a provider's device is damaged, a result is wrong, a job
   exfiltrates data, or illegal content passes through a provider's machine? Does an operator
   position as a mere conduit?
7. **Data residency and privacy.** Do personal-data rules apply to prompts that cross borders to a
   stranger's device? Is the operator a controller or processor, and what is the provider?
8. **Entity and jurisdiction.** Which legal entity and where (section 9)?
9. **Employment and worker status.** Does paying individuals for device time create any
   worker-classification question?
10. **Export controls.** Do restrictions on GPUs or model weights apply to providers abroad?

### Licence of this repo

`LICENSE` is Apache License 2.0 and `pyproject.toml` declares `license = "Apache-2.0"` [code]. In
plain terms, to be confirmed by counsel: it permits commercial use, modification, distribution and
running as a hosted service; it grants a patent licence from contributors; it requires keeping the
licence and `NOTICE`, stating changes, and does not grant trademark rights (so a product name and
brand are separate decisions). `NOTICE` records a port of a third-party project under the MIT
licence (`src/ml_stack/spec/` and its data files), which carries its own copyright-notice
requirement; `pyproject.toml` also lists `THIRD_PARTY_NOTICES.md` [code]. Model weights served on
the pool have their own licences (some restrict commercial use or hosting) and are not covered by
this repo's licence; that needs a per-model check. Closed operator code on top of Apache-2.0 code is
permitted; whether to open it is decision 4 in section 9.

## 6. Phased plan

Each gate is "true, measured how". No revenue figures. Sizes are rough effort in agent-weeks of one
focused lead, **[plan]**.

| Phase | Scope | Size | Gate to leave it |
|---|---|---|---|
| 0 Private pools | The home pool: your own paired devices, free runs, credits as a record | in progress | A pool of 2+ real devices completes reviewed work end to end with credits recorded; measured by the existing credit and allocation tests plus a live cross-device run |
| 1 Invited guests | A named guest runs work on a pool; per-guest metering and isolation; still free | 4-6 | A guest identity cannot read another's data or cache (tested against a real second identity); every guest job's usage flows through the existing credit and allocation path with a tenant id; a real guest used it for two weeks |
| 2 Managed pool, small teams | Operator-run accounts and invites for a team's own devices; first money (subscription or credit purchase); abuse and support | 8-12 | One paying team runs for a month with the operator able to onboard and suspend; settlement and refund tested against a real sandbox payment account; counsel has answered section 5 for the chosen jurisdiction |
| 3 Open market | Stranger providers and consumers; redundant execution, canaries, holdback, disputes; narrow job types | 20+ | Redundant execution catches a provider that returns wrong output in a red-team run; a hostile consumer job fails to escape or egress in the redteam suite; payout and tax reporting live; the section 2.4 niche question has evidence from phases 1-2 |

**Stays out, by phase.** Phase 0-1: payments, public sign-up, stranger providers. Phase 2: stranger
providers, open pricing, crypto. Phase 3: confidentiality claims for unattested providers,
training jobs, arbitrary container workloads (inference of named models only to start).

## 7. What changes in this repo's rules

`AGENTS.md` today says "There are no users but the owner ... unless he says otherwise" and bans
migrations for state nobody holds and branches for callers nobody has. **[code]** `AGENTS.md`
line 779. Under the owner's decision the rule changes by phase, not at once. Proposed wording for
the lead to add (this note does not edit `AGENTS.md` or `CLAUDE.md`). **[plan]**

```
**There are no users but the owner until a release phase says otherwise** (docs/service.md,
section 6). Phase 0 (the owner's own pools) is current: no migration, no compatibility, no
fallback for state or callers nobody holds. From the first release that an external person
runs (phase 1), three things start and only those: a persisted format or wire message that
leaves the owner's machines changes only with a version and a migration; an interface an
outside party calls is not renamed without a note in the release notes; a security-relevant
default stays refused. Everything else in the paragraph above still holds: no hypothetical
consumer, no wrapper for an old name, no fallback "just in case". The first external release
is named by the owner and recorded here. Design for the service now without building for it:
review new features against the checklist in docs/service.md, "Assumptions to bake in now".
It is a prompt, not a gate.
```

## 8. Assumptions to bake in now

A design checklist for whoever designs or reviews a feature. **It is not a gate.** No budget, no
required test, no blocking check; it must never slow ordinary development. Use it as a review
prompt, and when you notice a violation in code you are not changing, record it in `HANDOFF`
rather than fixing it on the spot. Fix opportunistically when you touch the file anyway.

Cost: S small, M a few files, L a design change. "Opportunistic" = can be done when the file is
touched, without its own task.

| # | Check | Existing code (verified) | Cost | Opportunistic |
|---|---|---|---|---|
| 1 | Providers and consumers are untrusted | Complies: sandbox deny-by-default; reviewer cannot be the worker (`task_credit.py`). Violates: one shared cluster key trusts every member (`docs/fleet.md`) | L for the cluster key; S elsewhere | Yes for new code |
| 2 | No single-owner or single-user assumption in ids, paths, stores or UI | Violates: reputation store is `reputation/u-<id>/graph.enc`; a device has one person-enrolled account; one `owner` token file (`work_reputation.fleet_route`) | M | Yes, when a store path is edited |
| 3 | Every job and request carries a tenant id | Violates: no `tenant` in `src/` (grep, zero hits); allocations carry worker, task, device only (`resource_allocations.assign`) | M | Yes, add the field as optional on new records |
| 4 | Every job's resource use flows through the existing credit and attribution path; do not add a parallel ledger | Complies: allocation, `verified_usage`, `economy.summary` (3.1). Violates: usage has only tokens and wall seconds | M | Yes: extend `economy.assessment` usage fields, never a second store |
| 5 | Keys and caches are per tenant | Violates: one keystore profile `reputation`; one cluster key; no per-tenant KV or prompt cache separation | L | No, design first |
| 6 | No host or central authority in the data path; an operator is optional and replaceable | Complies by design intent (`docs/mesh-board.md`, design only). Violates: workspace state is one `coordination.db` behind a file lock on one machine (`resource_allocations.py`) | L | No |
| 7 | Verified-work accounting is independent of who ran it | Complies: awards bind proposal and review hashes and refuse a self-review (`task_credit.py`); `work_dimensions` (unlanded) keeps views derived | S | Yes |
| 8 | Egress and sandbox policy default-deny for foreign work | Complies: `Net.deny()` is the policy default (`sandbox/policy.py`, `docs/sandbox.md`). Not checked: whether every launcher routes through it | S to M | Yes, route a launcher through it when touched |
| 9 | Secrets never in logs or on the board | Complies in part: `src/ml_stack/redact/` exists, keystore holds keys. Not audited across all log sites | M to audit; S per site | Yes |
| 10 | Clocks and ordering do not assume one machine | Violates: `held(.../coordination.lock)` is a same-machine file lock; `verified_at` is a local time. Design intent is a hybrid logical clock (`docs/mesh-board.md`, not built) | L | Only new code; avoid new wall-clock ordering |

## 9. Owner-only decisions

Each is one question; the cost is what choosing it commits.

1. **Who is the first customer?** (a) A named small team using its own devices (cheapest, tests
   wedge 1); (b) invited guests on the owner's own pool; (c) an open beta to strangers. Cost: (a)
   needs phase 1-2 tenancy and billing; (b) needs only phase 1; (c) needs all of section 4 first.
2. **Which payment rail?** (a) Card processor with connected accounts (handles most payout and
   tax-forms plumbing, fees per transaction, processor risk review); (b) crypto settlement (fewer
   gatekeepers, harder legal and tax position, a known sanctions burden); (c) invoice only, no
   self-serve, at phase 2 (no engineering, no scale). Cost: switching rails later means
   re-settling balances.
3. **Which jurisdiction and entity?** (a) A company in the owner's home jurisdiction; (b) a company
   elsewhere chosen for payments or privacy law; (c) no entity yet, free phases only. Cost: (a) and
   (b) legal fees and ongoing filings; (c) caps the plan at phase 1.
4. **Open-source or closed operator code?** (a) Open everything (consistent with the Apache-2.0
   repo, lowest trust cost for providers, copyable by competitors); (b) open the pool, close the
   operator (accounts, payments, abuse tooling); (c) close all new commercial code. Cost: (b)
   needs a clean repo boundary now; (c) changes the owner's stated relationship to this repo.
5. **Name and brand.** (a) Keep the working name; (b) rename (the existing
   `docs/poolside-refactor-plan.md` already plans a product rename); (c) a separate service brand
   over the open pool. Cost: the licence gives no trademark right, so a trademark search precedes
   any public use; a rename later touches every public surface.

### 9.1 Decisions (2026-10-08)

**Commercial direction: a real product from the start; section 5 is a checklist, not a gate.** Decided:
the pool is built as a real product from the first phase, and the lawyer questions in section 5 are worked
through as a checklist alongside the build rather than a gate that holds engineering until counsel has
answered. The phases are 0 to 3 as in section 6, and a phase's own gate in that table still applies before
leaving it (counsel's answer for the chosen jurisdiction is still a phase 2 gate before money moves).
Rejected: treating the service as an experiment until the legal position is settled, because it would keep
the verification, tenancy and accounting work unscheduled and the legal answers depend on what is built.

**Ledger path: signed journals and verified accounting now; an operator Merkle log at phase 2; no own chain
and no own token.** Decided, following 10.4: option 1 now, option 2 at phase 2, option 3 optional later,
option 4 never. Rejected: a chain or token of our own (option 4), because verification (3.4) is the hard
problem and no ledger solves it, and a token adds securities, money-transmission and custody exposure for no
trust the Merkle log does not already give. Option 5 (a payout choice at phase 3) remains tied to the
payment-rail question (question 2) and counsel.

**Constraint (2026-10-08): nothing on the plan depends on a paid Apple developer account until the
product takes off.** Hardware attestation on Apple silicon (app attest, a signed helper, notarized builds)
and anything that needs APNs or an iOS app are **blocked until the product takes off**. Consequence for the
phases: guests get Level 1 tenancy only (`docs/pool-encryption.md`, section 8.1), so every phase that
mentions confidentiality from a provider rests on the isolation and review of 4.1 and 4.2, not on Apple
attestation, and Level 2 on Apple silicon is not attempted now (decision of the same date).

## 10. Ledger options

The owner asked for the strongest honest case for a blockchain ledger under the compute economy,
then a red-team of it against this system. Tags as above.

### 10.1 The case for a shared ledger (steelman)

Among providers and consumers who trust no operator, a public ledger gives: **tamper-evident
balances** (no party can quietly edit history); **permissionless settlement** (anyone can pay
anyone without a processor approving them or a country's card network reaching them); **portable
reputation** (a provider's record is not locked inside one operator, so leaving the operator
costs nothing); and **no custodian** (funds sit in escrow code or in the parties' own keys, so
there is no operator balance sheet to lose, freeze or abscond with). For a market whose point is
having no host, this is the only option that removes the operator from the money path entirely.
That is a real property, and none of the other options offers it.

### 10.2 Red-team against this system

- **Oracle problem.** A chain cannot see that compute happened. Whether a job ran correctly is
  decided off-chain by redundant execution, attestation or spot checks. This repo has none of
  those [code: grep for `attestation`, `redundan`, `spot.check` over `src/` finds nothing
  relevant], only independent human or peer review of a task [code `task_credit.py`]. A chain
  records the verdict; it does not produce it. The decentralised projects in 2.2 face the same
  limit, and one source says many are off-chain compute with token incentives, not trustless
  [ext, 2.2, 2026-10-08].
- **Cost, latency, throughput.** Reported 2026 layer-2 fees are about $0.001-0.10 per transaction
  depending on the chain and source; layer-1 finality is about 16 minutes; optimistic rollups
  hold withdrawals 7 days, zero-knowledge rollups under an hour [ext: [eco.com L2 comparison](https://eco.com/support/en/articles/14798699-best-ethereum-l2s-in-2026-fees-tvl-tps-compared),
  aggregator pages, read 2026-10-08; the sources are marketing-heavy and the Ethereum
  Foundation roadmap figures for faster finality are plans, not current]. A cents-priced
  inference job cannot carry a per-job on-chain settlement without batching, and batching is an
  off-chain ledger again.
- **Token and securities exposure.** A native token raises whether it is a security, a commodity
  or a payment instrument (lawyer question L1). A US federal payment-stablecoin framework exists
  (P.L. 119-27), takes effect no later than 2027-01-18, and limits who may issue; it was silent in
  my sources on state money-transmitter interaction [ext: [CRS overview](https://www.everycrsreport.com/files/2026-08-20_IN12553_0b329134f6be0cc12f78017358552f4053d52c5e.html),
  [Gibson Dunn](https://www.gibsondunn.com/the-genius-act-a-new-era-of-stablecoin-regulation/), read 2026-10-08].
- **Key custody.** A provider who is not technical loses funds by losing a key and has no
  recovery; the existing system keeps secrets in the OS keystore on a managed device [code
  `reputation/sealed.py`]. A wallet per provider is a new support burden and a phishing target.
- **Fraud and collusion.** A ledger makes a fraudulent award permanent. Sybil providers, a
  consumer colluding with a provider to mint credit, and a collusive single reviewer are not
  stopped by consensus (3.2: one reviewer per task today).
- **No host.** A chain removes the operator from settlement only. Accounts, abuse handling, KYC
  where the law needs it, and disputes stay human work (3.6, 4.3, section 5). A public chain also
  makes every payment and balance public, which conflicts with consumer privacy.

### 10.3 Options compared

Criteria: what it trusts; cost to build and run; regulatory exposure as questions for a lawyer;
reversibility; fit by phase. Option 1 is partly design: the per-device signed journals are in
`docs/mesh-board.md`, "nothing is implemented" [docs]; the only Ed25519 code in `src/` verifies
release signatures (`fleet/signing.py`) [code]; no Merkle-tree code exists [code, grep].

| | 1 Journals + verified accounting | 2 Operator log, Merkle transparency, processor | 3 Option 2 + root anchored on a public chain, no token | 4 Own chain or rollup, token | 5 Stablecoin settlement, no own chain |
|---|---|---|---|---|---|
| Trusts | Each device for its own entries; the independent reviewer for credit [code] | The operator not to equivocate; auditors and witnesses catch it | As 2, plus a timestamp the operator cannot backdate | Validators, the token's economics, bridge code | The issuer, the chain, and the operator for verification |
| Build and run | Exists in part: credit path [code]; journals to build | Moderate: log service, signed tree heads, inclusion and consistency proofs (the formats in [RFC 9162](https://www.rfc-editor.org/rfc/rfc9162.html), read 2026-10-08); processor integration | Option 2 plus one periodic chain transaction; fees small | Highest: protocol, validators, audits, liquidity, ongoing security | Moderate: wallet or custodian integration, payout flow; chain fees per payout batch |
| Lawyer questions | None for money (free runs); privacy of evidence | Processor and connected-account rules; money transmission when holding funds (5.2) | As 2; any issue from using a public chain for timestamps only | Is the token a security or commodity? Is the operator an exchange or issuer? Licences per jurisdiction | Is the operator a money transmitter or virtual-asset provider? Which issuer rules apply? Sanctions screening of wallets |
| Reversibility | High: ledger is derived from evidence | High: a fiat refund is possible; the log itself is append-only | High: dropping anchoring loses only a timestamp | Low: a token, once held by strangers, is a liability | Medium: payouts final on-chain; can stop offering |
| Fit | Phase 0-1 | Phase 2 | Phase 2-3 if auditors ask for an external anchor | Phase 3 at earliest; unsupported by any phase gate | Phase 3, for providers who want to be paid without a card rail |

The witness idea for option 2: independent parties cosign the log's tree head so a log that shows
different views to different clients is caught; RFC 9162 itself says such protection is outside its
scope and a log is otherwise a trusted third party [ext: RFC 9162; [CoSi paper](https://arxiv.org/pdf/1503.08768), read 2026-10-08]. The
open question of what a single cosignature proves is live in the Sigsum community [ext, same
date], so witness policy must be designed, not copied.

### 10.4 Recommendation, and what would change it

Build to option 1 now, add option 2 at phase 2, and treat option 3 as an optional later step.
Do not build option 4. Offer option 5 as a payout choice at phase 3 only if the payment-rail
decision (section 9, question 2) and counsel allow it. Reasons: the hard problem is verification
(3.4), which no ledger solves; the existing credit path already derives balances from reviewed
evidence; a Merkle log over that evidence gives tamper-evidence without a token or a custodian
problem; fiat via a processor gives refunds and lawyers a familiar regime.

What would change it: (a) providers refusing any operator-held balance, which moves option 5 up;
(b) a customer or auditor requiring a public timestamp, which promotes option 3; (c) counsel
finding that holding funds needs a licence the owner will not seek, which pushes settlement to
direct consumer-to-provider stablecoin transfers; (d) evidence that real, verification-grade
redundant execution works at a cost below the job price, without which none of the open-market
options is safe regardless of ledger.

### 10.5 A pluggable settlement boundary (proposal, not built)

The credit path stays the single source of truth for what was earned (`task_credit.verify_task`
writes one award; `economy.summary` derives balance [code]). Settlement is a boundary after it:

```
class Settlement(Protocol):
    name: str                                    # "none" | "ledger" | "processor" | "stablecoin"
    def quote(self, award: Award, tenant: str) -> Money | None: ...   # price from award + measured usage
    def hold(self, tenant: str, quote: Money, ref: AwardRef) -> HoldId: ...     # idempotent on ref
    def settle(self, hold: HoldId, award: Award) -> SettlementReceipt: ...      # idempotent
    def release(self, hold: HoldId, reason: str) -> None: ...                  # refund or dispute outcome
    def receipt(self, ref: AwardRef) -> SettlementReceipt | None: ...
```

Rules: `AwardRef` is the existing immutable award id, so a retry returns the original receipt (the
award is already once-only [code]); the default `none` backend is today's free mode; a receipt is a
node linked to the award in the same encrypted graph, so balance stays derived and no parallel
ledger appears; a backend may reject a tenant but may never change an award or a review.
