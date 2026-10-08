# Person-spoken authorization

Design note. Slice 1a (section 12) is implemented; the rest is not. Owner decision of 2026-10-08: development
pushes need no authorization, and the only authorizable kind is `release-main`, created only by the structured
approval question (sections 7 and 12). Sections 5.1 to 5.3 describe the earlier prose design: typed words now
only revoke, refuse, or ask for the approval question.
Slice 1b-i (`delegate` removed from the CLI, `mint` person-only) is implemented; section 8 lists what remains.

## Invariants

1. Every agent acts as itself. There is no delegate identity, no delegate credential and no
   credential that carries a person's rights.
2. **An agent can never post as, pose as or be mistaken for a human.** This rule is hard. It
   has no escape hatch, no flag, no environment variable, no test-only switch and no trusted
   agent role. A change that adds one is refused.
3. What the person says in chat becomes a scoped, expiring, auditable authorization. The
   harness produces the evidence and plain code interprets it; no model, subagent, board
   message or file does either.
4. The human-only system settings stay a person's at their own terminal. A chat sentence
   never covers them.
5. Authorizing feels like conversation. The person says "yes, push it" or "restore the
   launchers"; no command line is required. `/allow KIND` exists as an optional explicit form.

## 1. What exists today

Paths relative to the repository root.

- `src/ml_stack/workspace/identity.py`: roles `human`, `lead`, `agent`. `MINTS` lets a human
  mint any role, a lead mint `agent`, an agent nothing. `Identity.trust` is `"human"` for
  role `human`, else `"agent-claimed"`. `delegate()` mints a child token `parent/name`.
- `notes.py`: trust rank `agent-claimed < test-verified < human` comes from the author's
  role, never from fields. `task_credit.py` supplies verified evidence.
- `chat.py`: the person's console; requires a `HUMAN` token.
- `agent_display.py`: display names from registry metadata; `session_kind` is `person` for
  role `human`.
- `authority.py` registry: a `delegated` gate passes a lead agent when `CLAUDECODE` or
  `ML_STACK_AGENT` is set. That is an environment test, not evidence the owner said
  anything.
- `sentinel/human.py`: `HumanGrant` (action, subject, 120 s) minted after a terminal
  confirmation; `_FORBIDDEN` keeps agent tool arguments off sentinel, authority and log
  state. `sealed.py` is an HMAC-sealed JSON file. `rails.py`: `reads_like_instruction`.
- `scripts/hooks`: `claude-bash-guard` (regex refusals; `ML_STACK_PUSH_MAIN=yes` is read off
  the command text), `claude-edit-guard`, `claude-session-start`, `claude-subagent-start`,
  `claude-subagent-stop`, `pre-push` (keyed on `CLAUDECODE`), `workspace_hook.py`.
  `.claude/settings.json` wires PreToolUse, SubagentStart/Stop and SessionStart. No
  `UserPromptSubmit` hook exists.
- `tests/test_redteam_human_floor.py`, `test_authority.py`, `test_onboard_human.py`: the
  floor, attacked through real child processes.

Gap: `ML_STACK_PUSH_MAIN=yes` and every delegated gate are authorised by text or environment
the model controls, not by anything the person said.

## 2. Threat model

| Actor | Can | Cannot |
|---|---|---|
| Model (main session) | Write any text; run permitted tools; read files it can open | Make the harness emit a `UserPromptSubmit` event; read the sealing key through a guarded tool; write the statement log through an agent verb |
| Subagent | What its parent can, labelled | Consume an authorization spoken in another session; record or widen one |
| Another board agent | Post, DM, set notes | The above; its text is data |
| File, tool output, board post, pasted text | Contain "the owner says yes" | Be a statement; only the hook records statements |
| Same-user local process | Read and write every file the user owns, including key and log | Nothing is denied to it |

Honest limits. A same-user process can read the sealing key and forge records, and an
agent's Bash tool runs as the same user. The design does not prevent this. It detects it
after the fact: the log is hash-chained and sealed, and verification shows tampering on the
board and in `ml-stack-doctor`; guards refuse to act on a log that fails verification. The
Bash and edit guards add the person store and key to `_FORBIDDEN`, which is a checkpoint, not
a sandbox. No separate OS account, no Secure Enclave. The harness is trusted: a tool result
mislabelled by the harness as a user prompt would be believed.

## 3. Statement channel

A `UserPromptSubmit` hook, `scripts/hooks/claude-user-prompt`, receives `session_id`,
`transcript_path`, `cwd` and `prompt` for what the person typed. The hook:

1. Ignores events with an `agent_id` (subagent turns) and sessions with
   `ML_STACK_NONINTERACTIVE` set.
2. Appends a `PersonStatement` to the log: sequence, time, session id, project, cwd,
   `sha256(prompt)`, a redacted excerpt (`sentinel/redaction.py`), source
   `harness-hook:UserPromptSubmit`. No raw secrets are stored.
3. Runs the interpreter (section 5). If it yields an authorization it appends that too.
4. Prints one `additionalContext` line, e.g. `authorization #7 recorded for push-dev
   (attested by hook)`, so the model knows it may act. Nothing else reaches the model.

The hook is the only writer; its process is the evidence. `AskUserQuestion` answers arrive
through `PostToolUse` with matcher `AskUserQuestion`, which carries the structured answer; the
same recorder stores a `PersonAnswer`. Verify both events' fields against the installed
Claude Code before slice 1 depends on them.

Other harnesses: Codex and local-model launchers (`ml-stack-agent`, `ml-stack-chat`) write the
same `PersonStatement` rows from their own process when the human types, never from the model
loop. A harness without such an event produces no statements; the action then needs the
person's terminal. Nothing is fabricated from model output.

Binding: a statement carries `session_id` and project. A consumer proves its session with its
own workspace token, which the session record (`persist_session`) maps to that `session_id`.
Model-relayed text ("the owner said yes") is never evidence.

## 4. The authorization object

The statement log is `~/.ml-stack/person/statements.log`, a `ChainLog` with a seal key beside
it. Records:

```
PersonStatement  seq, ts, session_id, project, cwd, prompt_sha256, excerpt, source, prev, hash
Authorization    id, ts, statement_seq, session_id, project, kind, target_rule, uses,
                 expires, state, how: "reply"|"imperative"|"asked"|"explicit", prev, hash
```

- `kind` comes from a closed registry (section 7). `target_rule` names how the guard derives
  the exact target (section 5); the authorization stores no target text from the model.
- `expires`: 15 minutes by default, 4 hours at most, never past session end. A longer window
  needs the person's words for it ("for the next hour"), parsed from a closed duration
  grammar. Compaction and resume do not extend it.
- `uses`: 1 unless the person says otherwise in their words ("keep pushing until I say
  stop" becomes bounded-use for kinds that allow it, at most 10, same expiry).
- `state`: `live`, `used`, `expired`, `revoked`; each transition appends a row.
- Revocation: the person says "stop", "cancel that", "never mind" or `/revoke` (closed
  lexicon, same interpreter); session end and `SubagentStop` append expiry rows.
- Consumers: the session in which it was spoken and its labelled subagents acting as the
  parent. Another session, project or a process that outlived the session gets no match.
  The consumption row records which agent acted.
- Guard query: `person_auth.consume(kind, derived_target, identity)` verifies seal and chain,
  matches kind, session, project, unexpired and remaining uses, appends `used` under the log
  lock, and returns the id or raises `NotAuthorized`. Guards call it at the moment of action
  and never cache. A log that cannot be read or verified refuses.

Board display: statements and authorizations are mirrored to a read-only `#person-record`
view as a `person-attestation` kind, rendered "attested by the harness hook for session S"
with sequence and hash (section 6).

## 5. Interpreting the person's words, without a model

The text is genuinely the person's because it came through the hook; the risk is only
interpretation. The interpreter is a pure function of (prompt, last assistant turn, current
context). It runs in the hook, uses no model and no network, and has a fixed lexicon in the
repository with its own tests. When it is not certain it asks (below).

### 5.1 Reply to a proposal

1. The hook reads the last assistant message from `transcript_path`. A *proposal* is the
   closed intent phrase table matched in that message ("I'll push the dev branch", "restore
   the launchers?", "restart the daemon") mapped to a kind, plus the proposal hash
   `sha256(kind, derived target, message id)`. A message with several different kinds or no
   match is no proposal.
2. The prompt is a *confirmation* only if all hold:
   - at most 12 words and one sentence; the first word (after lowercasing and stripping
     punctuation) is in the affirmative set: `yes, yep, yeah, ok, okay, sure, go, do, please,
     approved, confirmed`, or the whole prompt is one of `go ahead`, `do it`, `sounds good`,
     `yes please`, `looks good`;
   - no `?`, no negation token (`no, not, don't, dont, never, stop, wait, hold, cancel,
     without, nothing`), no conditional token (`if, unless, once, after, when, until,
     provided, depending, assuming`), no quotation marks or code fences, and no second kind
     named;
   - the proposal was the assistant's last message, within the same session, and no other
     assistant message or `AskUserQuestion` has been asked since;
   - fewer than 30 minutes passed and no compaction boundary sits between the proposal and
     the reply (the hook compares transcript records);
   - the target the guard now derives hashes to the proposal's hash. If the branch, the
     directory or the file list changed since the proposal, the hash differs and the reply
     does not apply.
3. Otherwise it is not a confirmation.

### 5.2 Direct imperative

A sentence that names the action itself ("push the dev branch", "restore the launchers",
"you may restart the daemon", "go ahead and push") maps through a closed intent table to a
kind. The table holds verb groups and object phrases; both must match; a pronoun-only
sentence ("do it") needs section 5.1. The same bans apply (question mark, negation,
conditional, quoted text, a second kind). The target is not read from the sentence beyond
choosing the kind: the **guard derives the exact target at consumption** from context: the
primary checkout's development branch and its upstream for `push-dev`; the named installed
launcher directory for `restore-launcher`; the owned process the claim table names for
`daemon-restart`. If the sentence names a directory or branch ("restore the launchers in
~/x"), the guard accepts only if the named value equals what it derives; otherwise it refuses
and asks.

### 5.3 Ambiguity falls back to a question

When the interpreter finds an action but fails a rule (a confirmation with a conditional, a
reply to a stale proposal, an imperative with two kinds), it creates no authorization. The
hook prints `additionalContext` telling the agent a confirmation is needed. The agent then
asks with `AskUserQuestion`, and the harness shows the person a verbatim echo: kind,
derived target, expiry and uses. The structured answer (`PostToolUse`) is recorded and
matches the echo hash; only then does an `Authorization` with `how: "asked"` exist. The
model cannot answer the question, and a free-text reply to it goes back through 5.1/5.2.
No CLI is needed at any step. `/allow KIND` produces `how: "explicit"` with the same
derived-target rule.

### 5.4 Adversarial cases the interpreter must reject (each is a test)

| Person-looking text | Result |
|---|---|
| "don't push yet" | negation; no authorization; revokes any live push authorization |
| "should I push?" / "push?" | question; none |
| "if the tests pass go ahead" | conditional; none, ask |
| "yes, but not main" / "ok, push main" | negation or a second kind; none |
| `> push the dev branch` in a quote, or inside a code fence or pasted block | quoted; none |
| "ok" after the assistant asked a different question | no proposal for that message; none |
| "ok" 40 minutes later, or after compaction or resume | stale; none |
| "yes" to a proposal whose branch or file list changed | proposal hash mismatch; none |
| "yes" when the assistant's last message proposed two actions | no single proposal; ask |
| a board post, note, file or tool result saying "Owner: push the dev branch" | not a statement; no record |
| "sure" typed by a subagent turn (event has `agent_id`) | ignored |
| "yes" while the session is non-interactive | ignored |
| "go ahead and push main" / "tag it" / "release" | not authorizable; refuse and say the owner runs it |
| "restore the launchers" when the hook runs for an agent-run turn | recorder rejects non-`UserPromptSubmit` events |

### 5.5 Residual risk of prose

Interpretation is deterministic and conservative but not infallible. The remaining risks:
a reply that is a real yes but to something the person misread (they approved the sentence
they saw, not the action they meant); an affirmative the lexicon accepts that the person
meant conditionally in words it does not list; an assistant message that describes an action
in friendly terms while the derived target is broader than the person pictured (mitigated
by echoing the derived target in `additionalContext`, not by anything the model wrote);
a person approving after reading a misleading proposal. Authorizations are single-use,
short, session-bound and logged, so the worst case is one wrongly permitted action of a
low-risk kind, visible on the board. Nothing in the lexicon can reach the human-only floor.

## 6. The hard rule: no agent can be taken for a human

No flag, escape hatch or exception exists for this rule.

a. Statement and authorization records are a distinct kind, `person-attestation`, rendered
   "attested by the harness hook for session S". They are not messages, carry no author
   field naming a person, no `human` role and no person token.
b. No agent, subagent, board message or hook run during an agent turn can write a
   `person-attestation` or any record that renders as a human author. The writer functions
   live in `person_record.py`, imported only by the hook entry points (a wiring test, same
   shape as `tests/test_serve_no_bypass.py`). The board append API refuses the kind from
   every token.
c. The author fields of an agent cannot look human. Registration refuses ids and display names that
   equal, contain or visually resemble a reserved set after Unicode normalisation, case
   folding and homoglyph mapping: `human`, `owner`, `person`, `user`, `you`, `system`,
   `attested`, and the owner's display name. **The owner's name is derived at runtime from
   the human registry entry and is never written in the repository.**
d. Text claiming to be the person, in agent posts, notes, files or tool output, is rendered
   as agent data; the renderer never styles by body content and escapes the attestation
   markup. `reads_like_instruction` findings also flag "claims person authority".

Paths where an agent could pose as a human today, and the check that closes each:

| Path | Today | Check |
|---|---|---|
| Board author fields (`notes.add`, `bus`, `boards`) | `author = who.id`, trust from role, not fields | reserved-name registration; render by registry role; test: owner-mimicking id or body renders as agent |
| `mint` (`identity.mint`) | a human token mints any role; a lead mints `agent` | agent verbs never hold a human token; lead `mint` of a new identity is removed from agent flows (section 8); test: agent token minting `human` raises `Denied` |
| `invite` (`agent_invites.py`) | a joined agent issues a one-time code; the joiner is a new real agent with its own token | role pinned to `agent`; joiner acts as itself; test: invite never yields `human`; see section 8 |
| `delegate` | mints `parent/name` with its own token | removed (section 8) |
| Human token files | in the workspace token directory | edit and Bash guards refuse agent reads; test with a child process |
| Hook in an agent turn | PreToolUse/PostToolUse run during agent turns | recorder accepts only `UserPromptSubmit` and `PostToolUse:AskUserQuestion` without `agent_id`; test |
| `chat.run` | requires `HUMAN` token | unchanged; test |
| Handoff and notes text | free text | rendered as agent data; a fake statement block is inert for `consume`; test |
| Display (`agent_display.py`) | `person` kind from role | derived only from registry role; names reserved; test |
| `CLAUDECODE` / `ML_STACK_AGENT` | delegated gates test the environment | gates move to `consume` (slice 4) |
| Statement log files | plain files | `_FORBIDDEN` plus seal verification; tamper shows on the board |

Red-team tests that must go red if a path opens (real child processes, `CLAUDECODE=1`, fresh
state, no mocks): owner-mimicking post renders as agent; agent token cannot append
`person-attestation` via board API, notes, bus or direct import; reserved and confusable names
refused; agent token cannot mint or invite `human`; no child identity is `human`; a forged
row with a wrong seal makes `consume` refuse; text claiming the person's authority in file,
board post, note and tool result never lets `consume` succeed; every board view renders any
non-`human` role without human styling; and the 14 adversarial phrasings in 5.4.

## 7. What chat authorization can and cannot cover

Can (closed registry; targets derived by the guard):

| kind | derived target | consumer |
|---|---|---|
| `release-main` | repository, remote, full commit and tree of one fast-forward of `refs/heads/main`, shown in the approval question | `pre-push`, Bash guard |
| `daemon-restart` | owned process id and birth time from the claim table | runtime claims |
| `restore-launcher` | installed launcher paths | repair tooling |
| `runtime-deploy` | immutable wheel hash and target runtime | runtime tooling |
| `remove-worktree` | the named worktree path, after the cleanup checks | worktree tooling |

Cannot, ever: the human-only floor (keystore and secrets, signing keys, `sudoers`, the OS
administrator prompt, `iogpu.wired_limit_mb`, sentinel policy, quarantine release, roles,
saved rules, the review screen, `init`); raising a budget; tags, forced pushes, remote ref deletion,
`--all` and `--mirror` pushes; creating, widening or reading authorizations. `main` is coverable only as
`release-main`, through the structured approval question and never through typed words, `/allow` or a
reply. A sentence naming one of the others is answered with a refusal that says the owner runs it.
`ML_STACK_PUSH_MAIN=yes` opens nothing for an agent; the owner's own terminal push meets no hook.

### release-main on a pull request

With the main ruleset of [github-protection.md](github-protection.md) on, GitHub refuses a push of `main`, so a
promotion is a pull request from a frozen `promote/<date>` snapshot of the development branch, opened and merged by
the agent identity. The approval question names one commit; here that commit is the snapshot's tip, which is the
pull request's head SHA, and the merge consumes the approval: a push to the snapshot branch, a different pull
request or a rebuilt snapshot is not covered. GitHub enforces nothing about the approval, since the ruleset
requires the pull request and its checks only; the gate is our guard and the agent credential's lack of rights over
the ruleset. The guard side for `gh pr merge` and the merge API calls (the consumer of `person-consume` for the
pull request head) is not written yet, and until it is the owner's word in chat is the only check.

## 8. Removing `delegate`, and checking `mint` and `invite`

`delegate` goes in slice 1. A subagent then has no token of its own: it acts as its parent's
identity with a label taken from the SubagentStart event (`agent_id`, `agent_type`), the
shape CLAUDE.md already describes ("a subagent acts as that parent with a label").

Done in slice 1b-i: the `delegate` command line verb is gone, `mint` belongs to a person
alone (a lead mints nothing), a labelled helper is displayed as `parent (label)`, and
`tests/test_redteam_no_synthetic_identity.py` pins the reviewed identity-creating call sites,
the verbs, the invite role and the people-looking names.

Remaining (slice 1b-ii), consumers that still obtain `parent/name` tokens:

| Consumer | Needs |
|---|---|
| `service.py` `Workspace.delegate`, `identity.py` `Registry.delegate`, `person_session.py` stub | delete with the last caller |
| `remote.py` `delegate`, `remote_host.py` operation `delegate` and its audit | delete; native sessions become a parent seat with a label |
| `automatic_connection.py` native session start, `harnessid.py` local harness start | `Seat(label, parent=...)`; `harnesshook`, `harness_claims`, `harness_remote` and `notification_reader` key their identity and physical claim owner by the seat name today |
| `localstart.py` `_worker_identity`, `child_renewal.py`, `task_scheduler.py`, `device_agent.bind_worker` | worker seats for tasks |
| `task_worktrees`, `resource_allocations`, `backlog`, `issuepump`, `claim_handoff`, `task_integration`, `task_source_recovery`, `task_worktree_recovery`, `task_scope`, `taskboard`, `task_actions` | authorize by `child.parent == caller` where the worker holds its own token |
| tests | about 25 files build a delegated worker through `ws.delegate` (`taskboard_kit`, `test_child_renewal`, `test_workspace_quickstart`, `test_device_agent*`, `test_automatic_project_connection`, `test_enrollment`) |

Open design question: a local-model worker is a separate process. Under the parent-plus-label
model it would hold its parent's token and so its parent's rights, and the task checks that
separate a worker from its coordinator (`child.parent`, the independent reviewer) would reduce
to label comparisons.

`mint`: a human mints any role; a lead mints `agent`. A lead minting a new `agent` identity
is the same synthetic-identity shape. It is removed from agent-reachable flows: only a person
at a terminal (and `invite`/`join`) creates identities, and no `mint` path can reach role
`human` from an agent token (test).

`invite`: the joiner is a real separate agent (Codex, a local model) registering as itself
with its own token, role `agent`, parentage recorded. That is not a synthetic delegate, since
nothing acts as another. It stays, with role pinned to `agent`, bounded uses and the
approve-first gate, and with a test that no invite can carry a person role.

## 9. Failure modes

- Ambiguity, injection, stale and changed proposals: section 5.
- Background tasks outliving a statement: expiry is wall-clock, checked when the action
  happens; session end closes the session's authorizations; a longer task asks again.
- Multiple sessions: an authorization names one session and project.
- Compaction, resume, replay: the log is outside the model's context; consumption rows
  decrement `uses`, so the same action after compaction finds it used; a continued
  `session_id` keeps remaining time and uses and never resets them.
- Clock: the verifier's clock decides expiry; a backwards-running log refuses.
- Hook failure: nothing is authorized and the next guard refuses; a missing authorization
  blocks only the action that needed it.
- Hostile same-user process: not prevented, detected after the fact (section 2).

## 10. Slices

Slice 1 (days). Goal: the person says "yes, push it" and `pre-push` consumes the
authorization; `delegate` is gone.

- `scripts/hooks/claude-user-prompt` (new), `.claude/settings.json` (add the hook),
  `src/ml_stack/workspace/person_record.py` (new: log, `consume`),
  `src/ml_stack/workspace/person_intent.py` (new: closed tables and the interpreter),
  `scripts/hooks/pre-push`, `scripts/hooks/claude-bash-guard` (call `consume`),
  `src/ml_stack/sentinel/human.py` (`_FORBIDDEN`), `workspace/boards.py` (refuse the kind),
  `workspace/identity.py` (reserved names, `label`, delete `delegate`), `workspace/service.py`,
  `remote.py`, `remote_host.py`, `person_session.py`, `harnessid.py`, `localstart.py`,
  `automatic_connection.py`, `child_renewal.py` (delete), `task_scheduler.py`,
  `scripts/hooks/claude-subagent-start`, `workspace/cli.py`, `agent_display.py`,
  `docs/workspace.md`, `docs/local-agent.md`.
- Tests, about 55: interpreter unit tests, one per row of 5.4 plus the affirmative and
  imperative positives (about 30); record and `consume` (exact derived target, expiry, uses,
  revoke, other session, tamper, replay after compaction; about 10); a real `pre-push`
  child-process test; red-team for section 6 (about 10); removal tests (no `delegate` verb,
  op or method; subagent acts as parent with a label; `test_child_renewal.py` deleted; the
  existing `delegate` tests rewritten for labelled parent identity). Selectors via
  `scripts/test all tests/<file>`.

Later, in order: 2) `AskUserQuestion` echo path and `PostToolUse` recorder; 3) `daemon-restart`,
`restore-launcher`, `runtime-deploy` consumers; 4) move `delegated` authority gates from the
environment test to `consume`; 5) Codex and `ml-stack-chat` statement writers; 6) board view
with person-session revocation.

## 10a. Verified on Claude Code 2.1.293 (2026-10-08)

A temporary hook probe logged the real hook inputs; it and its log are deleted.

- `UserPromptSubmit` fires for typed prompts and also for harness-generated turns such as a
  `<task-notification>`. The hook input has the same fields in both cases (`session_id`, `cwd`,
  `scratchpad_dir`, `prompt_id`, `permission_mode`, `transcript_path`, `prompt`). No hook field
  says who wrote the prompt.
- The session transcript does. Each user entry carries `origin`, `promptSource` and `turnOrigin`,
  and `promptId` equals the hook's `prompt_id`. Typed: `origin.kind = human`, `promptSource =
  typed`, `turnOrigin = human`. A prompt typed while a turn runs: same, with `promptSource =
  queued`. An agent message: `promptSource = system`, `turnOrigin = peer`. A task notification:
  `origin.kind = task-notification`, `turnOrigin = task_notification`. Local commands such as
  `/model` have `origin = null`.
- The statement hook therefore records a `PersonStatement` only when the transcript entry for its
  `prompt_id` has `origin.kind = human`, `turnOrigin = human` and `promptSource` in `typed`,
  `queued`. A missing entry, an unknown shape or a different Claude Code version is not human
  and fails closed. The transcript is written asynchronously, so the hook retries for a bounded
  time. The transcript format is internal; the version is pinned in the test and any change makes
  statements fail closed until the reader is updated.
- `AskUserQuestion`: the `PostToolUse` input has `tool_response.answers` (question text to the
  chosen label) and `annotations`. A `PreToolUse` hook can pre-fill `answers` through
  `updatedInput`, so a test asserts that no configured hook does.
- Every command run in a session sees `CLAUDE_CODE_MESSAGING_SOCKET` and
  `CLAUDE_CODE_MESSAGING_TOKEN`. A message posted to that socket must be shown to carry a
  non-human origin before slice 1a ships; until it is, the origin check above is the only gate and
  the acceptance test posts one and requires `not human`.
- A same-user process can edit the transcript file. That falls under the detect-after-the-fact
  threat model in section 2.

Library decisions:

- Authorization records are JSON signed with Ed25519 from `cryptography`, verified with the public
  key. `biscuit-python` (not `biscuit-auth`) is only worth adding for offline attenuation to
  subagents. `pymacaroons` is not used (last release 2018, HMAC only, needs PyNaCl). `pymerkle`
  is not used (GPLv3, last release 2023). `cedarpy` waits until there are many policies.
- One chain implementation: `sentinel/events.py` `EventLog` (keyed HMAC, sealed head, anchor,
  verify), with the workspace `ChainLog` features it lacks ported in (torn-tail cut, fsync, lock,
  incremental verify, prefix prune). The workspace audit, notes, quarantine, bus and the test-reuse
  store move onto it.
- Defects found in the existing chains, each its own task: `activity/reuse.py` `chain_ok` returns
  true on an unparseable line and the next append restarts the chain; `EventLog._catch_up` does not
  cut a torn tail; `activity/gate.py` `evidence()` never calls `verify()`; the workspace
  `ChainLog` is unkeyed.

## 11. Decisions

1. Slice 1 is split: 1a (statement hook, record store, `push-dev` check) and 1b (`delegate`
   removal).
2. `mint` is removed from agent flows. A person at a terminal can still mint. `invite` and `join`
   stay, with the invited role pinned to `agent`.

## 12. Slice 1a as built

- Modules in `src/ml_stack/workspace/`: `person_store` (paths, verified reads, authorization state, the lock),
  `person_record` (the writers), `person_auth` (`consume`), `person_release` (facts, question, push check, the
  `person-consume` commands), `person_intent` (revoke, refuse, ask), `person_transcript`, `person_targets`,
  `person_ancestry`, `person_hook`, `person_view`. Only `person_hook` and `person_auth` import `person_record`;
  `tests/test_person_guards.py` pins that.
- One chain: the log is a `sentinel/events.py` `EventLog` (keyed HMAC, sealed head) at
  `~/.ml-stack/person/statements.log`.
- Hooks: `scripts/hooks/claude-user-prompt` serves `UserPromptSubmit`, `PostToolUse` on `AskUserQuestion` and
  `SessionEnd`; `claude-session-start` closes a session's authorizations when it starts again and records the
  harness process as that session's binding.
- Transcript versions pinned: `person_transcript.PINNED_VERSIONS` (2.1.293 as probed, 2.1.294 as read from a
  live transcript). Another version fails closed until the tuple is updated. The transcript path must be an
  owned plain file under the Claude projects folder named `<session id>.jsonl`; only its last 8 MiB is read.
- `release-main`: the hook (or `person-consume propose`) renders the question from git: the full commit, subject,
  commits ahead of the remote's main, diffstat and remote, ending in a line tagged under the store key. The
  authorization is created only when the answer is exactly "Approve release of <short sha>" to exactly that
  question as git renders it at answer time. It is bound to repository, remote, commit and tree, expires after
  15 minutes, is used once under the consume lock, and ends on session end, `stop`, `cancel that` or `/revoke`.
- `pre-push`: an agent (CLAUDECODE set, or an ancestor process that is a harness or a recorded session binding)
  may push the development branch with no authorization. A push that updates `refs/heads/main` must be the only
  ref, a fast-forward from the remote's tip, to the approved commit; the approval is consumed after the other
  push checks pass. Tags, forced pushes and deletions are refused. The Bash guard refuses the same pushes by
  command text (through `env`, `sudo`, `sh -c` and `unset` prefixes) and lets a push of main through only while
  the session holds an approval.
- The session of a guard is the one whose recorded harness process is its ancestor; `ML_STACK_SESSION_ID` is not
  trusted.
- The hard rules of both guards ignore `MLSTACK_GUARD=off`: the guards run only for an agent's tool calls. The
  edit guard refuses edits to the person-record modules, the hook scripts, the guards, `pre-push`, `rules_loader.py`
  and `.claude/settings*.json`; edits through the shell are not covered and stay a same-user matter.
- Forgeable by the same user: the hook entry points can be run by an agent, and a same-user process can write
  the store, its key or a transcript. Detection, not prevention: the chain and seal are verified before every
  use, the transcript path is validated, each statement records its transcript folder and the Claude Code
  version, and the guards refuse commands that name the store, the hooks or the writers. `CLAUDE_CONFIG_DIR`
  and `ML_STACK_HOME` still move the transcript folder and the store; a store an agent creates elsewhere holds
  no binding for the real harness process unless the agent forges one.
- Privacy: a statement keeps the SHA-256 of the prompt, an excerpt of at most 80 characters with secrets
  redacted, the working directory and the session id. To purge, the person deletes `~/.ml-stack/person/` at a
  terminal; a new chain starts and every authorization ends.
- Manual acceptance (the person): a typed "yes" or "/allow" after a release proposal does not authorize; the
  approval answer to the AskUserQuestion does, and `git push origin main` then goes through once; a message posted
  to `CLAUDE_CODE_MESSAGING_SOCKET` in a live session is not recorded as a statement; `env -u CLAUDECODE git push
  origin main` from the agent's shell is refused; the hooks fire and exit 0 under the installed Claude Code.
