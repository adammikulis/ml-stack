# Person-spoken authorization

Design note. Nothing here is implemented. Status: proposal for the owner.

## Invariants

1. Every agent acts as itself. There is no delegate identity, no delegate credential and no
   credential that carries a person's rights.
2. **An agent can never post as, pose as or be mistaken for a human.** This rule is hard. It
   has no escape hatch, no flag, no environment variable, no test-only switch and no
   "trusted" agent role. A change that adds one is refused.
3. A person's chat sentence can become a scoped, expiring, auditable authorization. The
   harness produces the evidence; no model, subagent, board message or file does.
4. Human-only system settings (AGENTS.md "System settings and the authority registry")
   remain a person's at their own terminal. A chat sentence never covers them.

## 1. What exists today

Read from the tree; paths relative to the repository root.

- `src/ml_stack/workspace/identity.py`: roles `human`, `lead`, `agent`. `MINTS` lets a human
  token mint any role, a lead mint `agent`, an agent mint nothing. `Identity.trust` returns
  `"human"` for role `human`, else `"agent-claimed"`. `delegate()` mints a child token
  `parent/name`, role `agent`, weaker than the parent, no further delegation.
- `src/ml_stack/workspace/notes.py`: `RANK = agent-claimed < test-verified < human`; a note's
  trust comes from the author's role, never from fields. `task_credit.py` supplies the
  verified evidence rank.
- `src/ml_stack/workspace/chat.py`: `run` requires a `HUMAN` token; this is the person's
  console.
- `src/ml_stack/workspace/agent_display.py`: display names from registry metadata;
  `session_kind` is `person` when role is `human`.
- `src/ml_stack/authority.py` registry: each gate is `person` or `delegated`. A delegated
  gate passes a lead agent "acting on the owner's instruction" (`CLAUDECODE` or
  `ML_STACK_AGENT` set, no helper label). This is an environment-variable test, not evidence
  that the owner said anything. It is the weakest link this design replaces for chat-level
  permission.
- `src/ml_stack/sentinel/human.py`: `HumanGrant` (action, subject, expiry, `GRANT_TTL_S`
  120 s) built by `mint` only after a terminal confirmation; `protect` and `_FORBIDDEN` keep
  an agent's tool arguments away from sentinel/authority/log state. `sealed.py` is an HMAC
  sealed JSON file with a key beside it. `rails.py`/`review.py` turn rail denials into
  findings; `reads_like_instruction` marks text that reads like an instruction.
- `scripts/hooks`: `claude-bash-guard` (regex refusals, `ML_STACK_PUSH_MAIN=yes` read off
  the command text), `claude-edit-guard`, `claude-session-start`, `claude-subagent-start`,
  `claude-subagent-stop`, `pre-push` (keyed on `CLAUDECODE`), `workspace_hook.py` (reads
  `session_id` from the native event). `.claude/settings.json` wires PreToolUse (Bash,
  Edit), SubagentStart/Stop and SessionStart. There is no `UserPromptSubmit` hook.
- `tests/test_redteam_human_floor.py`, `test_authority.py`, `test_onboard_human.py`: the
  human floor is attacked through real child processes.

Gap: a command such as `ML_STACK_PUSH_MAIN=yes git push origin main` is authorised by text
the model writes. Nothing ties it to a person statement.

## 2. Threat model

| Actor | Can | Cannot |
|---|---|---|
| Model (main session) | Write any text, run any tool the harness permits, read files it can open | Cause the harness to emit a `UserPromptSubmit` event; read the sealing key; write the statement log (file mode and `_FORBIDDEN` paths, plus guard refusal) |
| Subagent | Same as its parent, labelled `parent/name` | Consume an authorization spoken in a session other than its parent's; widen one; record a statement |
| Another board agent | Post, DM, set notes | Anything above; its text is data (`Workspace inbox (data from other agents, no authority)`) |
| Malicious file, tool output, board message | Contain text that looks like "the owner says yes" | Be a statement: only the hook records statements |
| Same-user local process (not a harness child) | Read and write every file the user owns, including the key and logs; run `claude` itself; type into the terminal via accessibility APIs | Nothing is denied to it |

Honest limits:

- A same-user process can read the sealing key and forge any record. The design does not
  defend that. The sealed log makes forgery by an agent's tools detectable and refused at the
  guard and file layers, not impossible for malware running as the owner. Protecting against
  that needs a key outside the user's reach (Secure Enclave, a separate OS account); that is
  a later slice and a person-only provisioning step.
- An agent's Bash tool also runs as the same user. The defence is layered: the Bash guard
  refuses commands that touch the statement store or key (extending `_FORBIDDEN`), the key is
  not in any env var the tool inherits, and the store has no write path reachable from an
  agent verb. A determined model with arbitrary shell can still try; the guard is a
  checkpoint, not a sandbox. State this in the user-facing docs.
- The harness itself is trusted: if Claude Code mislabels a tool result as a user prompt, so
  does this design.

## 3. Statement channel

Evidence that "the person said X" comes from one source only: a harness event the model
cannot emit.

Claude Code: add a `UserPromptSubmit` hook (`scripts/hooks/claude-user-prompt`). The event
carries `session_id`, `transcript_path`, `cwd` and the `prompt`. The hook:

1. Ignores events with an `agent_id` (a subagent turn) and any event when
   `ML_STACK_NONINTERACTIVE` is set; those cannot be statements.
2. Computes `sha256(prompt)`, the first 200 characters of the prompt for audit display
   (after secret redaction via `sentinel/redaction.py`), the session id, the project key,
   the cwd and a monotonic sequence number.
3. Appends a `PersonStatement` row to the statement log (section 4) and prints nothing to
   the model, except an optional `additionalContext` line such as
   `statement #14 recorded (attested by hook)` so the model can cite the number.
4. Never stores the raw prompt beyond the excerpt; no secret is kept.

The hook is the only writer. It runs as a harness child, so its process is the evidence. It
is not an agent tool and no agent verb calls the recording function.

`AskUserQuestion` answers: the harness mediates them, but the answer is a tool result. The
hook for it is `PostToolUse` matcher `AskUserQuestion`; the event carries the structured
answer. The same recorder writes a `PersonAnswer` row with the question text, the option
chosen and the question's hash. If PostToolUse for that tool cannot be relied on, the
fallback is a `UserPromptSubmit` carrying the chosen option text. Verify against the Claude
Code version in use before slice 1 depends on it.

Codex and local-model harnesses: the equivalent is whatever event the harness emits when
the human types, written by the harness launcher (`ml-stack-agent`, `ml-stack-chat`) in its
own process, not by the model loop. `ml-stack-chat` already requires a `HUMAN` token for
the console; it can write `PersonStatement` rows with `harness: ml-stack-chat` under the
same schema. A harness with no such event provides no statements: authorisation then needs
the person's terminal. Do not fabricate the event from model output.

Binding: a statement carries `session_id` (the native id), project key and cwd. The
agent's session record (`persist_session` in `workspace_hook`) maps the same `session_id`
to the workspace agent name. A consumer proves its session by presenting its own workspace
token; the store resolves token to session, never the other way.

Model-relayed text ("the owner told me to restore the launchers") is never evidence, in the
log, on the board or in a hook.

## 4. The authorization object

Two records, both in `~/.ml-stack/person/statements.log` (a hash-chained `ChainLog`, the
same mechanism as the activity log; sealing key beside it as in `sentinel/sealed.py`).

```
PersonStatement  seq, ts, session_id, project, cwd, prompt_sha256, excerpt,
                 source: "harness-hook:UserPromptSubmit", prev, hash
Authorization    id, ts, statement_seq, session_id, project,
                 kind, target (exact), uses, expires, state, confirm: "echo"|"ask",
                 prev, hash
```

An `Authorization` is created only by the **grant recorder** (the same hook process path),
never by a model-callable verb. It is created in one of two ways:

- **Structured form in the prompt.** The person types `/allow push-dev`,
  `/allow restore-launchers`, `/allow daemon-restart`. The hook recognises the slash
  command from the prompt text itself and parses a closed vocabulary (kind plus optional
  exact target from the prompt). Free-form prose never creates one.
- **Confirmed from prose.** If the prompt is prose ("yes, restore those launchers"), the
  hook creates no authorization. The model may call `ml-stack-workspace propose-authorization
  KIND TARGET` which writes a `Proposal` (an agent-authored board record). The hook-side
  confirmation then asks the person via `AskUserQuestion` with a verbatim rendering of kind,
  exact target, expiry and uses. Only the structured answer to that question, recorded by the
  PostToolUse hook and matching the proposal's hash, creates the `Authorization`. The model
  cannot answer the question.

Scope fields:

- `kind`: from a closed registry (section 6). Unknown kinds are refused.
- `target`: exact string(s): branch name, absolute path list, command argv hash, port,
  worktree path. Globs are not accepted. A target that differs by one byte does not match.
- `uses`: 1 by default; at most 5; bounded-use only for kinds that list it.
- `expires`: default 15 minutes, hard maximum 4 hours, always at or before the session end.
  Compaction and resume do not extend it.
- `state`: `live`, `used`, `expired`, `revoked`. Transitions append rows; nothing is edited.

Who may consume: the session in which the statement was spoken (resolved by token to
`session_id`) and its labelled subagents acting as the parent (`parent/name` identity whose
parent's session matches). A different session, a different project or a background process
that outlived the session gets no match. Subagents consume with their own identity; the
consumption row records the child, so audit shows which agent acted.

Revocation: the person types `/revoke ID` or `/revoke all`; the hook appends the rows. A new
`UserPromptSubmit` event whose prompt is `/revoke` is honoured even if the model is
mid-turn. `SessionEnd` and `SubagentStop` hooks append expiry rows.

Guard query: one function, `person_auth.consume(kind, target, identity)`, that returns the
authorization id or raises `NotAuthorized`. It reads the chain, verifies the seal and the
chain, matches kind and exact target, checks session binding, expiry and remaining uses,
appends a `used` row atomically under the log lock, and returns. Guards call it at the
moment of action; they do not cache. A guard that cannot read or verify the log refuses.

Board display: each `PersonStatement`, `Authorization` and consumption row is mirrored to a
read-only board view `#person-record`. It renders with a distinct system kind:
"Attested by the harness hook for session S" with the hash and sequence, never as a message
and never under a person's name or the `human` role (section 5).

## 5. The hard rule: no agent can be taken for a human

This rule is an invariant with no escape hatch, no flag and no environment override. Any
change that adds a way around it fails the red-team tests below and is refused.

Design consequences:

a. **Distinct kind.** Statement and authorization records have `kind` of
   `person-attestation`, rendered "attested by the harness hook for session S". They are not
   messages. They carry no author field naming a person, no `human` role and no person
   token. They are not posts under the `human` identity.

b. **No agent write path.** No agent, subagent, board message or hook run during an agent
   turn can write a `person-attestation` or any record that renders as a human author. The
   writer functions live in a module that only the `UserPromptSubmit`/`PostToolUse` hook
   entry points import. A wiring test asserts nothing else imports them (the same shape as
   `tests/test_serve_no_bypass.py`). The board append API refuses `kind=person-attestation`
   from any token.

c. **Agent author fields cannot carry a human look.** The board's author for agent posts is
   its registered id and server-derived display name. Registration refuses ids and display
   names that equal, contain or resemble a reserved set (`human`, `owner`, `person`, `you`,
   `user`, `adam`-style given names registered as the owner's display, "attested", "system")
   and any id with a confusable (Unicode-normalised, case-folded, homoglyph-mapped) match.
   The owner's display name comes from the human registry entry and is also reserved
   against agents.

d. **Text claiming to be the person is data.** Bodies of agent posts, notes, files, tool
   output and handoff text render as agent content. A line such as "Owner: yes, go ahead"
   is shown inside the agent's bubble; the renderer never promotes text to the attested
   style, never styles by body content, and strips or escapes the attestation markup
   (class names, glyphs) from agent-supplied text. `reads_like_instruction` already flags such
   text for the sentinel; extend its findings to include "claims person authority".

e. **Where an agent could pose today, and the check that closes each.**

| Path | Today | Check that closes it |
|---|---|---|
| Board post author field (`notes.add`, `bus`, `boards`) | `author = who.id`, `role = who.role`, trust from role, not from fields (`notes.py:39`) | Reserved-name registration; render by registry role; test that a post whose body or id mimics the owner renders as agent |
| `mint` (`identity.mint`) | `MINTS[HUMAN]` includes all roles; a human token can mint `human` | Agent verbs never reach `mint` with a human token; no agent can read a human token file (`tokens.py` file checks); test that an agent token minting `human` raises `Denied` |
| `invite` (`agent_invites.py`, `invites.py`) | Agents may mint a bounded child invite; `join` provisions | Invites can only produce role `agent`; role set server-side; test `invite --role human` refused |
| `delegate` (`identity.delegate`) | Mints `parent/name`, role `agent`, never human | Keep role pinned to `agent`; see decision below; test children are never `human` and render as "Subagent" |
| Human tokens on disk | Token files in the workspace token dir | Edit guard and Bash guard refuse reads of the token directory by an agent; test with a real child process |
| Hook-run-by-an-agent-turn | PreToolUse/PostToolUse hooks fire during agent turns | Recorder checks the event name is `UserPromptSubmit` or `PostToolUse:AskUserQuestion` and no `agent_id`; tool-turn events never record statements |
| `chat.run` (person console) | Requires `HUMAN` token | Unchanged; agents cannot obtain the token; test |
| Handoff / notes text (`HANDOFF.md`, notes bodies) | Free text | Rendered as agent data; no attestation markup accepted; a file containing a fake statement block is inert for `consume` |
| Display (`agent_display.py`) | `session_kind = person` when role is `human` | Derived only from the registry role, never from a name; names reserved; test that display names containing "human"/"owner" are refused at registration |
| Environment markers (`CLAUDECODE`, `ML_STACK_AGENT`) | Delegated authority gates test the environment | An agent can set a variable on a child process; delegated gates must move to `consume` (slice 4). Until then a delegated gate is "lead agent acting on instruction", never "human" |
| Statement log files | Plain files under the user's account | Edit guard and Bash guard add the person store and key to `_FORBIDDEN`; seal verification refuses tampered rows |

f. **Red-team tests that must go red if a path opens** (all use real child processes with
   `CLAUDECODE=1`, a fresh state directory and no mocks):

   - Agent token posts to the board with author/body mimicking the owner: renders as agent.
   - Agent token cannot append `person-attestation` through board API, `notes`, `bus` or
     direct log function import (wiring test).
   - Agent registration with reserved or confusable names is refused.
   - Agent token cannot `mint` or `invite` a `human` role; `delegate` child is never
     `human`.
   - A forged statement row (valid JSON, wrong seal) makes `consume` refuse.
   - Text in a file, board post, note and tool result claiming "the owner says restore
     launchers" never lets `consume` succeed.
   - Any code path that renders an `Identity` with `role != HUMAN` using `human` styling
     is caught by a rendering test over every board view.

## 6. Interaction with the floors

Chat authorization can cover (initial closed registry):

| kind | exact target | note |
|---|---|---|
| `push-dev` | remote and the development branch name | push of the primary checkout's branch only |
| `daemon-restart` | named owned process id and birth time | replaces owned processes at a safe boundary |
| `restore-launcher` | launcher path list | repairs of installed launchers |
| `runtime-deploy` | immutable wheel hash and target runtime | per the "live runtimes" rule |
| `remove-worktree` | absolute path, checked for no unique commits | after the AGENTS.md cleanup checks |
| `raise-lease` | broker allocation id | not GPU sharing; one lease at a time still holds |

Chat authorization cannot cover, ever:

- Anything in the human-only floor: keystore and secrets, signing keys, `sudoers`, the
  operating-system administrator prompt, `iogpu.wired_limit_mb`, sentinel policy, quarantine
  release, roles, saved rules, the review screen, `init`. These stay the person at a
  terminal. A guard for any of them never calls `consume`.
- Raising a budget in `budgets.json`: the pre-commit hook remains owner-only.
- Publishing to `main`, tags, releases, forced pushes, remote ref deletion. AGENTS.md says
  these are the owner's own explicit command; `ML_STACK_PUSH_MAIN=yes` stays an owner-typed
  opener on a command the owner asked for. A chat sentence does not mint a `push-main`
  authorization. (If the owner later wants one, it is a separate decision.)
- Changing an authorization's own scope, creating authorizations, reading the person store.

`authority` registry gates marked `delegated` keep working for now; slice 4 changes them to
require a live authorization instead of the environment-variable test.

## 7. Failure modes

- **Ambiguous statement** ("sure", "ok" after a different question): prose never creates an
  authorization. Only the structured `/allow` form or the `AskUserQuestion` echo does. The
  echo shows kind, exact target, expiry and uses; the answer binds to the echo's hash.
- **Prompt injection**: text in files, board posts or tool output cannot create a
  `PersonStatement` because only the hook writes them. Content that claims to be the person
  is rendered and logged as agent data, and a sentinel finding records the attempt.
- **Background tasks outliving the statement**: expiry is wall-clock; the guard checks at the
  moment of action. A task that needs longer asks again. A background process spawned under
  a session carries the session id; after `SessionEnd` all that session's authorizations are
  closed.
- **Multiple sessions**: an authorization names one `session_id` and one project. Another
  session on the same project cannot consume it.
- **Compaction, resume, replay**: the log is outside the model's context; compaction cannot
  recreate a statement. Consumption rows decrement `uses`, so replaying the same action
  after compaction finds the authorization used. A resumed session has a new or continued
  `session_id`; if the id continues, remaining time and uses continue and do not reset.
- **Clock skew**: the log uses the writer's monotonic sequence with timestamp; expiry uses
  the verifier's clock and refuses on a backwards-running log.
- **Hook failure**: if the recorder cannot write, nothing is authorized and the next guard
  refuses. A notification outage does not block completed local tools; a missing
  authorization blocks only the action that needed it.
- **Hostile same-user process**: not defended (section 2).

## 8. Decision on `delegate`

`delegate` mints a weaker child token `parent/name`. It is not a person-authority path: the
child is role `agent`, labelled, and cannot mint or delegate. It does not violate "every
agent posts as itself" as long as the child is displayed as a subagent of its parent and
never a person. But the owner said he does not want an explicit delegate credential for
person authority, and CLAUDE.md says subagents join the workspace automatically by acting
as the parent with a label. Recommendation: remove the `delegate` command and the
`delegate` workspace method once the SubagentStart hook provides the labelled identity on
its own, and fold `child_renewal.py` into that flow. Until then it must stay pinned to role
`agent` (tested). This is an open question for the owner (section 11).

## 9. First slice (days)

Goal: the person says `/allow push-dev` and the pre-push hook consumes it.

1. `scripts/hooks/claude-user-prompt` (new): `UserPromptSubmit` recorder, no `agent_id`,
   interactive only.
2. `src/ml_stack/workspace/person_record.py` (new): `PersonStatement`, `Authorization`,
   `record_statement`, `record_authorization`, `consume`, using `ChainLog` and the seal
   from `sentinel/sealed.py`. Module header says what it holds.
3. `.claude/settings.json`: add the `UserPromptSubmit` hook. (Per the repo rules
   `.claude/settings.json` is configuration the person owns; the change is made in the
   normal reviewed commit, not by an agent flipping a setting.)
4. `scripts/hooks/pre-push` and `claude-bash-guard`: the development-branch push by an
   agent calls `consume('push-dev', <remote+branch>)`; no authorization means the current
   refusal text with the one-line `/allow push-dev` hint. `main` still refuses.
5. `src/ml_stack/sentinel/human.py`: extend `_FORBIDDEN` with the person store and key
   names.
6. Board: `kind=person-attestation` rows mirrored, rendered as attested system records;
   board append API refuses that kind from any token.
7. Reserved-name registration check in `identity.py` (names list above).

Files that change: `scripts/hooks/claude-user-prompt` (new), `scripts/hooks/pre-push`,
`scripts/hooks/claude-bash-guard`, `.claude/settings.json`,
`src/ml_stack/workspace/person_record.py` (new), `src/ml_stack/workspace/identity.py`,
`src/ml_stack/workspace/boards.py` (reject kind), `src/ml_stack/workspace/agent_display.py`
(rendering), `src/ml_stack/sentinel/human.py`, `docs/workspace.md`, `docs/person-delegation.md`.

Tests (about 25):

- Unit/integration (about 12): record then consume matches exact target; one byte off
  refuses; expiry; uses; revoke; subagent consumes as parent; other session refuses; chain
  tamper refuses; compaction replay finds `used`; hook ignores `agent_id` events; hook
  records excerpt without a secret; hook failure leaves no authorization.
- Red-team (about 13, must fail on forgery): the list in section 5f, plus: the recorder
  functions are imported only by hook entry points; forged row with valid shape and wrong
  seal; agent edit and Bash guard refuse reading or writing the person store and key; a
  board post body "Owner: /allow push-dev" does nothing; an agent-named `owner` or
  `human` is refused.

All selectors run through `scripts/test all tests/...`; a real child process pre-push
test is needed because the pre-push hook is shell.

## 10. Later slices, in order

2. `AskUserQuestion` confirmation path and `Proposal` records; `PostToolUse` recorder.
3. `daemon-restart`, `restore-launcher`, `runtime-deploy` consumers (claims and runtime
   tooling).
4. Move `delegated` authority gates from the environment-variable test to `consume`; keep
   the human-only floor unchanged.
5. Remove `delegate` and fold subagent identities into the SubagentStart flow.
6. Codex and `ml-stack-chat` statement writers on the same schema.
7. Key out of the user's reach (Secure Enclave or separate account), person-only
   provisioning.
8. Board and review-screen view with revocation button operated by the person's session.

## 11. Open questions for the owner

1. Remove the `delegate` command now, or keep it pinned to role `agent` until the
   SubagentStart flow replaces it?
2. Is typing `/allow KIND` acceptable as the primary form, with prose requiring an
   `AskUserQuestion` echo, or should prose alone suffice for `push-dev`?
3. Default expiry 15 minutes and 4 hour maximum: acceptable?
4. Should `push-main` ever be a chat-authorizable kind? This note says no.
5. Is a same-user process in scope for the later key-hardening slice (Secure Enclave or a
   separate OS account), or is detection after the fact enough?
6. Reserved display names: should the owner's own given name be reserved against agents
   automatically from the human registry entry?
