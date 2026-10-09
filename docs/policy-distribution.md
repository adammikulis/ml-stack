# Library-owned agent policy

Design note. Nothing here is implemented. It builds on `docs/person-delegation.md` (statements,
`consume`, the human floor) and `docs/session-liveness.md` (tier table, coordinator eligibility);
both define rules this note moves into a registry that code enforces.

Problem: the working rules are prose in this repository's `AGENTS.md` and `CLAUDE.md`, and the
guards that enforce part of them sit in this repository's `scripts/`. A project that uses
ml-stack as a library has its own `AGENTS.md`, `CLAUDE.md`, tests and agents. It gets none of the
rules unless it copies prose and scripts, and a copy drifts, is forgotten, or is edited away by
the agent it was meant to bind.

Principles: a rule that can be code is code in the wheel and its prose is generated from the same
record; a consumer may add rules and tighten parameters but never weaken a hard rule, and no flag
says otherwise; drift is compared against what the agent cannot edit in the same change (the
installed wheel, the base branch); this repository is its own first consumer.

## 1. Inventory

Classes: **a** library invariant in code; **b** library guard (hook, gate, runner) configured by
the project; **c** generated text, a delivery channel for a, b and e rules rather than a separate
kind of rule; **d** project policy; **e** judgment prose. "Today": `none` = prose only, `design` =
specified in a note, not built.

| # | Rule | Today | Class | Should live |
|---|---|---|---|---|
| 1 | Read AGENTS.md, then CLAUDE.md; first response names scope and model | prose + `briefing.py` text | c | registry `brief` rules rendered by `briefing.py`/`onboard.brief` |
| 2 | "If I can't use it, it's not done"; activation owner | prose + `docs/tasks.md` completion gate | e (a for the task gate) | judgment line in block; task gate stays code |
| 3 | Managed workers get no extra permissions; a spawned subagent is its own identity under its parent | code (`identity`, `agent_display`) | a | stays code, registry row with test refs |
| 4 | Subagent prompt carries the workspace line | `onboard.BRIEF` text | c | generated from registry |
| 5 | Main session coordinates; subagents never elect themselves | prose; design (`session-liveness` 4-5) | a | coordinator lease code |
| 6 | Lowest model tier never coordinates; unknown model ineligible | design | a (table: d) | `model_tiers` loader hard; table content is the project's |
| 7 | announce joined/done, 200-char lines, 6 per 10 min | code (`limits`, hooks announce) | a | code |
| 8 | Read the inbox first | `claude-session-start` | b | `ml-stack-policy hook session-start` |
| 9 | Claims before mutating; no stealing a live claim | code (`claims`, harness reserve) | a | code; hard |
| 10 | Named-file staging, no `git add -A`/`commit -a` | `claude-bash-guard` | b | packaged bash guard |
| 11 | No commit by an agent in the primary checkout; worktrees beside, not inside | `primary-only`, `worktreerules` | b | packaged guard; dev branch derived |
| 12 | No editable install of a live runtime | `worktreerules.INSTALL`, `doctor` | a/b | guard + doctor |
| 13 | `main`, tags, force, ref deletion are the owner's; opener is the owner's | bash guard + `pre-push` | b, **hard** | packaged `pre-push` and bash guard |
| 14 | Agents push the dev branch themselves; fetch before and after | `pre-push`, prose | b/e | `pre-push` plus block text |
| 15 | Landing queue: <10 commits/branch, <=20 behind, ~10 worktrees, finish before starting | prose | b (numbers d) | `policy check landing` (reads git; tighten-only numbers) |
| 16 | Merged worktree removed; cleanup checks | `pushed` hook, `worktree_lifecycle` | b | packaged |
| 17 | Hook failures are incidents; installed-hook smoke after cutover | `hook_diagnostics`, `doctor hooks`, prose | a/e | code + `policy selftest` |
| 18 | Scoped tests through the broker; workers run no full or shared suite | `scripts/test`, `testslots`; worker rule is prose | b | shipped runner; worker-tier refusal is new |
| 19 | Shared gates once per batch, scaled to the diff | prose | e/b | tree-hash gate record (test-reuse branch) |
| 20 | No live API/net test without the person's env; no real keystore in tests | `conftest`, markers | b, **hard** | shipped pytest plugin |
| 21 | A budget only falls; `--allow-increase` refused for agents | `budgets-only-fall`, `test_budgets`, `scripts/budgets` | b, **hard** | packaged ratchet hook, generalised |
| 22 | Hard size gates 900 / 500 lines | `deep_files`, `deep_components`, edit guard | b | packaged; numbers d, tighten-only |
| 23 | Duplicate bodies, mutation survivors, ruff/pyright budgets | `scripts/gates`, `mutate` | b | gate framework shipped; project adds its own |
| 24 | Layer and wiring tests | `tests/test_layers.py`, `test_wiring.py` | d (framework b) | project layer map; checker shipped |
| 25 | Edit guard: duplicate function, raw HTTP, docstring >12, params >8 | `claude-edit-guard` | b | packaged; `OWNED` table d |
| 26 | One compute job per device; no server by hand; leases only from the broker | `serve.grant`, bash guard, `server-starts` gate, test | a, **hard** | code |
| 27 | Do not drive a model at an edited checkout; own Chromium, never the person's | prose; `Window` code | b/e | add `claude-in-chrome` tool matcher refusal |
| 28 | Default models, decider, MTP, llama.cpp head | prose | d | project config; library default in code |
| 29 | Comments say what, not why | edit guard (docstring length only) | b partial, e | lint advisory; rest judgment |
| 30 | Commit subject `feat:`/`fix:`/`chore:`; banned phrases | prose (commit-msg checks names only) | b (set d) | packaged `commit-msg` |
| 31 | Judgment: delegation, commit before mutate, graph default, user-facing text, HANDOFF, reporting, saying it works, library vs app, root causes | prose | e | one-line pointers in the block |
| 32 | An agent is never a human; no flag | code + red-team tests | a, **hard** | code; no registry parameter |
| 33 | Authority gates; keystore, sudoers, OS prompt are a person's | `authority.py`, `sentinel/human.py` | a, **hard** | code |
| 34 | Authorization only from the harness's statement record | design (`person-delegation`) | a, **hard** | code |
| 35 | Never a real person in the tree or a commit message | `no-real-names`, `commit-msg`, `redact` | b, **hard** | packaged; name sources and fixtures d |
| 36 | Project stance (no users but the owner, nothing cemented), harness names, model order, trailer | prose | d | project policy file |
| 37 | Pre-commit chain; `.claude/settings.json` wiring | `install-hooks.*`, hand-edited JSON | b | `ml-stack-policy install` |

## 2. The durable home

### 2.1 Layout

```
src/ml_stack/policy/
  rules.toml        the registry (package data; ships in the wheel)
  hard.py           HARD_IDS frozenset, written in code
  registry.py       Rule, Param, load(), validate()
  config.py         project policy file: load, tighten-only validation
  render.py         marker blocks (AGENTS, CLAUDE), briefing text, hashes
  lock.py           policy.lock: write, read, verify
  install.py        plan/apply: git hooks, harness settings entries
  check.py          the drift checks behind `ml-stack-doctor policy check`
  diff.py           lock versus installed registry
  hooks/            bash_guard, edit_guard, pre_push, commit_msg, ratchet, session_start
  conformance/      pytest plugin + canary cases consumers run
  cli.py            ml-stack-policy
```

It sits in the `core` layer (standard library, `worktreerules` folded in, `hook_diagnostics`,
`checks`). Wiring: console script `ml-stack-policy` and a `policy` group in `ml-stack-doctor`;
`cli/reference.py` gains `HELP`/`TABLE` rows so the command-table drift check covers them.

Hooks run from the wheel: the settings entry is `ml-stack-policy hook bash-guard`. No script is
copied into the consumer, so there is no copy to drift and no editable path. Hook start-up
imports only `ml_stack.policy.hooks.*`; cold-start time is measured in slice 2 (not measured here).

### 2.2 Rule record

```toml
[[rule]]
id         = "git.main-is-the-owners"
since      = "0.2"
severity   = "hard"             # hard | required | advisory
class      = "guard"            # invariant | guard | judgment
scope      = "always"           # always | uses:serve | uses:workspace | ...
audience   = "all"              # all | lead | subagent
harness    = "any"              # any | claude-code | codex | local
brief      = true               # included in the generated briefing
summary    = "An agent pushes the development branch by name. main, tags, force and ref deletion are the owner's."
detail     = "docs/policy/git.md#main"      # optional, shown by `explain`
enforced_by = [
  { kind = "hook", ref = "ml_stack.policy.hooks.pre_push" },
  { kind = "hook", ref = "ml_stack.policy.hooks.bash_guard:PUSH_MAIN" },
]
proof      = ["tests/test_pre_push.py", "tests/test_bash_guard.py::test_push_main"]
params     = {}
```

A parameter:

```toml
[[rule]]
id = "size.python-lines"
severity = "required"
params.limit = { default = 900, tighten = "lower", floor = 100 }
```

`tighten` is one of `lower`, `higher`, `superset`, `subset`, `fixed`. A parameter with no
`tighten` cannot be overridden.

Registry self-tests (`tests/test_policy_registry.py`): unique well-formed ids; every `ref` resolves
and every `proof` path collects; a rule with enforcement has a proof; a `judgment` rule has none;
the hard ids equal `hard.HARD_IDS`; summaries are one line without block markers.

### 2.3 The hard-rule set

`hard.HARD_IDS` is a frozenset in Python, not data. Editing `rules.toml` cannot add a hard rule
to it or take one out: the registry test fails if the two disagree, and the loader refuses a
`rules.toml` whose hard ids differ from the code (so a tampered data file in a wheel is refused
at import of the policy package).

| Id | Content | Enforced by |
|---|---|---|
| `person.agent-is-never-human` | No flag, exception or escape hatch | identity, board, renderer; red-team tests |
| `person.human-floor` | keystore and secrets, sudoers, OS admin prompt, machine settings | `authority.py`, `sentinel/human.py` |
| `person.statement-only` | Authorization comes only from the harness-recorded statement | `person_record.consume` (design) |
| `person.no-real-names` | No real person in tree, tests, commit messages | `no-real-names`, `commit-msg` |
| `git.main-is-the-owners` | main, tags, force, deletion not pushed by an agent | `pre-push`, bash guard |
| `ratchet.only-falls` | A number in a ratchet file is not raised by an agent | `ratchet` hook |
| `claims.no-steal` | A live claim is not taken; recovery is verified | `claims.py` |
| `coordination.lowest-tier-never` | Lowest tier and unknown models never coordinate | `model_tiers` (design) |
| `serve.broker-only` | Compute is leased from the broker; no server by hand | `serve.grant`, bash guard |
| `test.no-live-no-keystore` | Tests do not call paid/public endpoints or the real keystore | pytest plugin |
| `runtime.immutable` | No editable install behind a live runtime | guard, doctor |
| `policy.tamper-protected` | Policy files, lock, hook wiring and generated blocks are not agent-writable | edit guard, bash guard, `ratchet` |

Membership is the owner's decision (open question 9). Hard rules with `scope = uses:*` bind a
project only when it imports that package; `person.*`, `git.*`, `ratchet.*`, `policy.*` always bind.

### 2.4 Project policy file

`.ml-stack/policy.toml`, committed:

```toml
requires = "ml-stack>=0.9"             # floor; the lock records the exact version

[docs]
agents = "AGENTS.md"                   # block inserted under the first H1
claude = "CLAUDE.md"                   # claude-code-only rules

[git]
development_branch = "auto"            # branch the primary checkout is on
commit_prefixes    = ["feat", "fix", "chore"]   # subset of the library's list

[params]
"size.python-lines"  = 600             # tighter than 900: accepted
"edit.max-parameters" = 6

[models]
tiers = ".ml-stack/model_tiers.json"   # project-owned table (rule 8)

[[local]]                              # a project rule; ids must start with "local."
id = "local.no-print"
severity = "required"
summary = "No print calls under src/."
enforced_by = [{ kind = "gate", ref = "scripts/gates/print_calls.py" }]
proof = ["tests/test_budgets.py::test_metric_is_within_its_budget"]
```

Validation (`config.validate`, run by `policy check`, the pre-commit chain and every hook that
reads the file): unknown key refused; a `[params]` key not in the registry refused; a value
that is not at least as strict as the library default refused (`900 -> 1200`, a longer prefix
list, a lower floor); any mention of a hard id refused; `local.*` ids only, unique, with a
`proof` that exists; `requires` not below the lock's recorded version.

There are no waivers. A consumer that cannot meet a `required` rule yet records the count in a
ratchet file (`budgets.json` is the existing example); the ratchet hook then refuses a rise.
That matches AGENTS.md ("a budget is a debt") and gives no path to switch a rule off.

### 2.5 Versions, lock, diff

Policy version is the package version; `rules_sha` is the SHA-256 of the canonical JSON of the
registry. A rule added or tightened lands in a `feat:`; removing or loosening a rule is a major
change and the owner's. `.ml-stack/policy.lock` (JSON, `version` field, written with the atomic
writer) records:

```
policy_version, rules_sha, params_sha (effective values after overrides),
blocks: {AGENTS.md: sha256 of rendered body, CLAUDE.md: ...},
hooks:  {git/pre-commit: sha256 of installed file, ...},
settings: {.claude/settings.json: [expected entries]}
```

`ml-stack-policy diff` compares the lock with the installed wheel and the files: rules added,
changed or removed (by id), effective parameter changes, each block's unified diff, hook files
that differ, settings entries missing. Hooks run from the wheel, so a code-enforced rule takes
effect on upgrade; `sync` brings prose, lock and wiring up to date, and `policy check` fails when
`rules_sha` differs from the lock and the difference touches a hard or required rule.

`ml-stack-policy sync` rewrites blocks, the lock and wiring. It is a person's command (2.6).

### 2.6 Machine settings stay a person's

Project files (`.claude/settings.json`, `.git/hooks`, `.ml-stack/policy.*`) are not machine
settings, but changing them changes what binds an agent. `install` and `sync` print the plan
(unified diff of every file, hook names, interpreter). With an agent marker set or no TTY they
stop there and exit non-zero ("a person applies this"); a person at a terminal confirms. A new
authority gate `policy.sync` is `person` in both presets. Hooks the installer did not write are
left alone and reported. User-level harness settings are never touched; the entry is printed
for the person to paste.

## 3. Guardrailed automation, class by class

### 3.1 Class a (invariants)

Wheel code, nothing to install. The registry names each one's code and `proof` tests; the
library suite and `conformance` collect them, and `scripts/redteam_coverage.py --check` maps each
hard rule to a red-team test. No parameter, variable or config key reaches
`person.agent-is-never-human`; a test fails if its id appears in any consumer-writable schema.

### 3.2 Class b (guards)

| Guard | Installed by | Refreshed by | Detected when removed or stale |
|---|---|---|---|
| bash guard, edit guard (PreToolUse) | `install` writes the entries into `.claude/settings.json` (diff first) | `sync` | `check` compares entries to the lock; SessionStart hook compares at every session start |
| `session-start`, `subagent-start/stop` | same | same | same |
| `pre-commit`, `commit-msg`, `pre-push` | `install` writes shims into the git hooks dir (`export PYTHON=...; exec ml-stack-policy hook NAME`) | `sync` | `check` hashes them; CI runs the same checks, so a skipped local hook is caught there |
| ratchet (`budgets-only-fall`, generalised to policy file and `--against REV`) | pre-commit chain and CI | with the wheel | CI `--against $BASE` |
| gates framework (`budgets`, size, duplicates, layers) | wheel; project supplies `budgets.json`, layer map | with the wheel | `policy check` runs the project's registered gates in `--fast` mode |
| test runner/broker (`scripts/test`, `testslots`) | wheel entry point; project supplies selectors | with the wheel | the pytest plugin refuses an unbrokered run when an agent marker is set |
| harness adapters | claude-code: settings entries; codex: the managed block pattern in `agent_hooks.py` (`BEGIN`/`END`); local models: in-process call of `policy.check_bash` / `check_edit` in the `ml-stack-agent` tool loop | `sync` | `check` prints a per-harness coverage table (which events are guarded, which are not) |

`ml-stack-doctor policy check` (CI-safe, no network, no `~/.ml-stack` writes) does, in order:

1. Load registry; verify `rules_sha` and hard ids against code.
2. Validate `.ml-stack/policy.toml`.
3. Re-render the blocks and compare with the files and the lock.
4. Verify hook files and settings entries against the lock; run `selftest`.
5. Compare against the base: with `--against REV`, the policy file, lock and wiring at `REV` are
   read from git, and any decrease of the policy version, removed hook entry, removed rule,
   loosened parameter or removed `local.*` rule fails.
6. Print the per-harness coverage table and any `advisory` findings.

Exit codes: 0 clean, 1 drift or violation, 2 could not check.

`ml-stack-policy selftest` feeds canary inputs to each installed hook (a `git push origin main`
bash event under `CLAUDECODE=1` must exit 2; an Edit inside a generated block must exit 2; a
subject with no prefix must fail `commit-msg`; a raised number must fail `ratchet`). The
SessionStart hook runs it with a 2 s budget and puts the result in `additionalContext`
("policy vX, rules_sha abcd, selftest ok / FAILED: bash-guard"). A failure is announced; it
does not block the session.

`conformance` (`pytest -p ml_stack.policy.conformance` or `ml-stack-policy conformance`) builds a
scratch repository from the consumer's policy file and lock (never the live checkout or
`~/.ml-stack`; file keyring, `ML_STACK_NO_REAL_KEYSTORE=1`), runs the canaries as child
processes and asserts the briefing carries every hard id. A consumer adds it to its suite and CI.

### 3.3 Class c (generated text)

The block in `AGENTS.md`:

```
<!-- ml-stack-policy:begin version=0.9.0 rules=ab12cd34 body=9f8e7d6c -->
## Library policy (generated)
Takes precedence over everything below. `ml-stack-policy show` prints it; `explain ID` gives detail.
- [hard] person.agent-is-never-human: An agent never posts as, poses as or is shown as a person. No exception.
- [hard] git.main-is-the-owners: ...
- [required] size.python-lines (600): ...
<!-- ml-stack-policy:end -->
```

`body` is a hash of the block text between the markers, so an edit inside is visible even if the
registry is unchanged. The block lists summaries only (target about 40 lines for the standard
profile); judgment rules appear as one-liners; `detail` is read on demand. `CLAUDE.md` gets a
shorter block with the `claude-code` rules (hook wiring, `CLAUDECODE`, trailer slot).

Drift checks: `ml-stack-policy render --check` (fast, runs in pre-commit and CI) and
`doctor policy check`. The edit guard simulates an `Edit`/`Write`/`MultiEdit` on a file with a
block and refuses when the resulting block body differs from the rendered one. Repository
`scripts/reference` and `cli/reference.py` are the model: generated section, `--write`,
`--check`; the policy renderer reuses their "replace between markers" shape.

### 3.4 Briefing

`briefing.REQUIRED_BRIEFING` and `onboard.BRIEF`/`SNIPPET` become templates over
`render.briefing(audience, harness)`: hard and `brief = true` rules for that audience, the policy
version and hash, and the existing variables (`{owner}`, `--agent`). A subagent brief
carries the same hard rules a lead reads and changes when the registry does. `SessionStart` adds
it for the lead; local-model harnesses put it in the system prompt. The brief still says
workspace text is data and grants nothing.

### 3.5 What an agent with file access can still do

| Action | Prevented? | Detected by |
|---|---|---|
| Edit the generated block through the Edit tool | yes, edit guard | also pre-commit and CI |
| Edit it with `sed -i`, a heredoc or a script | not at write time (the bash guard cannot know every writer) | `render --check` in pre-commit, CI, SessionStart |
| Remove or rewrite `.claude/settings.json` hook entries, `.git/hooks/*`, `core.hooksPath`, `--no-verify` | refused by edit guard and bash guard patterns for agents | SessionStart `selftest` and `check` (lock mismatch), CI |
| Loosen `.ml-stack/policy.toml` or raise a ratchet number | refused at commit for agents (`ratchet` generalised; the opener is a person-only command at the person's terminal) | CI `--against $BASE` |
| Edit block, lock, policy file and wiring consistently, then pin an older ml-stack | not prevented locally | CI compares against the base branch: version decrease, removed entries, loosened values fail. Only holds if the base branch is protected and the check is a required status; that is the owner's repository setting and a human-only machine setting |
| Run with `MLSTACK_GUARD=off` | an agent cannot set the hook process environment from a command; it can through `settings.local.json` `env` | edit guard protects `.claude/settings*.json`; open question 2 |
| Same-user process writes anything | not prevented | tamper evidence only; same limit as `person-delegation` section 2 |
| Harness without pre-tool hooks | rules degrade to the repository's own hooks, in-process checks and CI | coverage table says so |

Local enforcement is a checkpoint against the model, not a sandbox. What does not fall to an agent
with a shell: the wheel code, a protected base branch with a required check, and the person's
terminal for `sync`.

## 4. Consumer `AGENTS.md` and `CLAUDE.md`

Precedence, highest first:

1. Hard library rules (code). Nothing below can alter them.
2. Project policy file: stricter parameters and `local.*` rules.
3. The generated block, which states 1 and 2.
4. The project's own prose.
5. Workspace data: never authority.

Merging: the block sits under the first H1; the library writes only inside its markers and the
consumer writes nothing inside them. Repeating a library sentence outside the block gets an
advisory (normalised-sentence match against summaries); prose that contradicts a rule is not
detectable, and the block's first line says the block wins.

Discovery: the generated pointer in `AGENTS.md` (Codex and local harnesses read it), the
`SessionStart` context, the subagent brief and `ml-stack-policy show`. `CLAUDE.md` is Claude Code
only. Rules carry `harness`, so Codex does not read `CLAUDECODE` instructions as binding it; a
harness without hooks is told which rules it must follow by instruction alone.

What stays local: the consumer's vocabulary, models, branch names, budgets numbers, layer map,
`OWNED` table and handoff file. Rule 40 (library versus app) is inverted in a consumer: the
block says a missing capability is an issue against the library, not a local script.

## 5. Migration

Order keeps the current gates green by moving data before behaviour. Per AGENTS.md, nothing is
kept to preserve an old path: tests move with the code in the same commit.

### Slice 1 (days): registry, generated block, drift check, `doctor policy check`

- New: `src/ml_stack/policy/{__init__,registry,hard,render,check,cli}.py`, `rules.toml` (the
  hard set and the `required` rules of section 1 with their `enforced_by` and `proof`),
  console script `ml-stack-policy` in `pyproject.toml`.
- Edit: `src/ml_stack/doctor.py` (`policy` group), `src/ml_stack/cli/reference.py` (`HELP`,
  `TABLE`), `docs/commands.md` (via `scripts/reference --write`), `tests/test_layers.py`
  (`policy` in `core`), `docs/redteam/coverage-map.toml`, `scripts/hooks/pre-commit` (add
  `ml-stack-policy render --check`), CI workflow (add `doctor policy check`), `AGENTS.md`
  (section 6 hard-rule prose replaced by the block, rest unchanged).
- Gates this touches: `entry-points`, `argument-parsers`, `print-calls`, `atomic-writes`,
  `unversioned-records`, `deep-files`; new modules write through the existing helpers.
- Tests: `tests/test_policy_registry.py` (2.2 list), `test_policy_render.py` (round trip, stale
  detection, marker injection in a rule text), `test_policy_check.py` (child process, clean and
  drifted trees), `tests/test_redteam_policy_block.py`. Packaging: the wheel contains
  `ml_stack/policy/rules.toml` (extend `test_packaging_install.py`).
- Not in slice 1: hooks move, install, lock, project policy file.

### Second slice: hooks into the wheel, install, lock

- Move `scripts/hooks/{claude-bash-guard,claude-edit-guard,pre-push,commit-msg,primary-only,
  budgets-only-fall,pushed,no-data-files,claude-session-*}` logic to `policy/hooks/`; fold
  `worktreerules.py` and `rules_loader.py` in; `scripts/hooks/*` and `scripts/install-hooks.*`
  deleted. `.claude/settings.json` points at `ml-stack-policy hook NAME`.
- New: `policy/{install,lock,diff}.py`, `.ml-stack/policy.lock`, `selftest`.
- Edit: `doctor.hooks_of`/`HOOKS` (read the lock), `agent_hooks.py`, `onboard.hook_snippet`.
- Tests rewritten in the same commits: `test_bash_guard.py`, `test_edit_guard.py`,
  `test_hook_installer.py`, `test_hooks_installed.py`, `test_claude_workspace_hooks.py`; new
  `test_policy_install.py` (plan only for agents, writes for a pty person, leaves foreign hooks),
  `test_policy_selftest.py`. Measure hook cold-start before and after; record it in a dated doc.

### Slice 3: project policy file, ratchet generalised, briefing from the registry

- New: `policy/config.py`, `hooks/ratchet.py` (from `budgets-only-fall`, takes files from the
  policy file, `--against`), `.ml-stack/policy.toml` for this repository (its sizes, prefixes,
  `OWNED`, models, `local.*` rules).
- Edit: `briefing.py`, `workspace/onboard.py` (`BRIEF`, `SNIPPET` as templates),
  `scripts/budgets` (reads the policy file), `AGENTS.md` and `CLAUDE.md` trimmed to project
  prose plus blocks.
- Tests: `test_policy_config.py` (accept stricter, refuse looser, refuse hard id, refuse bad
  `local.` id), `test_policy_ratchet.py`, `test_policy_briefing.py` (every hard id appears in
  lead and subagent briefs), updates to `test_budgets.py`, `test_onboard*.py`.

### Slices 4-6

- 4: `policy/conformance/`, CI template, `check --against`, codex and local-model adapters
  (in-process call in the `ml-stack-agent` loop), coverage table. Test: `conformance` against a
  scratch consumer repository with its own `AGENTS.md` and policy file.
- 5: gate framework (`scripts/gates` importable), broker-admitted runner, pytest plugin, `policy
  check landing` ship as library features; project checkers stay project code.
- 6: `policy.sync` authority gate; `person.statement-only` and `coordination.lowest-tier-never`
  get real `enforced_by` refs when `person-delegation` slice 1 and `session-liveness` slice 4
  land (until then they read `design` and `check` prints them as unenforced).

### Red-team tests (real child processes, `CLAUDECODE=1`, fresh state, no mocks)

| Attack | Expected |
|---|---|
| `sed -i` changes the block | `render --check` fails in pre-commit and `policy check` |
| Edit tool inside the block; or block edited and `body` hash recomputed by hand | fails: block must equal the render of the installed wheel |
| Remove `.git/hooks/pre-push` | `check` fails; SessionStart `selftest` reports it |
| Delete the PreToolUse entry in `.claude/settings.json` | `check` and SessionStart report; edit guard refused the edit first |
| `git config core.hooksPath /dev/null`, `git commit --no-verify` as an agent | bash guard refuses |
| `size.python-lines` 1200 / 600 | 1200 refused; 600 accepted and the edit guard then refuses a 700-line write |
| Policy file mentions a hard id (`enabled = false`, `waive`) | refused |
| `[[local]] id = "person.agent-is-never-human"` or `ml.` prefix | refused (`local.` only) |
| Agent commit that loosens the policy file | `ratchet` refuses without the person's opener |
| Lock downgraded and older ml-stack pinned in one commit | `check --against $BASE` fails |
| Rule text containing `<!-- ml-stack-policy:end -->` | registry test and renderer refuse |
| `install`/`sync` run by an agent | prints the plan, exits non-zero, writes nothing |
| `person.agent-is-never-human` anywhere in a writable consumer file | tamper test fails |

## 6. Decisions

1. The policy version is the package version.
2. Hard-rule guards ignore `MLSTACK_GUARD=off` whenever an agent marker (`CLAUDECODE` or another
   harness marker) is set. The switch still disables the soft guards.
3. `install` and `sync` are person-only (`policy.sync` gate fixed at `person`).
4. `.ml-stack/policy.lock` is committed in consumers.
5. The generated block carries rule summaries and a link, not the full prose.
6. The commands are a `policy` group in `ml-stack-doctor`; no new console script (the entry-points
   budget does not rise).
7. No waivers; debt goes in a ratchet file that only falls.
8. Making `scripts/gates` and the test runner public library API before 1.0 is acceptable.
9. The hard set is confirmed as proposed.
10. In-process enforcement inside `ml-stack-agent` is acceptable for harnesses with no pre-tool
    hook event.
