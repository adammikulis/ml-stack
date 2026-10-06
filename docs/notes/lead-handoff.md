# Lead handoff (Codex and local Qwen models)

State as of the last merge on `0.2dev` (70eadc0b). Rules first: no version number anywhere (the
owner sets it), no push, tag or release by an agent, no real Keychain in tests, budgets and
red-team counts only fall, human-only floor untouched, agent messages short. Everything in
CLAUDE.md applies; this note only says where things stand.

## How work lands
* Work in your own worktree and branch from `integration/dev` (the lead's staging branch); the
  workers test their own affected behavior. The lead reviews the consolidated integration batch,
  runs shared structural/security gates once before publication, and handles full end-to-end
  verification. The lead schedules the full tier in the background after the batch.
* Generated files are regenerated, never hand-merged: `python3 scripts/redteam_coverage.py --write`,
  `python3 scripts/reference --write`; roles codemod `docs/notes/rename-roles-codemod.py` after a merge.
* Announce on the workspace: `ml-stack-workspace announce joined|milestone|done|blocked TEXT`; the
  lead's identity is `claude-code`. Subagents act as `--agent claude-code --label NAME`.

## Landed on 0.2dev
Boards, threads, DMs, subscriptions, live chat; quiet defaults (#announcements, bounded reads,
nudge); Requests inbox; roles `read-only` / `approve-first` / `plan-and-go`; wired-memory command
and slider; model identity (verified vs claimed); files on boards via the graph (attach, file,
file search); agent-made invites (bounded); harness launchers `ml-stack-codex` / `ml-stack-claude`
(broker lease, 256K, pre-tool gate); local-model agents (`ml-stack-workspace agent start`, chat
and coding profiles); `ml-stack-serve up` as a broker lease with a grant required to start a
server; test-speed work (gate tier, ratchet); Codex's scheduler chain.

## Open, in priority order
1. **Full tier is not trustworthy until Codex fixes the scheduler crash**: `KeyError 'minimum'`
   in scripts/testslots.py `_grant` kills a worker (xdist INTERNALERROR) so most tests never run;
   also one 'waiting' line per test floods output. Then run `scripts/test full` twice on a quiet
   machine and lower `tests/full-tier-time.json` with `scripts/test ratchet --update`. Unmerged
   commit on feat/test-speed (693c0c00, `-rfE` in scripts/test): merge it.
2. Codex: unified chat UI (one composer, sidebar, Coding mode), Board in the Fleet shell; the
   Agents panel and Requests panel mount (routes: `localroute.respond`, `ml_stack.inbox.route`).
3. Never run Flash-Next tests; the 27B (Qwen3.8-27B-UD-Q4_K_XL, 256K, q8 KV, MTP) comparison via
   `scripts/compare-harnesses` has not been run with real numbers (first-turn tokens, cache ratio).
4. Not run: a Linux pass (`scripts/test-on-linux`); the real macOS admin prompt for the wired
   limit; the Claude Code pre-tool hook live; Requests approvals with the keystore on.
5. Known gaps: the coding agent path (`agent start --profile coding` -> Codex harness) has not run
   end to end; held files raise no Requests item; embedding search over files; invite policy
   cannot see what an external agent read; a `notice_storm` test flake under load.
6. The self-improving loop (`docs/notes/self-improvement-loop.md`) and compute plan
   (`docs/notes/compute-plan.md`) are designs, not built: first measure utilisation and write the
   27B profile, then the usage-aware router, then the idle worker.

## Owner-only
Version number; push/tag/release; GitHub issues 9, 12, 14; publication; the wired-limit raise for
Flash-Next (`ml-stack-serve memory --for ... --apply`, needs the administrator password); the
decision to move `0.2dev` to `main`.
