# Agent instructions

The rules for every coding agent working in this repository (Claude Code, Codex, a local model,
any subagent) are in [CLAUDE.md](CLAUDE.md). Read it first and follow it; this file adds nothing
to it and is only here so tools that look for `AGENTS.md` find the same rules.

Most often needed from it:

* Work in your own git worktree on your own branch; never edit the primary checkout.
* Join the workspace and use it: `ml-stack-workspace connect` (a person runs it) gives you a paste;
  a subagent needs no invite and acts as its parent with `--label` (section "Subagents join the
  workspace automatically").
* No version numbers anywhere (the owner sets them), worktree agents commit tested leaves; the coordinator reviews and publishes the integration development branch after required checks. No force push, main push, tag or release.
* Tests never touch the real Keychain; budgets and red-team counts only fall.
