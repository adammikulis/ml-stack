# Agent instructions

The rules for every coding agent working in this repository (Claude Code, Codex, a local model,
any subagent) are in [CLAUDE.md](CLAUDE.md). Read it first and follow it; this file adds nothing
to it and is only here so tools that look for `AGENTS.md` find the same rules.

Most often needed from it:

* Work in your own git worktree on your own branch; never edit the primary checkout.
  Create it beside the checkout, not inside it. Before reporting completion, land the work,
  check for unique files and commits, remove the worktree and merged branch, prune, and verify
  the path is absent from `git worktree list`. See CLAUDE.md, "Worktrees", for the full checks.
* Join the workspace and use it: `ml-stack-workspace connect` (a person runs it) gives you a paste;
  a subagent needs no invite and acts as its parent with `--label` (section "Subagents join the
  workspace automatically").
* No version numbers anywhere (the owner sets them), no push, tag or release.
* Tests never touch the real Keychain; budgets and red-team counts only fall.
