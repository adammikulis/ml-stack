# Agent sandbox

Status: prepared 2026-10-08, nothing installed. The owner decided agents run under an operating
system sandbox. This document is the target profile for each harness, what each option closes,
how our workflows behave under it, and the order in which the owner checks it. Machine settings are
a person's (AGENTS.md section 6): an agent runs `scripts/agent-sandbox prepare`, and the person
installs what it staged.

Every claim about sandbox behaviour is **PREDICTED** unless marked **VERIFIED**. An agent cannot
turn the sandbox on for its own session, so the first real run is the owner's (acceptance steps
below). Source for the Claude Code facts: https://code.claude.com/docs/en/sandboxing and the
settings reference, read 2026-10-08, Claude Code 2.1.294. macOS 26 behaviour of Claude Code's
Seatbelt profile is unverified. [sandbox.md](sandbox.md) covers `ml_stack.sandbox`, the policy
engine ml-stack uses for commands it starts itself; this document covers the harnesses.

## What the built-in sandbox covers

| Surface | Covered |
|---|---|
| Bash, PowerShell, Monitor commands and their children, subagent Bash | yes |
| Read, Edit, Write tools | no: permission rules only |
| WebFetch | no |
| Local MCP servers, LSP | no |
| Command hooks | no: they run as the person |

Two consequences shape the profile. A sandboxed command that can write a hook script hijacks the
next hook run, so hook scripts, settings files and git hooks are write-denied. The file tools are
outside the sandbox, so the same paths are also refused by `permissions.deny` rules.

## Three options

| | Built-in sandbox (managed settings) | sandbox-runtime around the whole process | Separate OS account |
|---|---|---|---|
| Cost | one root-owned JSON file | `srt` (Node) plus a launcher change for each harness | an account, its own checkout, credentials and caches |
| Memory | one Seatbelt/bubblewrap process per command | one wrapper per harness | a second login session |
| Friction | loopback ports, ssh, sibling worktrees (see matrix) | the same, and the harness itself is confined: its own API host must be listed | highest: every shared path needs a group, ACL or copy |
| File tools, MCP, hooks | not covered | covered | covered |
| Reading tokens and `.owner` of other agents | closed by `denyRead` (Bash only) | closed for every tool | closed by file mode |
| Forging transcripts, the person store | Bash closed; Edit/Write closed by permission rules | closed | closed |
| Editing hooks and settings | Bash closed by `denyWrite`; file tools by rules; managed file is root-owned | closed | closed (owner-owned checkout) |
| pty tricks (a command attaching to the person's terminal) | open: sandbox does not isolate ttys | open | closed: different user cannot open the person's ttys |
| launchd, cron, at persistence | LaunchAgents and systemd dirs denied; `cron` and `at` spools are root-owned system paths, not listed | same | closed: the account's own crontab is the agent's |
| pip install into runtimes | `~/.ml-stack/runtimes` and `~/.pyenv/versions` denied | same | closed |
| gh and ssh credentials | `~/.config/gh`, `~/.ssh`, `SSH_AUTH_SOCK` denied or scrubbed | same | closed: nothing to read |
| Residual | a command that is allowed to reach the board can write the board's files (below) | same | the agent still has the board token it needs |

Recommendation: install the built-in sandbox now (all of it is one root-owned file, and
`scripts/agent-sandbox prepare` already produces it). Use sandbox-runtime for the local-model
harness, which Claude Code's setting does not reach. Revisit a separate account when the board is
reached through the daemon instead of direct file writes, because that is what the account would
protect that the sandbox cannot.

## Residual gaps the sandbox does not close

- **Board files.** `ml-stack-workspace` writes `~/.ml-stack/workspace` (board.db, audit.jsonl,
  claims.json) from inside the agent's own process. A sandboxed agent that may announce may also
  edit those files. Closing it needs the CLI to talk to the daemon (loopback TLS) with
  `workspace/` read-only to agents. Pending entry in HANDOFF.md.
- **The agent's own token.** `workspace/tokens/claude-code` stays readable (`allowRead`) because
  the commands that authenticate as the lead read it. Other agents' tokens and the person store
  are denied.
- **ttys.** No option except a separate account isolates terminals.
- **Hooks run unsandboxed.** Protected by write denial, not by isolation.
- **`cron` and `at`.** Seatbelt cannot name them as files; the profile relies on the system
  spool directories being root-owned. PREDICTED.

## Claude Code: managed settings

Written to `/Library/Application Support/ClaudeCode/managed-settings.json` (macOS) or
`/etc/claude-code/managed-settings.json` (Linux, WSL) by the person. Managed settings outrank
repository and user files. `allowUnsandboxedCommands: false` makes the sandbox admin-required
(Claude Code 2.1.285 and later): a repository's `excludedCommands`, `allowedDomains`,
`allowWrite`, `enabled: false` and `additionalDirectories` are ignored, and the
`dangerouslyDisableSandbox` retry is removed. `failIfUnavailable: true` stops Bash instead of
running unconfined.

The values come from `scripts/agent_sandbox_profile.py`; the file the person installs is whatever
`prepare` printed, not this summary.

| Key | Value |
|---|---|
| `sandbox.enabled`, `failIfUnavailable` | `true` |
| `sandbox.excludedCommands` | `ml-stack runtime ensure`, `ml-stack runtime rollback`, `ml-stack runtime restart-host`: the three deploy commands, guarded by the `runtime.deploy` authority gate (see the matrix) |
| `sandbox.allowUnsandboxedCommands`, `autoAllowBashIfSandboxed` | `false` |
| `sandbox.filesystem.allowWrite` | every checkout from `git worktree list` (explicit paths: Linux does not expand wildcards), the primary `.git`, the scratch and temp directories, `~/.ml-stack/{workspace,logs,activity,requests,holds,hook-diagnostics}` and the broker, server and lock files, `~/.cache/{dev-test-slots,ml_stack,huggingface,pip,ms-playwright}` |
| `sandbox.filesystem.denyWrite` | per checkout `scripts/hooks`, `.claude`, `.mcp.json`, `.githooks`, `.git/config`, `.git/hooks`; in the primary `.git`: `config`, `hooks`, `info`, `worktrees/*/config.worktree`, `worktrees/*/hooks`; `~/.claude`, `~/.codex`, `~/.ssh`, `~/.aws`, `~/.gnupg`, `~/.config/gh`, `~/.gitconfig`, shell rc files, `~/.pyenv/{versions,shims}`, `~/.local/bin`, `~/Library/LaunchAgents` (macOS) or the systemd user and autostart directories (Linux); under `~/.ml-stack`: `runtimes`, `keystore`, `hooks`, `gate`, `guard`, `machine-id`, `wired-limit.json`, and in `workspace` the tokens, local-agents, device-accounts, invites |
| `sandbox.filesystem.denyRead` | `~/.ssh`, `~/.aws`, `~/.gnupg`, `~/.config/gh`, `~/.netrc`, `~/.git-credentials`, `~/.npmrc`, `~/.pypirc`, browser profiles (Chrome, Firefox, Safari, Keychains, Cookies), and under `~/.ml-stack`: `workspace/tokens`, `workspace/local-agents`, `workspace/device-accounts.db*`, invites, `workspace-remote`, `keystore`, `server-keys.json`, `cluster.key.old`, `cluster.json`, `credentials.*`, `onboard` |
| `sandbox.filesystem.allowRead` | `~/.ml-stack/workspace/tokens/claude-code` |
| `sandbox.network.allowedDomains` | `huggingface.co`, `*.huggingface.co`, `hf.co`, `*.hf.co` (hub, CDN, LFS and xet hosts), `github.com`, `api.github.com`, `codeload.github.com`, `objects.githubusercontent.com`, `release-assets.githubusercontent.com`, `raw.githubusercontent.com`, `pypi.org`, `files.pythonhosted.org`, `127.0.0.1:8770` |
| `sandbox.network.allowLocalBinding` | `true` on macOS only |
| `sandbox.credentials.envVars.deny` | `CLAUDE_CODE_MESSAGING_SOCKET`, `CLAUDE_CODE_MESSAGING_TOKEN`, `ML_STACK_HOME`, `ML_STACK_CACHE`, `ML_STACK_AGENT`, `ML_STACK_WORKSPACE_AGENT`, `ML_STACK_NET_ALLOW_HOSTS`, `HF_ENDPOINT`, `HF_TOKEN`, `GH_TOKEN`, `GITHUB_TOKEN`, API keys, `SSH_AUTH_SOCK`, `PYTHONPATH`, `PYTHONSTARTUP`, `PYTHON_KEYRING_BACKEND`, `GIT_CONFIG_*`, `GIT_SSH_COMMAND`, `GIT_ASKPASS`, `PIP_*INDEX_URL`, `BASH_ENV`, `ENV` |
| `permissions.deny` | `Edit` and `Write` rules for every `denyWrite` path |
| `env` | `CLAUDE_CODE_SUBPROCESS_ENV_SCRUB=1` |

The host list is the one in `ml_stack.net.policy.ALLOWED` (hub and GitHub release hosts) plus
`pypi.org` and `files.pythonhosted.org` from `ml_stack.net.packages`. The geocoder is left out.
`*.hf.co` covers the LFS CDN and xet bridge hosts the hub redirects to; check the first real pull
against the proxy log and add a host if one is refused.

Unverified in the settings schema: the exact shape of `sandbox.credentials.envVars` (taken from
the owner's brief), and that `denyWrite` beats a parent `allowWrite` (the primary `.git` is
allowed and `.git/config` and `.git/hooks` inside it are denied). Acceptance steps 4 and 5 test
both. If `credentials.envVars` is rejected, `CLAUDE_CODE_SUBPROCESS_ENV_SCRUB=1` alone removes
credential-like variables and the model-facing launcher unsets the rest.

### Why `ML_STACK_HOME` and `ML_STACK_AGENT` are scrubbed

Both redirect where the workspace looks for its identity. An agent that sets `ML_STACK_HOME` to
a directory it controls can mint a state root with its own tokens; `ML_STACK_AGENT` selects which
token a command reads. The harness launcher sets the agent name before the sandbox starts, and the
managed file removes the variables from commands. `PYTHONPATH`, `PATH` shadowing, `BASH_ENV` and
`GIT_CONFIG_*` are the environment red-team's search-path and config-injection variables.

### Hooks

Hooks are unsandboxed and the repository wires them with `$CLAUDE_PROJECT_DIR/scripts/hooks/...`,
so each checkout's copy runs. Decision: **protect by `denyWrite` and permission rules**, not by
moving the scripts. The scripts import repository code (`rules_loader.py`, `ml_stack`), so a
root-owned copy would drift from the tree it guards, and `allowManagedHooksOnly` would disable the
repository's own hook entries unless the managed file duplicated them. The cost: an agent cannot
edit `scripts/hooks`; the owner or a Claude Code session without the sandbox does. When the policy
distribution installs hooks from a wheel (docs/policy-distribution.md section 3.2), move the
entries to the managed file and add `allowManagedHooksOnly`.

### Git across worktrees

A linked worktree's `.git` is a file pointing at `<primary>/.git/worktrees/<name>`. Commits write
the shared `objects`, `refs`, `logs` and that per-worktree directory, so the primary `.git` is
writable. Its `config`, `hooks`, `info` and each worktree's `config.worktree` and `hooks` are not,
which blocks the two ways a command makes git run code. `.claude/worktrees/agent-*` checkouts sit
inside the primary checkout and need nothing extra; sibling `../ml-stack-*` checkouts are listed by
`prepare` from `git worktree list`, so a new sibling needs `prepare` and an install again.

## Codex: config.toml

`prepare` stages a fragment for `/etc/codex/config.toml` (root-owned, so the agent cannot edit it):

```toml
sandbox_mode = "workspace-write"
approval_policy = "on-request"

[sandbox_workspace_write]
network_access = false
exclude_tmpdir_env_var = false
exclude_slash_tmp = false
writable_roots = [ ...the same write roots as above... ]
```

Seatbelt on macOS and bubblewrap on Linux and WSL enforce it. `network_access = false` means no
pip, no Hugging Face pull and no HTTPS git: those run through a person or an escalation
(`on-request`). `approval_policy = "never"` makes a refused command fail without asking; the owner
picks. Codex has no per-path deny list in this fragment, so `~/.ssh` and the token files are
protected only by what workspace-write leaves unreadable (reads are broad by default) and by the
checkout rule that `.git` and `.codex` under a writable root are read-only. PREDICTED; the
sibling-worktree `.git` pointer needs the primary `.git` as a writable root, which also makes
`config` and `hooks` writable. That is a known weaker result than Claude Code's; use
sandbox-runtime for Codex when it matters.

## Local-model harness: sandbox-runtime

`ml-stack-agent` and `ml-stack-chat` run as one process that also reads files and starts
commands. Wrap the whole process:

```
srt --settings "/Library/Application Support/ClaudeCode/srt-settings.json" -- ml-stack-agent ...
```

`srt-settings.json` carries the same `denyRead`, `allowWrite`, `denyWrite` and host list as the
Claude profile, plus `api.anthropic.com` for a Claude-backed harness. A local model reaches its
served port on loopback; the lease from `ml-stack-serve up` is taken by a process outside the
wrapper, and the wrapped agent connects to it. Reads outside `denyRead` are allowed (the
sandbox-runtime model, sandbox.md). For commands the agent loop starts itself, `SandboxedBash`
(`ml_stack.sandbox.tools`) applies a stricter allow-list policy and stays the default.

## Compatibility matrix

Expected result under the Claude Code profile. Only the unsandboxed behaviour of a command can be
VERIFIED from an agent; each sandboxed result is PREDICTED until the owner's steps run.

| Workflow | Expected result | Status | Fix when it fails |
|---|---|---|---|
| `scripts/test` lock directory `~/.cache/dev-test-slots` | works: in `allowWrite` | PREDICTED (command itself VERIFIED unsandboxed 2026-10-08) | add the directory |
| `scripts/test` loopback RPC endpoint (`testslots_rpc` binds `127.0.0.1` with an ephemeral port) | works on macOS with `allowLocalBinding`; on Linux and WSL an ephemeral port cannot be listed in `allowedDomains` and the connect is refused | PREDICTED | give the broker a fixed port setting and list it; until then run Linux tests from a person's terminal |
| `scripts/test` temp files and pytest tmp | works: temp directories in `allowWrite` | PREDICTED | add the directory the run reports |
| `scripts/land`, `git worktree add/remove`, commit in a sibling worktree | works for listed checkouts; a new sibling fails with `Operation not permitted` | PREDICTED | rerun `prepare`, install |
| Edit `scripts/hooks/*` or `.git/hooks` | refused by design | PREDICTED | the owner edits outside the sandbox |
| git fetch/push over HTTPS to github.com | works through the proxy; credentials come from a token the person's helper supplies, not `~/.config/gh` | PREDICTED | none, or `excludedCommands: ["git fetch *"]` in managed settings if the proxy breaks a TLS handshake |
| git over SSH | fails on macOS (SSH is blocked inside the sandbox; `~/.ssh` and `SSH_AUTH_SOCK` are denied) | PREDICTED | switch the remote to HTTPS |
| `git push` to a local bare repository (file transport) | works in a temp directory | **VERIFIED** unsandboxed (`tests/test_agent_sandbox.py`) | |
| `ml-stack-serve up/down`, a model server on a loopback port | the lease and a server started by the broker run outside the agent's sandbox; the agent's client connects to its port, which needs `allowLocalBinding` (macOS) or the port in `allowedDomains` (Linux) | PREDICTED | list the server port |
| GPU work (Metal) from an agent command | an agent does not start model servers by hand; the broker does | PREDICTED | none |
| pip through `ml_stack.net.packages` | reaches `pypi.org`, `files.pythonhosted.org`; installs only into a venv inside a checkout | PREDICTED | venv paths are under the checkout |
| `ml-stack runtime ensure`, `rollback`, `restart-host` | write `~/.ml-stack/runtimes` and the pyenv launchers, which the sandbox denies, so the staged settings list these three commands in `sandbox.excludedCommands` and they run outside it. The guard is the `runtime.deploy` authority gate, checked in the command: an agent passes only while the owner has delegated it (Dev), is refused while it is the person's (Prod, or after `ml-stack-workspace authority set person runtime.deploy`), and every pass or refusal is a `runtime.deploy` activity record naming the agent. Only these exact commands are excluded; `--allow-unmerged` and another repository stay a person's | PREDICTED (excluded-command matching of compound commands is unverified; a harness wrapped by sandbox-runtime cannot exclude commands and leaves the deploy to a person) | none |
| `packaging/build.py` | writes `dist/` and `.build-venv` inside the checkout, installs from PyPI | PREDICTED | none |
| playwright browsers | the browser binary in `~/.cache/ms-playwright` is readable; launching a headed window is not blocked by Seatbelt on the profile, but loopback servers need `allowLocalBinding` | PREDICTED | run headless; list the cache directory |
| the pool board client (`127.0.0.1:8770`, TLS) | works: the entry is in `allowedDomains` | PREDICTED | |
| `ml-stack-workspace announce/claim/inbox` | works: reads its token (allowRead), writes `workspace/` | PREDICTED | |
| hooks reading the transcript (`~/.claude/projects`) | unaffected: hooks run unsandboxed | PREDICTED (documented behaviour) | |
| Agent writes `~/.claude/projects/...` | refused | PREDICTED | |

## Acceptance steps for the owner

Do these at your own terminal, in order. Steps 1 to 3 run no sandbox.

1. `scripts/agent-sandbox prepare` in the primary checkout. Expect the staged directory, three
   file lines with 12-character digests, and one `sudo sh -c '...'` line.
2. Read `~/.ml-stack/agent-sandbox/staging/claude-managed-settings.json`. Expect
   `allowUnsandboxedCommands: false` and the worktree paths you use.
3. Run the printed `sudo` line. Expect `claude-managed-settings.json: OK` (and two more `OK`),
   no other output.
4. `scripts/agent-sandbox status`. Expect three `match` lines under `staged for device ...`.
5. Start `claude` in the primary checkout, run `/sandbox`. Expect the sandbox on and shown as
   admin-required (settings from managed policy).
6. In that session ask for `scripts/agent-sandbox probe` (a Bash call). Expect every line `PASS`
   or `SKIP`, then `this session is sandboxed`. A `FAIL` names the path or variable that stayed
   open.
7. Ask for `touch ~/.agent-test`. Expect `Operation not permitted`, with no retry offer.
8. Ask for `git commit --allow-empty -m "chore: test"` in a sibling worktree, then
   `echo x >> .git/hooks/x`. Expect the commit to work and the hook write to fail. This tests
   whether `denyWrite` beats the parent allow.
9. Ask for `env | grep -c CLAUDE_CODE_MESSAGING`. Expect `0`.
10. Ask for `ml-stack-workspace announce joined 'sandbox acceptance' --agent claude-code --label acceptance`.
    Expect the announcement to be recorded. Failure here means the token read or the `workspace`
    write is blocked.
11. Ask for `scripts/test all -n 1 tests/test_agent_sandbox.py`. Expect `22 passed`. On Linux a
    refused connection from the broker is the loopback gap in the matrix.
12. Ask for `git push` of a throwaway branch over HTTPS. Expect success; an SSH remote fails.
13. Ask for `curl -sI https://example.com`. Expect a proxy refusal.
14. Ask for `ml-stack-serve status`. Expect the status; if it cannot reach the server port,
    follow the matrix row.
15. For each failure, report the denied path or host from the sandbox's log line. `prepare` after
    the lists change replaces the staged files; the install command checks their digests again.
