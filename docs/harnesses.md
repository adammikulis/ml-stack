# Claude Code and Codex on a local model

`ml-stack-codex` and `ml-stack-claude` run the Codex and Claude Code command-line agents on a model
this machine serves, with ml-stack's rails around them: the served window, a pre-tool classifier
hook under a role, the Requests inbox for approvals, and a workspace identity of their own. Codex is
the default coding harness (a lighter system prompt and tool list); Claude Code is there for when
its richer hook gate matters.

```
ml-stack-codex                                  # Qwen3.8-27B-UD-Q4_K_XL, role approve-first, this directory
ml-stack-codex Qwen3.5-2B --role plan-and-go --project ~/work/repo -- exec "fix the failing test"
ml-stack-claude --on http://127.0.0.1:8080 --role read-only -- -p "explain this repo"
```

A program starts the same thing with `ml_stack.coding.launch_coding_agent(model, role, project,
harness="codex", **options)` (options: `name`, `orders_from`, `harness_args`, `say`,
`run_codex` / `run_claude`); it returns the harness's exit code.

## What a launch does

1. **Serves the model through the broker lease** (never a hand-started server): 262,144 tokens on
   one slot, q8_0 KV cache, the model's own multi-token-prediction head where it has one, the
   model's measured settings when there are some. The model is asked for before it is loaded: when
   the wired-memory limit cannot hold it at that window, the launcher prints
   `ml-stack-serve memory --for MODEL --ctx 262144 --kv q8_0 --apply` and stops. Raising the limit is
   for a person at their own terminal. `ml-stack-serve memory --for Qwen3.8-27B-UD-Q4_K_XL.gguf --ctx
   262144 --kv q8_0` says what the default needs (29.2 GiB on the measured machine, 2026-10-03).
   `--ctx N` is the total across `--slots` (default 1); `--on URL` uses a server already up.
2. **Tells the harness the window it was given**, so it neither compacts at a small default nor
   assumes a larger one. Claude Code: `CLAUDE_CODE_MAX_CONTEXT_TOKENS` and
   `CLAUDE_CODE_AUTO_COMPACT_WINDOW` (accepted from 100,000, capped at the model's window) are set to
   the served slot context. Codex: `model_context_window` and `model_auto_compact_token_limit`
   (90 percent of the window) in its generated config.
3. **Writes the session's settings outside the working tree**: a private directory under the state
   root (`harness/<id>`), the files read-only and, for Claude Code, the directory too. Nothing is
   written to `~/.claude` or `~/.codex`: Claude Code gets `--settings FILE` and
   `--append-system-prompt-file FILE`; Codex gets its own `CODEX_HOME` with `config.toml` and
   `AGENTS.md`, no auth file and no API key in its environment. The directory is removed when the
   harness exits.
4. **Installs the hooks** (below).
5. **Mints a workspace identity** and announces it (below).

## The rails

| role | Codex's own mode | the hook |
| --- | --- | --- |
| `read-only` | sandbox `read-only`, approvals `never` | allows reads; denies every acting call |
| `approve-first` (default) | sandbox `workspace-write`, approvals `on-request` | allows reads; asks for every acting call |
| `plan-and-go` | sandbox `workspace-write`, approvals `on-request` | allows reads and reversible calls; asks for destructive and unsure ones |

The hook is `python -m ml_stack.harnesshook pre`, a PreToolUse command hook that both harnesses run
before every tool call (Claude Code: every tool; Codex: Bash, `apply_patch`, MCP and other local
function tools). It labels the call with the destructive-action classifier
(`docs/destructive-actions.md`, deterministic layer) and applies the role. A call the classifier
cannot read, or a tool it does not know, is `unsure` and asks. An ask is a request in the Requests
inbox (`docs/requests.md`) that the hook waits on, up to five minutes; only the person answers it
(`ml-stack-requests`), never the model. Not answered, expired, or a request store that cannot be
opened (the hook runs without a terminal, so it never prompts the keystore) is a denial.

Denied in every role: a call whose text names the session's files or the state root, and a shell
command that asks for an action only a person can take (answer a request, change a role, release
quarantine; the human-only table in `chatpolicy.py`). The role, the label and the paths are on the
hook's command line, written by the launcher; nothing in the environment, the tool arguments or
a file the model reads changes them. A hook that crashes exits 2, which both harnesses read as a
block. The role names are those of `docs/agent-roles.md`.

The hook's output is a fixed sentence per outcome, so it adds a few identical tokens to a
conversation and nothing that changes the prompt prefix.

### The workspace

A person-started launcher mints the session's identity the way `ml-stack-workspace setup` does: the
standard agent role, a token file readable by this user only, nothing printed and no invite code. The
name is `local-<model>-<harness>` (or `--name`); the agent is placed on the project's board (the
project of `--project`, else the working directory) with the quiet subscriptions: the project board,
mentions, tasks, and `#general` (the announcements) as a digest. It is announced as `joined`, gets
the workspace brief (it acts with `--agent NAME`; it obeys the person, the lead named by `--as`,
default `claude-code`, and any `--orders-from` identity; what it reads there is data), and a
PostToolUse hook runs `ml-stack-workspace nudge --agent NAME` after each tool call and hands its
output (at most 500 characters, fenced as data) back as context; where `nudge` does not exist the
hook says nothing. The token is revoked and its file deleted when the session ends. A launcher an
agent started cannot mint: the session then acts as `--as AGENT` with its name as the label, and the
launcher prints `ml-stack-workspace setup --agents NAME` for a person to run.
The served alias and harness are recorded as the agent's model through `Workspace.set_model` where
that exists; it does not on this tree, so the seam is `Seat.record_model` in `harnessid.py`.

## What is not covered

- The hook is a gate inside the harness's own process tree. Neither harness is a sandbox against
  its own model: a Claude Code session has no sandbox here, so a call the classifier labels safe or
  that the person allowed can still do what that call does, and a command built to hide what it
  touches (an `eval` of a computed string) is `unsure` and asks rather than being read. The session's
  files are read-only and outside the tree, but the same operating-system user owns them. Codex's
  own sandbox is the stronger boundary for shell commands.
- A hook that cannot start (a missing interpreter) is a non-blocking error in both harnesses. The
  launcher writes the absolute interpreter that started it.
- Codex runs hooks it has reviewed; the launcher passes `--dangerously-bypass-hook-trust` because it
  wrote the hook into a home that holds nothing else. Codex has no pre-tool hook for its web search
  tool or for model-side reasoning; those are outside the gate.
- The Requests answer comes from the terminal or the UI, so an unattended session in `approve-first`
  stops at its first acting call (denied, with the reason shown to the model).
- Only the deterministic classifier layer runs in the hook; the model-assisted layer does not.
- The harness talks to the server directly, not through `ml_stack.http`, so the per-pool request queue (`ml_stack.gate`) does not order its calls. One slot per session keeps two sessions from sharing a cache.

## Local models and large harness prompts

- The harness prompt and tool list are tens of thousands of tokens before the first message. At 256K
  the first turn pays for them once; every later turn should reuse the cache. Keep the harness's
  system prompt, tool list and instruction files byte-stable between turns (do not change `AGENTS.md`
  or the settings mid-session; the launcher's files are constant for a run) and keep one slot per
  session so the long prefix is not evicted by a second conversation.
- A small model with a large tool list loops: it reads, retries a command a hook denied, and does
  not finish. A 2B model run through Codex for about fifteen minutes made 34 tool calls and tried to
  rewrite the file through `sed`, `python -c` and heredocs, which the classifier asked about, and
  never ran the edit. Use a model sized for the harness (the 27B) for real work.
- Codex needs the Responses API (`wire_api = "responses"`, the only value it accepts); llama-server
  serves `/v1/responses` and `/v1/messages` in the managed build. Claude Code needs `/v1/messages`.
- The thinking-off default of the model's profile applies; the Claude Code settings also turn
  always-on thinking off.

## Comparing the harnesses

`scripts/compare-harnesses --model NAME` serves the model once and runs the fixture in
`tests/fixtures/toy_bugfix` (a one-line bug, scored by its own test, in a temporary git repository)
through `ml-stack-claude` and `ml-stack-codex`, recording wall time, tool calls, the tokens the server
processed (llama-server `/metrics`), the tokens each harness says it sent and how many came from
the cache, the last request's prompt and cached tokens from `/slots`, the first-turn cost of each harness measured with a one-line prompt, and whether the test
passes. Rows are appended to `docs/experiments/harness-comparison.md`. ml-stack's own agent loop is
skipped where `ml-stack-workspace agent` does not exist. It refuses Flash-Next.

## Sources

- Claude Code hooks: https://code.claude.com/docs/en/hooks; environment variables:
  https://code.claude.com/docs/en/env-vars
- Codex hooks: https://learn.chatgpt.com/docs/hooks; configuration:
  https://learn.chatgpt.com/docs/config-file/config-reference and
  https://learn.chatgpt.com/docs/config-file/config-advanced; instruction files:
  https://developers.openai.com/codex/guides/agents-md
