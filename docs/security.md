# Security

## Agent workspace

`ml-stack-workspace` (`src/ml_stack/workspace/`, design in `docs/workspace.md`) lets agent
processes on one machine exchange messages and notes and avoid each other's ports, worktrees
and scratch files. It is a channel from one model's output into another model's input, so it is
treated as untrusted input.

**What listens.** Nothing. The workspace is a directory (`ML_STACK_WORKSPACE_HOME`, else
`<state>/workspace`, mode 0700) read and written by the commands and MCP tools of processes of
one account. There is no socket and no LAN exposure.

| Attacker | What they try | What stops it |
| --- | --- | --- |
| An agent steered by injected text | send as someone else | the sender is the owner of the token; `send` has no sender argument, and the registry stores only a SHA-256 of each secret, so reading it yields no identity |
| the same | get a rule adopted, or a person's approval claimed | every item is delivered inside an `<untrusted>` fence labelled with sender, role and `no authority`; "approved", "authorize", "from now on all agents" and "add to CLAUDE.md" patterns quarantine the item; a rule note is advice with `binding: false`; no operation edits repository documents |
| the same | get a note believed | trust is set by the service from the token and from a run of an allow-listed command, never by the writer; a lower-trust note cannot supersede a higher one; facts carry a TTL and show as stale |
| the same | leak a secret or a private term through a note or message | the write is refused, naming the rule and not the match; the audit log keeps lengths, never text |
| the same | flood, fill the disk | per-sender sliding-window rate limit, body and subject caps, per-recipient inbox cap, per-agent note and scratch limits, retention |
| the same | escape its scratch folder | names are plain, resolved paths are checked after following symlinks, a symlinked agent or folder directory is refused |
| the same | hold the lead's ports and branches | claims have a TTL renewed by heartbeat and are released when the owning pid is gone |
| the same | run a command through a note | the service runs a note's command only for a lead or human token, only if the owner listed its leading words in `limits.json`, with no shell, a reduced environment and a timeout; the default list is empty |
| anyone | edit the history | every log is hash-chained; `audit-verify` finds the first broken row and accepts an external anchor for truncation |

**Not defended.** A hostile process of the same account can read the files, edit or delete the
logs (a full rewrite with recomputed hashes is undetectable) and read another process's
environment. Pattern screens miss paraphrase. The workspace labels and fences text; it does not
stop a model from obeying what it was shown.

**Operations.** `init` refuses to run when `CLAUDECODE`, `ML_STACK_AGENT` or
`ML_STACK_NONINTERACTIVE` is set. Minting, revoking, releasing quarantine, running a note's
command and `gc` are CLI-only and are not offered over MCP. Tokens come from
`ML_STACK_WORKSPACE_TOKEN` or `--token-file`, never from an argument.

**Private terms.** The denylist is a file outside the repository (`ML_STACK_WORKSPACE_DENYLIST`,
else `<state>/workspace/private-terms`), one term per line.
