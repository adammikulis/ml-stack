# Agent workspace

A local message bus, shared notes, per-agent scratch folders and an ownership registry, so that
separate agent processes and a lead session coordinate through ml-stack instead of through a
person relaying text and instead of colliding on scratch files, ports, branches and servers.

    ml-stack workspace init                         # a person, at a terminal, once
    ml-stack workspace mint lead-1 --role lead      # prints that agent's token once
    export ML_STACK_WORKSPACE_TOKEN=...             # per agent process
    ml-stack workspace send reviewer task "check the lease tests"
    ml-stack workspace watch --once --timeout 600   # run in the background; exits on a message
    ml-stack workspace claim port 8081 --pid $$     # released when this shell exits
    ml-stack workspace who port 8081

## Decision record

Status: accepted for the MVP (issue 37).

### Context

Agents running on one machine share a lead who relays messages by hand, and they collide on
`/tmp` files, ports, worktrees and servers. A shared channel fixes both, and it is also a new
way to attack the people and models that read it: anything one agent writes lands in another
agent's context. The channel therefore has to be designed as an untrusted-input surface first
and a convenience second.

### Threat model

| Who or what | Can do | Defence |
| --- | --- | --- |
| An agent steered by injected text (a web page, a library, a file) | writes messages and notes in its own name, tries to impersonate another sender, tries to get a rule adopted, tries to leak a secret through a note | sender comes from the token, never from the payload; reads are fenced as data; flagged text is quarantined; secrets are refused on write |
| A compromised or buggy agent | floods the bus, fills the disk, claims everything, writes outside its folder | rate limits, size caps, retention, claim TTL, scratch confinement |
| A reader model | obeys what it reads | every item is delivered fenced as untrusted data labelled with its sender and trust level; there is no message type that confirms, approves, grants or changes a setting |
| Another process of the same user | reads or edits the files directly | not defended: see "What this does not do" |
| A process of another user or another machine | reaches the bus | no listener exists; the state directory is mode 0700 |

### Trust rules

1. Everything read from the workspace is untrusted text written by an agent. It never carries a
   person's authority and never confirms, approves, grants or changes settings on anyone's
   behalf. Even a record written under a `human` token is delivered as data: only the
   repository's own files and a person's direct messages bind.
2. A note is advice. An agent may propose a rule; the workspace has no operation that writes
   into repository documents, so promoting a rule is a separate step a person reviews.
3. Every note has one of three trust levels, set by the service and never by the writer:
   `human` (written under a `human` token, which only `init` at a terminal or another human
   token can mint), `test-verified` (the service ran the note's re-derive command, which the
   owner allow-listed, and it exited 0; the command, time, exit code and output digest are
   recorded) and `agent-claimed` (everything else).
4. Facts rot. A note may carry a re-derive command and a `ttl_s`; once its age passes the TTL
   (or its last verification does) it is shown as `stale`, and a stale `test-verified` note
   drops to `agent-claimed` for display. A lower-trust note cannot supersede a higher-trust one.
5. Every write is scanned. A credential or a term on the owner's denylist
   (`ML_STACK_WORKSPACE_DENYLIST`, else `<state>/workspace/private-terms`, one term per line,
   never committed) refuses the write with a message naming the category and not the match.
6. Every sender holds a capability token minted by a person or the lead: `mlws1.<id>.<secret>`.
   The registry stores only a SHA-256 of the secret, so reading the registry does not let a
   process pose as another agent. Tokens expire (default 24 hours) and can be revoked. Roles:
   `human` mints anyone, `lead` mints and revokes `agent` tokens, `agent` mints nothing.
   Per-sender rate limit, body and subject caps, inbox cap and retention are in `limits.json`.
7. Text is screened before it reaches a model: injection and authority-claim patterns put a
   message or note into quarantine; the recipient sees a placeholder with the quarantine id.
   Quarantine is on by default and cannot be switched off; only a `human` token releases an
   item, and a released item is still delivered fenced. The same text is also run through
   `ml_stack.guard`'s secret and injection patterns (imported directly) and held in the
   sentinel's quarantine (`ml_stack.sentinel`); the workspace's own checks run in addition.
8. All logs are hash-chained JSONL (`prev` and `hash` per row, sequence numbers, fsync on each
   append). `ml-stack-workspace audit-verify` reports the first broken row, and accepts the
   head printed by `audit-head` as an external anchor to catch truncation of the tail.
9. The service cannot interrupt a running model turn. Delivery happens when an agent next reads.
   `watch --once --timeout N` blocks until something arrives and exits, so a lead that starts it
   as a background command is woken by its exit.

### Decisions

* **No daemon for 0.3.0.** A hostile process of the same user is documented as not defended
  (see "What this does not do"). The core is a library over a state directory plus a CLI and MCP tools. Callers
  are processes of one user on one machine, so a socket would add a listener and signed requests
  without adding a boundary: the token never crosses a wire. A loopback daemon with `macauth`
  signed requests is the next step if agents ever need to run as another user or in a sandbox.
* **One ordered bus log.** `bus.jsonl` holds every message with a global sequence number. An
  agent's inbox is the rows addressed to it or to `*`; its outbox the rows it sent. A per-agent
  cursor file records what it acknowledged. One log gives one total order and one chain.
* **Token hash, not shared key.** A shared MAC key in a file would let any agent that can read
  it sign as anyone. Hashed per-agent secrets mean an agent can only be the identity whose
  secret it was given.
* **Flat command names.** One command per verb (`notes-add`, `scratch-new`, `audit-verify`),
  no nested sub-commands, so an allow-list or a permission rule can name each exactly.
* **Notes run commands only from an allow-list.** A note's re-derive command is untrusted text.
  `note verify` runs it only when its program and leading arguments match an entry in the
  owner's `limits.json` `verify_allow`, with no shell, a scrubbed environment and a timeout.
  The default list is empty.
* **Search reuses `graph.search`.** Notes become a small node list and are ranked with
  `graph.search.lexical` per word, fused with `rrf`. Embeddings are not used.
* **Claims die with their owner.** A claim has a TTL renewed by heartbeat and may name a pid;
  a claim whose pid is gone is released the next time anyone reads the registry.

### What this does not do

* It is not a boundary against a hostile process of the same user: such a process can edit the
  files, delete the log or read an agent's environment. The chain detects partial edits and
  truncation against an anchor; it cannot detect a full rewrite by someone who recomputes it.
* It does not stop a model from obeying text it was shown. It labels, fences and screens it.
* Pattern screens are heuristics and miss paraphrase; quarantine is for what they do catch.

## Using it

Every command takes `--json` and `--token-file`. Exit codes: 0 done, 2 bad input, 3 refused or
not allowed, 4 rate limited, 5 claim conflict, 6 a log is damaged.

| Area | Commands |
| --- | --- |
| identity | `init`, `mint NAME [--role agent\|lead\|human] [--ttl-hours H]`, `revoke NAME`, `whoami` |
| messages | `send TO TYPE BODY [--subject S] [--reply-to SEQ] [--ttl SECONDS]`, `inbox [--ack] [--raw]`, `wait --timeout S`, `watch [--once] [--timeout S]`, `outbox`, `ack SEQ`, `thread ROOT` |
| notes | `notes-add KIND TITLE BODY [--source --tags --supersedes --verify-cmd --ttl-days]`, `notes-search QUERY [--kind] [--all]`, `notes-get ID`, `notes-verify ID --cwd DIR` |
| scratch | `scratch-new NAME`, `scratch-ls`, `scratch-path NAME [REL]`, `scratch-rm NAME` |
| claims | `claim KIND KEY [--ttl S] [--pid N]`, `release KIND KEY`, `heartbeat`, `who KIND KEY`, `claims` |
| safety | `quarantine-ls`, `quarantine-release QID` (human token), `audit-verify [--anchor HASH]`, `audit-head`, `gc` |
| view | `status` (counts, live claims, the machine's test-slot queue, read only) |

Message types are `task`, `status`, `handoff`, `question`, `answer`, `claim`, `release` and
`note`. Note kinds are `decision`, `rule`, `fact` and `question`. Claim kinds are `branch`,
`worktree`, `port`, `file` and `server`.

The state directory holds `agents.json` (token hashes), `bus.jsonl`, `notes.jsonl`,
`quarantine.jsonl`, `audit.jsonl` (all chained), `cursors/`, `claims.json`, `rates.json`,
`scratch/<agent id>/<name>/` and the owner's `limits.json` and `private-terms`.

`limits.json` overrides these defaults: message body 16 KiB, subject 200 characters, note body
8 KiB, 30 writes per 60 s per sender, 500 unread per inbox, 500 notes per agent, 7 days of
messages, 24 hour agent tokens, 15 minute claims, 256 MiB and 16 folders of scratch per agent
with a 3 day expiry, and an empty `verify_allow` list.

MCP tools (`ml-stack-mcp`) read the sender's token from `ML_STACK_WORKSPACE_TOKEN` in the
agent's own process. Read-only: `workspace_status`, `_inbox` (does not mark read), `_thread`,
`_notes_search`, `_note_get`, `_who_owns`, `_claims`, `_scratch_ls`, `_scratch_path`,
`_audit_verify`. Writes: `workspace_send`, `_ack`, `_note_add`, `_claim`, `_heartbeat`,
`_scratch_new`; destructive: `_release`, `_scratch_rm`.

## Status of this implementation

Implemented and tested with real files and processes (101 tests in `tests/test_workspace_*.py`):
the chained logs (ordering across four concurrent processes, restart durability, torn-line cut,
a SIGKILLed sender, hand edits, tail truncation against an anchor), tokens and roles, forged
and revoked and expired tokens, rate limit, size caps, secret and private-term refusal,
quarantine and human release, fenced delivery, notes with trust levels, TTL, supersession rules
and allow-listed verification, scratch confinement (traversal, symlinks, other agents),
claims (conflict, nesting, TTL, heartbeat, release on process death, a four-process race), the
CLI with `--json`, the MCP tools and their annotations, and the read-only test-slot view.

Mutation check of the guards: 60 mutations applied one at a time to a copy of the tree, 50
caught. Survivors: `chain-no-fsync` (a crash-durability property a test cannot observe without
killing the machine), `chain-no-prev-check` and `chain-no-seq-check` (the hash covers `prev`
and a skipped sequence number changes it, so each check is covered by the other): all
three are acknowledged as unobservable and left alone. The other seven survivors (`name-allows-reserved`,
`note-kind-unchecked`, `quarantine-text-any`, `secrets-workspace-token`, `markers-rule-promotion`,
`denylist-comments`, `neutralise-fence-tags`) are pinned by `tests/test_workspace_rules.py`, each
confirmed to fail when its mutation is applied. The remaining mutations in the list were not run.

The adapters run against the real `sentinel` store and `guard` patterns in the suite
(`tests/test_workspace_surface.py`, no stand-ins). The MCP SDK transport was not exercised here.

Left out: see "The agent workspace" in `HANDOFF.md`.
