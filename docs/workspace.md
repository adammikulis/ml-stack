# Agent workspace

A local message bus, shared notes, per-agent scratch folders and an ownership registry, so that
separate agent processes and a lead session coordinate through ml-stack instead of through a
person relaying text and instead of colliding on scratch files, ports, branches and servers.

    ml-stack workspace init                         # a person, at a terminal, once
    ml-stack workspace mint lead-1 --role lead      # prints that agent's token once
    export ML_STACK_WORKSPACE_TOKEN=...             # per agent process
    ml-stack workspace send reviewer task "check the lease tests"
    ml-stack workspace watch --once --timeout 600   # run in the background; exits on a message

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
   Only a `human` token releases it, and a released item is still delivered fenced. When
   `ml_stack.guard` and `ml_stack.sentinel` are installed the same text is also run through the
   guard's patterns and held in the sentinel's quarantine; without them the local checks stand.
8. All logs are hash-chained JSONL (`prev` and `hash` per row, sequence numbers, fsync on each
   append). `ml-stack workspace audit verify` reports the first broken row, and accepts the
   head printed by `audit head` as an external anchor to catch truncation of the tail.
9. The service cannot interrupt a running model turn. Delivery happens when an agent next reads.
   `watch --once --timeout N` blocks until something arrives and exits, so a lead that starts it
   as a background command is woken by its exit.

### Decisions

* **No daemon.** The core is a library over a state directory plus a CLI and MCP tools. Callers
  are processes of one user on one machine, so a socket would add a listener and signed requests
  without adding a boundary: the token never crosses a wire. A loopback daemon with `macauth`
  signed requests is the next step if agents ever need to run as another user or in a sandbox.
* **One ordered bus log.** `bus.jsonl` holds every message with a global sequence number. An
  agent's inbox is the rows addressed to it or to `*`; its outbox the rows it sent. A per-agent
  cursor file records what it acknowledged. One log gives one total order and one chain.
* **Token hash, not shared key.** A shared MAC key in a file would let any agent that can read
  it sign as anyone. Hashed per-agent secrets mean an agent can only be the identity whose
  secret it was given.
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

## Status of this implementation

See the end of this file for what is implemented, tested and left out.
