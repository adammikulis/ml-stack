# The activity log

One log per user of what the stack did: what agents and people did, what was asked and answered,
what coordinated and what was refused, and what ran. Every record is encrypted, chained to the one
before it, and read back with `ml-stack-log`. It is evidence for the person at the keyboard, not
context for a model: no tool a model is offered returns a record.

## What is recorded

A record carries a time, an actor (`person`, `agent:<label>` or `system`), a session id, a kind, a
subject, an outcome, up to twelve references (ids, hosts, claim names) and up to sixteen small
metadata values. Text is cut to 200 characters, secrets are masked by sentinel's redaction, and
control and bidirectional characters are written as visible escapes (`\x1b`, `‮`). A record is
at most 4 KB. Bodies are never stored: a metadata name such as `body`, `text`, `content`,
`arguments`, `output` or `result` is dropped.

| Kind | Written when | Metadata |
|---|---|---|
| `agent.tool_call` | a chat tool call ends | tool, outcome (`ok`, `error`, `blocked`, `no_such_tool`), role, rail that blocked it, argument names, size of the result |
| `approval.asked`, `approval.answered` | a call asks the person | tool, role, answer (`allow_once`, `always`, `never`, `no`) |
| `approval.rule_fired` | a saved rule decided a call | tool, verdict, the rule in words |
| `role.changed` | the person types `/role` | new role, old role |
| `rule.added`, `rule.removed`, `rule.flipped`, `rule.tainted_setting` | the person edits the rules | tool, verdict, the rule in words |
| `workspace.message`, `workspace.claim`, `workspace.event` | the workspace audit trail grows | from, to, type, thread, size; claim name; never the body |
| `security.*` | any sentinel event: keystore operations, quarantine and release, dialog outcomes | kind, severity, source, outcome |
| `reputation.observed` | a source misbehaves | source, event, resulting state |
| `model.lease` | the broker grants, shares or refuses a lease | model, quant, who asked, purpose, port, context, parallel, MTP and draft flags, weight |
| `net.download` | a download is kept or held | host, size, SHA-256, kind, scan result |
| `runtime.deploy` | `ml-stack runtime ensure`, `rollback` or `restart-host` finishes or is refused | commit, command, result, agent, whether a person ran it, and `delegated` when an agent passed the `runtime.deploy` authority gate |
| `bench.run` | a bench run is kept | label, command, model, build, rows, store and key |
| `test.result` | `scripts/test` finishes a pytest run | git tree hash, tier, command hash, passed/failed/skipped, seconds |
| `activity.off`, `activity.off_refused`, `activity.gap`, `activity.export` | the log itself changed | cause, count |

`test.result` is keyed by the hash of the tree the run saw (everything not ignored, working tree
included). `activity.evidence(tree, tier)` returns the latest passing record, so "the full tier is
green on tree X" is a record the tool wrote.

Not recorded: the text of messages, prompts, tool arguments and results, file contents, model
output, and anything a person types.

## Where it lives

`~/.ml-stack/activity/u-<uid>/` (the state root moves it): `activity.log` and rotated
`activity.log.1`..`.7`, `activity.log.head` (the sealed chain head) and its key, `salt`,
`drops.json` (a count and a cause, no text). The directory is mode 0700, the files 0600, and the
directory is registered with sentinel so no tool call may name it.

Each line holds only a time and the chain (`seq`, `prev`, `hash`); the rest is AES-256-GCM under the
`activity` subkey of the user's master key in the OS keystore. Reading needs the key; verifying the
chain does not. Rotation is by size (1 MB) and age (7 days), at most eight files, and the chain
carries across them. Files whose newest record is older than `ML_STACK_ACTIVITY_RETENTION_DAYS`
(default 90, 0 keeps until the file limit) are removed from the old end and the chain continues
from the last record removed.

The chain is the sentinel event log's: each record's MAC covers the one before it, so an edited,
removed, reordered or cut-off record is found by `verify`. The MAC key sits beside the log, so
someone who can write the directory can also rebuild the chain; `ml-stack-log stats` prints the head
(`count hash`) to keep somewhere out of reach, and `verify --anchor FILE` checks it. Rebuilt lines
still cannot be read, because the encryption key is not on disk: they show as unreadable.

## Reading it

    ml-stack-log                      the last 50 records
    ml-stack-log today                since local midnight
    ml-stack-log tail -f              follow
    ml-stack-log --since 2h --agent scout --kind approval --session chat-3 --grep serve_up
    ml-stack-log tail --timeline      each session as a story
    ml-stack-log show 3fa91c04b2d1    one record in full (id or sequence number)
    ml-stack-log related alpha path:/w/a.py   what agent alpha did that names that claim
    ml-stack-log stats                counts by kind, actor and outcome, size, dropped records, head
    ml-stack-log verify               check the chain; says which file and line breaks it
    ml-stack-log export --json FILE   write records to a new file (a person at a terminal only)

Records are never reordered or hidden by the viewer, only filtered as asked. Everything printed is
escaped and secret-masked again. `related` builds a graph of the records in memory (events tied to
actors, sessions, subjects and references, the shape in `docs/graph.md`) and walks it; nothing
relational is stored besides the log.

A process an agent started (`CLAUDECODE`, `ML_STACK_AGENT`, `ML_STACK_NONINTERACTIVE`) may read, and
sees the records inside an `<activity-log-data>` fence with `<` and `>` replaced, as data and not
instructions. It cannot export, and its request to turn the log off is refused and recorded.

## When it cannot write

`record` never raises into the caller. A full disk, a locked keystore or a damaged file drops the
record and counts it (`drops.json`, `stats`); the next record that does get written is preceded by
an `activity.gap` record saying how many were lost and why. A process with no person present
(`ML_STACK_NONINTERACTIVE`) reads the master key only after `ml-stack-security unlock`; until then
its records are dropped and counted.

## Turning it off

`ML_STACK_ACTIVITY=off` in a person's own process stops that process writing. The first thing it
does is write an `activity.off` record. In an agent's process the variable is ignored and an
`activity.off_refused` record is written. Nothing else turns the log off, and no tool, MCP call or
chat request can read, edit, truncate or delete it.

## For code that feeds it

    from ml_stack import activity
    activity.record("net.download", subject="host:example.org", outcome="ok",
                    refs={"sha256": digest}, meta={"size": size})

`activity.bind_session(id)` sets the session id (the chat does); `activity.attach()` mirrors sentinel
events and catches the log up with the workspace audit trail. Both are idempotent.
