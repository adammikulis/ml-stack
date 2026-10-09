# Agent roles and saved rules

`poolhouse-chat` is the one agent command. `poolhouse-chat` is a conversation; `poolhouse-chat
"run benchmarks with quince-2b"` (or `--task`) is a task: the agent asks what the task leaves
open, shows its plan once, asks go, runs the calls the plan names and ends on `done`, printing
a cost line. What differs between the two is a role, a row in `poolhouse.roles.ROLES`.

| role | tools | who is asked | default for | limits (calls, GPU time) |
| --- | --- | --- | --- | --- |
| `read-only` | the reads (status, find, history, views); nothing that starts, stops, downloads or writes | nobody: nothing acts | | 60, 0 s |
| `approve-first` | reads and the acting tools | every acting call | conversation | 200, none |
| `plan-and-go` | reads and the acting tools | an acting call the approved plan does not name | task | 200, 6 h |

`read-only` may look and never change anything; `approve-first` may propose changes, each one waiting for the person's approval (or a saved Always rule); `plan-and-go` may carry out the plan it states, still bound by the destructive-action classifier, the taint rail and the person-only floor.

`--role NAME` picks one at the start; `/role NAME` changes it from the prompt. Only the
person's typed line does either: the model has no tool for it, its text and tool output are
never read as commands, and asking it to change the role prints the command to type. A role
can never hold the person-only floor (release or purge quarantine, approve a host, grants, the
sentinel and guard policy, baselines, the role and the rules): those tools do not exist.
`tests/test_roles.py` walks every role against every tool and asserts the floor is absent.

A plan step names a tool first and every value the call will carry
(`bench_run ["run", "quince-2b.gguf", "--sample", "10"]`). The person's go covers a call when
the tool matches and each of its values appears in a step; the step then covers that one call.
A call outside the plan asks. A run that has read text from outside (a model card, a log) asks
for every acting call whatever the role, in one question that names why. A path outside
Poolhouse's state always asks. A call the destructive-action classifier labels destructive or unsure asks in every role, and no
Always rule can be saved for it ([destructive actions](destructive-actions.md)).

## Every question has three answers

```
! serve_up(model="quince-2b.gguf"): serve_up will start a model server (takes GPU and memory).
allow it? 1) allow this time  2) always allow  3) never allow  [Enter = no] > 2
  this rule: Always allow serve_up for model quince-2b.gguf in the approve-first role
  save it? [y/N] > y
  saved. /rules lists and edits the rules.
```

A rule is a tool, one pattern per argument of the call (exact, or with `*` and `?`; no regular
expressions), a verdict, an optional role and a created date, and a count of how often it
fired. A call matches only when it carries exactly the arguments the rule names. A never rule
beats an always rule, which beats asking, and a never rule stops a call even inside a plan-and-go
plan. Always is not offered, and an existing always rule does not apply, for a downloaded
model (its size is not known first), a path outside Poolhouse's state, a wildcard in a value, a
tool a feature added that asks for itself, or a run that has read outside text (unless the
rule was given `/rules tainted N`; the taint rail still asks its own question, merged into the
same prompt).

Rules live in `~/.poolhouse/agent-rules.json` (`schema_version` 2; version 1 files with the earlier role names reader, operator and runner are read as `read-only`, `approve-first` and `plan-and-go` and written in the new names on the next save), mode 0600, written atomically, only by the
answer at this prompt and by:

| | |
| --- | --- |
| `/rules` or `poolhouse-chat rules` | the numbered rules in words, with count and date |
| `/rules remove 3` | delete rule 3; the call asks again |
| `/rules flip 3` | always <-> never |
| `/rules tainted 3` | let always rule 3 apply after outside text was read |
| `/rules clear` | delete every rule |

Each change is logged (`agent-rules-events.jsonl` beside the file). A file that cannot be read
whole, has an unknown shape or tool, or is readable by others, matches nothing: every call asks
and `/rules` says why. No tool can write it: a tool argument naming it is refused.

## Adding a feature

`chat.extensions(person, mem)` returns a `roles.Extension`: `tools` (schema and callable pairs),
`context` (text added to the system message when a session starts), `start` (text put in front of
the first message of a session and of each `/new`), `commands` (slash commands for the person,
such as `/memory`), `reads` (tool names every role may call), `asks` (names that change
something; roles that act ask first, and no always rule is offered) and `asks_itself` (names
that change something and ask the person inside the tool: only roles that act are offered them,
the rail does not ask a second time, and no always rule exists). `remember` is in `asks_itself`
and `recall` in `reads`, so the `read-only` role recalls but cannot remember.
