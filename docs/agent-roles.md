# Agent roles and saved rules

`ml-stack-chat` is the one agent command. `ml-stack-chat` is a conversation; `ml-stack-chat
"run benchmarks with quince-2b"` (or `--task`) is a task: the agent asks what the task leaves
open, shows its plan once, asks go, runs the calls the plan names and ends on `done`, printing
a cost line. What differs between the two is a role, a row in `ml_stack.roles.ROLES`.

| role | tools | who is asked | default for | limits (calls, GPU time) |
| --- | --- | --- | --- | --- |
| `reader` | the reads (status, find, history, views); nothing that starts, stops, downloads or writes | nobody: nothing acts | | 60, 0 s |
| `operator` | reads and the acting tools | every acting call | conversation | 200, none |
| `runner` | reads and the acting tools | an acting call the approved plan does not name | task | 200, 6 h |

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
ml-stack's state always asks.

## Every question has three answers

```
! serve_up(model="quince-2b.gguf"): serve_up will start a model server (takes GPU and memory).
allow it? 1) allow this time  2) always allow  3) never allow  [Enter = no] > 2
  this rule: Always allow serve_up for model quince-2b.gguf in the operator role
  save it? [y/N] > y
  saved. /rules lists and edits the rules.
```

A rule is a tool, one pattern per argument of the call (exact, or with `*` and `?`; no regular
expressions), a verdict, an optional role and a created date, and a count of how often it
fired. A call matches only when it carries exactly the arguments the rule names. A never rule
beats an always rule, which beats asking, and a never rule stops a call even inside a runner's
plan. Always is not offered, and an existing always rule does not apply, for a downloaded
model (its size is not known first), a path outside ml-stack's state, a wildcard in a value, a
tool a feature added that asks for itself, or a run that has read outside text (unless the
rule was given `/rules tainted N`; the taint rail still asks its own question, merged into the
same prompt).

Rules live in `~/.ml-stack/agent-rules.json`, mode 0600, written atomically, only by the
answer at this prompt and by:

| | |
| --- | --- |
| `/rules` or `ml-stack-chat rules` | the numbered rules in words, with count and date |
| `/rules remove 3` | delete rule 3; the call asks again |
| `/rules flip 3` | always <-> never |
| `/rules tainted 3` | let always rule 3 apply after outside text was read |
| `/rules clear` | delete every rule |

Each change is logged (`agent-rules-events.jsonl` beside the file). A file that cannot be read
whole, has an unknown shape or tool, or is readable by others, matches nothing: every call asks
and `/rules` says why. No tool can write it: a tool argument naming it is refused.

## Adding a feature

`chat.extensions()` returns a `roles.Extension`: `tools` (schema and callable pairs), `context`
(text added to the system message when a session starts), `reads` (tool names every role may
call) and `asks` (names that change something; roles that act ask first, and no always rule is
offered). A tool that asks the person itself belongs in `reads`.
