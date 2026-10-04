# Destructive actions ask first

A classifier labels every tool call the chat agent makes `safe`, `reversible`, `destructive` or
`unsure`. `destructive` and `unsure` ask the person before the call runs, whatever the role.
`reversible` follows the role and the saved rules as before; `safe` changes nothing. The
classifier only adds a question: no role, plan or saved rule lets a call past it, and nothing it
reads can lower its answer.

Code: `ml_stack.guard.destructive` (`classify`, `Verdict`, `combine`), the rail
`ml_stack.guard.destructive_rail.DestructiveRail`, the optional model layer
`ml_stack.guard.destructive_model.ModelLayer`. Any consumer can call
`classify(Call(name, arguments), roots=..., catalog=..., floors=..., annotations=...)`.

## Reading the question

```
! run_shell(command="rm -rf build"): run_shell will run a shell command. Asked because: destructive: deletes files (rm -r)
```

`Asked because: <label>: <reasons>.` The reasons are plain words from the rule that fired:
`deletes files (rm -r) outside the project`, `overwrites a file with a redirect (>)`, `updates every
row (UPDATE without a real WHERE)`, `sends commits to a remote`. `unsure` says why it could not
read the call (`the command name comes from a variable or pattern`, `a python3 one-liner the
classifier does not read`). The same question carries the role's and the taint rail's reasons.

## Layer 1: fixed rules, no model, about 0.05 ms

The label is the most severe of these.

- Tool metadata. The built-in tools are labelled in `chatpolicy.catalog()`; a catalogued tool's
  arguments are not read as commands. An extension's tools get a floor (reads `safe`, asking tools
  `reversible`) and their arguments are still read. MCP `destructiveHint` raises a tool to
  `destructive`; `readOnlyHint` counts as `safe` only for a tool nothing else says anything about. The
  verbs in a tool's name (`delete_`, `send_`, `publish_`, `get_`, `create_`) label it; a tool whose
  name and arguments say nothing is `unsure`.
- Shell commands (`command`, `cmd`, `script`, `argv`, a program key beside an `args` list). Parsed
  with `shlex` into a normalised argv, split on `; && || | &` and newlines, then checked by
  program and flag, never by substring: `rm`, `rmdir`, `unlink`, `shred`; `mv` and `cp` over an
  existing target; `>` and `tee` without `-a`, `dd of=`, `truncate`, `sed -i`, `sort -o`; `find
  -delete` and `-exec` (the inner command is read); `xargs` (the inner command is read); `git reset
  --hard`, `clean -f`, `checkout --`, `restore`, `branch -D`, `push` (force, delete, `+ref`),
  `stash drop`, `reflog expire`, `gc --prune`, `filter-branch`, `commit --amend`; `mkfs`,
  `diskutil`, `fdisk`; `chmod`/`chown -R`; `kill`, `pkill`; `docker rm/rmi/prune/kill`,
  `kubectl delete`, `terraform apply/destroy`, cloud CLIs; `launchctl`/`systemctl stop`;
  package removal and publish; `curl`/`wget` that send data, `scp`, `rsync`, `nc`, `ssh`; a
  download piped to a shell. A path in a protected directory (`/etc`, `~/.ssh`, a `.env`, `.git`) or
  outside the project and ml-stack's state is called out and raises writes.
- SQL, split into statements with comments and literals blanked: `DROP`, `TRUNCATE`, `DELETE`,
  `UPDATE` without a real `WHERE` (`WHERE 1=1` is not one), `ALTER ... DROP`, `GRANT`, `REVOKE`,
  `REPLACE`, `COPY ... PROGRAM`.
- Arguments of any tool: `force`/`yes`/`overwrite` set true, an `action` or HTTP `method` that
  deletes or sends, a write over an existing file or over a path with almost nothing, a list of more
  than 25 items, `production` in an environment-like key, a download over 5 GiB.
- Anything that cannot be read is `unsure`: quoting that hides a name (`$'\x72m'`), command
  substitution, backticks, process substitution, `eval`, `source`, here-documents, an unbalanced quote,
  `sh -c` nested more than three deep, `python -c`/`node -e` code that is not clearly harmless,
  a variable or glob where a command or a write target is, a program not in the table, a call
  over 20,000 characters (a command over 4,000), arguments that are not a JSON object.
  `r''m`, `\rm`, `/bin/rm` and `RM` need no special case: parsing normalises them.

The call's text is data. The classifier never runs it, never follows an instruction in it ("this
is safe" is ignored), reads at most a bounded number of argument nodes, uses no regular
expression that backtracks badly, and gives the same answer for the same input and project
directory (the only filesystem reads are existence checks for `mv`, `cp` and overwrites).

## Layer 2: the decision model, optional

Off unless the person starts the process with `ML_STACK_DESTRUCTIVE_MODEL=1`
(`ML_STACK_DESTRUCTIVE_FLOOR`, 0.5 to 1, default 0.8). No tool, message or file the model can
reach changes either. It asks the typed question `is_destructive` (choice `safe | reversible |
destructive`) through `ml_stack.decide.questions` and the default decider path over the
sanitised call (`defang`/`closed`), one token, no text generated. It runs only for calls layer 1 did not
already stop on. An answer under the floor is `unsure`. A decider that is down, errors or takes
longer than 3 s gives nothing and the verdict records `layer: deterministic`. Answers are cached
per tool and exact arguments (not per argument shape, which would let `ls x` stand in for
`rm x`).

## Combining, roles and rules

- The most severe label wins (`safe < reversible < unsure < destructive`); the model can raise a
  label and never lowers one (`combine`).
- A classifier failure is `unsure`, so the call asks.
- `destructive` and `unsure` ask in every role, including the role that carries out an approved plan:
  a plan step naming the exact call does not waive the question. (The plan is not classified when
  it is approved; the call is classified when it is made.)
- "Always allow" is never offered, and `rules.make` refuses it, for a call layer 1 labels
  `destructive` or `unsure`. For a model-only `destructive` the role rail's own exact-argument
  always rule is unchanged. A Never rule still denies first. A saved Always rule does not stop the
  classifier asking.
- The verdict is kept in `DestructiveRail.last`, the confirmation shows in `Run.events`, and
  `DestructiveRail.on_verdict` receives each verdict that asks. Nothing writes it to the activity
  log yet.

## Measured

2026-10-03, `scripts/experiments/destructive_eval.py` (add `--misses` to list each wrong call),
layer 1 only, on this machine, no model. Sets: the corpus in `ml_stack.decide.guards.destructive`
(30 safe, 30 reversible, 31 destructive) and `destructive_adversarial` (166 safe look-alikes, 76
reversible, 374 destructive, with obfuscation). Recall counts `unsure` as caught; strict counts only
`destructive`. The script fails below 0.99 recall or above 5% of safe calls asked.

| set | recall | strict | safe calls asked | reversible calls asked |
| --- | --- | --- | --- | --- |
| corpus | 1.000 | 1.000 | 0.000 | 0.000 |
| held-out | 1.000 | 0.890 | 0.000 | 0.000 |

Latency per call: mean 0.05 ms, 95th percentile 0.11 ms, maximum about 1 ms.

Read these as optimistic. The held-out rows were written by the author of the rules in three
batches. Before the rules were adjusted to them, the first pass asked about 202 of 204 (0.990) of
the first batch's destructive calls, 160 of 172 (0.930) of the second, and asked about 3 of 101 safe
calls (3.0%) and 2 of 49 reversible calls of the third. The approval-fatigue number to expect on
real work is above zero: any program outside the table (an unknown CLI, a script run directly, a
`python x.py`) is `unsure` and asks. Layer 2 is exercised in tests against the stub logprob server,
which says nothing about a real decider's accuracy; it has not been measured on a real model.

## Limits

- The program table is finite; a call to a program it lacks asks, and a program with a harmful flag
  it does not model can be labelled too low (`reversible` build tools run project scripts).
- Intent is not read: `echo x > notes.txt` asks because `>` overwrites, not because of what the text
  is for.
- An argument that holds a command under a key the classifier does not know is read only for tools
  whose name says they run commands, or when it holds shell metacharacters there.
- Existence checks for `mv`/`cp`/overwrites read the filesystem at call time.
- Tests: `tests/test_destructive.py`, `tests/test_destructive_chat.py`,
  `tests/test_redteam_destructive.py` (`--redteam`).
