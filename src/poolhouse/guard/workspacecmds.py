"""Effects of the maintained workspace CLI's bounded messaging and read commands."""
from __future__ import annotations

from poolhouse.guard.harm import Finding

READ = {"tasks": (0, 0), "task": (1, 1), "inbox": (0, 0), "thread": (1, 1), "who": (2, 2), "roster": (0, 0),
        "status": (0, 0), "nudge": (0, 0)}
WRITE = {"task-claim": (2, 2), "task-heartbeat": (1, 1),
         "task-checkpoint": (2, 2), "task-submit": (2, 2), "task-review": (2, 2), "task-credit": (1, 1), "send": (3, 3), "announce": (2, 2), "claim": (2, 2)}


def effect(args: list[str]) -> list[Finding]:
    if not args:
        return [Finding("unsure", "workspace command is missing")]
    command, words = args[0], []
    rest = iter(args[1:])
    for word in rest:
        if word == "--agent":
            if not next(rest, ""):
                return [Finding("unsure", "workspace identity flag has no value")]
        elif word.startswith("--agent=") or word == "--json" or (word == "--peek" and command == "inbox"):
            continue
        elif word.startswith("-"):
            return [Finding("unsure", "workspace flag is not in the bounded command profile")]
        else:
            words.append(word)
    bounds = READ.get(command) or WRITE.get(command)
    if bounds is None or not bounds[0] <= len(words) <= bounds[1]:
        return [Finding("unsure", "workspace command or arguments are outside the messaging profile")]
    label = "safe" if command in READ else "reversible"
    return [Finding(label, f"workspace {command} reads messages" if command in READ else
                    f"workspace {command} updates authenticated messages or claims")]
