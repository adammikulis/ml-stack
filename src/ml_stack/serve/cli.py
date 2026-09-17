"""``ml-stack-serve`` -- see what is serving, put a model up, take one down.

Every subcommand parses its arguments and prints; the work is in `ml_stack.serve.ops`, and
each command group's own parsing lives beside it: `status_cli`, `lifecycle_cli` (up, down,
escalate, build), `profile_cli`, `fit_cli`, `machine_cli` (memory, limits, reclaim),
`broker_cli` (broker, queue).
"""

from __future__ import annotations

from ml_stack.command import Group
from ml_stack.serve import (
    broker_cli,
    fit_cli,
    lifecycle_cli,
    machine_cli,
    profile_cli,
    status_cli,
)
from ml_stack.serve.ops import DEFAULT_ROOT

__all__ = ["COMMANDS", "DEFAULT_ROOT", "main"]

COMMANDS = Group(
    "ml-stack-serve",
    "See which model is being served on this machine, put one up, take it down.")
main = COMMANDS.run

COMMANDS.add(
    "status", status_cli.cmd_status,
    help="what is serving, what its draft head is keeping, and what a lease would do",
    options=status_cli.OPTIONS)

COMMANDS.add(
    "up", lifecycle_cli.cmd_up,
    help="serve a model, or adopt the one already serving it",
    options=lifecycle_cli.OPTIONS_UP)

COMMANDS.add(
    "profile", profile_cli.cmd_profile,
    help="the settings a model scored best with for each workload: what to serve "
         "it with, and how to ask it",
    options=profile_cli.OPTIONS)

COMMANDS.add(
    "fit", fit_cli.cmd_fit,
    help="how many people fit at a given context, from measured KV numbers",
    options=fit_cli.OPTIONS)

COMMANDS.add(
    "memory", machine_cli.cmd_memory,
    help="how much a model may use here, and whether that survives a reboot",
    options=machine_cli.OPTIONS_MEMORY)

COMMANDS.add(
    "limits", machine_cli.cmd_limits,
    help="how much of this machine ml-stack may take",
    options=machine_cli.OPTIONS_LIMITS)

COMMANDS.add(
    "reclaim", machine_cli.cmd_reclaim,
    help="stop the servers nobody is using",
    options=machine_cli.OPTIONS_RECLAIM)

COMMANDS.add(
    "down", lifecycle_cli.cmd_down,
    help="stop a server started on this machine",
    options=lifecycle_cli.OPTIONS_DOWN)

COMMANDS.add(
    "escalate", lifecycle_cli.cmd_escalate,
    help="grow the slots a running server holds, keeping every live conversation",
    options=lifecycle_cli.OPTIONS_ESCALATE)

COMMANDS.add(
    "build", lifecycle_cli.cmd_build,
    help="build llama-server from llama.cpp's own master (or download the newest "
         "release), and switch to it once it is verified",
    options=lifecycle_cli.OPTIONS_BUILD)

COMMANDS.add(
    "broker", broker_cli.cmd_broker,
    help="run the one process on this machine every model server is asked for",
    options=broker_cli.OPTIONS_BROKER)

COMMANDS.add(
    "queue", broker_cli.cmd_queue,
    help="the servers the broker holds, who holds each, and who is waiting",
    options=broker_cli.OPTIONS_QUEUE)


if __name__ == "__main__":
    raise SystemExit(main())
