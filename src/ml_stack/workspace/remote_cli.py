"""Shell access to an authoritative project board."""

from argparse import Namespace
from pathlib import Path

from ml_stack.command import flag, option
from ml_stack.workspace import connection_details, harness_remote
from ml_stack.workspace.automatic_connection import connect
from ml_stack.workspace.identity import Denied
from ml_stack.workspace.project_connection import bind
from ml_stack.workspace.remote import RemoteWorkspace

OPTIONS = [option("json"), flag("action", choices=("connect", "connection", "join", "whoami", "agents", "boards",
            "read", "post", "send", "inbox", "announce", "claims", "claim", "release", "heartbeat", "history", "use")),
           flag("arguments", nargs="*"), flag("--host", default=""),
           flag("--project-id", default=""), flag("--cluster-key", default=""),
           flag("--cluster", default=""),
           flag("--name", default=""), flag("--agent", default=""),
           flag("--token-file", default=""), flag("--model", default=""),
           flag("--harness", default=""), flag("--label", default=""),
           flag("--project-root", default="."), flag("--ttl", type=float, default=0.0),
           flag("--limit", type=int, default=20), flag("--ack", action="store_true")]


def run(args):
    """Run a project-scoped remote operation."""
    if args.action == "connect":
        if args.arguments or args.host or args.project_id:
            raise ValueError("connect discovers this project on its Dev cluster")
        return connect(Path(args.project_root), args.name or args.agent,
                       claim=(args.model, args.harness), cluster=args.cluster,
                       cluster_key=Path(args.cluster_key) if args.cluster_key else None)
    remote = RemoteWorkspace(args.host, args.project_id,
                             cluster_key=Path(args.cluster_key) if args.cluster_key else None,
                             cluster=args.cluster)
    values, action = args.arguments, args.action
    if action == "join":
        if len(values) != 1:
            raise ValueError("join needs the selected project's invitation code")
        return remote.join(values[0], args.name, model=args.model, harness=args.harness)
    completion = ((action == 'send' and len(values) >= 2 and values[1] == 'done')
                  or (action == 'announce' and values and values[0] == 'done'))
    if (action in {'claim', 'release', 'heartbeat'} or completion) and not args.agent:
        raise Denied('direct claims and completion require an explicit agent identity')
    token = remote.token(agent=args.agent, token_file=args.token_file)
    if action == "connection":
        if values:
            raise ValueError("connection takes no positional arguments")
        return connection_details.details(remote, token)
    if action == "use":
        return bind(remote, Path(args.project_root), args.agent, args.cluster)
    if action in {"claim", "release", "heartbeat"}:
        if (action == "heartbeat" and values) or (action != "heartbeat" and len(values) != 2):
            raise ValueError(f"invalid arguments for remote {action}")
        kind, key = values if values else ('', '')
        if kind in ('area', 'file', 'worktree'):
            key = str((Path(args.project_root) / key).resolve())
        request = Namespace(cmd=action, agent=args.agent, kind=kind, key=key,
                            ttl=args.ttl, pid=0, note=args.label, label=args.label)
        return harness_remote.cli_command(remote, token, request)
    if action in {"whoami", "agents", "claims", "history"}:
        return remote.call(action, token)
    if action == "boards":
        return remote.call("board.list", token)
    if action == "inbox":
        return remote.call("inbox", token, ack=args.ack, limit=min(max(args.limit, 1), 100))
    if action == "read" and len(values) == 1:
        return remote.call("board.read", token, values[0], limit=min(max(args.limit, 1), 100))
    if action == "post" and len(values) >= 2:
        return remote.call("send", token, values[0], "note", " ".join(values[1:]), label=args.label)
    if action == "send" and len(values) >= 3:
        if values[1] == 'done':
            harness_remote.cli_command(remote, token, Namespace(cmd='send', agent=args.agent))
        return remote.call("send", token, values[0], values[1], " ".join(values[2:]), label=args.label)
    if action == "announce" and len(values) >= 2:
        if values[0] == 'done':
            harness_remote.cli_command(remote, token, Namespace(cmd='announce', agent=args.agent))
        return remote.call("announce", token, values[0], " ".join(values[1:]), args.label)
    raise ValueError(f"invalid arguments for remote {action}")
