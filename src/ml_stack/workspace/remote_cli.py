"""Shell access to an authoritative project board."""

from pathlib import Path

from ml_stack.command import flag, option
from ml_stack.workspace.remote import RemoteWorkspace

OPTIONS = [option("json"), flag("action", choices=("join", "whoami", "agents", "boards",
            "read", "post", "send", "inbox", "announce", "claims", "claim", "heartbeat")),
           flag("arguments", nargs="*"), flag("--host", required=True),
           flag("--project-id", required=True), flag("--cluster-key", default=""),
           flag("--name", default=""), flag("--agent", default=""),
           flag("--token-file", default=""), flag("--model", default=""),
           flag("--harness", default=""), flag("--label", default=""),
           flag("--limit", type=int, default=20), flag("--ack", action="store_true")]


def run(args):
    """Run a project-scoped remote operation."""
    remote = RemoteWorkspace(args.host, args.project_id,
                             cluster_key=Path(args.cluster_key) if args.cluster_key else None)
    values, action = args.arguments, args.action
    if action == "join":
        if len(values) != 1:
            raise ValueError("join needs the selected project's invitation code")
        return remote.join(values[0], args.name, model=args.model, harness=args.harness)
    token = remote.token(agent=args.agent, token_file=args.token_file)
    if action in {"whoami", "agents", "claims", "heartbeat"}:
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
        return remote.call("send", token, values[0], values[1], " ".join(values[2:]), label=args.label)
    if action == "announce" and len(values) >= 2:
        return remote.call("announce", token, values[0], " ".join(values[1:]), args.label)
    if action == "claim" and len(values) == 2:
        return remote.call("claim", token, *values, pid=0)
    raise ValueError(f"invalid arguments for remote {action}")
