"""Compose project workspace hosting with the device daemon."""

import sys

from ml_stack import runtime
from ml_stack.fleet import daemon
from ml_stack.workspace.remote_host import WorkspaceHost


def main(argv: list[str] | None = None) -> int:
    runtime.forward("ml_stack.cli.daemon", list(sys.argv[1:] if argv is None else argv))
    return daemon.run(argv, workspace_factory=WorkspaceHost)


if __name__ == "__main__":
    raise SystemExit(main())
