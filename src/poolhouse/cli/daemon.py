"""Compose project workspace hosting with the device daemon."""

import sys

from poolhouse import runtime
from poolhouse.fleet import daemon
from poolhouse.workspace.remote_host import WorkspaceHost


def main(argv: list[str] | None = None) -> int:
    runtime.forward("poolhouse.cli.daemon", list(sys.argv[1:] if argv is None else argv))
    return daemon.run(argv, workspace_factory=WorkspaceHost)


if __name__ == "__main__":
    raise SystemExit(main())
