"""Compose project workspace hosting with the device daemon."""

from ml_stack.fleet import daemon
from ml_stack.workspace.remote_host import WorkspaceHost


def main(argv: list[str] | None = None) -> int:
    return daemon.main(argv, workspace_factory=WorkspaceHost)


if __name__ == "__main__":
    raise SystemExit(main())
