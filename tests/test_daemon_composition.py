"""Device daemon composition supplies the project workspace service."""

from ml_stack.cli import daemon as composition
from ml_stack.fleet import autostart
from ml_stack.workspace.remote_host import WorkspaceHost


def test_daemon_entrypoint_injects_project_host(monkeypatch):
    received = []

    def start(argv, *, workspace_factory):
        received.append((argv, workspace_factory))
        return 7

    monkeypatch.setattr(composition.daemon, "main", start)
    assert composition.main(["--no-web"]) == 7
    assert received == [(["--no-web"], WorkspaceHost)]


def test_autostart_uses_composed_daemon(monkeypatch):
    monkeypatch.setattr(autostart.shutil, 'which', lambda name: None)
    assert autostart._executable()[-1] == "ml_stack.cli.daemon"
