"""Project source controls refuse forged browser access before publication."""

import http.client
import json
import threading
from types import SimpleNamespace

import pytest

from poolhouse.fleet import routes
from poolhouse.fleet.api import Daemon, make_handler
from poolhouse.fleet.jobs import JobRunner
from poolhouse.fleet.ui import UI
from poolhouse.http import Server


@pytest.mark.parametrize("ui_header,cookie", [(False, ""), (False, "forged"), (True, ""), (True, "forged")])
@pytest.mark.parametrize("path", ["/ui/projects", "/ui/projects/" + "a" * 32 + "/checkout"])
def test_project_ui_refuses_forged_browser_access(tmp_path, monkeypatch, ui_header, cookie, path):
    touched = []
    ui = UI(cluster_key_path=tmp_path / "cluster.key")
    ui.projects = SimpleNamespace(share=lambda *_args: touched.append("share"))
    monkeypatch.setattr(routes, "in_cluster", lambda _path: True)
    runner = JobRunner(tmp_path / "jobs")
    server = Server(("127.0.0.1", 0), make_handler(Daemon(runner, tmp_path, "test-daemon", ui=ui)))
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    headers = {"Content-Type": "application/json", "Origin": "https://hostile.invalid"}
    if ui_header:
        headers["X-Poolhouse-UI"] = "1"
    if cookie:
        headers["Cookie"] = "poolhouse=" + cookie
    connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=5)
    try:
        connection.request("POST", path, body=json.dumps({"action": "share", "candidate_id": "../../private"}), headers=headers)
        response = connection.getresponse()
        assert response.status == (401 if ui_header else 403), response.read()
        assert not touched
    finally:
        connection.close()
        server.shutdown()
        server.server_close()
        worker.join(2)
        runner.shutdown()
