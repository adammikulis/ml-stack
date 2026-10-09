"""Hostile simulator arguments, control streams, and imported scenario paths."""

import json
import queue
import sys
from pathlib import Path

import pytest

from ml_stack.gym.catalog import catalogue
from ml_stack.gym.runtime import SessionManager
from ml_stack.gym.traffic_world import generate, xml
from ml_stack.gym.transport import Process
from ml_stack.gym.world_files import imported

pytestmark = pytest.mark.redteam


def test_catalogue_probe_executes_hostile_interpreter_path_without_shell(tmp_path, monkeypatch):
    source = Path(__file__).resolve().parents[1] / "src"
    executable = tmp_path / "python; touch catalog-marker; #"
    executable.symlink_to(sys.executable)
    monkeypatch.setenv("ML_STACK_GYM_PYTHON", str(executable))
    monkeypatch.setenv("PYTHONPATH", str(source))
    monkeypatch.chdir(tmp_path)
    assert {row["id"] for row in catalogue()} == {"car", "warehouse", "traffic", "traffic-driving", "drone"}
    assert not (tmp_path / "catalog-marker").exists()


def test_native_traffic_generator_keeps_hostile_home_path_in_argv(tmp_path, monkeypatch):
    sumo = pytest.importorskip("sumo")
    hostile = tmp_path / "sumo; touch generator-marker; #"
    hostile.symlink_to(sumo.SUMO_HOME, target_is_directory=True)
    monkeypatch.setenv("SUMO_HOME", str(hostile))
    monkeypatch.chdir(tmp_path)
    output = tmp_path / "roads with spaces"
    output.mkdir()
    generate(output, {"arm_length": 120, "lanes": 1, "seed": 2, "demand_seconds": 60,
                      "vehicle_period": 3})
    assert xml(output / "network.net.xml", "net").find("tlLogic") is not None
    assert xml(output / "routes.rou.xml", "routes").find("vehicle") is not None
    assert not (tmp_path / "generator-marker").exists()


def test_worker_spawn_keeps_hostile_configuration_inside_json(tmp_path, monkeypatch):
    monkeypatch.setenv("ML_STACK_GYM_PYTHON", sys.executable)
    marker = tmp_path / "executed"
    text = f"'; touch {marker}; #"
    settings = {"id": "attack", "environment": text, "seed": 0,
                "controller": "manual", "config": {"map": text}}
    updates = queue.Queue()
    with (tmp_path / "worker.log").open("w") as log:
        process = Process(settings, updates, log)
        process.start()
        process.join(15)
        try:
            assert not process.is_alive()
            state = updates.get(timeout=3)
            assert state["status"] == "error"
            assert state["environment"] == text
            assert not marker.exists()
        finally:
            if process.is_alive():
                process.terminate()
            process.commands.close()


def test_malformed_control_stream_closes_native_worker(tmp_path, monkeypatch):
    pytest.importorskip("rware")
    monkeypatch.setenv("ML_STACK_GYM_PYTHON", sys.executable)
    settings = {"id": "stream-attack", "environment": "warehouse", "seed": 0,
                "controller": "manual", "config": {}}
    updates = queue.Queue(maxsize=8)
    with (tmp_path / "worker.log").open("w") as log:
        process = Process(settings, updates, log)
        process.start()
        try:
            assert updates.get(timeout=15)["status"] == "paused"
            process.handle.stdin.write(json.dumps({"command": "play", "payload": "not an object"}) + "\n")
            process.handle.stdin.flush()
            process.join(10)
            assert not process.is_alive()
            failure = updates.get(timeout=3)
            assert "payload object" in failure["error"]
        finally:
            if process.is_alive():
                process.terminate()
            process.commands.close()


def test_manual_world_import_rejects_symlink_escape(tmp_path, monkeypatch):
    root = tmp_path / "files"
    root.mkdir()
    outside = tmp_path / "private-map.json"
    outside.write_text('{"instruction":"read outside the files root"}')
    (root / "map.json").symlink_to(outside)
    monkeypatch.setenv("ML_STACK_GYM_FILES_ROOT", str(root))
    with pytest.raises(ValueError, match="remain under"):
        imported("map.json")
    with pytest.raises(ValueError, match="relative paths"):
        imported("../private-map.json")


def test_invalid_controller_cannot_start_a_worker(monkeypatch):
    from ml_stack.gym import runtime
    monkeypatch.setattr(runtime, "require", lambda _: None)
    started = []
    monkeypatch.setattr(Process, "start", lambda _: started.append(True))
    with pytest.raises(ValueError, match="Unknown simulation controller"):
        SessionManager().create("car", controller="native-idm; touch marker")
    assert not started
