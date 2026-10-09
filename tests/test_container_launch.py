"""Typed container identities and isolation resources are exact."""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_container_launch import (
    ContainerProof,
    ContainerRun,
    source_digest,
    validate_base_image,
    validate_container,
)

pytestmark = pytest.mark.redteam


def row():
    return {"Id": "c" * 64, "Image": "sha256:" + "d" * 64, "State": {"Running": True},
            "Config": {"Labels": {"ml-stack.test-run": "owner"}, "WorkingDir": "/work",
                       "Cmd": ["sleep", "infinity"], "Entrypoint": None, "User": "1000:1000", "Volumes": None},
            "HostConfig": {"IpcMode": "private", "Init": True, "NetworkMode": "none", "ReadonlyRootfs": True,
                           "CapDrop": ["ALL"], "SecurityOpt": ["no-new-privileges"], "Tmpfs": {"/tmp": "rw,nosuid,nodev,noexec,size=1073741824"}},
            "Mounts": []}


def validate(value, running=True):
    validate_container(value, ContainerProof("c" * 64, "owner", "sha256:" + "d" * 64, Path("/stage"), "deps", True), running=running)


def test_foreign_identity_and_image_are_refused():
    original = row()
    validate(original)
    for field, changed in (("Id", "e" * 64), ("Image", "sha256:" + "e" * 64)):
        value = copy.deepcopy(original)
        value[field] = changed
        with pytest.raises(RuntimeError):
            validate(value)
    value = copy.deepcopy(original)
    value["Config"]["Labels"]["ml-stack.test-run"] = "foreign"
    with pytest.raises(RuntimeError):
        validate(value)
    original["State"]["Running"] = False
    with pytest.raises(RuntimeError):
        validate(original)
    validate(original, running=False)


def test_unsafe_namespaces_and_mounts_are_refused():
    changes = {"Privileged": True, "PidMode": "host", "NetworkMode": "default", "IpcMode": "host",
               "CapAdd": ["NET_ADMIN"], "Devices": [{"PathOnHost": "/dev/gpu"}], "VolumesFrom": ["foreign"],
               "PortBindings": {"80/tcp": []}, "ReadonlyRootfs": False, "CapDrop": [], "SecurityOpt": [],
               "Tmpfs": {"/tmp": "rw"}}
    for key, changed in changes.items():
        value = row()
        value["HostConfig"][key] = changed
        with pytest.raises(RuntimeError):
            validate(value)
    value = row()
    value["Mounts"] = [{"Type": "bind", "Source": "/", "Destination": "/host", "RW": False}]
    with pytest.raises(RuntimeError):
        validate(value)
    for key, changed in (("Entrypoint", ["arbitrary"]), ("User", "0"), ("Volumes", {"/host": {}})):
        value = row()
        value["Config"][key] = changed
        with pytest.raises(RuntimeError):
            validate(value)


def test_tracked_stage_refuses_symlinks_unlisted_files_and_missing_entries(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "one").write_text("first")
    source_digest(source, ("one",))
    with pytest.raises(RuntimeError, match="incomplete"):
        source_digest(source, ("one", "missing"))
    (source / "extra").write_text("unlisted")
    with pytest.raises(RuntimeError, match="unlisted"):
        source_digest(source, ("one",))
    (source / "extra").unlink()
    (source / "one").unlink()
    (source / "one").symlink_to(tmp_path / "outside")
    with pytest.raises(RuntimeError, match="redirected"):
        source_digest(source, ("one",))


def test_base_image_startup_behavior_is_refused_before_creation(tmp_path):
    validate_base_image({})
    for changed in ({"Entrypoint": ["arbitrary"]}, {"User": "1000"}, {"Volumes": {"/host": {}}},
                    {"Healthcheck": {"Test": ["CMD", "arbitrary"]}}, {"Env": ["BASH_ENV=/malicious"]}):
        launch = object.__new__(ContainerRun)
        launch.source = lambda: tmp_path
        launch.platform = ""
        launch.image = "python:3.13"
        calls = []

        def call(*arguments, records=calls, config=changed, **options):
            records.append(arguments)
            if arguments[0] not in ("pull", "image"):
                raise AssertionError("unadmitted image reached container creation")
            return SimpleNamespace(stdout=json.dumps([{"Id": "sha256:" + "d" * 64, "Config": config}]))

        launch.call = call
        with pytest.raises(RuntimeError, match="startup behavior"):
            launch.prepare()
        assert [command[0] for command in calls] == ["pull", "image"]


def test_wrapper_and_selector_admission_remain_exact():
    with pytest.raises(RuntimeError, match="maintained pytest interpreter"):
        ContainerRun(["bash", "/arbitrary", "tests/test_layers.py"], {})
    with pytest.raises(RuntimeError, match="fixture admission"):
        ContainerRun([sys.executable, "-m", "pytest", "tests/arbitrary.py"], {})


@pytest.mark.parametrize("failure", ["construct", "recheck", "terminal_close"])
def test_failed_native_preparation_closes_owned_resources(monkeypatch, tmp_path, failure):
    import contextlib

    import test_kernel_isolation
    import test_kernel_lifecycle
    import test_terminal_bank
    import testslots_runner

    calls = []
    primary = RuntimeError("native image preparation failed")

    class Terminals:
        token = "test-token"
        endpoint = "test-endpoint"
        identity = (1, 2, 3)

        def __init__(self, *arguments):
            calls.append("terminal_create")

        def close(self):
            calls.append("terminal_close")
            if failure == "terminal_close":
                raise RuntimeError("terminal close failed")

    class Confined:
        wrapped = SimpleNamespace(argv=["test-command"])

        def __init__(self, *arguments, **options):
            self.environment = {}
            calls.append("confined_create")
            if failure == "construct":
                raise primary

        def recheck_images(self):
            calls.append("recheck")
            raise primary

        def finish(self):
            calls.append("confined_close")

    admission = SimpleNamespace(endpoint="test-endpoint", token="test-token", identity=(1, 2, 3),
                                terminal_admission=lambda: None, finish=lambda: calls.append("admission_close"))
    monkeypatch.setattr(testslots_runner.sys, "platform", "darwin")
    monkeypatch.setattr(testslots_runner, "environment_for", lambda value: {"DEV_TEST_BUDGET": "1"})
    monkeypatch.setattr(testslots_runner.testslots, "slots_dir", lambda: tmp_path)
    monkeypatch.setattr(testslots_runner.testslots, "_reject_nested", lambda: None)
    monkeypatch.setattr(testslots_runner.testslots, "lease", lambda *args, **kwargs: contextlib.nullcontext())
    monkeypatch.setattr(testslots_runner.testslots_rpc, "UnixAdmission", lambda *args: admission)
    monkeypatch.setattr(test_terminal_bank, "TerminalBank", Terminals)
    monkeypatch.setattr(test_kernel_isolation, "ConfinedRun", Confined)
    monkeypatch.setattr(test_kernel_lifecycle, "held_shutdown", lambda: calls.append("held_shutdown"))
    monkeypatch.setattr(test_kernel_lifecycle, "retired_probe", lambda value: calls.append("retired_probe"))

    def spawn(*arguments, **options):
        raise AssertionError("failed preparation launched pytest")

    monkeypatch.setattr(testslots_runner.subprocess, "Popen", spawn)
    with pytest.raises(RuntimeError) as caught:
        testslots_runner.run_pytest(["test-command"], 1)
    assert calls.count("admission_close") == 1
    assert calls.count("terminal_close") == 1
    assert calls.count("confined_close") == (0 if failure == "construct" else 1)
    if failure == "terminal_close":
        assert caught.value.__context__ is primary
    else:
        assert caught.value is primary
