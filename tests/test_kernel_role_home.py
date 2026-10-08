"""Native immutable role-home and finite state-path checks."""
import io
import json
import os
import socket
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import test_kernel_assets as loader
import test_kernel_role_smoke as smoke

from ml_stack.files import write_json


def test_role_home_atomic_state_and_immutable_fences(tmp_path):
    bank = json.loads(Path(os.environ["ML_STACK_TEST_HOLDER_ROLE_BANK"]).read_text())
    node = "tests/test_kernel_role_home.py::test_role_home_atomic_state_and_immutable_fences"
    assert set(bank["cases"]) == {node}
    entry = bank["cases"][node]
    home = Path(entry["home"])
    prefix = Path(entry["prepared"]["runtime"]["prefix"])
    assert prefix.parents[3] == home
    asset = prefix / "pyvenv.cfg"
    before = asset.read_bytes()
    write_json(home / "broker.json", {"generation": "first"})
    write_json(home / "broker.json", {"generation": "second"})
    assert json.loads((home / "broker.json").read_text()) == {"generation": "second"}
    for name in ("other.json", "badabcdefgh.tmp", "tmpabcdefg.tmp", "tmpabcdefghi.tmp",
                 "tmpabcdefgh.bad", "tmpabcdefgh.tmp\n"):
        with pytest.raises(PermissionError):
            (home / name).write_bytes(b"unlisted")
    with pytest.raises(PermissionError):
        (prefix / "lib/python3.13/site-packages/ml_stack/serve/broker.py").read_bytes()
    for operation in (lambda: asset.write_bytes(b"changed"), lambda: asset.chmod(0o600),
                      asset.unlink, lambda: asset.rename(home / "broker-handoff.json"),
                      lambda: (tmp_path / "alias").hardlink_to(asset),
                      lambda: prefix.chmod(0o700), lambda: (home / "runtimes").chmod(0o755),
                      lambda: home.chmod(0o755), lambda: home.parent.chmod(0o755),
                      lambda: prefix.rename(home / "moved"),
                      lambda: (home / "runtimes").rename(home / "moved-family"),
                      lambda: home.rename(tmp_path / "moved-home")):
        with pytest.raises(PermissionError):
            operation()
    channel = home / "holder-channels" / "owned.sock"
    with socket.socket(socket.AF_UNIX) as listener:
        listener.settimeout(1)
        listener.bind(str(channel))
        listener.listen(1)
        with socket.socket(socket.AF_UNIX) as caller:
            caller.settimeout(1)
            caller.connect(str(channel))
            accepted, _ = listener.accept()
            with accepted:
                accepted.settimeout(1)
                caller.sendall(b"owned")
                assert accepted.recv(5) == b"owned"
    channel.unlink()
    redirected = home / "holder-channels/redirected"
    redirected.symlink_to(asset)
    try:
        with pytest.raises(PermissionError):
            redirected.write_bytes(b"changed through alias")
    finally:
        redirected.unlink()
    for forbidden in (home / "forbidden.sock", home.parent / "forbidden.sock",
                      home / "runtimes/bad.sock", home / "runtimes/darwin-arm64-cpython-313/bad.sock"):
        assert len(os.fsencode(forbidden)) < 104
        with socket.socket(socket.AF_UNIX) as listener, pytest.raises(PermissionError):
            listener.bind(str(forbidden))
    assert asset.read_bytes() == before


def test_fixed_normal_role_imports_and_runtime_verification():
    bank = json.loads(Path(os.environ["ML_STACK_TEST_HOLDER_ROLE_BANK"]).read_text())
    node = "tests/test_kernel_role_home.py::test_fixed_normal_role_imports_and_runtime_verification"
    assert set(bank["cases"]) == {node}
    entry = bank["cases"][node]
    descriptor = entry["prepared"]["runtime"]
    home, prefix = Path(entry["home"]), Path(descriptor["prefix"])
    assert prefix.parents[3] == home and entry["variant"] == "normal"
    report = json.loads(Path(os.environ["ML_STACK_TEST_ROLE_IMPORT_REPORT"]).read_text())
    assert report["status"] == "passed" and report["exit"] == 0 and report["case"] == node
    value = report["identity"]
    assert value["prefix"] == str(prefix) and value["version"] == "0.1.0" and value["stamp"] == "a" * 40
    assert Path(value["package"]) == prefix / "lib/python3.13/site-packages/ml_stack/__init__.py"
    assert Path(value["holder"]).is_relative_to(prefix) and Path(value["psutil"]).is_relative_to(prefix)
    assert set(bank["original_variant_files"]) == {"normal", "slow", "stale"}
    for original in bank["original_variant_files"].values():
        with pytest.raises(PermissionError):
            Path(original).read_bytes()
    with pytest.raises(PermissionError):
        (prefix / "lib/python3.13/site-packages/ml_stack/serve/broker.py").read_bytes()


def test_smoke_initial_capture_failure_never_releases_gate(monkeypatch):
    pipe = io.BytesIO()
    killed = []
    process = SimpleNamespace(pid=123, stdin=pipe, stdout=io.BytesIO(), stderr=io.BytesIO(),
                              returncode=-9, kill=lambda: killed.append(123), wait=lambda **kw: -9)
    monkeypatch.setattr(smoke, "start_process", lambda *a, **kw: process)
    monkeypatch.setattr(smoke, "capture_generation", lambda p: (_ for _ in ()).throw(PermissionError()))
    monkeypatch.setattr(smoke, "owned_group", lambda *a: pytest.fail("uncaptured group was inspected"))
    state = {}
    with pytest.raises(PermissionError):
        smoke.execute(["fixed"], {}, time.monotonic() + 10, state)
    assert killed == [123] and state["gate_released"] is False
    assert state["cleanup"] == "pre-gate direct child reaped"
    assert pipe.closed and process.stdout.closed and process.stderr.closed


def test_smoke_changed_generation_never_signals_group(monkeypatch):
    process = SimpleNamespace(pid=123)
    owner = SimpleNamespace(create_time=lambda: 2, uids=lambda: SimpleNamespace(real=os.getuid()))
    monkeypatch.setattr(smoke.psutil, "Process", lambda pid: owner)
    monkeypatch.setattr(smoke, "terminate_process_group", lambda *a, **kw: pytest.fail("changed group signaled"))
    with pytest.raises(RuntimeError, match="generation changed"):
        smoke.owned_group(process, (1, os.getuid(), 123, 123), time.monotonic() + 10)


def test_smoke_unknown_membership_never_signals_or_claims_cleanup(monkeypatch):
    process = SimpleNamespace(pid=123, stdin=io.BytesIO(), stdout=io.BytesIO(), stderr=io.BytesIO(),
                              returncode=0, wait=lambda **kw: 0)
    monkeypatch.setattr(smoke, "start_process", lambda *a, **kw: process)
    monkeypatch.setattr(smoke, "capture_generation", lambda p: (1, os.getuid(), 123, 123))
    monkeypatch.setattr(smoke, "read_output", lambda *a: (b"", b""))
    monkeypatch.setattr(smoke.psutil, "Process", lambda pid: SimpleNamespace(status=lambda: smoke.psutil.STATUS_ZOMBIE))
    monkeypatch.setattr(smoke, "owned_group", lambda *a: (_ for _ in ()).throw(RuntimeError("unknown membership")))
    state = {}
    with pytest.raises(RuntimeError, match="unknown membership"):
        smoke.execute(["fixed"], {}, time.monotonic() + 10, state)
    assert state["gate_released"] is True and state["cleanup"] == "unknown"
    assert state["cleanup_error"] == "RuntimeError"
    assert process.stdout.closed and process.stderr.closed


def test_owned_smoke_reaps_exited_leader_and_known_stdout_child():
    report = json.loads(Path(os.environ["ML_STACK_TEST_ROLE_CONTROL_REPORT"]).read_text())
    assert report["status"] == "passed" and report["expected_output_timeout"] is True
    assert report["cleanup"] == "captured group checked and leader reaped"
    assert report["exit"] == 0 and report["gate_released"] is True
    assert report["child_group"] == report["group"] == report["pid"]
    assert report["child_sid"] == report["sid"] == report["pid"]
    assert report["child_uid"] == report["uid"] == os.getuid()
    assert report["child_started"] >= report["started"]
    assert report["child_terminal"] in {"gone", "zombie"}


def test_loader_omits_only_the_verified_first_self_install_name(tmp_path, monkeypatch):
    image = tmp_path / "native.so"
    image.write_bytes(b"owned fixture")
    name = "@rpath/fixture.so"
    monkeypatch.setattr(loader, "owned_image", lambda *a: (image, set(), ()))
    monkeypatch.setattr(loader, "inspect_tool", lambda *a: f"{image}:\n{name}\n")
    monkeypatch.setattr(loader, "inspect_images", lambda *a: (
        f"{image}:\n\t{name} (compatibility version 0.0.0, current version 0.0.0)\n"
        "\t/usr/lib/libSystem.B.dylib (compatibility version 1.0.0, current version 1.0.0)\n"))
    assert loader.image_dependencies([image], image, (), None, time.monotonic() + 5) == [Path('/usr/lib/libSystem.B.dylib')]


@pytest.mark.parametrize("hostile", ["real-rpath", "same-name-later", "misplaced", "malformed", "missing-header", "identity-drift"])
def test_loader_refuses_unverified_or_changed_install_name(tmp_path, monkeypatch, hostile):
    image = tmp_path / "native.so"
    image.write_bytes(b"owned fixture")
    name = "@rpath/fixture.so"
    names = f"{image}:\n{name}\n"
    dependencies = [name]
    if hostile == "real-rpath":
        dependencies.append("@rpath/external.dylib")
    elif hostile == "same-name-later":
        dependencies.append(name)
    elif hostile == "misplaced":
        dependencies = ["/usr/lib/libSystem.B.dylib", name]
    elif hostile == "malformed":
        names += "second-install-name\n"
    elif hostile == "missing-header":
        names = name + "\n"
    monkeypatch.setattr(loader, "owned_image", lambda *a: (image, set(), ()))
    def inspection(*args):
        if hostile == "identity-drift":
            image.chmod(0o400)
        return names
    monkeypatch.setattr(loader, "inspect_tool", inspection)
    monkeypatch.setattr(loader, "inspect_images", lambda *a: str(image) + ":\n" + ''.join(
        f"\t{dependency} (compatibility version 0.0.0, current version 0.0.0)\n" for dependency in dependencies))
    with pytest.raises(RuntimeError):
        loader.image_dependencies([image], image, (), None, time.monotonic() + 5)
