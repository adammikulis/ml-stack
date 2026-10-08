"""Supervisor confinement policy and reserved resource bounds."""
from __future__ import annotations

import hashlib
import json
import os
import re
import socket
import stat
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import test_kernel_holder as holder
import test_kernel_isolation as isolation
import testslots_rpc
from confinement import needs_confinement
from test_kernel_attestation import CHECKS, Attestation, validate_result
from test_kernel_endpoint import connection, verify_socket
from test_kernel_holder import Assets, ancestry, private_directory
from test_kernel_lifecycle import observe_exit
from test_kernel_regex import execute

from ml_stack.sandbox import path_language
from ml_stack.sandbox.policy import Net, Policy
from ml_stack.sandbox.seatbelt import (
    SYMLINKS,
    SYSTEM_READ,
    _ancestors,
    _existing,
    executable_links,
    profile,
    quote,
)


@needs_confinement
def test_holder_namespace_allows_only_its_owned_unix_channels():
    root = Path(os.environ["DEV_TEST_HOLDER_NAMESPACE"])
    channel = root / "owned.sock"
    outside = Path(os.environ["DEV_TEST_DENIED_ENDPOINT"]).with_name("x.sock")
    assert len(os.fsencode(outside)) < 104 and not outside.is_relative_to(root)
    with socket.socket(socket.AF_UNIX) as listener:
        listener.settimeout(2)
        listener.bind(str(channel))
        listener.listen(1)
        try:
            with socket.socket(socket.AF_UNIX) as client:
                client.settimeout(2)
                client.connect(str(channel))
                client.sendall(b"owned")
                stream, _ = listener.accept()
                with stream:
                    stream.settimeout(2)
                    assert stream.recv(5) == b"owned"
            with socket.socket(socket.AF_UNIX) as denied, pytest.raises(PermissionError):
                denied.bind(str(outside))
            with pytest.raises(PermissionError):
                subprocess.run(["/usr/bin/true"], check=True, timeout=2, close_fds=True)
        finally:
            channel.unlink()


@needs_confinement
def test_holder_namespace_refuses_redirect_owner_and_path_bounds(tmp_path):
    root = Path(os.environ["DEV_TEST_HOLDER_NAMESPACE"])
    path = root / "invalid"
    path.mkdir(mode=0o700)
    try:
        path.chmod(0o777)
        with pytest.raises(RuntimeError, match="private and owned"):
            private_directory(path)
        path.chmod(0o700)
        linked = root / "link"
        linked.symlink_to(path)
        try:
            with pytest.raises(RuntimeError, match="redirected"):
                private_directory(linked)
        finally:
            linked.unlink()
        with pytest.raises(RuntimeError, match="too long"):
            private_directory(tmp_path)
        observed = path.stat()
        class Foreign:
            def __fspath__(self):
                return str(path)
            def resolve(self, *, strict):
                return self
            def lstat(self):
                return SimpleNamespace(st_mode=stat.S_IFDIR | 0o700, st_uid=observed.st_uid + 1)
        with pytest.raises(RuntimeError, match="private and owned"):
            private_directory(Foreign())
    finally:
        path.rmdir()


@needs_confinement
def test_holder_assets_refuse_replaceable_ancestry(tmp_path):
    parent = tmp_path / "parent"
    parent.mkdir(mode=0o700)
    prefix = parent / "prefix"
    prefix.mkdir(mode=0o500)
    parent.chmod(0o777)
    try:
        with pytest.raises(RuntimeError, match="foreign or writable"):
            ancestry(prefix)
    finally:
        parent.chmod(0o700)
    assets = Assets([], set(), set(), set(), {}, ancestry(prefix))
    parent.chmod(0o777)
    try:
        with pytest.raises(RuntimeError, match="ancestry changed"):
            assets.recheck()
    finally:
        parent.chmod(0o700)
    redirected = tmp_path / "redirected"
    redirected.symlink_to(parent)
    with pytest.raises(RuntimeError, match="redirected"):
        ancestry(redirected / "prefix")


@needs_confinement
@pytest.mark.parametrize("pid", [True, 0, -1, "foreign"])
def test_retired_probe_rejects_invalid_process_identity(pid):
    with pytest.raises(RuntimeError, match="invalid owned probe PID"):
        observe_exit(pid, [], -1)


@pytest.mark.parametrize("field,value", [("nonce", "replayed"), ("profile", "foreign"), ("checks", []),
                                         ("extra", True)], ids=["nonce", "profile", "failed-probe", "extra"])
@needs_confinement
def test_precollection_refuses_forged_or_incomplete_report(field, value):
    result = {"nonce": "owned", "profile": "pinned", "checks": CHECKS}
    result[field] = value
    with pytest.raises(RuntimeError, match="invalid precollection"):
        validate_result(result, "owned", "pinned")


@needs_confinement
def test_precollection_bad_nonce_does_not_release_collection():
    evidence = Attestation.__new__(Attestation)
    evidence.report, output = os.pipe()
    acknowledgement, evidence.release = os.pipe()
    receiver = os.dup(acknowledgement)
    os.set_blocking(receiver, False)
    evidence.descriptors = {evidence.report, output, acknowledgement, evidence.release}
    evidence.pass_fds = (output, acknowledgement)
    evidence.confined = SimpleNamespace(observation_seconds=1)
    evidence.nonce, evidence.profile = "owned", "pinned"
    evidence.target = evidence.denied = None
    os.write(output, json.dumps({"nonce": "replay", "profile": "pinned", "checks": CHECKS}).encode() + b"\n")
    try:
        with pytest.raises(RuntimeError, match="invalid precollection"):
            evidence.launched(SimpleNamespace(poll=lambda: None))
        with pytest.raises(BlockingIOError):
            os.read(receiver, 1)
    finally:
        evidence.close()
        os.close(receiver)


@pytest.mark.parametrize("field,value", [("heavy", 1), ("phase", "other"), ("label", "x" * 257),
                                         ("parent", "foreign"), ("extra", True), ("token", []),
                                         ("operation", "execute")],
                         ids=["heavy-int", "phase", "long-label", "parent", "extra-field", "token-list", "operation"])
@needs_confinement
def test_admission_rejects_invalid_fields_before_mutation(field, value):
    request = {"operation": "ready", "token": "a" * 48, "parent": None,
               "label": "pytest", "phase": "test", "heavy": False}
    request[field] = value
    with pytest.raises(ValueError):
        testslots_rpc.validate_request(request)


@needs_confinement
def test_terminal_schema_rejects_paths_and_oversized_data():
    request = {"operation": "terminal", "token": "a" * 48, "parent": "b" * 48,
               "terminal_token": "c" * 48, "terminal_operation": "write", "terminal": "d" * 48,
               "data": "ab" * 4096}
    testslots_rpc.validate_request(request)
    with pytest.raises(ValueError):
        testslots_rpc.validate_request({**request, "data": "ab" * 4097})
    with pytest.raises(ValueError):
        testslots_rpc.validate_request({**request, "path": "/tmp/foreign"})


@needs_confinement
@pytest.mark.parametrize("value", [1, False, "true", None])
def test_admission_release_requires_exact_boolean(value):
    with pytest.raises(ValueError):
        testslots_rpc.validate_release({"release": value})


@needs_confinement
def test_ended_admission_cannot_transfer_terminal():
    admission = testslots_rpc.Admission.__new__(testslots_rpc.Admission)
    admission.guard = threading.Lock()
    admission.active = {"a" * 48}
    with admission.terminal_admission("a" * 48):
        pass
    admission.active.clear()
    with pytest.raises(PermissionError, match="inactive"), admission.terminal_admission("a" * 48):
        pytest.fail("inactive admission entered descriptor transfer")


@needs_confinement
def test_supervisor_rejects_unknown_request_fields_over_actual_rpc():
    with connection(os.environ["DEV_TEST_PYTEST_ENDPOINT"], os.environ["DEV_TEST_PYTEST_IDENTITY"]) as channel, channel.makefile("rwb") as stream:
        testslots_rpc._write(stream, {"operation": "ready", "token": os.environ["DEV_TEST_PYTEST_TOKEN"],
                                     "parent": None, "label": "pytest", "phase": "test", "heavy": False,
                                     "command": "forged"})
        assert "invalid admission fields" in json.loads(stream.readline(65537))["error"]
    with testslots_rpc.request("ping") as response:
        assert response == {"ping": True}


@needs_confinement
def test_admission_socket_identity_change_is_refused_before_connection():
    identity = json.loads(os.environ["DEV_TEST_PYTEST_IDENTITY"])
    identity[1] += 1
    with pytest.raises(PermissionError, match="identity changed"):
        connection(os.environ["DEV_TEST_PYTEST_ENDPOINT"], json.dumps(identity))
    with testslots_rpc.request("ping") as response:
        assert response == {"ping": True}


@needs_confinement
def test_owned_retained_socket_nodes_are_private_and_inode_bound():
    for endpoint, identity in (("DEV_TEST_PYTEST_ENDPOINT", "DEV_TEST_PYTEST_IDENTITY"),
                               ("DEV_TEST_PTY_ENDPOINT", "DEV_TEST_PTY_IDENTITY")):
        path = os.environ[endpoint].removeprefix("unix:")
        verify_socket(path, json.loads(os.environ[identity]))
        with pytest.raises(PermissionError):
            Path(path).unlink()


@needs_confinement
@pytest.mark.parametrize("request_count,permits", [(65536, 128), (0, 0)])
def test_admission_rejects_exhausted_requests_without_starting_handler(request_count, permits):
    admission = testslots_rpc.Admission.__new__(testslots_rpc.Admission)
    admission.guard = threading.Lock()
    admission.handlers = threading.BoundedSemaphore(permits)
    admission.request_count = request_count
    closed = []
    admission.shutdown_request = closed.append
    admission.process_request("owned-test-connection", ("127.0.0.1", 1))
    assert closed == ["owned-test-connection"]
    assert admission.request_count == request_count


@needs_confinement
def test_reserved_descriptor_rejects_foreign_admission(monkeypatch):
    reservation = testslots_rpc.terminal_request("allocate")
    try:
        with monkeypatch.context() as change:
            change.setenv("DEV_TEST_REMOTE_LEASE", "f" * 48)
            with pytest.raises(RuntimeError, match="descriptor response"):
                testslots_rpc.terminal_slave(reservation["terminal"])
        descriptor = testslots_rpc.terminal_slave(reservation["terminal"])
        os.close(descriptor)
    finally:
        testslots_rpc.terminal_request("release", terminal=reservation["terminal"])


@needs_confinement
def test_unknown_fixture_tier_is_refused_before_collection():
    with pytest.raises(RuntimeError, match="fixture admission"):
        isolation.check_selectors([sys.executable, "-m", "pytest", "tests/test_workspace_live.py"])
    isolation.check_selectors([sys.executable, "-m", "pytest", "tests/test_layers.py"])


@needs_confinement
def test_literal_source_grants_do_not_read_neighbor_files(tmp_path):
    source = tmp_path / "source.py"
    source.write_text("pass")
    policy = Policy(read_files=(str(source),), read_dirs=(str(tmp_path),), net=Net.only(12345)).validated()
    text = profile(policy, sys.executable, "source-policy")
    assert f'(literal "{source}")' in text
    assert f'(subpath "{tmp_path}")' not in text
    assert 'localhost:12345' in text
    assert 'localhost:*' not in text
    assert 'network-bind' not in text


@needs_confinement
def test_inherited_credentials_and_loader_paths_are_removed():
    environment = isolation.filtered_environment({"PATH": "/usr/bin:/bin", "SSH_AUTH_SOCK": "/secret.sock",
                                                  "ML_STACK_HOME": "/real-state", "PYTHONPATH": "/secret",
                                                  "DEV_TEST_ARBITRARY_TOKEN": "private"})
    assert environment == {"PATH": "/usr/bin:/bin"}


@needs_confinement
@pytest.mark.skipif(sys.platform != "darwin", reason="supervisor terminal admission")
def test_terminal_bank_requires_its_token_and_limits_input(monkeypatch):
    import testslots_rpc
    token = os.environ["DEV_TEST_PTY_TOKEN"]
    with monkeypatch.context() as change:
        change.setenv("DEV_TEST_PTY_TOKEN", "f" * 48)
        with pytest.raises(RuntimeError, match="token"):
            testslots_rpc.terminal_request("allocate")
    assert os.environ["DEV_TEST_PTY_TOKEN"] == token
    reserved = testslots_rpc.terminal_request("allocate")
    try:
        assert "slave" not in reserved and "device" not in reserved
        with pytest.raises(RuntimeError, match="bound"):
            testslots_rpc.terminal_request("write", terminal=reserved["terminal"], data=(b"a" * 4097).hex())
    finally:
        testslots_rpc.terminal_request("release", terminal=reserved["terminal"])


@needs_confinement
@pytest.mark.skipif(sys.platform != "darwin", reason="macOS kernel confinement")
def test_kernel_denies_parent_storage_for_test_and_descendant():
    canary = os.environ["DEV_TEST_CONFINEMENT_CANARY"]
    with pytest.raises(PermissionError):
        Path(canary).read_bytes()
    with pytest.raises(PermissionError):
        Path(canary).write_bytes(b"forged")
    code = "from pathlib import Path; import sys; Path(sys.argv[1]).write_bytes(b'descendant')"
    result = subprocess.run([sys.executable, "-c", code, canary], capture_output=True, text=True, timeout=10)
    assert result.returncode != 0 and "PermissionError" in result.stderr


@needs_confinement
@pytest.mark.skipif(sys.platform != "darwin", reason="macOS reserved descriptor transfer")
def test_reserved_slave_descriptor_supports_io_without_path_permission():
    import testslots_rpc
    reservation = testslots_rpc.terminal_request("allocate")
    slave = testslots_rpc.terminal_slave(reservation["terminal"])
    try:
        assert os.isatty(slave)
        os.write(slave, b"reserved-terminal\n")
        response = testslots_rpc.terminal_request("read", terminal=reservation["terminal"])
        assert b"reserved-terminal" in bytes.fromhex(response["data"])
        with pytest.raises(PermissionError):
            os.open(reservation["path"], os.O_RDWR)
    finally:
        os.close(slave)
        testslots_rpc.terminal_request("release", terminal=reservation["terminal"])


@needs_confinement
@pytest.mark.skipif(sys.platform != "darwin", reason="macOS inherited kernel confinement")
def test_kernel_denies_symlink_escape_descendant_read_and_other_unix_endpoint(tmp_path):
    canary = os.environ["DEV_TEST_CONFINEMENT_CANARY"]
    link = tmp_path / "escape"
    link.symlink_to(canary)
    with pytest.raises(PermissionError):
        link.read_bytes()
    code = "from pathlib import Path; import sys; Path(sys.argv[1]).read_bytes()"
    result = subprocess.run([sys.executable, "-c", code, canary], capture_output=True, text=True, timeout=10)
    assert result.returncode != 0 and "PermissionError" in result.stderr
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
        connection.settimeout(1)
        with pytest.raises(PermissionError):
            connection.connect(os.environ["DEV_TEST_DENIED_ENDPOINT"])
    import testslots_rpc
    with testslots_rpc.request("ready") as response:
        assert response == {"ready": True}


@needs_confinement
@pytest.mark.skipif(sys.platform != "darwin", reason="macOS static preflight")
def test_preflight_does_not_import_pytest_or_conftest():
    code = ("import sys; from pathlib import Path; from testselectors import validate; "
            "validate(['tests/test_test_kernel_isolation.py'], Path.cwd()); "
            "assert 'pytest' not in sys.modules; assert 'conftest' not in sys.modules")
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr


@needs_confinement
@pytest.mark.skipif(sys.platform != "darwin", reason="explicit host-admin package boundary")
def test_host_admin_index_rule_keeps_other_paths_and_world_writes_refused(monkeypatch):
    import grp

    import test_kernel_assets as assets
    monkeypatch.setattr(grp, "getgrnam", lambda name: SimpleNamespace(gr_gid=80))
    info = SimpleNamespace(st_mode=0o40775, st_uid=os.getuid(), st_gid=80)
    assert assets.trusted_host_index(Path("/opt/homebrew/Cellar"), info)
    assert assets.trusted_host_index(Path("/opt/homebrew/opt"), info)
    assert not assets.trusted_host_index(Path("/opt/homebrew/Cellar/package"), info)
    info.st_mode = 0o40777
    assert not assets.trusted_host_index(Path("/opt/homebrew/Cellar"), info)
    info.st_mode = 0o40775
    info.st_uid = 987654
    assert not assets.trusted_host_index(Path("/opt/homebrew/Cellar"), info)


@needs_confinement
@pytest.mark.skipif(sys.platform != "darwin", reason="macOS descriptor path validation")
@pytest.mark.parametrize("alias", ["direct", "symlink", "hardlink"])
def test_supervisor_output_refuses_protected_destinations(tmp_path, alias):
    from test_kernel_outputs import junit_sink
    protected = tmp_path / "protected"
    protected.mkdir()
    target = protected / "result.xml"
    target.write_bytes(b"unchanged")
    target.chmod(0o600)
    destination = target
    if alias != "direct":
        destination = tmp_path / "alias.xml"
        if alias == "symlink":
            destination.symlink_to(target)
        else:
            os.link(target, destination)
    with pytest.raises(RuntimeError):
        junit_sink(destination, (protected,))
    assert target.read_bytes() == b"unchanged"


@needs_confinement
@pytest.mark.skipif(sys.platform != "darwin", reason="macOS descriptor path validation")
def test_supervisor_output_pins_both_streams_and_bounds_writes(tmp_path):
    from test_kernel_outputs import LIMIT, OutputSink
    protected = tmp_path / "protected"
    protected.mkdir()
    for name in ("stdout", "stderr"):
        path = protected / name
        with path.open("wb") as stream, pytest.raises(RuntimeError, match="protected roots"):
            OutputSink(stream.fileno(), (protected,))
    path = tmp_path / "allowed"
    with path.open("wb") as stream:
        sink = OutputSink(stream.fileno(), (protected,))
        try:
            sink.write(b"bounded")
            sink.total = LIMIT
            with pytest.raises(RuntimeError, match="exceeds bound"):
                sink.write(b"x")
        finally:
            sink.close()
    assert path.read_bytes() == b"bounded"


@needs_confinement
def test_supervisor_storage_refuses_protected_temp_before_creation(tmp_path, monkeypatch):
    protected = tmp_path / "protected"
    protected.mkdir()
    monkeypatch.setattr(isolation.tempfile, "gettempdir", lambda: str(protected))
    monkeypatch.setattr(isolation, "protected_roots", lambda environment: (protected,))
    with pytest.raises(RuntimeError, match="storage overlaps"):
        isolation.supervisor_storage({"ML_STACK_HOME": str(protected)})
    assert list(protected.iterdir()) == []


@needs_confinement
def test_supervisor_storage_refuses_ancestors_of_protected_roots(tmp_path):
    from ml_stack.activity.source_snapshot import validate_storage
    protected = tmp_path / "protected"
    protected.mkdir()
    with pytest.raises(RuntimeError, match="storage overlaps"):
        validate_storage(tmp_path, (protected,))


@needs_confinement
def test_source_inventory_does_not_execute_inherited_fsmonitor():
    import json
    manifest = Path(os.environ["DEV_TEST_CONFINEMENT_CANARY"]).parent / "bootstrap.json"
    value = json.loads(manifest.read_text())["git_observation"]
    assert set(value) == {"nonce", "source", "tree", "source_tree"}
    assert len(value["nonce"]) == len(value["source"]) == 64
    assert len(value["tree"]) == 40
    assert len(value["source_tree"]) == 40


@pytest.fixture
def artifact_outputs(tmp_path, monkeypatch):
    from test_kernel_outputs import ArtifactOutputs

    from ml_stack.activity import source_snapshot
    protected = tmp_path / "protected"
    protected.mkdir()
    storage = tmp_path / "storage"
    storage.mkdir(mode=0o700)
    monkeypatch.setattr(source_snapshot, "protected_roots", lambda environment: (protected,))
    monkeypatch.setattr(source_snapshot.tempfile, "gettempdir", lambda: str(storage))
    artifacts = ArtifactOutputs({})
    try:
        yield artifacts
    finally:
        artifacts.close()


@needs_confinement
def test_artifact_namespace_is_fresh_private_and_pins_all_outputs(artifact_outputs):
    artifacts = artifact_outputs
    assert artifacts.directory.stat().st_mode & 0o777 == 0o700
    assert set(artifacts.sinks) == {"console.log", "junit.xml", "evidence.result.json"}
    for name, sink in artifacts.sinks.items():
        sink.write(name.encode())
        assert (artifacts.directory / name).read_bytes() == name.encode()
        assert (artifacts.directory / name).stat().st_mode & 0o777 == 0o600
    assert artifacts.boundary()["run_id"] == artifacts.run_id


@needs_confinement
def test_artifact_redirect_restores_descriptors_after_failure(artifact_outputs, tmp_path):
    saved = [(fd, os.dup(fd)) for fd in (1, 2)]
    try:
        for fd in (1, 2):
            fixture = os.open(tmp_path / str(fd), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            try:
                os.dup2(fixture, fd)
            finally:
                os.close(fixture)
        before = [os.fstat(fd) for fd in (1, 2)]
        with pytest.raises(RuntimeError, match="fixture failure"), artifact_outputs.redirect():
            os.write(1, b"private stdout\n")
            os.write(2, b"private stderr\n")
            raise RuntimeError("fixture failure")
        assert [(os.fstat(fd).st_dev, os.fstat(fd).st_ino) for fd in (1, 2)] == [(v.st_dev, v.st_ino) for v in before]
    finally:
        for fd, original in saved:
            os.dup2(original, fd)
            os.close(original)
    console = (artifact_outputs.directory / "console.log").read_bytes()
    assert b"private stdout" in console and b"private stderr" in console


@needs_confinement
def test_artifact_replaced_output_cannot_be_presented_as_pinned_evidence(artifact_outputs):
    path = artifact_outputs.directory / "evidence.result.json"
    path.rename(path.with_suffix(".original"))
    path.write_text("forged")
    path.chmod(0o600)
    with pytest.raises(RuntimeError, match="changed"):
        artifact_outputs.boundary()


@needs_confinement
def test_artifact_console_overflow_is_aggregate_across_duplicate_descriptors(artifact_outputs):
    from test_kernel_outputs import LIMIT, OutputSink
    console = artifact_outputs.sinks["console.log"]
    duplicate = OutputSink(console.fd, artifact_outputs.roots, private=True)
    try:
        os.ftruncate(console.fd, LIMIT)
        with pytest.raises(RuntimeError, match="exceeds bound"):
            duplicate.write(b"x")
    finally:
        duplicate.close()


@needs_confinement
def test_literal_metadata_compression_preserves_exact_permission_union(tmp_path):
    first = str(tmp_path / 'quoted"file')
    second = str(tmp_path / "second")
    directory = str(tmp_path / "directory")
    policy = Policy(read_files=(first, second), read_metadata=(first, directory, directory),
                    read_dirs=(directory,), strict=True)
    text = profile(policy, sys.executable, "literal-union")
    metadata = set()
    for line in text.splitlines():
        if line.startswith("(allow file-read-metadata "):
            metadata.update(json.loads(value) for value in re.findall(r'\(literal ("(?:[^"\\]|\\.)*")\)', line))
    reads = [*_existing(SYSTEM_READ), sys.executable, os.path.realpath(sys.executable)]
    original = {*_ancestors(reads), *_ancestors([*policy.read_files, *policy.read_dirs, *policy.read_metadata]),
                *executable_links(sys.executable), *SYMLINKS, *policy.read_metadata, *policy.read_dirs}
    assert original | set(policy.read_files) == metadata | set(policy.read_files)
    assert first not in metadata and second not in metadata
    assert str(tmp_path / "directory" / "unlisted") not in metadata
    assert str(tmp_path / "second-sibling") not in metadata
    assert '(subpath "' + str(tmp_path) not in text


@needs_confinement
def test_namespace_only_probes_do_not_admit_immutable_runtime_assets(tmp_path, monkeypatch):
    channel = tmp_path / "channels"
    monkeypatch.setattr(holder, "namespace", lambda protected: channel)
    monkeypatch.setattr(holder, "compile_namespace", lambda root, control: {"exit": 0})
    def refused(path):
        raise AssertionError("runtime inventory requested")
    monkeypatch.setattr(holder, "read_manifest", refused)
    environment = {}
    assets, actual, compilation = holder.prepare(
        ["tests/test_test_kernel_isolation.py::test_holder_namespace_allows_only_its_owned_unix_channels"],
        tmp_path, environment, (), ())
    assert assets is None and actual == channel and compilation == {"exit": 0}
    assert "ML_STACK_TEST_HOLDER_MANIFEST" not in environment
    with pytest.raises(AssertionError, match="runtime inventory requested"):
        holder.prepare([next(iter(holder.RUNTIME_PROBES))], tmp_path, {}, (), ())


@needs_confinement
def test_actual_interpreter_sites_include_inherited_system_packages_only(tmp_path, monkeypatch):
    local, inherited, user = (tmp_path / name for name in ("local", "inherited", "user"))
    for path in (local, inherited, user):
        path.mkdir()
    monkeypatch.setattr(isolation.sysconfig, "get_paths", lambda: {"purelib": str(local)})
    monkeypatch.setattr(isolation.site, "getsitepackages", lambda: [str(local), str(inherited)])
    monkeypatch.setattr(isolation.site, "getusersitepackages", lambda: str(user))
    actual = isolation.interpreter_read_roots((tmp_path / "state",), tmp_path / "account")
    assert set(actual) == {str(local), str(inherited)} and str(user) not in actual


@needs_confinement
@pytest.mark.parametrize("hostile", ["user", "state", "account-parent"])
def test_actual_interpreter_sites_refuse_user_and_protected_roots(tmp_path, monkeypatch, hostile):
    paths = {"user": tmp_path / "user", "state": tmp_path / "state", "account-parent": tmp_path}
    for name in ("user", "state"):
        paths[name].mkdir()
    monkeypatch.setattr(isolation.sysconfig, "get_paths", dict)
    monkeypatch.setattr(isolation.site, "getsitepackages", lambda: [str(paths[hostile])])
    monkeypatch.setattr(isolation.site, "getusersitepackages", lambda: str(paths["user"]))
    with pytest.raises(RuntimeError, match="overlap protected account or user-site"):
        isolation.interpreter_read_roots((paths["state"],), tmp_path / "account")


@needs_confinement
def test_native_regex_matches_only_complete_listed_paths():
    value = json.loads(Path(os.environ["DEV_TEST_REGEX_REPORT"]).read_text())
    assert set(value) == {"literal", "regex"}
    for result in value.values():
        assert result["exit"] == 0 and result["listed"] == 9 and result["denied"] == 17
        assert len(result["profile"]) == 64


@needs_confinement
def test_exact_trie_preserves_all_complete_paths_and_native_text():
    paths = ["/owned/a", "/owned/ab", "/owned/a/child", '/owned/quote"file', "/owned/back\\slash",
             "/owned/meta$[]()+*?{}^.|", "/owned/café", "/owned/漢字"]
    paths.extend(f"/owned/group/{number}" for number in range(129))
    values, receipt = path_language.exact_language(paths + paths[:2], quote)
    recovered = set()
    ordered = sorted(set(paths))
    for offset in range(0, len(ordered), path_language.GROUP):
        recovered.update(path_language.terminals(path_language.tree(ordered[offset:offset + path_language.GROUP])))
    assert recovered == set(paths) and receipt["paths"] == len(set(paths))
    assert receipt["input_sha256"] == receipt["terminals_sha256"]
    assert path_language.terminals(path_language.radix(ordered)) == set(paths)
    assert path_language.terminals(path_language.radix(list(reversed(ordered)))) == set(paths)
    assert len(values) == 1 and receipt["representation"] == "subtree-radix"
    assert all('regex-quote' in value for value in values)
    assert all('"^"' in value and '"$"' in value for value in values)


@needs_confinement
def test_exact_trie_refuses_terminal_substitution(monkeypatch):
    monkeypatch.setattr(path_language, "freeze_radix", lambda raw: path_language.Node("/forged", True, ()))
    with pytest.raises(path_language.LanguageError, match="terminal language differs"):
        path_language.expressions(["/owned"], quote)


@needs_confinement
@pytest.mark.parametrize("bound", ["MAX_PATHS", "MAX_PATH_BYTES", "MAX_INPUT", "MAX_OUTPUT", "MAX_NODES", "MAX_DEPTH"])
def test_exact_trie_bounds_refuse_without_permission_fallback(monkeypatch, bound):
    monkeypatch.setattr(path_language, bound, 0 if bound == "MAX_DEPTH" else 1)
    with pytest.raises(path_language.LanguageError, match="bound"):
        path_language.expressions(["/owned/a", "/owned/ab", "/owned/ac"], quote)


@pytest.mark.parametrize("path", ["relative", "/owned/../other", "/owned//other", "/owned/", "/owned\n"],
                         ids=["relative", "parent", "empty", "trailing", "newline"])
@needs_confinement
def test_exact_trie_refuses_ambiguous_or_control_path_text(path):
    with pytest.raises(path_language.LanguageError, match="malformed"):
        path_language.expressions([path], quote)


@needs_confinement
def test_verified_holder_inventory_runs_only_the_fixed_normal_interpreter():
    manifest = json.loads(Path(os.environ["ML_STACK_TEST_HOLDER_MANIFEST"]).read_text())
    runtime = manifest["variants"]["normal"]["runtime"]
    code = "import json,sys,ml_stack;print(json.dumps({'prefix':sys.prefix,'file':ml_stack.__file__,'version':ml_stack.__version__}))"
    result = execute([runtime["python"], "-I", "-c", code], dict(os.environ), time.monotonic() + 5)
    assert result["prefix"] == runtime["prefix"] and result["version"] == runtime["version"]
    assert Path(result["file"]).is_relative_to(Path(runtime["prefix"]))
    for name in ("slow", "stale"):
        unselected = Path(manifest["variants"][name]["runtime"]["prefix"]) / "pyvenv.cfg"
        with pytest.raises(PermissionError):
            unselected.read_bytes()
    outside = Path(os.environ["DEV_TEST_DENIED_ENDPOINT"]).with_name("x.sock")
    assert len(os.fsencode(outside)) < 104
    with socket.socket(socket.AF_UNIX) as denied, pytest.raises(PermissionError):
        denied.bind(str(outside))
    with pytest.raises(PermissionError):
        subprocess.run(["/usr/bin/true"], check=True, timeout=2, close_fds=True)


@needs_confinement
def test_runtime_variant_mapping_is_explicit_and_mixed_selection_is_union(monkeypatch):
    normal = next(iter(holder.RUNTIME_PROBES))
    slow = "tests/test_holder_protocol.py::test_slow_independent_verification_is_cancelled_without_waiting_or_orphaning"
    stale = "tests/test_holder_protocol.py::test_completed_preparation_cannot_extend_a_changed_broker_generation"
    plain = "tests/test_holder_protocol.py::test_actual_kernel_peer_is_the_connected_owned_child"
    assert holder.required_variants([normal]) == {"normal"}
    assert holder.required_variants([slow + "[slow]", stale]) == {"slow", "stale"}
    assert holder.required_variants([normal, slow, stale]) == {"normal", "slow", "stale"}
    assert holder.required_variants([plain]) == set()
    monkeypatch.setattr(holder, "VARIANTS_BY_NODE", {normal: frozenset({"normal"})})
    with pytest.raises(RuntimeError, match="mapping is incomplete"):
        holder.required_variants([normal])


@needs_confinement
def test_native_read_components_compile_independently():
    report = json.loads(Path(os.environ["DEV_TEST_COMPILER_REPORT"]).read_text())
    assert set(report) == {"literal_files", "files", "metadata", "directories"}
    assert report["literal_files"]["input_sha256"] == report["files"]["input_sha256"]
    assert all(value["exit"] == 0 for value in report.values())


@needs_confinement
def test_exact_radix_partition_keeps_natural_disjoint_terminal_union(monkeypatch):
    paths = ["/owned/prefix", "/owned/prefix/child", "/owned/prefix/childish"]
    paths.extend(f"/owned/{package}/module{number}.py" for package in ("alpha", "beta") for number in range(40))
    monkeypatch.setattr(path_language, "MAX_EXPRESSION_PATHS", 8)
    monkeypatch.setattr(path_language, "MAX_EXPRESSION_NODES", 16)
    monkeypatch.setattr(path_language, "MAX_EXPRESSION_BYTES", 1024)
    root = path_language.radix(sorted(paths))
    groups = path_language.partition(root, quote)
    recovered = set()
    for node, value in groups:
        complete = path_language.terminals(node)
        assert not recovered & complete and len(complete) <= 8
        assert len(value.encode()) <= 1024
        recovered.update(complete)
    values, receipt = path_language.exact_language(paths, quote)
    assert recovered == set(paths) and len(values) == len(groups) > 1
    assert receipt["input_sha256"] == receipt["terminals_sha256"]
    assert receipt["max_expression_bytes"] <= 1024


@needs_confinement
def test_exact_radix_partition_refuses_oversized_singleton(monkeypatch):
    monkeypatch.setattr(path_language, "MAX_EXPRESSION_BYTES", 1)
    with pytest.raises(path_language.LanguageError, match="singleton expression exceeds bound"):
        path_language.expressions(["/owned/one"], quote)


@needs_confinement
def test_explicit_literal_holder_inventory_runs_only_normal():
    test_verified_holder_inventory_runs_only_the_fixed_normal_interpreter()


@needs_confinement
def test_literal_experiment_preserves_exact_native_terminal_union(tmp_path):
    from test_kernel_compile import literal_language, replace_exact_file_language
    paths = ["/owned/a", "/owned/ab", "/owned/a/child", '/owned/quote"file',
             "/owned/back\\slash", "/owned/café", "/owned/meta$[]()+*?{}^.|", "/owned/漢字"]
    lines, receipt = literal_language(paths + paths[:1])
    assert len(lines) == 1 and all("(literal " + quote(path) + ")" in lines[0] for path in paths)
    assert receipt["paths"] == len(paths) and receipt["input_sha256"] == receipt["terminals_sha256"]
    path = tmp_path / "profile.sb"
    original = ["(deny default)", '(allow file-read* (literal "/system"))',
                '(allow file-read-metadata (literal "/owned"))']
    generated = ["(allow file-read* " + value + ")" for value in path_language.expressions(paths, quote)]
    path.write_text("\n".join([*original, *generated]) + "\n")
    assert replace_exact_file_language(path, paths) == receipt
    assert path.read_text().splitlines() == [*original, *lines]
    path.write_text("\n".join([*original, *generated, *generated]) + "\n")
    with pytest.raises(RuntimeError, match="generated exact file language changed"):
        replace_exact_file_language(path, paths)


@needs_confinement
def test_pinned_normal_package_identity_uses_only_verified_import_files():
    test_verified_holder_inventory_runs_only_the_fixed_normal_interpreter()
    manifest = json.loads(Path(os.environ["ML_STACK_TEST_HOLDER_MANIFEST"]).read_text())
    prefix = Path(manifest["variants"]["normal"]["runtime"]["prefix"])
    with pytest.raises(PermissionError):
        (prefix / "lib/python3.13/site-packages/ml_stack/serve/broker.py").read_bytes()


@needs_confinement
def test_static_identity_closure_retains_full_integrity_and_refuses_missing_assets(tmp_path):
    node = next(iter(holder.IDENTITY_PROBES))
    assert holder.identity_selected([node])
    assert not holder.identity_selected([next(iter(holder.RUNTIME_PROBES))])
    with pytest.raises(RuntimeError, match="cannot mix"):
        holder.identity_selected([node, next(iter(holder.RUNTIME_PROBES))])
    prefix = tmp_path / "normal"
    files = [prefix / relative for relative in holder.IDENTITY_FILES]
    link = prefix / "bin/python"
    directories = {prefix, link.parent}
    for path in files:
        directories.update(parent for parent in path.parents if parent.is_relative_to(prefix))
    extra = prefix / "unlisted"
    identities = dict.fromkeys([*files, extra], (1,))
    assets = holder.Assets([*files, extra], directories, {link, extra}, set(), identities.copy(), admitted_variants=("normal",))
    manifest = {"variants": {"normal": {"runtime": {"prefix": str(prefix)}}}}
    holder.project_identity(assets, manifest)
    assert assets.files == files and assets.metadata == {link}
    assert assets.identities == identities and assets.scope == "package-identity-import"
    assets.files.pop()
    with pytest.raises(RuntimeError, match="not in verified inventory"):
        holder.project_identity(assets, manifest)


@needs_confinement
def test_runtime_projection_keeps_all_verified_identities_without_unselected_grants(tmp_path):
    manifest = {"variants": {}}
    files, directories, metadata = [], set(), set()
    for name in ("normal", "slow", "stale"):
        prefix = tmp_path / name
        prefix.mkdir()
        python = prefix / "python"
        python.write_bytes(b"fixture")
        files.append(python)
        directories.add(prefix)
        metadata.add(python)
        manifest["variants"][name] = {"runtime": {"prefix": str(prefix), "python": str(python)}}
    identities = dict.fromkeys(files, (1,))
    assets = holder.Assets(files, directories, metadata, set(files), identities.copy())
    holder.project_variants(assets, manifest, {"normal"})
    assert assets.identities == identities and assets.verified_variants == ("normal", "slow", "stale")
    assert assets.admitted_variants == ("normal",)
    assert assets.files == [tmp_path / "normal" / "python"]
    assert assets.directories == {tmp_path / "normal"}
    assert assets.metadata == assets.executables == {tmp_path / "normal" / "python"}


@needs_confinement
def test_role_case_allocation_keeps_parameters_and_generations_separate():
    from test_kernel_role_home import CASE_FILE, selected_cases
    base = CASE_FILE + "test_actual_immutable_runtime_receipt_is_verified_outside_the_socket_deadline"
    assert selected_cases([base]) == {base + "[broker]": "normal", base + "[holder]": "normal"}
    assert selected_cases([base + "[holder]"]) == {base + "[holder]": "normal"}
    with pytest.raises(RuntimeError, match="unreviewed fixture parameter"):
        selected_cases([base + "[foreign]"])
    assert not selected_cases(["tests/test_holder_protocol.py::test_unreviewed_role"])


@needs_confinement
def test_role_copy_has_independent_inode_and_refuses_changed_content(tmp_path):
    from test_kernel_role_home import copy_file
    source = tmp_path / "source"
    source.write_bytes(b"immutable fixture")
    source.chmod(0o400)
    record = {"mode": 0o400, "size": source.stat().st_size,
              "sha256": hashlib.sha256(source.read_bytes()).hexdigest()}
    target = tmp_path / "copied"
    copy_file(source, target, record, time.monotonic() + 5)
    assert target.read_bytes() == source.read_bytes()
    assert source.stat().st_ino != target.stat().st_ino
    assert target.stat().st_nlink == 1 and stat.S_IMODE(target.stat().st_mode) == 0o400
    with pytest.raises(RuntimeError, match="content changed"):
        copy_file(source, tmp_path / "forged", {**record, "sha256": "0" * 64}, time.monotonic() + 5)
    source.chmod(0o600)
    with pytest.raises(RuntimeError, match="source file identity changed"):
        copy_file(source, tmp_path / "public", record, time.monotonic() + 5)


@needs_confinement
@pytest.mark.parametrize("change", ["identity", "writable", "hardlinked", "unpinned"])
def test_role_native_image_requires_its_exact_readonly_verified_pin(tmp_path, change):
    from test_kernel_assets import owned_image
    image = tmp_path / "fixture.so"
    image.write_bytes(b"\xcf\xfa\xed\xfe" + b"fixture")
    image.chmod(0o400)
    pinned = {image: holder.identity(image.lstat())}
    actual, _, _ = owned_image(image, (), pinned)
    assert actual == image
    if change == "identity":
        pinned[image] = (0, *pinned[image][1:])
    elif change == "writable":
        image.chmod(0o600)
    elif change == "hardlinked":
        (tmp_path / "alias").hardlink_to(image)
    else:
        pinned = {}
    with pytest.raises(RuntimeError, match=r"copied image changed|outside interpreter package roots"):
        owned_image(image, (), pinned)

