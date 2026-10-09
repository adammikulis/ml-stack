"""Reviewed offline holder assets and one invocation-owned Unix channel namespace.

The namespace is scoped authority, not a process or listener cardinality limit.
It is retained after the run because descendant exhaustion is not established.
"""
from __future__ import annotations

import hashlib
import json
import os
import selectors
import stat
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

from ml_stack.platform import start_process, terminate_process_group
from ml_stack.sandbox.policy import Net, Policy
from ml_stack.sandbox.seatbelt import Seatbelt

SOURCE = "03a4ff9101896ea5673726c664bbfd5a172fa906"
MANIFEST = "a479fb1c2e7939be92e98e3e95d94d7b7a2c7643adb2c9e1cfeece2c005e06b0"
MAX_BYTES = 4 * 1024 * 1024
MAX_FILES = 16384
MAX_CONTENT = 2 * 1024 * 1024 * 1024
HOLDER_NODES = frozenset({
    'tests/test_holder_protocol.py::test_actual_kernel_peer_is_the_connected_owned_child',
    'tests/test_holder_protocol.py::test_socket_is_private_and_removed_on_context_failure',
    'tests/test_holder_protocol.py::test_existing_channel_is_never_replaced',
    'tests/test_holder_protocol.py::test_unsupported_platform_refuses_before_socket_creation',
    'tests/test_holder_protocol.py::test_replaceable_storage_refuses_before_channel_creation',
    'tests/test_holder_protocol.py::test_inherited_listener_reports_creator_not_request_handler',
    'tests/test_holder_protocol.py::test_session_signature_proves_only_fresh_pinned_key_possession',
    'tests/test_holder_protocol.py::test_session_substitution_refuses_even_with_existing_broker_mac',
    'tests/test_holder_protocol.py::test_unknown_key_and_wrong_broker_peer_refuse',
    'tests/test_holder_protocol.py::test_current_lease_and_broker_generation_are_rechecked_before_signing',
    'tests/test_holder_protocol.py::test_transcript_parser_keeps_strict_security_bounds',
    'tests/test_holder_protocol.py::test_protected_gap_never_autostarts_another_broker',
    'tests/test_holder_protocol.py::test_existing_authenticated_broker_remains_reachable_during_protection',
    'tests/test_holder_protocol.py::test_exec_child_inheriting_listener_cannot_supply_the_pinned_session_signature',
    'tests/test_holder_protocol.py::test_actual_owned_socket_frames_are_bounded',
    'tests/test_holder_protocol.py::test_runtime_role_and_fixed_argv_cannot_be_supplied_by_a_receipt',
    'tests/test_holder_protocol.py::test_actual_immutable_runtime_receipt_is_verified_outside_the_socket_deadline',
    'tests/test_holder_protocol.py::test_fork_inheriting_key_cannot_sign_as_original_holder',
    'tests/test_holder_protocol.py::test_session_representation_never_exposes_existing_token_or_private_key',
    'tests/test_holder_protocol.py::test_optional_observation_failure_keeps_original_lease_until_owned_stop',
    'tests/test_holder_protocol.py::test_poll_observation_error_cancels_and_reaps_owned_preparation',
    'tests/test_holder_protocol.py::test_malformed_preparation_runtime_is_unavailable_not_a_holder_failure',
    'tests/test_holder_protocol.py::test_actual_owned_preparation_and_mutual_session_exchange_keep_original_lease',
    'tests/test_holder_protocol.py::test_slow_independent_verification_is_cancelled_without_waiting_or_orphaning',
    'tests/test_holder_protocol.py::test_completed_preparation_cannot_extend_a_changed_broker_generation',
})

RUNTIME_PROBES = frozenset({"tests/test_test_kernel_isolation.py::test_verified_holder_inventory_runs_only_the_fixed_normal_interpreter",
                          "tests/test_test_kernel_isolation.py::test_native_read_components_compile_independently",
                          "tests/test_test_kernel_isolation.py::test_explicit_literal_holder_inventory_runs_only_normal",
                          "tests/test_kernel_role_home.py::test_role_home_atomic_state_and_immutable_fences",
                          "tests/test_kernel_role_home.py::test_fixed_normal_role_imports_and_runtime_verification"})
IDENTITY_PROBES = frozenset({"tests/test_test_kernel_isolation.py::test_pinned_normal_package_identity_uses_only_verified_import_files"})
IDENTITY_FILES = ("pyvenv.cfg", "lib/python3.13/site-packages/ml_stack/__init__.py",
                  "lib/python3.13/site-packages/ml_stack/__pycache__/__init__.cpython-313.pyc",
                  "lib/python3.13/site-packages/ml_stack-0.1.0.dist-info/METADATA")

VARIANTS_BY_NODE = {
    'tests/test_holder_protocol.py::test_actual_immutable_runtime_receipt_is_verified_outside_the_socket_deadline': frozenset({'normal'}),
    'tests/test_holder_protocol.py::test_actual_kernel_peer_is_the_connected_owned_child': frozenset(),
    'tests/test_holder_protocol.py::test_actual_owned_preparation_and_mutual_session_exchange_keep_original_lease': frozenset({'normal'}),
    'tests/test_holder_protocol.py::test_actual_owned_socket_frames_are_bounded': frozenset(),
    'tests/test_holder_protocol.py::test_completed_preparation_cannot_extend_a_changed_broker_generation': frozenset({'stale'}),
    'tests/test_holder_protocol.py::test_current_lease_and_broker_generation_are_rechecked_before_signing': frozenset(),
    'tests/test_holder_protocol.py::test_exec_child_inheriting_listener_cannot_supply_the_pinned_session_signature': frozenset(),
    'tests/test_holder_protocol.py::test_existing_authenticated_broker_remains_reachable_during_protection': frozenset(),
    'tests/test_holder_protocol.py::test_existing_channel_is_never_replaced': frozenset(),
    'tests/test_holder_protocol.py::test_fork_inheriting_key_cannot_sign_as_original_holder': frozenset(),
    'tests/test_holder_protocol.py::test_inherited_listener_reports_creator_not_request_handler': frozenset(),
    'tests/test_holder_protocol.py::test_malformed_preparation_runtime_is_unavailable_not_a_holder_failure': frozenset(),
    'tests/test_holder_protocol.py::test_optional_observation_failure_keeps_original_lease_until_owned_stop': frozenset(),
    'tests/test_holder_protocol.py::test_poll_observation_error_cancels_and_reaps_owned_preparation': frozenset(),
    'tests/test_holder_protocol.py::test_protected_gap_never_autostarts_another_broker': frozenset(),
    'tests/test_holder_protocol.py::test_replaceable_storage_refuses_before_channel_creation': frozenset(),
    'tests/test_holder_protocol.py::test_runtime_role_and_fixed_argv_cannot_be_supplied_by_a_receipt': frozenset(),
    'tests/test_holder_protocol.py::test_session_representation_never_exposes_existing_token_or_private_key': frozenset(),
    'tests/test_holder_protocol.py::test_session_signature_proves_only_fresh_pinned_key_possession': frozenset(),
    'tests/test_holder_protocol.py::test_session_substitution_refuses_even_with_existing_broker_mac': frozenset(),
    'tests/test_holder_protocol.py::test_slow_independent_verification_is_cancelled_without_waiting_or_orphaning': frozenset({'slow'}),
    'tests/test_holder_protocol.py::test_socket_is_private_and_removed_on_context_failure': frozenset(),
    'tests/test_holder_protocol.py::test_transcript_parser_keeps_strict_security_bounds': frozenset(),
    'tests/test_holder_protocol.py::test_unknown_key_and_wrong_broker_peer_refuse': frozenset(),
    'tests/test_holder_protocol.py::test_unsupported_platform_refuses_before_socket_creation': frozenset(),
    'tests/test_test_kernel_isolation.py::test_verified_holder_inventory_runs_only_the_fixed_normal_interpreter': frozenset({'normal'}),
    'tests/test_test_kernel_isolation.py::test_native_read_components_compile_independently': frozenset({'normal'}),
    'tests/test_test_kernel_isolation.py::test_explicit_literal_holder_inventory_runs_only_normal': frozenset({'normal'}),
    'tests/test_test_kernel_isolation.py::test_pinned_normal_package_identity_uses_only_verified_import_files': frozenset({'normal'}),
    'tests/test_kernel_role_home.py::test_role_home_atomic_state_and_immutable_fences': frozenset({'normal'}),
    'tests/test_kernel_role_home.py::test_fixed_normal_role_imports_and_runtime_verification': frozenset({'normal'}),
}


def selected(command: list[str]) -> bool:
    probes = {"tests/test_test_kernel_isolation.py::test_holder_namespace_allows_only_its_owned_unix_channels",
              "tests/test_test_kernel_isolation.py::test_holder_namespace_refuses_redirect_owner_and_path_bounds"}
    return any(part.split("[", 1)[0] in HOLDER_NODES | RUNTIME_PROBES | IDENTITY_PROBES | probes for part in command)


def required_variants(command: list[str]) -> set[str]:
    if set(VARIANTS_BY_NODE) != HOLDER_NODES | RUNTIME_PROBES | IDENTITY_PROBES:
        raise RuntimeError("holder confinement: runtime resource mapping is incomplete")
    selected = {part.split("[", 1)[0] for part in command}
    return set().union(*(VARIANTS_BY_NODE[node] for node in selected & VARIANTS_BY_NODE.keys()))


def project_variants(assets, manifest: dict, variants: set[str]) -> None:
    if not variants or not variants <= manifest["variants"].keys():
        raise RuntimeError("holder confinement: unreviewed variant set")
    prefixes = [Path(manifest["variants"][name]["runtime"]["prefix"]) for name in sorted(variants)]
    assets.files = [path for path in assets.files if any(path.is_relative_to(root) for root in prefixes)]
    assets.directories = {path for path in assets.directories if any(path.is_relative_to(root) for root in prefixes)}
    assets.metadata = {path for path in assets.metadata if any(path.is_relative_to(root) for root in prefixes)}
    assets.executables = {Path(manifest["variants"][name]["runtime"]["python"]).resolve(strict=True) for name in variants}
    assets.verified_variants = tuple(sorted(manifest["variants"]))
    assets.admitted_variants = tuple(sorted(variants))


def project_identity(assets, manifest: dict) -> None:
    if assets.admitted_variants != ("normal",):
        raise RuntimeError("holder identity: exact normal variant required")
    prefix = Path(manifest["variants"]["normal"]["runtime"]["prefix"])
    files = [prefix / relative for relative in IDENTITY_FILES]
    link = prefix / "bin/python"
    directories = {prefix, link.parent}
    for path in files:
        directories.update(parent for parent in path.parents if parent.is_relative_to(prefix))
    if not set(files) <= set(assets.files) or not directories <= assets.directories or link not in assets.metadata:
        raise RuntimeError("holder identity: static import closure is not in verified inventory")
    assets.files, assets.directories, assets.metadata = files, directories, {link}
    assets.scope = "package-identity-import"


def identity_selected(command: list[str]) -> bool:
    resources = {part.split("[", 1)[0] for part in command} & VARIANTS_BY_NODE.keys()
    consuming = {node for node in resources if VARIANTS_BY_NODE[node]}
    if consuming & IDENTITY_PROBES and not consuming <= IDENTITY_PROBES:
        raise RuntimeError("holder identity: cannot mix identity and complete runtime resource probes")
    return bool(consuming) and consuming <= IDENTITY_PROBES


def private_directory(path: Path) -> tuple[int, int]:
    if path.resolve(strict=True) != path or len(os.fsencode(path)) > 40:
        raise RuntimeError("holder confinement: namespace is redirected or too long")
    info = path.lstat()
    if (not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid()
            or stat.S_IMODE(info.st_mode) != 0o700):
        raise RuntimeError("holder confinement: namespace must be private and owned")
    return info.st_dev, info.st_ino


def namespace(protected: tuple[Path, ...]) -> Path:
    from ml_stack.activity.source_snapshot import validate_storage
    # Short Darwin sockaddr_un names cannot fit the OS account temporary path.
    base = validate_storage(Path("/private/tmp"), protected)
    result = Path(tempfile.mkdtemp(prefix="mlh-", dir=base))
    private_directory(result)
    return result


def identity(info: os.stat_result) -> tuple[int, ...]:
    return (info.st_dev, info.st_ino, info.st_uid, info.st_mode, info.st_nlink,
            info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def ancestry(path: Path) -> dict[Path, tuple[int, ...]]:
    if path.resolve(strict=True) != path:
        raise RuntimeError("holder confinement: asset ancestry is redirected")
    records = {}
    for parent in (path, *path.parents):
        info = parent.lstat()
        sticky = parent == Path("/private/tmp") and info.st_uid == 0 and info.st_mode & stat.S_ISVTX
        if (not stat.S_ISDIR(info.st_mode) or info.st_uid not in (0, os.getuid())
                or (info.st_mode & 0o022 and not sticky)):
            raise RuntimeError("holder confinement: asset ancestry is foreign or writable")
        records[parent] = (info.st_dev, info.st_ino, info.st_uid, info.st_gid, info.st_mode)
    return records


def read_manifest(path: Path) -> dict:
    if path.resolve(strict=True) != path:
        raise RuntimeError("holder confinement: manifest path is redirected")
    for parent in path.parents:
        info = parent.stat()
        sticky = info.st_uid == 0 and info.st_mode & stat.S_ISVTX
        if info.st_uid not in (0, os.getuid()) or (info.st_mode & 0o022 and not sticky):
            raise RuntimeError("holder confinement: manifest ancestry is replaceable")
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        before = os.fstat(fd)
        if (not stat.S_ISREG(before.st_mode) or before.st_uid != os.getuid()
                or before.st_nlink != 1 or stat.S_IMODE(before.st_mode) != 0o400
                or before.st_size > MAX_BYTES):
            raise RuntimeError("holder confinement: manifest is not a sealed owned file")
        with os.fdopen(os.dup(fd), "rb") as stream:
            raw = stream.read(MAX_BYTES + 1)
        if (len(raw) > MAX_BYTES or identity(before) != identity(os.fstat(fd))
                or identity(before) != identity(path.lstat())
                or hashlib.sha256(raw).hexdigest() != MANIFEST):
            raise RuntimeError("holder confinement: unreviewed or changed manifest")
    finally:
        os.close(fd)
    value = json.loads(raw)
    if (type(value) is not dict or set(value) != {"version", "source_commit", "variants"}
            or type(value["version"]) is not int or value["version"] != 1
            or value["source_commit"] != SOURCE
            or type(value["variants"]) is not dict
            or set(value["variants"]) != {"normal", "slow", "stale"}):
        raise RuntimeError("holder confinement: manifest source or schema changed")
    return value


@dataclass
class Assets:
    files: list[Path]
    directories: set[Path]
    metadata: set[Path]
    executables: set[Path]
    identities: dict[Path, tuple[int, ...]]
    ancestors: dict[Path, tuple[int, ...]] = field(default_factory=dict)
    compilation: dict | None = None
    verified_variants: tuple[str, ...] = ()
    admitted_variants: tuple[str, ...] = ()
    scope: str = "complete-prefix"
    role_cases: dict = field(default_factory=dict)
    role_rules: list[str] = field(default_factory=list)
    role_channels: tuple[str, ...] = ()
    role_bank_sha256: str | None = None
    role_namespace: str | None = None

    def proof(self) -> dict:
        return {"manifest": MANIFEST, "source_commit": SOURCE,
                "nodes": len(self.identities), "verified_variants": self.verified_variants,
                "admitted_variants": self.admitted_variants, "admitted_files": len(self.files), "executables": [str(path) for path in sorted(self.executables)],
                "scope": self.scope, "admitted_files_sha256": hashlib.sha256(json.dumps(sorted(map(str, self.files))).encode()).hexdigest(),
                "minimal_compilation": self.compilation,
                "role_cases": {case: {"home": entry["home"], "variant": entry["variant"]}
                               for case, entry in self.role_cases.items()},
                "role_bank_sha256": self.role_bank_sha256, "role_namespace": self.role_namespace}

    def recheck(self) -> None:
        for path, expected in self.identities.items():
            if identity(path.lstat()) != expected:
                raise RuntimeError("holder confinement: prepared asset changed")
        for path, expected in self.ancestors.items():
            info = path.lstat()
            if (info.st_dev, info.st_ino, info.st_uid, info.st_gid, info.st_mode) != expected:
                raise RuntimeError("holder confinement: prepared ancestry changed")


def prepare(command: list[str], control: Path, environment: dict[str, str],
            protected: tuple[Path, ...], admitted: tuple[Path, ...]) -> tuple[Assets | None, Path | None, dict | None]:
    if not selected(command):
        return None, None, None
    wanted = required_variants(command)
    identity_mode = identity_selected(command)
    if not wanted:
        channel_root = namespace(protected)
        compilation = compile_namespace(channel_root, control)
        environment.update(TMPDIR=str(channel_root), DEV_TEST_HOLDER_NAMESPACE=str(channel_root))
        return None, channel_root, compilation
    source = Path("/private/tmp/ml-stack-holder-fixture-prefixes-68a5f5/holder-manifest.json")
    manifest = read_manifest(source)
    assets = verify_assets(manifest, protected, admitted)
    from test_kernel_role_home import selected_cases
    if selected_cases(command):
        return prepare_roles(manifest, assets, command, (control, protected, admitted), environment), None, None
    project_variants(assets, manifest, wanted)
    if identity_mode:
        project_identity(assets, manifest)
    copied = control / "holder-manifest.json"
    fd = os.open(copied, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o400)
    try:
        with os.fdopen(os.dup(fd), "w") as stream:
            json.dump(manifest, stream, sort_keys=True)
        os.fsync(fd)
    finally:
        os.close(fd)
    assets.files.append(copied)
    assets.identities[copied] = identity(copied.lstat())
    channel_root = namespace(protected)
    compilation = compile_namespace(channel_root, control)
    environment.update(TMPDIR=str(channel_root), ML_STACK_TEST_HOLDER_MANIFEST=str(copied),
                       DEV_TEST_HOLDER_NAMESPACE=str(channel_root))
    assets.compilation = compilation
    return assets, channel_root, compilation


def prepare_roles(manifest: dict, originals: Assets, command: list[str], context: tuple,
                  environment: dict[str, str]) -> Assets:
    from test_kernel_role_home import FENCE_NODE, copy_cases, state_rules
    control, protected, admitted = context
    bank_root, bank = copy_cases(manifest, command, protected, time.monotonic() + 30)
    copied = verify_assets({"variants": {case: entry["prepared"] for case, entry in bank["cases"].items()}},
                           protected, admitted)
    originals.recheck()
    inventory = json.loads(Path(__file__).with_name("test_kernel_holder_imports.json").read_text())
    if (set(inventory) != {"version", "source_commit", "roots", "files"}
            or type(inventory["version"]) is not int or inventory["version"] != 1 or inventory["source_commit"] != SOURCE
            or type(inventory["files"]) is not list or len(inventory["files"]) != 475
            or any(type(path) is not str or not path or len(path.encode()) > 4096
                   or Path(path).is_absolute() or ".." in Path(path).parts or str(Path(path)) != path
                   for path in inventory["files"])
            or len(set(inventory["files"])) != 475):
        raise RuntimeError("holder role: static candidate inventory changed")
    files, directories, metadata = [], set(), set()
    fences = FENCE_NODE in command
    if fences and set(bank["cases"]) != {FENCE_NODE}:
        raise RuntimeError("holder role: fence probe cannot mix with import consumers")
    for entry in bank["cases"].values():
        prefix = Path(entry["prepared"]["runtime"]["prefix"])
        paths = [prefix / relative for relative in (["pyvenv.cfg"] if fences else inventory["files"])]
        if not set(paths) <= set(copied.files):
            raise RuntimeError("holder role: candidate read is outside verified copied files")
        files.extend(paths)
        directories.add(prefix / "bin")
        metadata.add(prefix / "bin/python")
        for path in paths:
            directories.update(parent for parent in path.parents if parent.is_relative_to(prefix))
        home = Path(entry["home"])
        metadata.update(parent for parent in prefix.parents if parent.is_relative_to(home))
        copied.role_rules.extend(state_rules(home, slow=entry["variant"] == "slow"))
    if not directories <= copied.directories or not metadata <= copied.metadata | copied.ancestors.keys():
        raise RuntimeError("holder role: candidate directory is outside verified copy")
    copied.files, copied.directories, copied.metadata = files, directories, metadata
    copied.identities.update(originals.identities)
    copied.ancestors.update(originals.ancestors)
    copied.verified_variants = tuple(sorted(manifest["variants"]))
    copied.admitted_variants = tuple(sorted({entry["variant"] for entry in bank["cases"].values()}))
    copied.scope, copied.role_cases = "immutable-role-fences" if fences else "candidate-holder-role-imports", bank["cases"]
    copied.role_namespace = str(bank_root)
    copied.role_channels = tuple(str(Path(entry["home"]) / "holder-channels") for entry in bank["cases"].values())
    for channel in copied.role_channels:
        copied.ancestors.update(ancestry(Path(channel)))
    path = control / "holder-role-bank.json"
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o400)
    try:
        raw = json.dumps(bank, sort_keys=True).encode()
        copied.role_bank_sha256 = hashlib.sha256(raw).hexdigest()
        with os.fdopen(os.dup(fd), "wb") as stream:
            stream.write(raw)
        os.fsync(fd)
    finally:
        os.close(fd)
    copied.files.append(path)
    copied.identities[path] = identity(path.lstat())
    environment["ML_STACK_TEST_HOLDER_ROLE_BANK"] = str(path)
    copied.recheck()
    return copied


def compile_namespace(channel_root: Path, control: Path) -> dict:
    """Diagnostic compilation only; this establishes no channel behavior."""
    deadline = time.monotonic() + 5
    policy = Policy(name="holder-unix-compile", strict=True, net=Net.deny(),
                    unix_namespaces=(str(channel_root),), env={"PATH": os.defpath, "LC_ALL": "C"}).validated()
    wrapped = Seatbelt().wrap(["/usr/bin/true"], policy, profile_path=control / "minimal-unix.sb")
    process = start_process(wrapped.argv, env=dict(policy.env), close_fds=True,
                            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output = bytearray()
    try:
        with selectors.DefaultSelector() as ready:
            ready.register(process.stdout, selectors.EVENT_READ)
            ready.register(process.stderr, selectors.EVENT_READ)
            while ready.get_map():
                remaining = deadline - time.monotonic() - 1
                if remaining <= 0 or not (events := ready.select(remaining)):
                    raise TimeoutError("holder confinement: minimal compilation deadline exceeded")
                for event, _ in events:
                    block = os.read(event.fileobj.fileno(), 4096)
                    if not block:
                        ready.unregister(event.fileobj)
                    output.extend(block)
                    if len(output) > 8192:
                        raise RuntimeError("holder confinement: minimal compiler output exceeds bound")
        status = process.wait(timeout=max(.001, deadline - time.monotonic() - 1))
        if status:
            raise RuntimeError(f"holder confinement: minimal Unix profile exit {status}: "
                               + output.decode("utf-8", errors="replace"))
        return {"exit": status, "profile": hashlib.sha256((control / "minimal-unix.sb").read_bytes()).hexdigest()}
    finally:
        process.stdout.close()
        process.stderr.close()
        if process.poll() is None:
            try:
                terminate_process_group(process, force=True)
            except ProcessLookupError:
                pass
            finally:
                process.wait(timeout=max(.001, deadline - time.monotonic()))


def verify_assets(manifest: dict, protected: tuple[Path, ...], admitted: tuple[Path, ...],
                  seconds: float = 30) -> Assets:
    deadline = time.monotonic() + seconds
    assets = Assets([], set(), set(), set(), {})
    total = 0
    for variant in manifest["variants"].values():
        if type(variant) is not dict or set(variant) != {"runtime", "files"}:
            raise RuntimeError("holder confinement: invalid variant")
        runtime, records = variant["runtime"], variant["files"]
        if (type(runtime) is not dict or set(runtime) != {"prefix", "python", "commit", "version", "identity"}
                or runtime["commit"] != "a" * 40 or runtime["version"] != "0.1.0"
                or runtime["identity"] != "darwin-arm64-cpython-313"
                or type(records) is not list or not 1 <= len(records) <= MAX_FILES):
            raise RuntimeError("holder confinement: runtime fixture stamp changed")
        prefix = Path(runtime["prefix"])
        if (not prefix.is_absolute() or prefix.resolve(strict=True) != prefix
                or any(prefix.is_relative_to(root) or root.is_relative_to(prefix) for root in protected)
                or runtime["python"] != str(prefix / "bin/python")):
            raise RuntimeError("holder confinement: invalid runtime prefix")
        assets.ancestors.update(ancestry(prefix))
        names = set()
        for record in records:
            if time.monotonic() > deadline:
                raise TimeoutError("holder confinement: asset deadline exceeded")
            relative = record.get("path")
            if (type(relative) is not str or not relative or Path(relative).is_absolute()
                    or ".." in Path(relative).parts or relative in names):
                raise RuntimeError("holder confinement: invalid inventory path")
            names.add(relative)
            path = prefix if relative == "." else prefix / relative
            info = path.lstat()
            assets.identities[path] = identity(info)
            if (info.st_uid != os.getuid() or record["uid"] != info.st_uid
                    or record["mode"] != stat.S_IMODE(info.st_mode) or record["size"] != info.st_size):
                raise RuntimeError("holder confinement: inventory identity changed")
            if stat.S_ISLNK(info.st_mode):
                target = path.resolve(strict=True)
                if (record["type"] != "symlink" or record["target"] != str(path.readlink())
                        or not (target.is_relative_to(prefix) or any(target.is_relative_to(root) for root in admitted))):
                    raise RuntimeError("holder confinement: unadmitted runtime symlink")
                assets.metadata.add(path)
            elif info.st_mode & 0o222:
                raise RuntimeError("holder confinement: runtime asset is writable")
            elif stat.S_ISDIR(info.st_mode) and record["type"] == "directory":
                assets.directories.add(path)
            elif stat.S_ISREG(info.st_mode) and record["type"] == "file" and info.st_nlink == 1:
                total += info.st_size
                if total > MAX_CONTENT:
                    raise RuntimeError("holder confinement: asset content bound exceeded")
                verify_file(path, info, record["sha256"], deadline)
                assets.files.append(path)
            else:
                raise RuntimeError("holder confinement: unsupported inventory node")
        actual = inventory(prefix, deadline)
        if actual != names:
            raise RuntimeError("holder confinement: incomplete runtime inventory")
        assets.executables.add(Path(runtime["python"]).resolve(strict=True))
    assets.recheck()
    return assets


def verify_file(path: Path, info: os.stat_result, expected: str, deadline: float) -> None:
    digest = hashlib.sha256()
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        if identity(os.fstat(fd)) != identity(info):
            raise RuntimeError("holder confinement: asset changed before read")
        with os.fdopen(os.dup(fd), "rb") as stream:
            while block := stream.read(65536):
                digest.update(block)
                if time.monotonic() > deadline:
                    raise TimeoutError("holder confinement: asset deadline exceeded")
        if identity(os.fstat(fd)) != identity(info) or digest.hexdigest() != expected:
            raise RuntimeError("holder confinement: asset digest changed")
    finally:
        os.close(fd)


def inventory(prefix: Path, deadline: float) -> set[str]:
    actual = {"."}
    for parent, directories, files in os.walk(prefix, followlinks=False):
        if time.monotonic() > deadline:
            raise TimeoutError("holder confinement: inventory deadline exceeded")
        actual.update(str((Path(parent) / name).relative_to(prefix)) for name in (*directories, *files))
        if len(actual) > MAX_FILES:
            raise RuntimeError("holder confinement: runtime inventory bound exceeded")
    return actual
