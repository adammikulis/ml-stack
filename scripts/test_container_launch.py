"""Supervisor-created isolated Linux pytest containers."""
from __future__ import annotations

import hashlib
import json
import os
import pwd
import re
import secrets
import shutil
import stat
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from test_kernel_isolation import check_selectors

from ml_stack.activity.source_snapshot import SourceSnapshot, private_namespace

ROOT = Path(__file__).resolve().parent.parent
GUEST_TMP = Path("/tmp")  # noqa: S108 - supervisor-created private container tmpfs
TMP_OPTIONS = "rw,nosuid,nodev,noexec,size=1073741824"


@dataclass(frozen=True)
class ContainerProof:
    identifier: str
    owner: str
    image: str
    source: Path
    volume: str
    runtime: bool


class ContainerRun:
    def __init__(self, command, environment):
        check_selectors(command)
        self.arguments = command[3:]
        account = Path(pwd.getpwuid(os.getuid()).pw_dir).resolve()
        self.environment = {"PATH": os.defpath, "HOME": str(account), "LC_ALL": "C"}
        self.image = environment.get("ML_STACK_LINUX_IMAGE", "python:3.13")
        self.volume = environment.get("ML_STACK_LINUX_VOLUME", "ml-stack-linux-suite")
        self.platform = environment.get("ML_STACK_LINUX_PLATFORM", "")
        self.single = environment.get("ML_STACK_LINUX_SINGLE", "")
        self.rebuild = environment.get("ML_STACK_LINUX_REBUILD", "")
        if (not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:/@-]{0,255}", self.image)
                or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", self.volume)
                or self.platform not in ("", "linux/amd64", "linux/arm64")
                or self.single not in ("", "1") or self.rebuild not in ("", "1")):
            raise RuntimeError("container admission: invalid maintained container configuration")
        binary = shutil.which("docker")
        if binary is None:
            raise RuntimeError("container admission: Docker is required")
        self.binary = str(Path(binary).resolve(strict=True))
        info = Path(self.binary).stat()
        if info.st_uid not in (0, os.getuid()) or info.st_mode & 0o022:
            raise RuntimeError("container admission: writable Docker executable")
        socket = account / ".docker/run/docker.sock" if sys.platform == "darwin" else Path("/var/run/docker.sock")
        info = socket.stat()
        if not stat.S_ISSOCK(info.st_mode) or info.st_uid not in (0, os.getuid()):
            raise RuntimeError("container admission: an owned local Docker daemon is required")
        self.socket = socket
        self.socket_proof = asset_identity(socket)
        self.binary_proof = asset_identity(Path(self.binary), content=True)
        self.daemon = "unix://" + str(socket)
        self.control = private_namespace(environment, "ml-stack-container-")
        self.config = self.control / "docker"
        self.config.mkdir(mode=0o700)
        (self.config / "config.json").write_text('{"auths":{}}')
        self.owner = secrets.token_hex(24)
        self.setup = self.runtime = self.runtime_image = None
        self.bridge = self.bridge_process = None

    def argv(self, *arguments):
        return [self.binary, "--host", self.daemon, "--config", str(self.config), *arguments]

    def recheck_assets(self):
        if (asset_identity(self.socket) != self.socket_proof
                or asset_identity(Path(self.binary), content=True) != self.binary_proof):
            raise RuntimeError("container admission: local daemon or Docker executable changed")

    def call(self, *arguments, capture=True, timeout=30):
        self.recheck_assets()
        result = subprocess.run(self.argv(*arguments), env=self.environment, check=True, text=True,
                                capture_output=capture, timeout=timeout)
        self.recheck_assets()
        return result

    def source(self):
        checkout = self.control / "source"
        checkout.mkdir(mode=0o700)
        with_snapshot = SourceSnapshot(ROOT, self.control)
        try:
            for name in with_snapshot.tracked:
                path = with_snapshot.path(name)
                info = path.lstat()
                if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o022 or info.st_nlink != 1:
                    raise RuntimeError("container admission: tracked source must be owned plain files")
                content = pinned_source(path, info, with_snapshot)
                target = checkout / os.fsdecode(name)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(content)
                target.chmod(0o555 if info.st_mode & 0o111 else 0o444)
            self.source_manifest = tuple(os.fsdecode(name) for name in with_snapshot.tracked)
            self.source_digest = source_digest(checkout, self.source_manifest)
        finally:
            with_snapshot.close()
        return checkout

    def prepare(self):
        source = self.source()
        self.call("pull", *(["--platform", self.platform] if self.platform else []), self.image, capture=False, timeout=600)
        rows = json.loads(self.call("image", "inspect", self.image).stdout)
        self.base_image = rows[0]["Id"]
        base = rows[0].get("Config", {})
        validate_base_image(base)
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", self.base_image):
            raise RuntimeError("container admission: immutable image identity unavailable")
        result = self.call("run", "-d", "--init", "--no-healthcheck", "--label", "ml-stack.test-run=" + self.owner,
                           "-v", str(source) + ":/src:ro", "-v", self.volume + ":/venv",
                           "-e", "PIP_DISABLE_PIP_VERSION_CHECK=1", "-e", "PIP_CACHE_DIR=/venv/pipcache",
                           "-e", "IMAGE=" + self.base_image, "-e", "REBUILD=" + self.rebuild,
                           "-e", "HOST_UID=1000", "-e", "HOST_GID=1000", "-w", "/work", self.base_image,
                           "sleep", "infinity")
        self.setup = container_id(result.stdout)
        self.validate(self.setup, runtime=False)
        self.call("exec", self.setup, "bash", "/src/scripts/test-on-linux-setup", *self.arguments,
                  capture=False, timeout=3600)
        self.call("exec", self.setup, "/usr/local/bin/python", "/src/scripts/test_container_seal.py",
                  self.source_digest, capture=False, timeout=600)
        if source_digest(source, self.source_manifest) != self.source_digest:
            raise RuntimeError("container admission: staged source proof changed")
        self.runtime_image = self.call("commit", self.setup).stdout.strip()
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", self.runtime_image):
            raise RuntimeError("container admission: invalid committed runtime image")
        self.call("rm", "-f", self.setup)
        self.setup = None
        result = self.call("run", "-d", "--init", "--user", "1000:1000", "--no-healthcheck", "--read-only", "--network", "none", "--cap-drop", "ALL",
                           "--security-opt", "no-new-privileges", "--label", "ml-stack.test-run=" + self.owner,
                           "--tmpfs", str(GUEST_TMP) + ":" + TMP_OPTIONS, "-w", "/work",
                           self.runtime_image, "sleep", "infinity")
        self.runtime = container_id(result.stdout)
        self.validate(self.runtime, runtime=True)

    def validate(self, identifier, *, runtime, running=True):
        rows = json.loads(self.call("inspect", identifier).stdout)
        if len(rows) != 1:
            raise RuntimeError("container admission: ambiguous container identity")
        row = rows[0]
        expected = self.runtime_image if runtime else self.base_image
        validate_container(row, ContainerProof(identifier, self.owner, expected, self.control / "source", self.volume, runtime), running=running)

    def command(self, environment):
        from test_container_bridge import HostBridge
        self.validate(self.runtime, runtime=True)
        self.recheck_assets()
        self.bridge_process = subprocess.Popen(self.argv("exec", "-i", "--user", "1000:1000", self.runtime,
                                           "/priv-venv/bin/python", "/work/scripts/test_container_bridge.py",
                                           "--guest", "--directory", str(GUEST_TMP / "admission")),
                                           env=self.environment, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                           stderr=subprocess.PIPE, close_fds=True)
        self.bridge = HostBridge(self.bridge_process, environment["DEV_TEST_PYTEST_ENDPOINT"][5:],
                                 json.loads(environment["DEV_TEST_PYTEST_IDENTITY"]),
                                 expected_guest=(str(GUEST_TMP / "admission/admission.sock"), 1000))
        endpoint, identity = self.bridge.start()
        arguments = [*self.arguments, "-n", "0" if self.single else environment["DEV_TEST_WORKERS"]]
        values = {key: environment[key] for key in ("DEV_TEST_PYTEST_TOKEN", "DEV_TEST_WORKERS", "DEV_TEST_BUDGET", "DEV_TEST_WAIT_S",
                                              "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                                              "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS") if key in environment}
        values.update(DEV_TEST_PYTEST_ENDPOINT="unix:" + endpoint, DEV_TEST_PYTEST_IDENTITY=json.dumps(identity),
                      DEV_TEST_SLOTS_DIR=str(GUEST_TMP / "slots"), DEV_TEST_REMOTE_BROKER=str(GUEST_TMP / "slots"),
                      PYTHONPATH="/work/src:/work/scripts", PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1")
        options = [part for key, value in values.items() for part in ("-e", key + "=" + value)]
        return self.argv("exec", "--user", "1000:1000", *options, self.runtime, "bash", "-c",
                         'source /test-env; exec /priv-venv/bin/python -m pytest -p testslots_pytest "$@"',
                         "bash", *arguments), self.environment

    def close(self):
        errors = []
        if self.bridge is not None:
            try:
                self.bridge.close()
            except (OSError, RuntimeError, ValueError, subprocess.SubprocessError, ExceptionGroup) as exc:
                errors.append(exc)
        if self.bridge_process is not None and self.bridge_process.poll() is None:
            try:
                self.bridge_process.terminate()
                try:
                    self.bridge_process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self.bridge_process.kill()
                    self.bridge_process.wait(timeout=5)
            except (OSError, RuntimeError, ValueError, subprocess.SubprocessError, ExceptionGroup) as exc:
                errors.append(exc)
        for identifier, runtime in ((self.runtime, True), (self.setup, False)):
            if identifier is not None:
                try:
                    self.validate(identifier, runtime=runtime, running=False)
                    self.call("rm", "-f", identifier)
                except (OSError, RuntimeError, ValueError, subprocess.SubprocessError, ExceptionGroup) as exc:
                    errors.append(exc)
        if self.runtime_image is not None:
            try:
                self.call("image", "rm", self.runtime_image)
            except (OSError, RuntimeError, ValueError, subprocess.SubprocessError, ExceptionGroup) as exc:
                errors.append(exc)
        if errors:
            raise ExceptionGroup("container admission: cleanup failed; ownership proof retained at " + str(self.control), errors)
        shutil.rmtree(self.control)


def source_digest(root, manifest):
    digest = hashlib.sha256()
    actual = set()
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise RuntimeError("container admission: redirected staged source")
        if path.is_file():
            relative = str(path.relative_to(root))
            actual.add(relative)
            if relative not in manifest:
                raise RuntimeError("container admission: unlisted staged source")
            digest.update(relative.encode() + b"\0" + path.read_bytes())
    if actual != set(manifest):
        raise RuntimeError("container admission: incomplete tracked source inventory")
    return digest.hexdigest()


def validate_base_image(config):
    forbidden = {"LD_PRELOAD", "LD_LIBRARY_PATH", "PYTHONHOME", "PYTHONPATH", "PYTHONSTARTUP", "BASH_ENV", "ENV"}
    if not isinstance(config, dict) or any(value.split("=", 1)[0] in forbidden for value in config.get("Env", [])):
        raise RuntimeError("container admission: image has unadmitted startup behavior")
    if (not isinstance(config, dict) or config.get("Entrypoint") or config.get("User") or config.get("Volumes")
            or config.get("Healthcheck", {}).get("Test", ["NONE"]) != ["NONE"]):
        raise RuntimeError("container admission: image has unadmitted startup behavior")


def validate_container(row, proof, *, running=True):
    identifier, owner, image = proof.identifier, proof.owner, proof.image
    source, volume, runtime = proof.source, proof.volume, proof.runtime
    host, config = row.get("HostConfig", {}), row.get("Config", {})
    if (row.get("Id") != identifier or (running and not row.get("State", {}).get("Running"))
            or row.get("Image") != image or config.get("Labels", {}).get("ml-stack.test-run") != owner
            or config.get("WorkingDir") != "/work" or config.get("Cmd") != ["sleep", "infinity"]
            or config.get("Entrypoint") or config.get("User", "") != ("1000:1000" if runtime else "") or config.get("Volumes")
            or config.get("Healthcheck", {}).get("Test", ["NONE"]) != ["NONE"]):
        raise RuntimeError("container admission: foreign or altered container image")
    if (host.get("Privileged") or host.get("PidMode") or host.get("IpcMode") != "private"
            or host.get("Devices") or host.get("DeviceRequests") or host.get("CapAdd")
            or host.get("VolumesFrom") or host.get("PortBindings") or not host.get("Init")):
        raise RuntimeError("container admission: unsafe container privileges")
    mounts = row.get("Mounts", [])
    if runtime:
        if (mounts or host.get("NetworkMode") != "none" or not host.get("ReadonlyRootfs")
                or host.get("CapDrop") != ["ALL"] or host.get("SecurityOpt") != ["no-new-privileges"]
                or host.get("Tmpfs") != {str(GUEST_TMP): TMP_OPTIONS}):
            raise RuntimeError("container admission: unsafe test runtime resources")
    else:
        if host.get("NetworkMode") != "default" or host.get("SecurityOpt") or host.get("Tmpfs"):
            raise RuntimeError("container admission: unsafe setup resources")
        if len(mounts) != 2 or {item.get("Destination") for item in mounts} != {"/src", "/venv"}:
            raise RuntimeError("container admission: unexpected setup mounts")
        for mount in mounts:
            if mount["Destination"] == "/src":
                valid = mount.get("Type") == "bind" and mount.get("Source") == str(source) and not mount.get("RW")
            else:
                valid = mount.get("Type") == "volume" and mount.get("Name") == volume and mount.get("RW")
            if not valid:
                raise RuntimeError("container admission: altered setup mount")


def container_id(value):
    identifier = value.strip()
    if not re.fullmatch(r"[0-9a-f]{64}", identifier):
        raise RuntimeError("container admission: invalid supervisor-created container identity")
    return identifier


def asset_identity(path, content=False):
    from test_kernel_assets import owned_ancestors
    owned_ancestors(path)
    info = path.stat()
    value = (info.st_dev, info.st_ino, info.st_uid, info.st_mode, info.st_size, info.st_mtime_ns)
    if content:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        with os.fdopen(descriptor, "rb") as stream:
            digest = hashlib.sha256(stream.read(128 * 1024 * 1024 + 1)).hexdigest()
            after = os.fstat(stream.fileno())
            if after.st_size > 128 * 1024 * 1024 or (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns) != (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns):
                raise RuntimeError("container admission: Docker executable changed during sealing")
        value += (digest,)
    return value


def pinned_source(path, info, snapshot):
    from ml_stack.activity.source_snapshot import FILE_LIMIT, TOTAL_LIMIT
    if info.st_size > FILE_LIMIT or snapshot.total + info.st_size > TOTAL_LIMIT:
        raise RuntimeError("container admission: source bytes exceed bounds")
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(descriptor, "rb") as stream:
        before = os.fstat(stream.fileno())
        content = stream.read(FILE_LIMIT + 1)
        after = os.fstat(stream.fileno())
        current = path.lstat()
        proofs = {(node.st_dev, node.st_ino, node.st_size, node.st_mtime_ns, node.st_mode, node.st_nlink)
                  for node in (info, before, after, current)}
        if len(proofs) != 1 or len(content) != info.st_size or time.monotonic() >= snapshot.deadline:
            raise RuntimeError("container admission: source changed during bounded staging")
    snapshot.total += len(content)
    return content
