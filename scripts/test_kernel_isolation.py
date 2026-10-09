"""Supervisor-owned macOS pytest confinement."""
from __future__ import annotations

import contextlib
import math
import os
import secrets
import shutil
import site
import socket
import stat
import sys
import sysconfig
import tempfile
from dataclasses import replace
from pathlib import Path

import test_browser_admission
from test_fixture_plan import FixturePlan
from test_kernel_holder import (
    identity as holder_asset_identity,
    prepare as prepare_holder,
    private_directory,
)
from test_kernel_selectors import admitted_node

from ml_stack.activity.source_snapshot import actual_git, git_environment
from ml_stack.sandbox.path_language import exact_language
from ml_stack.sandbox.policy import Net, Policy
from ml_stack.sandbox.seatbelt import Seatbelt, quote

GATE = frozenset({"test_layers.py", "test_budgets.py", "test_budget_floors.py",
                  "test_budget_ratchet.py", "test_redteam_human_floor.py", "test_requests_floor.py",
                  "test_onboard_human.py", "test_cli_reference.py", "test_test_kernel_isolation.py"})
PUBLIC_ENV = frozenset({"PATH", "LANG", "LC_ALL", "LC_CTYPE", "TERM", "CI", "CLAUDECODE",
                        "DEV_TEST_BUDGET", "DEV_TEST_HEAVY_LANES", "DEV_TEST_LANE_WAIT_S",
                        "DEV_TEST_LOAD_HIGH", "DEV_TEST_RESERVED_CORES", "DEV_TEST_WAIT_S",
                        "DEV_TEST_WORKERS", "DEV_TEST_PYTEST_ENDPOINT", "DEV_TEST_PYTEST_TOKEN",
                        "DEV_TEST_REMOTE_BROKER", "DEV_TEST_SLOTS_DIR", "DEV_TEST_PTY_TOKEN",
                        "DEV_TEST_PTY_ENDPOINT", "DEV_TEST_PYTEST_IDENTITY", "DEV_TEST_PTY_IDENTITY"})


def filtered_environment(environment: dict[str, str]) -> dict[str, str]:
    threads = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
               "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")
    return {key: value for key, value in environment.items() if key in PUBLIC_ENV or key in threads}


def check_selectors(command: list[str]) -> FixturePlan | None:
    if len(command) < 4 or Path(command[0]).resolve() != Path(sys.executable).resolve() or command[1:3] != ["-m", "pytest"]:
        raise RuntimeError("test confinement: expected the maintained pytest interpreter invocation")
    selected = []
    index = 3
    while index < len(command):
        part = command[index]
        if part in ("-q", "-rfE", "--redteam", "--color=no", "--slow"):
            index += 1
            continue
        if part in ("-n", "-p"):
            if index + 1 >= len(command) or (part == "-p" and command[index + 1] != "testslots_pytest"):
                raise RuntimeError("test confinement: unadmitted pytest plugin or worker option")
            if part == "-n" and not command[index + 1].isdigit():
                raise RuntimeError("test confinement: invalid worker count")
            index += 2
            continue
        if part.startswith("--junitxml="):
            index += 1
            continue
        target = part.split("::", 1)[0]
        if (Path(target).parent != Path("tests") or Path(target).name not in GATE or ".." in Path(target).parts) and not admitted_node(part):
            return test_browser_admission.source_plan(command, dict(os.environ))
        selected.append(target)
        index += 1
    if not selected:
        raise RuntimeError("test confinement: explicit reviewed gate selectors are required")


def worktree_pointer(repo: Path) -> Path:
    pointer = repo / ".git"
    entry = pointer.lstat()
    if not stat.S_ISREG(entry.st_mode) or entry.st_uid != os.getuid() or entry.st_mode & 0o022:
        raise RuntimeError("test confinement: an owned plain worktree pointer is required")
    return pointer


def source_metadata(repo: Path, directories: set[Path], aliases: set[Path]) -> set[Path]:
    nodes = {*aliases, worktree_pointer(repo)}
    for directory in sorted(directories):
        if not directory.is_relative_to(repo):
            continue
        if directory.resolve() != directory:
            raise RuntimeError("test confinement: redirected source directory")
        with os.scandir(directory) as entries:
            for index, entry in enumerate(entries):
                if index >= 4096 or len(nodes) >= 65536:
                    raise RuntimeError("test confinement: source metadata inventory exceeds bounds")
                nodes.add(Path(entry.path))
    return nodes


def source_reads(repo: Path, environment: dict[str, str]) -> tuple[list[Path], set[Path]]:
    from ml_stack.activity.source_snapshot import SourceSnapshot, private_namespace
    storage = private_namespace(environment, "ml-stack-inventory-")
    snapshot = None
    try:
        snapshot = SourceSnapshot(repo, storage)
        files = [snapshot.path(name) for name in snapshot.tracked if snapshot.path(name).is_file()]
    finally:
        if snapshot is not None:
            snapshot.close()
        storage.rmdir()
    files.extend((Path(__file__).resolve(), repo / "scripts/test_terminal_bank.py", repo / "scripts/test_kernel_assets.py",
                  repo / "tests/test_test_kernel_isolation.py"))
    files = [path.resolve() for path in files if path.is_file() and path.resolve().is_relative_to(repo)]
    directories = {repo}
    slots = environment.get("DEV_TEST_SLOTS_DIR")
    if slots:
        directories.add(Path(slots).resolve())
    for path in files:
        directories.update(parent for parent in path.parents if parent.is_relative_to(repo))
    return files, directories


def interpreter_read_roots(protected: tuple[Path, ...], account: Path) -> tuple[str, ...]:
    """Return actual interpreter library roots without user-site authority."""
    values = [value for key, value in sysconfig.get_paths().items()
              if key in ("stdlib", "platstdlib", "purelib", "platlib")]
    values.extend(site.getsitepackages())
    if len(values) > 16:
        raise RuntimeError("test confinement: interpreter library root count exceeds bound")
    user = site.getusersitepackages()
    user_sites = [Path(value).resolve() for value in ([user] if isinstance(user, str) else user)]
    assets = {Path(value).resolve() for value in values if Path(value).is_dir()}
    for asset in assets:
        if (asset == account or account.is_relative_to(asset) or any(
                asset.is_relative_to(root) or root.is_relative_to(asset) for root in protected)
                or any(asset.is_relative_to(root) or root.is_relative_to(asset) for root in user_sites)):
            raise RuntimeError("test confinement: interpreter assets overlap protected account or user-site roots")
    return tuple(str(path) for path in sorted(assets))


class ConfinedRun:
    """Own private roots and a kernel launch profile for one pytest invocation."""

    def __init__(self, command: list[str], environment: dict[str, str], endpoint: str, *, terminal_endpoint: str = "", artifacts=None):
        self.outputs = []
        self.relays = []
        self.relay_error = None
        self.artifacts = artifacts
        self.browser = test_browser_admission.CURRENT.get()
        self.denied_listener = None
        self.bootstrap = None
        self.control = None
        self.holder_namespace = None
        with contextlib.ExitStack() as cleanup:
            cleanup.callback(self.discard_partial)
            self.prepare(command, environment, endpoint, terminal_endpoint)
            cleanup.pop_all()

    def discard_partial(self) -> None:
        """Release everything a failed preparation acquired, including its control and holder directories."""
        with contextlib.ExitStack() as stack:
            for path in (self.holder_namespace, self.control):
                if path is not None:
                    stack.callback(shutil.rmtree, path, True)
            self.close_outputs()
            if self.denied_listener is not None:
                stack.callback(self.denied_listener.close)
            if self.bootstrap is not None:
                stack.callback(self.bootstrap.close)

    def prepare(self, command: list[str], environment: dict[str, str], endpoint: str, terminal_endpoint: str) -> None:
        import pwd
        check_selectors(command)
        base = supervisor_storage(environment)
        self.control = Path(tempfile.mkdtemp(prefix="ml-stack-confined-", dir=base)).resolve()
        self.scratch = self.control / "scratch"
        self.scratch.mkdir(mode=0o700)
        roots = {name: self.scratch / name for name in ("home", "state", "cache", "tmp")}
        for root in roots.values():
            root.mkdir(mode=0o700)
        self.prepare_canaries()
        repo = Path(__file__).resolve().parent.parent
        self.environment = filtered_environment(environment)
        config_keys = {"GIT_CONFIG_NOSYSTEM", "GIT_CONFIG_GLOBAL", "GIT_CONFIG_SYSTEM"}
        self.environment.update({key: value for key, value in git_environment().items() if key in config_keys})
        self.environment.update(HOME=str(roots["home"]), ML_STACK_HOME=str(roots["state"]),
                                XDG_CACHE_HOME=str(roots["cache"]), ML_STACK_CACHE=str(roots["cache"]), TMPDIR=str(roots["tmp"]),
                                PYTHONPATH=os.pathsep.join((str(repo / "src"), str(repo / "scripts"))),
                                PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1",
                                PYTHON_KEYRING_BACKEND="keyring.backends.null.Keyring",
                                DEV_TEST_CONFINEMENT_CANARY=str(self.canary), DEV_TEST_DENIED_ENDPOINT=self.denied_endpoint)
        files, directories = source_reads(repo, environment)
        self.holder_assets, self.holder_namespace, self.holder_compilation = prepare_holder(
            command, self.control, self.environment, protected_roots(environment),
            (Path(sys.prefix).resolve(), Path(sys.base_prefix).resolve()))
        if self.holder_assets is not None:
            files.extend(self.holder_assets.files)
            directories.update(self.holder_assets.directories)
        self.observation_seconds = float(environment.get("DEV_TEST_WAIT_S", "3600"))
        if not math.isfinite(self.observation_seconds) or self.observation_seconds <= 0:
            raise ValueError("test confinement: invalid observation wait bound")
        self.observation_seconds = min(15.0, self.observation_seconds)
        protected = protected_roots(environment)
        arguments = self.prepare_outputs(command, protected)
        account = Path(pwd.getpwuid(os.getuid()).pw_dir).resolve()
        protected = protected_roots(environment)
        prefixes = interpreter_read_roots(protected, account)
        for prefix in (sys.prefix, sys.base_prefix):
            config = Path(prefix).resolve() / "pyvenv.cfg"
            if config.is_file():
                files.append(config)
            files.extend(Path(prefix).resolve().glob("lib/*.dylib"))
        from test_kernel_assets import interpreter_assets
        files, aliases, self.images = interpreter_assets(files, protected, self.observation_seconds)
        self.prepare_role_images(files, aliases, protected)
        self.prepare_git(files, aliases, protected)
        self.loader_metadata = tuple(sorted(aliases))
        policy = Policy(name="maintained-pytest", read=prefixes, write=(str(self.scratch),),
                        read_files=tuple(str(path) for path in files),
                        read_metadata=(*(str(path) for path in source_metadata(repo, directories, aliases)),
                                       *(str(path) for path in (self.holder_assets.metadata if self.holder_assets is not None else ())),
                                       endpoint[5:], terminal_endpoint),
                        read_dirs=tuple(str(path) for path in sorted(directories)),
                        exec=tuple(str(Path(path).resolve()) for path in
                                   (sys.executable, str(self.git), "/bin/sh",
                                    *(sorted(self.holder_assets.executables) if self.holder_assets is not None else ()))),
                        net=Net.deny(), env=self.environment, strict=True,
                        unix_sockets=(endpoint[5:], terminal_endpoint),
                        unix_namespaces=(() if self.holder_namespace is None else (str(self.holder_namespace),))).validated()
        if self.holder_namespace is not None:
            policy = replace(policy, write=(*policy.write, str(self.holder_namespace))).validated()
            self.holder_identity = private_directory(self.holder_namespace)
        policy = self.role_policy(policy)
        policy = self.browser_policy(policy)
        if self.artifacts is not None and (self.holder_assets is not None or self.holder_namespace is not None):
            self.artifacts.proof["holder_resources"] = {
                **(self.holder_assets.proof() if self.holder_assets is not None else {"scope": "namespace-only"}),
                "minimal_compilation": self.holder_compilation, "namespace": str(self.holder_namespace),
                "namespace_identity": getattr(self, "holder_identity", None)}
        self.launch_profile(arguments, policy)

    def role_policy(self, policy: Policy) -> Policy:
        if self.holder_assets is None or not self.holder_assets.role_channels:
            return policy
        channels = self.holder_assets.role_channels
        return replace(policy, write=(*policy.write, *channels),
                       unix_namespaces=(self.holder_assets.role_namespace,)).validated()

    def prepare_role_images(self, files: list[Path], aliases: set[Path], protected: tuple[Path, ...]) -> None:
        if self.holder_assets is None or not self.holder_assets.role_cases:
            return
        if self.holder_assets.scope == "immutable-role-fences":
            return
        from test_kernel_assets import loader_assets
        native = [path for path in self.holder_assets.files if path.name.endswith(".so")]
        if len(native) != 3 * len(self.holder_assets.role_cases):
            raise RuntimeError("holder role: native image closure changed")
        pinned = {path: self.holder_assets.identities[path] for path in native}
        libraries, image_aliases, identities = loader_assets([Path(sys.executable), *native], protected,
                                                            self.observation_seconds, pinned=pinned)
        files.extend(libraries)
        aliases.update(image_aliases)
        self.images.update(identities)

    def prepare_git(self, files: list[Path], aliases: set[Path], protected: tuple[Path, ...]) -> None:
        from test_kernel_assets import loader_assets
        self.git = actual_git()
        git_files, git_aliases, git_images = loader_assets([self.git], protected, self.observation_seconds)
        files.extend(git_files)
        aliases.update(git_aliases)
        self.images.update(git_images)
        info = self.git.stat()
        self.git_identity = (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns,
                             info.st_uid, info.st_gid, info.st_mode, info.st_nlink)
        self.environment["PATH"] = os.pathsep.join((str(self.git.parent), str(Path(sys.executable).parent), os.defpath))

    def prepare_outputs(self, command: list[str], protected: tuple[Path, ...]) -> list[str]:
        from test_kernel_outputs import OutputSink, junit_sink
        if self.artifacts is None:
            for stream in (sys.stdout, sys.stderr):
                self.relays.append(OutputSink(stream.fileno(), protected))
        else:
            self.artifacts.validate()
            for _ in range(2):
                self.relays.append(OutputSink(self.artifacts.sinks["console.log"].fd, protected, private=True))
        arguments = []
        for part in command:
            if part.startswith("--junitxml="):
                if self.outputs:
                    raise RuntimeError("test confinement: duplicate JUnit destination")
                original = Path(part.split("=", 1)[1])
                if self.artifacts is not None and original != self.artifacts.directory / "junit.xml":
                    raise RuntimeError("test confinement: strict JUnit target is not an inert artifact")
                destination = junit_sink(original, protected)
                confined = self.scratch / "junit.xml"
                self.outputs.append((confined, destination))
                part = "--junitxml=" + str(confined)
            arguments.append(part)
        arguments.extend(("-o", "cache_dir=" + str(self.scratch / "pytest-cache")))
        return arguments

    def launch_profile(self, arguments, policy: Policy) -> None:
        from test_kernel_role_smoke import CONTROL_NODE, observe_control
        if CONTROL_NODE in arguments:
            report, receipt = observe_control(self, policy)
            self.environment["ML_STACK_TEST_ROLE_CONTROL_REPORT"] = str(report)
            policy = replace(policy, read_files=(*policy.read_files, str(report))).validated()
            if self.artifacts is not None:
                self.artifacts.proof["owned_process_control"] = receipt
        from test_kernel_role_home import IMPORT_NODE
        if IMPORT_NODE in arguments:
            from test_kernel_role_smoke import observe as observe_role
            report, receipt = observe_role(self, policy)
            self.environment["ML_STACK_TEST_ROLE_IMPORT_REPORT"] = str(report)
            policy = replace(policy, read_files=(*policy.read_files, str(report))).validated()
            self.holder_assets.identities[report] = holder_asset_identity(report.lstat())
            if self.artifacts is not None:
                self.artifacts.proof["holder_import_smoke"] = receipt
        from test_kernel_compile import NODE as COMPILE_NODE, observe as compile_components
        if any(part.split("[", 1)[0] == COMPILE_NODE for part in arguments):
            report = compile_components(self, policy)
            self.environment["DEV_TEST_COMPILER_REPORT"] = str(report)
            policy = replace(policy, read_files=(*policy.read_files, str(report))).validated()
        from test_kernel_regex import NODE, observe
        if any(part.split("[", 1)[0] == NODE for part in arguments):
            result, report = observe(self, policy)
            self.environment["DEV_TEST_REGEX_REPORT"] = str(report)
            policy = replace(policy, read_files=(*policy.read_files, str(report))).validated()
            if self.artifacts is not None:
                self.artifacts.proof["native_regex"] = result
        from test_kernel_attestation import Attestation
        self.bootstrap = Attestation(self, arguments)
        policy = replace(policy, read_files=(*policy.read_files, str(self.bootstrap.code), str(self.bootstrap.helper), str(self.bootstrap.manifest))).validated()
        self.policy = policy
        if self.artifacts is not None:
            self.artifacts.proof["exact_read_language"] = exact_language(policy.read_files, quote)[1]
        self.wrapped = Seatbelt().wrap(self.bootstrap.argv, policy, profile_path=self.control / "profile.sb")
        if self.holder_assets is not None and self.holder_assets.role_rules:
            with (self.control / "profile.sb").open("a") as stream:
                stream.write("\n" + "\n".join(self.holder_assets.role_rules) + "\n")
        from test_kernel_compile import LITERAL_NODE, replace_exact_file_language
        if any(part.split("[", 1)[0] == LITERAL_NODE for part in arguments):
            receipt = replace_exact_file_language(self.control / "profile.sb", policy.read_files)
            if self.artifacts is not None:
                self.artifacts.proof["exact_read_language"] = receipt
        if self.browser is not None:
            self.browser.seal_profile(self.control / "profile.sb")
        self.bootstrap.seal(self.control / "profile.sb")
        if self.artifacts is not None:
            self.artifacts.proof.update(profile=self.bootstrap.profile, bootstrap=self.bootstrap.code_hash,
                                        manifest=self.bootstrap.manifest_hash)

    def browser_policy(self, policy: Policy) -> Policy:
        if self.browser is None:
            return policy
        self.browser.prepare(self.control, self.environment)
        policy = self.browser.policy(policy)
        if self.artifacts is not None:
            self.artifacts.proof["browser_resources"] = self.browser.proof()
        return policy

    def recheck_images(self) -> None:
        if self.browser is not None:
            self.browser.recheck()
        info = self.git.stat()
        identity = (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns,
                    info.st_uid, info.st_gid, info.st_mode, info.st_nlink)
        if actual_git() != self.git or identity != self.git_identity:
            raise RuntimeError("test confinement: owned Git changed before launch")
        if self.holder_assets is not None:
            self.holder_assets.recheck()
        if self.holder_namespace is not None and private_directory(self.holder_namespace) != self.holder_identity:
            raise RuntimeError("holder confinement: namespace identity changed")
        for path, expected in self.images.items():
            info = path.stat()
            if (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns) != expected:
                raise RuntimeError("test confinement: loader image changed before launch")

    def prepare_canaries(self) -> None:
        self.canary = self.control / "denied-canary"
        self.canary.write_bytes(secrets.token_bytes(32))
        self.canary.chmod(0o600)
        self.canary_identity = self.canary.read_bytes()
        self.denied_endpoint = str(self.control / "denied.sock")
        self.denied_listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.denied_listener.bind(self.denied_endpoint)
        self.denied_listener.listen(1)
        self.denied_listener.setblocking(False)

    def finish(self) -> None:
        try:
            if self.holder_namespace is not None and private_directory(self.holder_namespace) != self.holder_identity:
                raise RuntimeError("holder confinement: namespace identity changed")
            canary_unchanged = self.canary.read_bytes() == self.canary_identity
            self.promote_outputs()
            self.relays[1].write(f"test confinement: retained supervisor artifacts at {self.control}\n".encode())
            if self.relay_error is not None:
                raise RuntimeError("test confinement: output relay failed") from self.relay_error
            if not canary_unchanged:
                raise RuntimeError("test confinement: supervisor canary changed")
        finally:
            self.close_outputs()
            self.denied_listener.close()
            self.bootstrap.close()

    def promote_outputs(self) -> None:
        for source, destination in self.outputs:
            try:
                fd = os.open(source, os.O_RDONLY | os.O_NOFOLLOW)
            except FileNotFoundError:
                continue
            with os.fdopen(fd, "rb") as stream:
                info = os.fstat(stream.fileno())
                if not stat.S_ISREG(info.st_mode) or info.st_size > 16 * 1024 * 1024:
                    raise RuntimeError("test confinement: invalid bounded JUnit output")
                value = stream.read(16 * 1024 * 1024 + 1)
                if len(value) > 16 * 1024 * 1024:
                    raise RuntimeError("test confinement: JUnit output exceeds bound")
            if destination.validate() != destination.identity:
                raise RuntimeError("test confinement: JUnit destination changed")
            os.ftruncate(destination.fd, 0)
            destination.write(value)

    def close_outputs(self) -> None:
        while self.outputs:
            _, sink = self.outputs.pop()
            sink.close()
        while self.relays:
            self.relays.pop().close()


def supervisor_storage(environment: dict[str, str]) -> Path:
    from ml_stack.activity.source_snapshot import validate_storage
    return validate_storage(Path(tempfile.gettempdir()), protected_roots(environment))


def protected_roots(environment: dict[str, str]) -> tuple[Path, ...]:
    from ml_stack.activity.source_snapshot import protected_roots as roots
    return roots(environment)


def validate_control_roots(environment: dict[str, str]) -> None:
    import pwd
    account = Path(pwd.getpwuid(os.getuid()).pw_dir).resolve()
    slots = Path(environment.get("DEV_TEST_SLOTS_DIR", str(account / ".cache/dev-test-slots"))).expanduser().resolve()
    if any(slots.is_relative_to(root) or root.is_relative_to(slots) for root in protected_roots(environment)):
        raise RuntimeError("test confinement: CPU admission storage overlaps protected roots")
