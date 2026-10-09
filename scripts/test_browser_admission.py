"""Owned source plans, per-test grants and exact browser kernel resources."""
from __future__ import annotations

import contextlib
import contextvars
import json
import os
import stat
import sys
from dataclasses import replace
from pathlib import Path

from test_fixture_plan import FixturePlan, invocation_selectors
from test_kernel_browser import prepare as prepare_assets
from test_kernel_browser_endpoints import EndpointBank

from poolhouse.activity.source_snapshot import SourceSnapshot, private_namespace, protected_roots
from poolhouse.fleet.discovery import primary_ip
from poolhouse.sandbox.policy import Policy
from poolhouse.sandbox.seatbelt import quote

CURRENT = contextvars.ContextVar('test_fixture_resources', default=None)


@contextlib.contextmanager
def bind(run):
    token = CURRENT.set(run)
    try:
        yield
    finally:
        CURRENT.reset(token)


def source_plan(command: list[str], environment: dict[str, str]) -> FixturePlan:
    root = Path(__file__).resolve().parent.parent
    storage = private_namespace(environment, 'poolhouse-fixture-plan-')
    snapshot = None
    try:
        snapshot = SourceSnapshot(root, storage)
        files = [snapshot.path(name) for name in snapshot.tracked if snapshot.path(name).is_file()]
        return FixturePlan(command, root, files)
    finally:
        if snapshot is not None:
            snapshot.close()
        storage.rmdir()


class BrowserRun:
    def __init__(self, command: list[str], environment: dict[str, str], admission, workers: int, *, plan: FixturePlan | None = None):
        self.plan = source_plan(command, environment) if plan is None else plan
        self.selectors = tuple(invocation_selectors(command))
        self.admission = admission
        self.assets = None
        self.bank = None
        self.process = None
        self.environment = {}
        self.profile_identity = None
        admission.fixture_resources = self.grant
        if self.plan.resources & {'browser-endpoints', 'lan-negative'}:
            address = primary_ip() if 'lan-negative' in self.plan.resources else None
            if 'lan-negative' in self.plan.resources and not address:
                raise RuntimeError('browser confinement: no verified LAN interface for declared fixture')
            count = min(64 if address else 128, max(8, len(self.plan.selected) * 2 + workers * 2))
            self.bank = EndpointBank(count, self.active, lan_address=address)
            self.environment = {'DEV_TEST_BROWSER_ENDPOINT': self.bank.endpoint,
                                'DEV_TEST_BROWSER_IDENTITY': json.dumps(self.bank.identity),
                                'DEV_TEST_BROWSER_TOKEN': self.bank.token,
                                'DEV_TEST_BROWSER_CAPACITY': str(len(self.bank.ports))}

    def grant(self, label: str, phase: str) -> frozenset[str]:
        if phase == 'collection':
            return frozenset()
        base = label.split('[', 1)[0]
        if base not in self.plan.node_resources or not any(
                label == selector or ('[' not in selector and (base == selector or base.startswith(selector + '::')))
                for selector in self.selectors):
            raise PermissionError('browser confinement: admission label is outside source invocation')
        return self.plan.node_resources[base]

    @contextlib.contextmanager
    def active(self, identifier: str):
        with self.admission.terminal_admission(identifier):
            granted = self.admission.active_resources.get(identifier)
            if not isinstance(granted, frozenset):
                raise PermissionError('browser confinement: active test has no source-bound resource receipt')
            yield granted

    def prepare(self, control: Path, environment: dict[str, str]) -> None:
        self.assets = prepare_assets(self.plan.resources, control, environment,
                                     protected_roots(environment), (Path(sys.prefix).resolve(), Path(sys.base_prefix).resolve()))
        environment.update(self.environment)
        if self.assets is not None:
            environment.update(self.assets.environment)
        self.recheck()

    def policy(self, policy: Policy) -> Policy:
        files, directories, metadata, executables, sockets = [], [], [], [], []
        if self.assets is not None:
            files = list(map(str, self.assets.files))
            directories = list(map(str, self.assets.directories))
            metadata = list(map(str, self.assets.metadata))
            executables = list(map(str, self.assets.executables))
        if self.bank is not None:
            metadata.append(self.bank.endpoint)
            sockets.append(self.bank.endpoint)
        return replace(policy, read_files=(*policy.read_files, *files), read_dirs=(*policy.read_dirs, *directories),
                       read_metadata=(*policy.read_metadata, *metadata), exec=(*policy.exec, *executables),
                       unix_sockets=(*policy.unix_sockets, *sockets)).validated()

    def seal_profile(self, path: Path) -> None:
        self.recheck()
        if self.bank is None:
            return
        if path.resolve(strict=True) != path:
            raise RuntimeError('browser confinement: redirected profile artifact')
        descriptor = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_NOFOLLOW)
        with os.fdopen(descriptor, 'w') as stream:
            info = os.fstat(stream.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077 or info.st_nlink != 1:
                raise RuntimeError('browser confinement: profile is not an owned private artifact')
            for host, port in self.bank.addresses:
                endpoint = quote(f'{host}:{port}')
                stream.write(f'(allow network-outbound (remote ip {endpoint}))\n')
                stream.write(f'(allow network-inbound (local ip {endpoint}))\n')
            stream.flush()
            os.fsync(stream.fileno())
        self.profile_identity = path.stat().st_dev, path.stat().st_ino

    def recheck(self) -> None:
        self.plan.recheck()
        if self.assets is not None:
            self.assets.recheck()
        if self.bank is not None:
            self.bank.recheck()

    def proof(self) -> dict:
        return {'fixtures': self.plan.proof(),
                'assets': self.assets.proof() if self.assets is not None else None,
                'endpoints': self.bank.proof() if self.bank is not None else None}

    def close(self, *, descendants_drained: bool = False) -> None:
        if self.bank is not None:
            if descendants_drained is not True:
                raise RuntimeError('browser confinement: retained endpoint bank pending owned descendant drain proof')
            self.bank.close()
