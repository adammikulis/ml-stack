"""Source-bound fixture resource declarations and invocation plans."""
from __future__ import annotations

import ast
import hashlib
import json
import os
import stat
import time
from dataclasses import dataclass
from pathlib import Path

KINDS = frozenset({'browser', 'browser-endpoints', 'javascript', 'lan-negative'})
BUILTINS = frozenset({'tmp_path', 'tmp_path_factory', 'monkeypatch', 'request', 'capsys',
                     'capfd', 'caplog', 'recwarn', 'pytestconfig', 'cache', 'record_property',
                     'record_testsuite_property', 'doctest_namespace', 'worker_id', 'testrun_uid', 'self', 'cls'})


def resources(*names: str):
    if not names or any(name not in KINDS for name in names):
        raise ValueError('fixture plan: invalid literal resource declaration')
    return lambda function: function


def spelling(node) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = spelling(node.value)
        return prefix + '.' + node.attr if prefix else ''
    return ''


def literal(node):
    try:
        return ast.literal_eval(node)
    except (ValueError, TypeError, RecursionError):
        raise RuntimeError('fixture plan: dynamic fixture declaration') from None


def file_identity(path: Path) -> tuple[int, ...]:
    info = path.lstat()
    if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid()
            or info.st_mode & 0o022 or info.st_nlink != 1 or info.st_size > 4 * 1024 * 1024):
        raise RuntimeError('fixture plan: source must be bounded, owned and unlinked')
    return (info.st_dev, info.st_ino, info.st_mode, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def invocation_selectors(command: list[str]) -> list[str]:
    if len(command) < 4 or command[1:3] != ['-m', 'pytest']:
        raise RuntimeError('fixture plan: invocation must be maintained pytest')
    selected = []
    index = 3
    while index < len(command):
        item = command[index]
        if item.startswith(('-k', '-m', '--pyargs')):
            raise RuntimeError('fixture plan: filtered invocation requires maintained selection metadata')
        if item in {'-n', '-p'}:
            index += 1
            if index >= len(command) or (item == '-n' and not command[index].isdecimal()) or (item == '-p' and command[index] != 'testslots_pytest'):
                raise RuntimeError('fixture plan: unsupported pytest execution option')
        elif item in {'-q', '-rfE', '--redteam', '--color=no', '--slow'} or item.startswith('--junitxml='):
            pass
        elif item.startswith('tests/') and item.split('::', 1)[0].endswith('.py'):
            selected.append(item)
        else:
            raise RuntimeError('fixture plan: unsupported invocation selection')
        index += 1
    return selected


@dataclass(frozen=True)
class Definition:
    node: ast.FunctionDef | ast.AsyncFunctionDef
    key: tuple[str, ...]
    fixture: bool
    autouse: bool
    name: str
    resources: frozenset[str]
    requested: tuple[str, ...]
    parameters: frozenset[str]


class Module:
    def __init__(self, path: Path, data: bytes, root: Path):
        self.path = path
        self.root = root
        self.tree = ast.parse(data, filename=str(path))
        self.aliases = {}
        self.definitions = {}
        self.collect_aliases()
        self.collect(self.tree.body, (), self.module_markers())

    def collect_aliases(self) -> None:
        for node in self.tree.body:
            if isinstance(node, ast.Import):
                for alias in node.names:
                    self.aliases[alias.asname or alias.name.split('.')[0]] = alias.name
            elif isinstance(node, ast.ImportFrom):
                if any(alias.name == '*' for alias in node.names):
                    raise RuntimeError('fixture plan: wildcard fixture imports require explicit source metadata')
                namespace = node.module or ''
                if node.level:
                    parts = list(self.path.parent.relative_to(self.root).parts)
                    if parts and parts[0] == 'src':
                        parts.pop(0)
                    if node.level > len(parts):
                        raise RuntimeError('fixture plan: relative import escapes source package')
                    namespace = '.'.join([*parts[:len(parts) - node.level + 1], *namespace.split('.')])
                if not namespace:
                    raise RuntimeError('fixture plan: unresolved import namespace')
                for alias in node.names:
                    self.aliases[alias.asname or alias.name] = namespace + '.' + alias.name
            elif isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                value = self.canonical(node.value)
                if value:
                    self.aliases[node.targets[0].id] = value

    def module_markers(self):
        result = []
        for node in self.tree.body:
            if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == 'pytestmark' for target in node.targets):
                values = node.value.elts if isinstance(node.value, (ast.List, ast.Tuple)) else [node.value]
                for value in values:
                    call = value.func if isinstance(value, ast.Call) else value
                    if not self.canonical(call).startswith('pytest.mark.'):
                        raise RuntimeError('fixture plan: dynamic module fixture marks')
                result.extend(values)
        return tuple(result)

    def canonical(self, node) -> str:
        value = spelling(node)
        base, _, rest = value.partition('.')
        return self.aliases.get(base, base) + ('.' + rest if rest else '')

    def annotations(self, decorators) -> tuple[bool, bool, str | None, set[str], list[str], set[str]]:
        fixture, autouse, name = False, False, None
        kinds, requested, parameters = set(), [], set()
        for decorator in decorators:
            call = decorator if isinstance(decorator, ast.Call) else None
            canonical = self.canonical(call.func if call else decorator)
            if canonical == 'pytest.fixture':
                fixture = True
                values = {item.arg: literal(item.value) for item in call.keywords
                          if item.arg in {'autouse', 'name', 'scope', None}} if call else {}
                if None in values or type(values.get('autouse', False)) is not bool:
                    raise RuntimeError('fixture plan: dynamic fixture configuration')
                autouse, name = values.get('autouse', False), values.get('name')
                if name is not None and not isinstance(name, str):
                    raise RuntimeError('fixture plan: dynamic fixture name')
            elif canonical == 'test_fixture_plan.resources':
                if call is None or call.keywords:
                    raise RuntimeError('fixture plan: resource declaration must be literal')
                values = [literal(item) for item in call.args]
                if not values or any(not isinstance(item, str) or item not in KINDS for item in values):
                    raise RuntimeError('fixture plan: unknown resource declaration')
                kinds.update(values)
            elif canonical == 'resources' or canonical.endswith('.resources'):
                raise RuntimeError('fixture plan: unresolved resource decorator')
            elif canonical == 'pytest.mark.usefixtures':
                values = [literal(item) for item in call.args]
                if any(not isinstance(item, str) for item in values):
                    raise RuntimeError('fixture plan: dynamic usefixtures')
                requested.extend(values)
            elif canonical == 'pytest.mark.parametrize':
                names = literal(call.args[0])
                names = names.split(',') if isinstance(names, str) else names
                if not isinstance(names, (list, tuple)) or any(not isinstance(item, str) for item in names):
                    raise RuntimeError('fixture plan: dynamic parameter names')
                indirect = next((literal(item.value) for item in call.keywords if item.arg == 'indirect'), False)
                if type(indirect) is not bool and not isinstance(indirect, (list, tuple)):
                    raise RuntimeError('fixture plan: dynamic indirect parameters')
                parameters.update(item.strip() for item in names if indirect is not True and item.strip() not in (indirect or ()))
        return fixture, autouse, name, kinds, requested, parameters

    def collect(self, nodes, prefix: tuple[str, ...], inherited=()) -> None:
        for node in nodes:
            if isinstance(node, ast.ClassDef):
                relevant = node.name.startswith('Test') or any(
                    isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and any(self.canonical(decorator.func if isinstance(decorator, ast.Call) else decorator) == 'pytest.fixture'
                            for decorator in item.decorator_list) for item in node.body)
                if not relevant:
                    continue
                if node.bases:
                    raise RuntimeError('fixture plan: inherited test fixture scopes require explicit source metadata')
                self.collect(node.body, (*prefix, node.name), (*inherited, *node.decorator_list))
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                annotations = self.annotations(node.decorator_list)
                if not annotations[0] and node.name.startswith('test_'):
                    annotations = self.annotations((*inherited, *node.decorator_list))
                fixture, autouse, name, kinds, requested, parameters = annotations
                key = (*prefix, node.name)
                if key in self.definitions or node.name in self.aliases:
                    raise RuntimeError('fixture plan: ambiguous source binding')
                self.definitions[key] = Definition(node, key, fixture, autouse, name or node.name,
                                                   frozenset(kinds), tuple(requested), frozenset(parameters))

    def fixture(self, name: str, prefix=(), excluding=frozenset()) -> Definition | None:
        for scope in (prefix, ()):
            found = next((value for value in self.definitions.values() if value.fixture and value.name == name
                          and value.key[:-1] == scope and (self.path, value.key) not in excluding), None)
            if found is not None:
                return found
        return None


class FixturePlan:
    def __init__(self, command: list[str], root: Path, admitted_files):
        self.deadline = time.monotonic() + 30
        selectors = invocation_selectors(command)
        self.command = tuple(command)
        self.root = root.resolve(strict=True)
        self.admitted = frozenset(Path(path) for path in admitted_files)
        self.modules = {}
        self.sources = {}
        self.selected = []
        self.node_resources = {}
        self.resources = set()
        self.edges = set()
        if not selectors:
            raise RuntimeError('fixture plan: invocation has no explicit source selectors')
        for selector in selectors:
            self.select(selector)
        self.resources = frozenset(self.resources)
        self.recheck()

    def module(self, path: Path) -> Module:
        if time.monotonic() >= self.deadline:
            raise RuntimeError('fixture plan: planning deadline exceeded')
        if path in self.modules:
            return self.modules[path]
        if path not in self.admitted or not path.is_relative_to(self.root) or path.resolve(strict=True) != path:
            raise RuntimeError('fixture plan: source is outside maintained snapshot')
        if len(self.modules) >= 128:
            raise RuntimeError('fixture plan: source graph exceeds bound')
        before = file_identity(path)
        with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW), 'rb') as stream:
            data = stream.read(4 * 1024 * 1024 + 1)
            if len(data) > 4 * 1024 * 1024 or file_identity(path) != before:
                raise RuntimeError('fixture plan: source changed during bounded read')
        if sum(value[2] for value in self.sources.values()) + len(data) > 32 * 1024 * 1024:
            raise RuntimeError('fixture plan: source bytes exceed bound')
        self.sources[path] = (before, hashlib.sha256(data).hexdigest(), len(data))
        self.modules[path] = Module(path, data, self.root)
        return self.modules[path]

    def imported(self, module: Module, name: str):
        value = module.aliases.get(name)
        if value is None or '.' not in value:
            return None
        namespace, _, symbol = value.rpartition('.')
        relative = Path(*namespace.split('.')).with_suffix('.py')
        for prefix in ('tests', 'scripts', 'src', ''):
            path = self.root / prefix / relative
            if path in self.admitted:
                target = self.module(path)
                return target, symbol
        return None

    def scopes(self, module: Module):
        result = []
        for parent in reversed((module.path.parent, *module.path.parent.parents)):
            if parent.is_relative_to(self.root):
                path = parent / 'conftest.py'
                if path != module.path and (path.exists() or path.is_symlink()):
                    if path not in self.admitted:
                        raise RuntimeError('fixture plan: visible conftest is outside maintained snapshot')
                    result.append(self.module(path))
        return [*result, module]

    def resolve(self, name: str, scopes: list[Module], prefix=(), active=frozenset()) -> None:
        for module in reversed(scopes):
            definition = module.fixture(name, prefix, active)
            if definition is not None:
                self.visit(module, definition, scopes, active)
                return
            imported = self.imported(module, name)
            if imported is not None:
                target, symbol = imported
                definition = target.fixture(symbol, excluding=active)
                if definition is not None:
                    self.visit(target, definition, scopes, active)
                    return
        if name in BUILTINS:
            return
        if name == 'playwright':
            self.resources.add('browser')
            return
        raise RuntimeError(f'fixture plan: unresolved fixture {name}')

    def visit(self, module: Module, definition: Definition, scopes, active=frozenset()) -> None:
        key = (module.path, definition.key)
        if time.monotonic() >= self.deadline or key in active or len(active) >= 64:
            raise RuntimeError('fixture plan: cyclic or excessive fixture graph')
        self.edges.add((str(module.path.relative_to(self.root)), definition.key))
        self.resources.update(definition.resources)
        dependencies = [argument.arg for argument in (*definition.node.args.posonlyargs, *definition.node.args.args, *definition.node.args.kwonlyargs)
                        if argument.arg not in definition.parameters]
        dependencies.extend(definition.requested)
        for call in ast.walk(definition.node):
            if isinstance(call, ast.Call) and spelling(call.func).endswith('.getfixturevalue'):
                if len(call.args) != 1 or not isinstance(literal(call.args[0]), str):
                    raise RuntimeError('fixture plan: dynamic fixture request')
                dependencies.append(literal(call.args[0]))
        for dependency in dependencies:
            self.resolve(dependency, scopes, definition.key[:-1], active | {key})

    def select(self, selector: str) -> None:
        base = selector.split('[', 1)[0]
        file, *key = base.split('::')
        if '..' in Path(file).parts or not file.endswith('.py'):
            raise RuntimeError('fixture plan: invalid source selector')
        module = self.module(self.root / file)
        selected = [value for path, value in module.definitions.items() if path[-1].startswith('test_')
                    and (not key or path[:len(key)] == tuple(key))]
        if not selected:
            raise RuntimeError('fixture plan: selector does not resolve to a source test')
        scopes = self.scopes(module)
        for definition in selected:
            aggregate = self.resources
            self.resources = set()
            self.selected.append((file, definition.key))
            self.visit(module, definition, scopes)
            autouse = set()
            for scope in scopes:
                autouse.update(fixture.name for fixture in scope.definitions.values() if fixture.autouse
                               and (not fixture.key[:-1] or fixture.key[:-1] == definition.key[:-1]))
                for alias in scope.aliases:
                    imported = self.imported(scope, alias)
                    if imported is not None:
                        target, name = imported
                        fixture = target.fixture(name)
                        if fixture is not None and fixture.autouse:
                            autouse.add(alias)
            for name in autouse:
                self.resolve(name, scopes, definition.key[:-1])
            nodeid = file + '::' + '::'.join(definition.key)
            self.node_resources[nodeid] = frozenset(self.resources)
            aggregate.update(self.resources)
            self.resources = aggregate

    def recheck(self) -> None:
        for path, (expected, digest, _) in self.sources.items():
            if file_identity(path) != expected:
                raise RuntimeError('fixture plan: source identity changed before admission')
            with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW), 'rb') as stream:
                data = stream.read(4 * 1024 * 1024 + 1)
            if len(data) > 4 * 1024 * 1024 or hashlib.sha256(data).hexdigest() != digest or file_identity(path) != expected:
                raise RuntimeError('fixture plan: source content changed before admission')

    def proof(self) -> dict:
        self.recheck()
        return {'invocation_sha256': hashlib.sha256(json.dumps(self.command).encode()).hexdigest(),
                'resources': sorted(self.resources), 'selected': self.selected,
                'node_resources': {node: sorted(kinds) for node, kinds in sorted(self.node_resources.items())},
                'sources': {str(path.relative_to(self.root)): value[1] for path, value in self.sources.items()},
                'fixture_edges': sorted(self.edges)}
