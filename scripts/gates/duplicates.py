"""Functions whose bodies normalise to the same shape as another function's."""

from __future__ import annotations

import ast
import hashlib
from dataclasses import dataclass
from pathlib import Path

try:
    from . import Finding
except ImportError:  # loaded as a standalone file
    @dataclass(frozen=True)
    class Finding:  # type: ignore[no-redef]
        path: str
        line: int
        detail: str


NAME = "duplicate-bodies"
OWNER = ""

FLOOR = 4
SKIP_NAMES = {"__init__", "__repr__", "__str__", "__eq__", "__hash__", "__post_init__"}
SKIP_FIELDS = {"annotation", "returns", "type_comment", "decorator_list", "type_params",
               "ctx", "kind"}
PLAIN_BODY = (ast.Assign, ast.AnnAssign, ast.Pass)


@dataclass(frozen=True)
class Entry:
    """One function in the index: where it is, what it is called, how big it is."""

    path: str
    line: int
    name: str
    statements: int

    @property
    def where(self) -> str:
        return f"{self.path}:{self.line}"


def describe() -> str:
    return "functions under src/ml_stack with the same normalised body as another"


def module_imports(tree: ast.Module) -> set[str]:
    """The names bound to a module object by `import x` anywhere in a module."""
    bound: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                bound.add(alias.asname or alias.name.split(".")[0])
    return bound


def _bound_names(fn: ast.AST) -> set[str]:
    """Every name a function binds: arguments, assignments, loops, with, except."""
    names: set[str] = set()
    for node in ast.walk(fn):
        if isinstance(node, ast.arg):
            names.add(node.arg)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            names.add(node.id)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            names.add(node.name)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            names.update(node.names)
    return names


class _Normalise:
    """Turns a function body into a token stream with the local vocabulary removed."""

    def __init__(self, local: set[str], imports: set[str], loose: bool) -> None:
        self.local, self.imports, self.loose = local, imports, loose
        self.slots: dict[str, int] = {}
        self.out: list[str] = []

    def slot(self, name: str) -> str:
        if self.loose:
            return "v"
        return f"v{self.slots.setdefault(name, len(self.slots))}"

    def name(self, name: str) -> str:
        if name in self.local:
            return self.slot(name)
        if name in self.imports:
            return f"m:{name}"
        return "g" if self.loose else f"g:{name}"

    def dotted(self, node: ast.Attribute) -> str | None:
        parts: list[str] = []
        cursor: ast.AST = node
        while isinstance(cursor, ast.Attribute):
            parts.append(cursor.attr)
            cursor = cursor.value
        if isinstance(cursor, ast.Name) and cursor.id in self.imports and cursor.id not in self.local:
            return "m:" + ".".join([cursor.id, *reversed(parts)])
        return None

    def emit(self, node: ast.AST) -> None:
        if isinstance(node, ast.Name):
            self.out.append(self.name(node.id))
        elif isinstance(node, ast.arg):
            self.out.append(self.slot(node.arg))
        elif isinstance(node, ast.Attribute):
            whole = self.dotted(node)
            if whole:
                self.out.append(whole)
            else:
                self.out.append(f"attr:{node.attr}")
                self.emit(node.value)
        elif isinstance(node, ast.Constant):
            self.out.append(_literal(node.value))
        elif self.loose and isinstance(node, (ast.Tuple, ast.List, ast.Set)) and node.elts \
                and all(isinstance(item, ast.Constant) for item in node.elts):
            self.out.append("c:seq")
        elif isinstance(node, ast.keyword):
            self.out.append(f"kw:{node.arg}")
            self.emit(node.value)
        elif isinstance(node, ast.alias):
            self.out.append(f"m:{node.name}")
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            self.out.append(node.__class__.__name__)
            self.generic(node)
        else:
            self.out.append(node.__class__.__name__)
            self.generic(node)

    def generic(self, node: ast.AST) -> None:
        for field, value in ast.iter_fields(node):
            if field in SKIP_FIELDS or field == "name" and isinstance(value, str):
                continue
            if isinstance(value, list):
                self.out.append(f"[{len(value)}]")
                for item in value:
                    if isinstance(item, ast.AST):
                        self.emit(item)
                    elif isinstance(item, str) and field == "names":
                        self.out.append("s")
            elif isinstance(value, ast.AST):
                self.emit(value)
            elif isinstance(value, str) and field in {"module", "attr"}:
                self.out.append(f"m:{value}" if field == "module" else f"attr:{value}")


def _literal(value: object) -> str:
    if value is None:
        return "c:none"
    if isinstance(value, bool):
        return "c:bool"
    if isinstance(value, str):
        return "c:str"
    if isinstance(value, bytes):
        return "c:bytes"
    if isinstance(value, int):
        return "c:int"
    if isinstance(value, float):
        return "c:float"
    return "c:other"


def _body(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> list[ast.stmt]:
    body = list(fn.body)
    if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant) \
            and isinstance(body[0].value.value, str):  # type: ignore[attr-defined]
        body = body[1:]
    return body


def statements(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> int:
    """How many statements a function's body holds, docstring excluded."""
    return sum(1 for stmt in _body(fn) for _ in ast.walk(stmt) if isinstance(_, ast.stmt))


def signature(fn: ast.FunctionDef | ast.AsyncFunctionDef, imports: set[str], *,
              loose: bool = True) -> str:
    """The hash of a function's normalised body."""
    walker = _Normalise(_bound_names(fn), imports, loose)
    args = fn.args
    walker.out.append(f"args:{len(args.posonlyargs) + len(args.args) + len(args.kwonlyargs)}")
    for stmt in _body(fn):
        walker.emit(stmt)
    return hashlib.sha256("\x1f".join(walker.out).encode()).hexdigest()[:16]


def _dull(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    body = _body(fn)
    if not body:
        return True
    return all(isinstance(stmt, PLAIN_BODY) for stmt in body)


def functions_in(text: str, path: str, *, floor: int = FLOOR,
                 loose: bool = True) -> list[tuple[Entry, str]]:
    """Every function in one module worth hashing, with its signature."""
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return []
    imports = module_imports(tree)
    out: list[tuple[Entry, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.name in SKIP_NAMES or _dull(node):
            continue
        size = statements(node)
        if size < floor:
            continue
        out.append((Entry(path, node.lineno, node.name, size),
                    signature(node, imports, loose=loose)))
    return out


def python_files(root: Path) -> list[Path]:
    """Every library module under a checkout, tests excluded."""
    src = root / "src" / "ml_stack"
    return sorted(p for p in src.rglob("*.py")
                  if not p.name.startswith("test_") and "tests" not in p.parts)


def index(root: Path, *, floor: int = FLOOR, loose: bool = True) -> dict[str, list[Entry]]:
    """`{signature: [entry, ...]}` for every function under src/ml_stack."""
    groups: dict[str, list[Entry]] = {}
    for path in python_files(root):
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        for entry, sig in functions_in(text, str(path.relative_to(root)), floor=floor, loose=loose):
            groups.setdefault(sig, []).append(entry)
    return groups


def find(root: Path, *, floor: int = FLOOR, loose: bool = True) -> list[Finding]:
    """One finding per function that shares its normalised body with another."""
    groups = [members for members in index(root, floor=floor, loose=loose).values()
              if len(members) > 1]
    groups.sort(key=lambda members: (-len(members), -members[0].statements,
                                     members[0].path, members[0].line))
    out: list[Finding] = []
    for members in groups:
        for entry in members:
            others = ", ".join(f"{other.name} at {other.where}"
                               for other in members if other is not entry)
            out.append(Finding(entry.path, entry.line,
                               f"{entry.name} ({entry.statements} statements) has the same body as {others}"))
    return out


def _main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=describe())
    parser.add_argument("root", nargs="?", default=".", type=Path)
    parser.add_argument("--floor", type=int, default=FLOOR)
    parser.add_argument("--strict", action="store_true", help="keep global names apart")
    parser.add_argument("--groups", type=int, default=0, help="show only the N largest groups")
    args = parser.parse_args()

    groups = [members for members in
              index(args.root, floor=args.floor, loose=not args.strict).values()
              if len(members) > 1]
    groups.sort(key=lambda members: (-len(members), -members[0].statements))
    print(f"{len(groups)} groups, {sum(len(m) for m in groups)} functions, floor {args.floor}")
    for members in groups[: args.groups or len(groups)]:
        print(f"  {len(members)}x {members[0].statements} stmts  "
              + ", ".join(f"{one.name} {one.where}" for one in members))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
