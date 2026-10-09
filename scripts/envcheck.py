"""What this checkout needs from the machine for a stated purpose, and the one command that closes the gap.

Standard library only, because it runs before anything is installed. `scripts/preflight` prints
it; `scripts/test` runs it first. A purpose is a name from PURPOSES; docs/environment.md is the
table of what each one needs and why.

The Python packages a purpose needs are not a list kept here. They are the module-level imports
reachable from the files that purpose runs (a conftest that imports numpy at collection needs
numpy whether or not an extra says so), resolved against what is importable now.
"""

from __future__ import annotations

import ast
import contextlib
import hashlib
import importlib.util
import json
import re
import shutil
import subprocess
import sys
import tempfile
import tomllib
from functools import cache
from importlib import metadata
from pathlib import Path
from typing import NamedTuple

ROOT = Path(__file__).resolve().parent.parent
PURPOSES = ("runtime", "dev", "gate", "build")
ENTRY_FILES = {
    "dev": ("tests/conftest.py", "scripts/test"),
    "gate": ("scripts/budgets", "scripts/redteam_coverage.py", "scripts/reference"),
    "build": ("packaging/build.py",),
}
"""The files a purpose runs; their reachable module-level imports are what it needs installed."""
DIST_FOR = {"yaml": "pyyaml", "PIL": "pillow", "cv2": "opencv-python", "sklearn": "scikit-learn"}
"""Import name -> distribution name where they differ and the package is not installed to ask."""
STDLIB = frozenset(sys.stdlib_module_names)
CONDITIONAL = re.compile(r"TYPE_CHECKING|sys\.platform|os\.name|platform\.system")
"""An ``if`` on one of these guards an import that is not needed everywhere."""
NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


class Gap(NamedTuple):
    """One thing missing: what it is, the pip requirement that supplies it (empty when pip cannot), and the other fix."""

    what: str
    pip: str = ""
    fix: str = ""


def canonical(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def pyproject(root: Path = ROOT) -> dict:
    return tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))


def local_module(root: Path, here: Path, dotted: str) -> Path | None:
    """The file in this checkout that ``import dotted`` binds: under src/, or beside the importing file."""
    parts = dotted.split(".")
    for base in (root / "src", here.parent, root / "scripts"):
        for candidate in (base.joinpath(*parts).with_suffix(".py"), base.joinpath(*parts, "__init__.py"), base.joinpath(*parts)):
            if candidate.is_file():
                return candidate
    return None


def imports_of(path: Path) -> list[str]:
    """Modules imported at module level, ``try`` bodies and function bodies left out (those are optional by construction)."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError, UnicodeDecodeError):
        return []
    out: list[str] = []
    todo = list(tree.body)
    while todo:
        node = todo.pop()
        if isinstance(node, ast.Import):
            out += [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            out += [node.module, *(f"{node.module}.{alias.name}" for alias in node.names)]
        elif isinstance(node, ast.If) and not CONDITIONAL.search(ast.unparse(node.test)):
            todo += node.body + node.orelse
    return out


def walk(entries: tuple[str, ...], root: Path) -> tuple[dict[str, str], set[Path]]:
    """The third-party names the files in ``entries`` reach (name -> a file importing it) and every file visited."""
    seen: set[Path] = set()
    found: dict[str, str] = {}
    todo = [root / entry for entry in entries]
    while todo:
        path = todo.pop()
        if path in seen or not path.is_file():
            continue
        seen.add(path)
        for dotted in imports_of(path):
            top = dotted.split(".")[0]
            local = local_module(root, path, dotted)
            if local is not None:
                todo += [local, *(p for p in (local.parent / "__init__.py",) if p.is_file())]
            elif top not in STDLIB and local_module(root, path, top) is None:
                found.setdefault(top, str(path.relative_to(root)))
    return found, seen


def stamps(files: set[Path]) -> dict[str, list[int]]:
    return {str(p): [p.stat().st_mtime_ns, p.stat().st_size] for p in files if p.is_file()}


def third_party(entries: tuple[str, ...], root: Path = ROOT) -> dict[str, str]:
    """`walk`'s names, remembered in the temp directory until a file it read changes (the walk parses ~600 files)."""
    key = hashlib.sha256(f"{root}\0{entries}".encode()).hexdigest()[:16]
    memo = Path(tempfile.gettempdir()) / f"ml-stack-preflight-{key}.json"
    try:
        held = json.loads(memo.read_text(encoding="utf-8"))
        if held["files"] == stamps({Path(p) for p in held["files"]}):
            return held["found"]
    except (OSError, ValueError, KeyError):
        pass
    found, seen = walk(entries, root)
    with contextlib.suppress(OSError):
        memo.write_text(json.dumps({"found": found, "files": stamps(seen)}), encoding="utf-8")
    return found


def importable(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


@cache
def owners() -> dict[str, list[str]]:
    """Module -> the distributions that provide it, for what is installed here (about half a second to build)."""
    return dict(metadata.packages_distributions())


def distribution_of(module: str) -> str:
    """The distribution that provides ``module`` here, or the usual name for it when it is not installed."""
    found = owners().get(module)
    return found[0] if found else DIST_FOR.get(module, module)


def parsed(text: str) -> tuple[str, str] | None:
    """(name, version specifier) of a PEP 508 requirement that applies here, None when its marker does not hold.

    Without `packaging` (it is a dependency of the library, so a bare interpreter may lack it) the
    specifier is not checked and a requirement with a marker or an extra is taken to apply.
    """
    try:
        from packaging.requirements import Requirement
    except ImportError:
        match = NAME.match(text)
        return (match.group(0), "") if match and ";" not in text else None
    req = Requirement(text)
    return (req.name, str(req.specifier)) if req.marker is None or req.marker.evaluate() else None


def requirement_gaps(requirements: list[str]) -> list[Gap]:
    """The requirements (PEP 508 strings) that are absent or too old here; markers that do not hold here are skipped."""
    gaps = []
    for text in requirements:
        found = parsed(text)
        if found is None:
            continue
        name, spec = found
        try:
            have = metadata.version(name)
        except metadata.PackageNotFoundError:
            gaps.append(Gap(f"{name} is not installed", f"{name}{spec}"))
            continue
        if spec and not fits(have, spec):
            gaps.append(Gap(f"{name} {have} is installed and {spec} is wanted", f"{name}{spec}"))
    return gaps


def fits(version: str, spec: str) -> bool:
    """Whether ``version`` satisfies ``spec``; True when `packaging` is not there to say."""
    try:
        from packaging.specifiers import SpecifierSet
    except ImportError:
        return True
    return SpecifierSet(spec).contains(version, prereleases=True)


def extra_requirements(extra: str, root: Path = ROOT) -> list[str]:
    """The requirement strings of an optional-dependency group, the groups it names inside ``ml-stack[...]`` included."""
    extras = pyproject(root)["project"]["optional-dependencies"]
    out: list[str] = []
    for text in extras.get(extra, []):
        inner = re.fullmatch(r"ml-stack\[(.+)\]", text)
        out += [r for e in inner.group(1).split(",") for r in extra_requirements(e.strip(), root)] if inner else [text]
    return out


def python_gaps(root: Path = ROOT) -> list[Gap]:
    want = pyproject(root)["project"]["requires-python"]
    floor = tuple(int(n) for n in re.sub(r"[^0-9.]", "", want).split("."))
    if sys.version_info[:len(floor)] >= floor:
        return []
    return [Gap(f"Python {sys.version_info[0]}.{sys.version_info[1]} runs this and the project wants {want}",
                fix="pyenv install 3.13 && pyenv local 3.13   (or any interpreter >= 3.12)")]


def tool_gaps(names: tuple[str, ...]) -> list[Gap]:
    install = {"git": "install git (xcode-select --install, or apt-get install git)",
               "cargo": "install Rust with rustup: https://rustup.rs (the app needs the version app/Cargo.toml names)",
               "node": "install Node (brew install node, or pip install nodejs-wheel)"}
    return [Gap(f"{n} is not on PATH", fix=install[n]) for n in names if shutil.which(n) is None]


def pinned_tool_gaps(root: Path = ROOT) -> list[Gap]:
    """ruff and pyright at the versions the budgets were counted with: the pins file names them."""
    pins = [line.split("#", 1)[0].strip() for line in (root / "scripts/gates/pinned.txt").read_text(encoding="utf-8").splitlines()]
    return [Gap(g.what, g.pip) for g in requirement_gaps([p for p in pins if p])]


def import_gaps(purpose: str, root: Path = ROOT) -> list[Gap]:
    """Third-party modules the files this purpose runs import at module level and that cannot be imported here."""
    wanted = third_party(ENTRY_FILES.get(purpose, ()), root)
    return [Gap(f"{name} (imported by {by}) is not importable", distribution_of(name)) for name, by in sorted(wanted.items())
            if not importable(name)]


def rust_gaps(root: Path = ROOT) -> list[Gap]:
    """cargo at or above the rust-version the workspace declares."""
    if shutil.which("cargo") is None:
        return tool_gaps(("cargo",))
    text = (root / "app" / "Cargo.toml").read_text(encoding="utf-8")
    floor = re.search(r'rust-version\s*=\s*"([\d.]+)"', text)
    said = subprocess.run(["cargo", "--version"], capture_output=True, text=True, check=False).stdout
    have = re.search(r"(\d+)\.(\d+)", said)
    if floor and have and tuple(int(n) for n in have.groups()) < tuple(int(n) for n in floor.group(1).split(".")[:2]):
        return [Gap(f"cargo {have.group(0)} is older than the rust-version {floor.group(1)} app/Cargo.toml declares", fix="rustup update stable")]
    return []


def gaps(purpose: str, root: Path = ROOT) -> list[Gap]:
    """Everything ``purpose`` needs that is missing, in the order to fix it."""
    if purpose not in PURPOSES:
        raise ValueError(f"unknown purpose {purpose!r}; one of {', '.join(PURPOSES)}")
    out = python_gaps(root)
    if purpose != "gate":  # the gates job installs the checkers' own imports and not the library
        out += requirement_gaps(pyproject(root)["project"]["dependencies"])
    if purpose == "dev":
        out += requirement_gaps(extra_requirements("test", root)) + import_gaps(purpose, root) + tool_gaps(("git",))
    elif purpose == "gate":
        out += import_gaps(purpose, root) + pinned_tool_gaps(root) + tool_gaps(("git",))
    elif purpose == "build":
        out += requirement_gaps(["build", "hatchling"]) + import_gaps(purpose, root) + rust_gaps(root)
    return out


def command(found: list[Gap]) -> str:
    """The lines that fix the gaps: one pip install for everything pip supplies, then the other fixes."""
    best: dict[str, str] = {}
    for g in found:
        if g.pip:
            name = canonical(NAME.match(g.pip).group(0))
            best[name] = max(best.get(name, ""), g.pip, key=len)
    pips = sorted(best.values())
    lines = [f"{sys.executable} -m pip install " + " ".join(f"'{p}'" for p in pips)] if pips else []
    return "\n".join(lines + sorted({g.fix for g in found if g.fix}))


def report(purposes: list[str], root: Path = ROOT) -> tuple[int, str]:
    """(exit status, text): 0 and one line when nothing is missing, else 1 and each gap with the fix."""
    found = [g for p in purposes for g in gaps(p, root)]
    if not found:
        return 0, f"preflight: {' + '.join(purposes)} ready ({sys.executable}, Python {sys.version_info[0]}.{sys.version_info[1]})"
    seen, rows = set(), []
    for g in found:
        if g.what not in seen:
            seen.add(g.what)
            rows.append(f"  {g.what}")
    return 1, f"preflight: {' + '.join(purposes)} is not ready:\n" + "\n".join(rows) + "\nfix:\n" + "\n".join(
        f"  {line}" for line in command(found).splitlines())
