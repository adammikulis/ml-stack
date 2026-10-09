"""No test reaches a paid API or a public endpoint unless a person switched it on.

A ``live_api`` or ``live_net`` mark is the only way in, the mark is skipped without its
``POOLHOUSE_*`` switch, a credential in the environment switches nothing on, and every other
test is refused a connection beyond this machine and its LAN.
"""

from __future__ import annotations

import ast
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from poolhouse.testing import live

TESTS = Path(__file__).resolve().parent
MARKS = {"live_api", "live_net"}


def marked(node: ast.AST) -> bool:
    """Whether a def, class or module-level ``pytestmark`` carries a live mark."""
    decorators = getattr(node, "decorator_list", [])
    return any((isinstance(d, ast.Attribute) and d.attr in MARKS)
               or (isinstance(d, ast.Call) and isinstance(d.func, ast.Attribute)
               and d.func.attr in MARKS) for d in decorators)


def environment_read(node: ast.AST) -> str:
    """The credential name an expression reads from ``os.environ``, empty when it reads none."""
    key = None
    if isinstance(node, ast.Subscript) and ast.unparse(node.value) == "os.environ":
        key = node.slice
    elif isinstance(node, ast.Call) and ast.unparse(node.func) in ("os.environ.get", "os.getenv"):
        key = node.args[0] if node.args else None
    elif isinstance(node, ast.Compare) and isinstance(node.ops[0], (ast.In, ast.NotIn)) \
            and ast.unparse(node.comparators[0]) == "os.environ":
        key = node.left
    if isinstance(key, ast.Constant) and key.value in live.CREDENTIALS:
        return str(key.value)
    return ""


def sdk_import(node: ast.AST) -> str:
    """The paid SDK a statement or ``importorskip`` call imports, empty when it imports none."""
    names: list[str] = []
    if isinstance(node, ast.Import):
        names = [a.name for a in node.names]
    elif isinstance(node, ast.ImportFrom) and node.module:
        names = [node.module]
    elif isinstance(node, ast.Call) and ast.unparse(node.func).endswith(
            ("importorskip", "import_module", "__import__")) and node.args \
            and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
        names = [node.args[0].value]
    for name in names:
        for sdk in live.PAID_SDKS:
            if name == sdk or name.startswith(sdk + "."):
                return sdk
    return ""


def spawns_claude(node: ast.AST) -> bool:
    """Whether a call starts the ``claude`` command line, or looks it up on PATH."""
    if not isinstance(node, ast.Call):
        return False
    first = node.args[0] if node.args else None
    if ast.unparse(node.func) == "shutil.which":
        return isinstance(first, ast.Constant) and first.value == "claude"
    if ast.unparse(node.func).startswith("subprocess.") and isinstance(first, ast.List):
        head = first.elts[0] if first.elts else None
        return isinstance(head, ast.Constant) and head.value == "claude"
    return False


def findings(source: str) -> list[str]:
    """Every live call in a test module that no ``live_api``/``live_net`` mark covers."""
    tree = ast.parse(source)
    module_marked = any(
        isinstance(n, ast.Assign) and any(getattr(t, "id", "") == "pytestmark" for t in n.targets)
        and any(m in ast.unparse(n.value) for m in MARKS) for n in tree.body)
    out: list[str] = []

    def walk(node: ast.AST, covered: bool) -> None:
        covered = covered or marked(node)
        for found in (sdk_import(node), environment_read(node)):
            if found and not covered:
                out.append(f"line {getattr(node, 'lineno', 0)}: {found}")
        if spawns_claude(node) and not covered:
            out.append(f"line {getattr(node, 'lineno', 0)}: the claude command")
        for child in ast.iter_child_nodes(node):
            walk(child, covered)

    walk(tree, module_marked)
    return out


@pytest.mark.parametrize("snippet", [
    "import anthropic\n",
    "from openai import OpenAI\n",
    "import pytest\nanthropic = pytest.importorskip('anthropic')\n",
    "import os\nKEY = os.environ.get('ANTHROPIC_API_KEY')\n",
    "import os\nKEY = os.environ['OPENAI_API_KEY']\n",
    "import os\nHAVE = 'HF_TOKEN' in os.environ\n",
    "import os\nKEY = os.getenv('GEMINI_API_KEY')\n",
    "import subprocess\nsubprocess.run(['claude', '-p', 'hi'])\n",
    "import shutil\nCLAUDE = shutil.which('claude')\n",
])
def test_a_live_call_with_no_mark_is_found(snippet: str) -> None:
    assert findings(snippet)


@pytest.mark.parametrize("snippet", [
    "import pytest\n\n@pytest.mark.live_api\ndef test_x():\n    import anthropic\n",
    "import pytest\npytestmark = pytest.mark.live_api\nimport openai\n",
    "import os\n\ndef test_x(monkeypatch):\n    monkeypatch.setenv('ANTHROPIC_API_KEY', 'sk')\n",
    "def test_x(monkeypatch):\n    monkeypatch.delenv('OPENAI_API_KEY', raising=False)\n",
    "import os\nBASE = {'ANTHROPIC_API_KEY': 'sk-real', 'HOME': '/h'}\n",
    "import subprocess\nsubprocess.run(['git', 'status'])\n",
])
def test_a_marked_call_or_one_that_reads_nothing_is_not_found(snippet: str) -> None:
    assert findings(snippet) == []


def test_no_test_calls_a_paid_sdk_or_reads_a_credential_without_a_mark() -> None:
    bad = {path.name: found for path in sorted(TESTS.rglob("*.py"))
           if path.name != Path(__file__).name
           if (found := findings(path.read_text(encoding="utf-8")))}
    assert not bad, f"unmarked live calls (mark them live_api or live_net): {bad}"


def run_marked(tmp_path: Path, env: dict[str, str], mark: str) -> subprocess.CompletedProcess[str]:
    """Run a one-test module carrying ``mark`` under this suite's own conftest."""
    (tmp_path / "test_one.py").write_text(textwrap.dedent(f"""
        import pytest

        @pytest.mark.{mark}
        def test_it():
            print("RAN-LIVE")
        """), encoding="utf-8")
    return run_in(tmp_path, env)


def run_in(tmp_path: Path, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    """Run the modules in ``tmp_path`` under a copy of this suite's conftest."""
    conftest = (TESTS / "conftest.py").read_text(encoding="utf-8")
    here = "REPO = Path(__file__).resolve().parent.parent"
    assert here in conftest, "the copied conftest must be told where the repository is"
    conftest = conftest.replace(here, f"REPO = Path({str(TESTS.parent)!r})", 1)  # the copy sits in a temp directory
    (tmp_path / "conftest.py").write_text(conftest, encoding="utf-8")
    (tmp_path / "heavy-modules.txt").write_text("", encoding="utf-8")
    return subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-s", "-n", "0", "-p", "no:cacheprovider",
         "--rootdir", str(tmp_path), "-c", str(TESTS.parent / "pyproject.toml"), str(tmp_path)],
        capture_output=True, text=True, cwd=tmp_path, check=False,
        env={"PATH": "/usr/bin:/bin", "HOME": str(tmp_path),
             "PYTHONPATH": os.pathsep.join([str(TESTS.parent / "src"), str(TESTS.parent / "scripts"),
                                           str(TESTS.parent)]),
             **env})


@pytest.mark.parametrize("mark", ["live_api", "live_net"])
def test_a_credential_alone_does_not_run_a_live_test(tmp_path, mark) -> None:
    keys = dict.fromkeys(live.CREDENTIALS, "sk-present")
    done = run_marked(tmp_path, keys, mark)
    assert "RAN-LIVE" not in done.stdout and "1 skipped" in done.stdout, done.stdout + done.stderr


@pytest.mark.parametrize(("mark", "switch"), [("live_api", live.LIVE_API), ("live_net", live.LIVE_NET)])
def test_the_switch_runs_a_live_test_and_the_other_switch_does_not(tmp_path, mark, switch) -> None:
    other = live.LIVE_NET if switch == live.LIVE_API else live.LIVE_API
    assert "RAN-LIVE" in run_marked(tmp_path, {switch: "1"}, mark).stdout
    assert "RAN-LIVE" not in run_marked(tmp_path, {other: "1"}, mark).stdout


def test_a_test_that_reaches_a_public_host_is_refused_and_fails(tmp_path) -> None:
    (tmp_path / "test_one.py").write_text(textwrap.dedent("""
        import socket

        def test_it():
            socket.getaddrinfo("localhost", 80)
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as udp:
                udp.connect(("8.8.8.8", 53))
                print("UDP-CONNECT-SENDS-NOTHING")
            for call in (lambda: socket.getaddrinfo("api.anthropic.com", 443),
                         lambda: socket.create_connection(("8.8.8.8", 443), timeout=1),
                         lambda: socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                         .sendto(b"x", ("8.8.8.8", 53))):
                try:
                    call()
                except OSError as exc:
                    print("REFUSED", exc)
        """), encoding="utf-8")
    done = run_in(tmp_path, {})
    assert "UDP-CONNECT-SENDS-NOTHING" in done.stdout
    assert done.stdout.count("REFUSED a test reached") == 3, done.stdout + done.stderr
    assert done.returncode != 0 and "a real remote host was reached" in done.stdout


@pytest.mark.parametrize(("host", "beyond"), [
    ("api.anthropic.com", True), ("huggingface.co", True), ("8.8.8.8", True), ("2001:4860::1", True),
    ("localhost", False), ("127.0.0.1", False), ("::1", False), ("10.1.2.3", False),
    ("192.168.0.9", False), ("169.254.1.1", False), ("224.0.0.251", False),
    ("box.local", False), ("quenlow.example", False), ("", False),
])
def test_which_hosts_are_beyond_this_machine(host: str, beyond: bool) -> None:
    assert live.outside(host) is beyond
