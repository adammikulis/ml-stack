# Headless bundle: the daemon, and your browser for the interface.
import importlib.util
from pathlib import Path

def package_dir(name):
    return Path(importlib.util.find_spec(name).origin).parent

datas = [
    (str(package_dir("ml_stack.fleet") / "web"), "ml_stack/fleet/web"),
    (str(package_dir("ml_stack.contracts") / "_data"), "ml_stack/contracts/_data"),
]

# The commit this was built from, beside ml_stack.fleet.measuring, which answers it.
_built = Path("../.build-work/built-from")
if _built.is_file():
    datas.append((str(_built), "ml_stack/fleet"))

# The ml-stack wheels themselves, so the app can build a training environment on a
# machine that has never heard of this project.
_wheels = Path("../dist")
if _wheels.is_dir():
    datas += [(str(w), "wheels") for w in _wheels.glob("*.whl")]

hidden = [
    "ml_stack.fleet.daemon", "ml_stack.fleet.api", "ml_stack.fleet.jobs",
    "ml_stack.fleet.files", "ml_stack.fleet.device", "ml_stack.fleet.measuring",
    "ml_stack.fleet.peers", "ml_stack.fleet.launch",
    "ml_stack.fleet.ui", "ml_stack.fleet.autostart",
    "ml_stack.fleet.telemetry", "ml_stack.fleet.settings",
    "ml_stack.fleet.chat", "ml_stack.fleet.conversations", "ml_stack.fleet.llama",
    "ml_stack.contracts", "ml_stack.client", "ml_stack.media",
    # Reached only through a lazy import, so nothing static points at it.
    "ml_stack.serve", "psutil",
]

# fleet.ui's bench_state and bench_history import these to report a running measurement,
# inside a try/except that already reads "the bench is not installed here" -- true of a
# frozen daemon, whose bench runs as the detached ml-stack-bench process, never in this one.
# PyInstaller's static analysis cannot see that the import is optional, so without the
# exclude it bundles graph, world and ingest -- and the numpy they need -- for a path this
# binary never takes.
a = Analysis(["launcher-headless.py"], datas=datas, hiddenimports=hidden,
             excludes=["tkinter", "test", "unittest", "pydoc_data", "webview",
                       "numpy", "ml_stack.graph", "ml_stack.world", "ml_stack.bench",
                       "ml_stack.ingest"])
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, a.binaries, a.datas, [], name="ml-stack-headless",
          console=True, strip=False, upx=False)
