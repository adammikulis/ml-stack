# Headless bundle: the daemon, and your browser for the interface.
import importlib.util
from pathlib import Path

def package_dir(name):
    return Path(importlib.util.find_spec(name).origin).parent

datas = [
    (str(package_dir("ml_stack.fleet") / "web"), "ml_stack/fleet/web"),
    (str(package_dir("ml_stack.contracts") / "_data"), "ml_stack/contracts/_data"),
]

# The ml-stack wheels themselves, so the app can build a training environment on a
# machine that has never heard of this project.
_wheels = Path("../dist")
if _wheels.is_dir():
    datas += [(str(w), "wheels") for w in _wheels.glob("*.whl")]

hidden = [
    "ml_stack.fleet.daemon", "ml_stack.fleet.peers", "ml_stack.fleet.launch",
    "ml_stack.fleet.app", "ml_stack.fleet.ui", "ml_stack.fleet.autostart",
    "ml_stack.fleet.telemetry", "ml_stack.fleet.settings",
    "ml_stack.contracts", "ml_stack.client", "ml_stack.media",
]

a = Analysis(["launcher-headless.py"], datas=datas, hiddenimports=hidden,
             excludes=["tkinter", "test", "unittest", "pydoc_data", "webview"])
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, a.binaries, a.datas, [], name="ml-stack-headless",
          console=True, strip=False, upx=False)
