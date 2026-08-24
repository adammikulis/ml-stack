# PyInstaller spec: one self-contained ml-stack executable.
import importlib.util
from pathlib import Path

def package_dir(name):
    spec = importlib.util.find_spec(name)
    return Path(spec.origin).parent

datas = [
    (str(package_dir("ml_stack.fleet") / "web"), "ml_stack/fleet/web"),
    (str(package_dir("ml_stack.contracts") / "_data"), "ml_stack/contracts/_data"),
]

a = Analysis(
    ["launcher.py"],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=[
        "ml_stack.fleet.daemon", "ml_stack.fleet.peers", "ml_stack.fleet.launch",
        "ml_stack.fleet.ui", "ml_stack.fleet.autostart", "ml_stack.fleet.telemetry",
        "ml_stack.contracts", "ml_stack.client", "ml_stack.media",
    ],
    excludes=["tkinter", "test", "unittest", "pydoc_data"],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz, a.scripts, a.binaries, a.datas, [],
    name="ml-stack",
    console=True,
    strip=False,
    upx=False,
)
