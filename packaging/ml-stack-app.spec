# macOS .app bundle: double-clickable, no terminal.
import importlib.util
from pathlib import Path

def package_dir(name):
    return Path(importlib.util.find_spec(name).origin).parent

datas = [
    (str(package_dir("ml_stack.fleet") / "web"), "ml_stack/fleet/web"),
    (str(package_dir("ml_stack.contracts") / "_data"), "ml_stack/contracts/_data"),
]

a = Analysis(
    ["launcher.py"],
    datas=datas,
    hiddenimports=[
        "ml_stack.fleet.daemon", "ml_stack.fleet.peers", "ml_stack.fleet.launch",
        "ml_stack.fleet.ui", "ml_stack.fleet.autostart", "ml_stack.fleet.telemetry",
        "ml_stack.contracts", "ml_stack.client", "ml_stack.media",
    ],
    excludes=["tkinter", "test", "unittest", "pydoc_data"],
)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="ml-stack", console=False)
coll = COLLECT(exe, a.binaries, a.datas, name="ml-stack")
app = BUNDLE(
    coll,
    name="ml-stack.app",
    bundle_identifier="com.ml-stack.app",
    info_plist={
        "CFBundleName": "ml-stack",
        "CFBundleDisplayName": "ml-stack",
        "CFBundleShortVersionString": "0.1.0",
        "LSMinimumSystemVersion": "12.0",
        # No Dock icon or menu bar: this is a daemon with a web interface, and an app
        # that steals focus every login is one people quit.
        "LSUIElement": True,
        "NSHighResolutionCapable": True,
    },
)
