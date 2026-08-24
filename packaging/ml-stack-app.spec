# The app people double-click: a native window on the interface.
import importlib.util
from pathlib import Path

def package_dir(name):
    return Path(importlib.util.find_spec(name).origin).parent

datas = [
    (str(package_dir("ml_stack.fleet") / "web"), "ml_stack/fleet/web"),
    (str(package_dir("ml_stack.contracts") / "_data"), "ml_stack/contracts/_data"),
]

hidden = [
    "ml_stack.fleet.daemon", "ml_stack.fleet.peers", "ml_stack.fleet.launch",
    "ml_stack.fleet.app", "ml_stack.fleet.ui", "ml_stack.fleet.autostart",
    "ml_stack.fleet.telemetry", "ml_stack.fleet.settings",
    "ml_stack.contracts", "ml_stack.client", "ml_stack.media",
]

app_hidden = hidden + ["webview", "webview.platforms.cocoa", "webview.platforms.winforms",
                       "webview.platforms.gtk", "webview.platforms.qt"]

a = Analysis(["launcher.py"], datas=datas, hiddenimports=app_hidden,
             excludes=["tkinter", "test", "unittest", "pydoc_data"])
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
        "CFBundleShortVersionString": "0.1.1",
        "LSMinimumSystemVersion": "12.0",
        "NSHighResolutionCapable": True,
    },
)
