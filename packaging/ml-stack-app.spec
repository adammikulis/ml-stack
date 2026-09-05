# The app people double-click: a native window on the interface.
import importlib.util
import sys
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
    "ml_stack.fleet.chat", "ml_stack.fleet.conversations", "ml_stack.fleet.llama",
    "ml_stack.contracts", "ml_stack.client", "ml_stack.media",
    # Reached only through a lazy import, so nothing static points at it.
    "ml_stack.serve", "psutil",
]

app_hidden = hidden + ["webview", "webview.platforms.cocoa", "webview.platforms.winforms",
                       "webview.platforms.gtk", "webview.platforms.qt"]

a = Analysis(["launcher.py"], datas=datas, hiddenimports=app_hidden,
             excludes=["tkinter", "test", "unittest", "pydoc_data"])
pyz = PYZ(a.pure)

# Everywhere but macOS this is one file, sitting beside ml-stack-headless. A folder
# holding the executable with its runtime alongside is a folder someone drags the
# executable out of, and it then cannot find the Python it needs; and an installer
# copying "the binary" copies a directory, or nothing.
if sys.platform == "darwin":
    exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="ml-stack",
              console=False)
    coll = COLLECT(exe, a.binaries, a.datas, name="ml-stack")
    app = BUNDLE(
        coll,
        name="ml-stack.app",
        bundle_identifier="com.ml-stack.app",
        info_plist={
            "CFBundleName": "ml-stack",
            "CFBundleDisplayName": "ml-stack",
            "CFBundleShortVersionString": "0.1.8",  # x-release-please-version
            "LSMinimumSystemVersion": "12.0",
            "NSHighResolutionCapable": True,
        },
    )
else:
    exe = EXE(pyz, a.scripts, a.binaries, a.datas, [], name="ml-stack",
              console=False)
