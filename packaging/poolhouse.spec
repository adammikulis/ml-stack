# Headless bundle: the daemon, and your browser for the interface.
import importlib.util
import runpy
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, collect_submodules, copy_metadata


def package_dir(name):
    return Path(importlib.util.find_spec(name).origin).parent

datas = [
    (str(package_dir("poolhouse.fleet") / "web"), "poolhouse/fleet/web"),
    (str(package_dir("poolhouse.ui") / "assets"), "poolhouse/ui/assets"),
    (str(package_dir("poolhouse.contracts") / "_data"), "poolhouse/contracts/_data"),
    *copy_metadata("poolhouse"),
    *copy_metadata("openai-agents", recursive=True),
    *collect_data_files("agents"),
]

# The commit this was built from, beside poolhouse.fleet.measuring, which answers it.
_built = Path("../.build-work/built-from")
if _built.is_file():
    datas.append((str(_built), "poolhouse/fleet"))

# The poolhouse wheels themselves, so the app can build a training environment on a
# machine that has never heard of this project.
_wheels = Path("../dist")
if _wheels.is_dir():
    datas += [(str(w), "wheels") for w in _wheels.glob("*.whl")]

hidden = [
    "poolhouse.fleet.daemon", "poolhouse.fleet.api", "poolhouse.fleet.jobs",
    "poolhouse.fleet.files", "poolhouse.fleet.device", "poolhouse.fleet.measuring",
    "poolhouse.fleet.peers", "poolhouse.fleet.launch",
    "poolhouse.fleet.ui", "poolhouse.fleet.autostart",
    "poolhouse.fleet.telemetry", "poolhouse.fleet.settings",
    "poolhouse.agent.sdk_runtime", "poolhouse.agent.conversation",
    "poolhouse.fleet.sdk_chat", "poolhouse.client.sdk", "agents", "openai", "httpx",
    "poolhouse.fleet.chat", "poolhouse.fleet.conversations", "poolhouse.fleet.llama",
    "poolhouse.contracts", "poolhouse.client", "poolhouse.media",
    # Reached only through a lazy import, so nothing static points at it.
    "poolhouse.serve", "psutil", "ladybug._lbug", "ladybug._lbug_capi",
    "poolhouse.graph.store", "poolhouse.graph.cypher", "machineid",
    "poolhouse.workspace.coordinator", "poolhouse.workspace.coordinator_routes", "poolhouse.workspace.fleet_routes", "poolhouse.workspace.coding_routes", "poolhouse.mcp",
    "poolhouse.activity.fleet_routes", "poolhouse.workspace.work_reputation",
    # keyring picks its backend by importing these at run time. The PyInstaller hook collects them
    # today (checked in a frozen build); naming them keeps a change to that hook from silently
    # taking the keystore out of the app.
    "keyring.backends.macOS", "keyring.backends.Windows", "keyring.backends.SecretService",
]

hidden += collect_submodules("agents")
project_data, project_modules = runpy.run_path(str(Path(SPECPATH) / "frozen.py"))["collect_project"]()
datas += project_data
hidden += project_modules

a = Analysis(["launcher-headless.py"], datas=datas, hiddenimports=hidden,
             binaries=collect_dynamic_libs("ladybug"),
             excludes=["tkinter", "test", "unittest", "pydoc_data", "webview",
                       "numpy", "torch", "torch_geometric", "pandas", "polars", "pyarrow"])
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, a.binaries, a.datas, [], name="poolhouse-headless",
          console=True, strip=False, upx=False)
