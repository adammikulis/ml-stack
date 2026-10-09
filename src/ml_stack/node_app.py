"""Poolhouse.app: the stable macOS identity the LAN-enabled node runs under.

macOS ties Local Network permission to an app's bundle id and code signature. A bare cargo-built binary has neither, so every
build of it asks the person again. The bundle built here holds a copy of the verified node binary as its executable, with a
fixed bundle id and a signature whose designated requirement names that id and the signing certificate (`node_signing`), never
the binary's hash, so rebuilding the binary keeps the permission the person gave.

The bundle lives under `~/.ml-stack/apps/`, outside the repository. `BUNDLE_ID` is never changed: a new id is a new app to macOS
and loses every grant.

    ml-stack node build [--binary PATH] [--icon SVG]     (ml_stack.node_permission)
"""

from __future__ import annotations

import plistlib
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from ml_stack import node_binary, node_signing, runtime
from ml_stack.files import read_json, write_json
from ml_stack.home import home
from ml_stack.log import say, warn

BUNDLE_ID = "app.poolhouse.node"
NAME = "Poolhouse"
EXECUTABLE = "poolside-node"
USAGE = "Poolhouse finds and pairs with your other devices on this network."
BONJOUR = ["_poolhouse._tcp"]
SOURCE_RECORD = "source.json"
MINIMUM_SYSTEM = "13.0"
ICON_SVG = Path(__file__).parent / "fleet" / "web" / "poolhouse.svg"
ICON_SIZES = (16, 32, 128, 256, 512)
TOOL_TIMEOUT = 60.0


class AppError(OSError):
    """The bundle could not be built or signed."""


def apps_directory() -> Path:
    """Where the bundles live: outside every checkout."""
    return home() / "apps"


def bundle_path() -> Path:
    """The Poolhouse.app bundle."""
    return apps_directory() / f"{NAME}.app"


def executable_path(bundle: Path | None = None) -> Path:
    """The executable inside a bundle: the node binary, signed under the bundle's identity."""
    return (bundle or bundle_path()) / "Contents" / "MacOS" / EXECUTABLE


def info_plist(version: str = "0", *, icon: bool = True) -> dict:
    """The bundle's Info.plist as a dict: fixed id, no dock icon, and the keys macOS reads to show the Local Network prompt."""
    plist = {
        "CFBundleIdentifier": BUNDLE_ID, "CFBundleName": NAME, "CFBundleDisplayName": NAME, "CFBundleExecutable": EXECUTABLE,
        "CFBundlePackageType": "APPL", "CFBundleVersion": version, "CFBundleShortVersionString": version,
        "LSUIElement": True, "LSMinimumSystemVersion": MINIMUM_SYSTEM, "NSLocalNetworkUsageDescription": USAGE,
        "NSBonjourServices": list(BONJOUR),
    }
    if icon:
        plist["CFBundleIconFile"] = NAME
    return plist


def requirement(certificate_sha1: str) -> str:
    """The designated requirement: this bundle id, signed by this certificate. It names no cdhash, so a rebuilt binary still satisfies it."""
    return f'designated => identifier "{BUNDLE_ID}" and certificate leaf = H"{certificate_sha1.lower()}"'


def ad_hoc_requirement() -> str:
    """The designated requirement of the fallback signature, which has no certificate: the bundle id alone."""
    return f'designated => identifier "{BUNDLE_ID}"'


def launch_command(bundle: Path, args: list[str]) -> list[str]:
    """The command line that runs the node through the bundle's executable."""
    return [str(executable_path(bundle)), *args]


def uses_bundle(args: list[str]) -> bool:
    """Whether a node start with these arguments goes through Poolhouse.app: on macOS, when the network is on (`--lan`)."""
    return sys.platform == "darwin" and "--lan" in args


def _tool(argv: list[str]) -> None:
    done = subprocess.run(argv, capture_output=True, text=True, timeout=TOOL_TIMEOUT, check=False)
    if done.returncode:
        raise AppError(f"{argv[0]} failed: {(done.stderr or done.stdout).strip()[-400:]}")


def _icns(svg: Path, into: Path) -> bool:
    """Write Poolhouse.icns from the SVG with macOS's own renderer; False when it cannot (the bundle then has no icon)."""
    with tempfile.TemporaryDirectory(prefix="poolhouse-icon") as scratch:
        work = Path(scratch)
        try:
            _tool(["qlmanage", "-t", "-s", "1024", "-o", str(work), str(svg)])
            iconset = work / "Poolhouse.iconset"
            iconset.mkdir()
            for size in ICON_SIZES:
                for scale in (1, 2):
                    _tool(["sips", "-z", str(size * scale), str(size * scale), str(work / f"{svg.name}.png"), "--out",
                           str(iconset / f"icon_{size}x{size}{'@2x' if scale == 2 else ''}.png")])
            _tool(["iconutil", "-c", "icns", str(iconset), "-o", str(into)])
        except (AppError, OSError, subprocess.SubprocessError):
            return False
    return True


def current_source(bundle: Path | None = None) -> str:
    """The checksum of the node binary the bundle was last built from, or ''."""
    return str(read_json((bundle or bundle_path()) / "Contents" / "Resources" / SOURCE_RECORD, {}).get("sha256", ""))


def build(binary: Path, *, icon: Path | None = None, bundle: Path | None = None) -> dict:
    """Make the bundle around a node binary and sign it; returns {bundle, sha256, signing}. ``signing`` is "certificate" or "ad-hoc".

    The bundle is made and signed beside its place and copied into place only once signed, so a failure leaves the previous one standing.
    """
    bundle = bundle or bundle_path()
    staging = bundle.parent / ".building" / bundle.name
    shutil.rmtree(staging.parent, ignore_errors=True)
    sha = _assemble(staging, binary, icon or (ICON_SVG if ICON_SVG.is_file() else None))
    try:
        signing = node_signing.sign(staging, BUNDLE_ID, requirement, ad_hoc_requirement())
        shutil.rmtree(bundle, ignore_errors=True)
        shutil.copytree(staging, bundle, symlinks=True)
    finally:
        shutil.rmtree(staging.parent, ignore_errors=True)
    return {"bundle": str(bundle), "sha256": sha, "signing": signing}


def _assemble(bundle: Path, binary: Path, icon: Path | None) -> str:
    """Lay out the unsigned bundle; returns the checksum of the binary it holds."""
    executable = executable_path(bundle)
    executable.parent.mkdir(parents=True)
    shutil.copyfile(binary, executable)
    executable.chmod(0o755)
    resources = bundle / "Contents" / "Resources"
    resources.mkdir()
    has_icon = icon is not None and _icns(icon, resources / f"{NAME}.icns")
    version = str(read_json(binary.parent / node_binary.RECORD, {}).get("commit", "0"))[:12] or "0"
    (bundle / "Contents" / "Info.plist").write_bytes(plistlib.dumps(info_plist(version, icon=has_icon)))
    sha = node_binary.sha256(binary)
    write_json(resources / SOURCE_RECORD, {"sha256": sha})
    return sha


def bundled(binary: Path, sha: str) -> Path:
    """The bundle executable that stands for a verified node binary, building (and signing) the bundle when it is not that binary's."""
    if current_source() != sha or not executable_path().is_file():
        say(f"node: building {bundle_path()} from {sha[:12]}")
        build(binary)
    return executable_path()


def _selected_binary() -> Path:
    prefix = runtime.selection_prefix()
    if prefix is None:
        raise AppError("no runtime is selected, so there is no node binary to put in the bundle")
    return node_binary.verified(prefix)


def build_command(args) -> int:
    try:
        result = build(Path(args.binary) if args.binary else _selected_binary(), icon=Path(args.icon) if args.icon else None)
    except OSError as exc:
        warn(f"node app: {exc}")
        return 1
    say(f"{result['bundle']} signed ({result['signing']}), from node {result['sha256'][:12]}")
    return 0
