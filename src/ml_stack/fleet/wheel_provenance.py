"""Wheel revision markers and integrity manifests."""

import base64
import csv
import hashlib
import io
import zipfile
from pathlib import Path

ORIGIN = "source-checkout"


def wheel_commit(wheel: Path) -> str:
    """Return the wheel's bounded commit marker, or empty for an unstamped wheel."""
    with zipfile.ZipFile(wheel) as archive:
        name = "ml_stack/fleet/built-from"
        try:
            info = archive.getinfo(name)
        except KeyError:
            return ""
        if info.file_size > 100:
            raise ValueError("runtime wheel commit marker is too large")
        return archive.read(name).decode().strip()


def stamp(wheel: Path, commit: str, checkout: Path) -> None:
    """Record runtime provenance and update the wheel's integrity manifest."""
    with zipfile.ZipFile(wheel) as archive:
        contents = {name: archive.read(name) for name in archive.namelist()}
    records = [name for name in contents if name.endswith(".dist-info/RECORD")]
    if len(records) != 1:
        raise ValueError("wheel must contain one RECORD")
    record = records[0]
    contents["ml_stack/fleet/built-from"] = (commit + "\n").encode()
    contents[f"ml_stack/fleet/{ORIGIN}"] = (str(checkout.resolve()) + "\n").encode()
    manifest = io.StringIO(newline="")
    writer = csv.writer(manifest)
    for name, data in contents.items():
        if name != record:
            digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
            writer.writerow((name, f"sha256={digest}", str(len(data))))
    writer.writerow((record, "", ""))
    contents[record] = manifest.getvalue().encode()
    with zipfile.ZipFile(wheel, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in contents.items():
            archive.writestr(name, data)
