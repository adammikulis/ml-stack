"""Zip a built bundle. Used by the release workflow; works on every platform."""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path


def pack(source: Path, out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(source.rglob("*")):
            if path.is_dir():
                continue
            arc = path.relative_to(source)
            info = zipfile.ZipInfo.from_file(path, arc)
            # Carry the executable bit; a zip that loses it produces a download
            # nobody can run.
            info.external_attr = (path.stat().st_mode & 0xFFFF) << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            with path.open("rb") as fh:
                zf.writestr(info, fh.read())
    return out


if __name__ == "__main__":
    made = pack(Path(sys.argv[1]), Path(sys.argv[2]))
    print(f"{made}  {made.stat().st_size / 2**20:.1f} MB")
