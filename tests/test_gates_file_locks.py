"""File-lock detection and Windows handle transfer imports."""

from pathlib import Path

import pytest
from gates import file_locks


def _find(tmp_path: Path, source: str):
    path = tmp_path / "src" / "poolhouse" / "sample.py"
    path.parent.mkdir(parents=True)
    path.write_text(source, encoding="utf-8")
    return file_locks.find(tmp_path)


@pytest.mark.parametrize("source", [
    "import msvcrt\nmsvcrt.open_osfhandle(handle, flags)\n",
    "import msvcrt as crt\ncrt.open_osfhandle(handle, flags)\n",
    "from msvcrt import open_osfhandle as transfer\ntransfer(handle, flags)\n",
])
def test_windows_handle_transfers_are_allowed(tmp_path, source):
    assert not _find(tmp_path, source)


@pytest.mark.parametrize("source", [
    "import msvcrt\nmsvcrt.locking(fd, mode, count)\n",
    "import msvcrt as crt\ncrt.locking(fd, mode, count)\n",
    "from msvcrt import locking as lock\nlock(fd, mode, count)\n",
    "from msvcrt import open_osfhandle, locking\n",
    "from msvcrt import *\nlocking(fd, mode, count)\n",
    "import msvcrt\nmsvcrt.open_osfhandle(handle, flags)\nmsvcrt.locking(fd, mode, count)\n",
    "import msvcrt as crt\ngetattr(crt, operation)(fd, mode, count)\n",
    "import msvcrt\nother = msvcrt\nother.locking(fd, mode, count)\n",
    "import msvcrt\nconsume(msvcrt)\n",
    "import msvcrt\nmsvcrt.open_osfhandle = replacement\n",
    "import msvcrt\n",
    "import fcntl as locks\nlocks.flock(fd, mode)\n",
])
def test_locking_imports_and_module_escapes_are_refused(tmp_path, source):
    assert _find(tmp_path, source)


def test_aliased_windows_locking_call_is_identified(tmp_path):
    findings = _find(tmp_path, "from msvcrt import locking as lock\nlock(fd, mode, count)\n")
    assert any(finding.detail == "msvcrt.locking" for finding in findings)
