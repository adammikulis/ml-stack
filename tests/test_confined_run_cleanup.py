"""A failed ConfinedRun preparation releases what it acquired."""
import os
import shutil
import sys
import tempfile
from pathlib import Path

import pytest
import test_kernel_isolation as isolation


@pytest.mark.skipif(sys.platform != "darwin", reason="the Seatbelt kernel runs on macOS")
def test_failed_preparation_removes_the_control_directory(monkeypatch):
    storage = Path(tempfile.mkdtemp(prefix="cr", dir="/tmp"))
    try:
        monkeypatch.setattr(isolation, "supervisor_storage", lambda environment: storage)

        def refuse(*arguments, **options):
            assert list(storage.glob("poolhouse-confined-*")), "the control directory exists before the failure"
            raise RuntimeError("forced holder preparation failure")

        monkeypatch.setattr(isolation, "prepare_holder", refuse)
        command = [sys.executable, "-m", "pytest", "tests/test_layers.py"]
        with pytest.raises(RuntimeError, match="forced holder preparation failure"):
            isolation.ConfinedRun(command, {key: value for key, value in os.environ.items()
                                                if key not in ("POOLHOUSE_HOME", "POOLHOUSE_CACHE")}, "unused-endpoint")
        assert list(storage.iterdir()) == []
    finally:
        shutil.rmtree(storage, ignore_errors=True)
