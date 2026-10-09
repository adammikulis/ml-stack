"""No test can reach the person's own keystore (it once stored hundreds of Keychain items)."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests.conftest import real_keystore_classes


def test_the_real_keystore_backends_refuse_inside_a_test():
    backends = real_keystore_classes()
    if not backends:
        pytest.skip("no real keystore backend can be imported here")
    for backend in backends:
        for method in ("get_password", "set_password", "delete_password"):
            assert getattr(getattr(backend, method), "__name__", "") == "refuse_the_real_keystore", (backend, method)   # before any call
        with pytest.raises(RuntimeError, match="real OS keystore"):
            backend.set_password(object(), "poolhouse-test", "nobody", "x")


def test_a_fake_installed_through_keyrings_own_interface_still_works():
    import keyring

    from tests.memory_keys import MemoryRing

    before = keyring.get_keyring()
    keyring.set_keyring(MemoryRing())
    try:
        keyring.set_password("poolhouse-test", "nobody", "x")
        assert keyring.get_password("poolhouse-test", "nobody") == "x"
    finally:
        keyring.set_keyring(before)


def usable_here(cls: type) -> bool:
    """Whether keyring would load ``cls`` on this host: the macOS backend imports everywhere but runs on macOS."""
    try:
        cls.priority  # noqa: B018 - a backend that cannot run here raises when its priority is read
    except (RuntimeError, ImportError, OSError):
        return False
    return True


def test_a_process_a_test_starts_sees_the_real_keystore_as_absent():
    """The in-process guard cannot reach a child; the environment variable the suite sets does:
    a child whose active backend is the machine's own keystore finds none, and touches nothing."""
    from poolhouse import keystore

    real = [c for c in real_keystore_classes() if usable_here(c)]
    if not real:
        pytest.skip("no real keystore backend can run on this host")
    cls = real[0]
    code = ("from poolhouse import keystore; import sys\n"
            "sys.exit(0 if not keystore.default().available() else 3)")
    src = str(Path(__file__).resolve().parents[1] / "src")
    env = {**os.environ, "PYTHONPATH": src, "PYTHON_KEYRING_BACKEND": f"{cls.__module__}.{cls.__name__}"}
    off = subprocess.run([sys.executable, "-c", code], env=env, check=False, timeout=60)
    assert off.returncode == 0
    assert keystore.is_real(cls())
