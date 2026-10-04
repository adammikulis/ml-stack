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
            backend.set_password(object(), "ml-stack-test", "nobody", "x")


def test_a_fake_installed_through_keyrings_own_interface_still_works():
    import keyring

    from tests.memory_keys import MemoryRing

    before = keyring.get_keyring()
    keyring.set_keyring(MemoryRing())
    try:
        keyring.set_password("ml-stack-test", "nobody", "x")
        assert keyring.get_password("ml-stack-test", "nobody") == "x"
    finally:
        keyring.set_keyring(before)


def test_a_process_a_test_starts_sees_the_real_keystore_as_absent():
    """The in-process guard cannot reach a child; the environment variable the suite sets does:
    a child whose active backend is the machine's own keystore finds none, and touches nothing."""
    from ml_stack import keystore

    real = real_keystore_classes()
    if not real:
        pytest.skip("no real keystore backend on this host")
    cls = real[0]
    code = ("from ml_stack import keystore; import sys\n"
            "sys.exit(0 if not keystore.default().available() else 3)")
    src = str(Path(__file__).resolve().parents[1] / "src")
    env = {**os.environ, "PYTHONPATH": src, "PYTHON_KEYRING_BACKEND": f"{cls.__module__}.{cls.__name__}"}
    off = subprocess.run([sys.executable, "-c", code], env=env, check=False, timeout=60)
    assert off.returncode == 0
    assert keystore.is_real(cls())
