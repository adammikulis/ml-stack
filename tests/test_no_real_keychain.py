"""No test can reach the person's own keystore (it once stored hundreds of Keychain items)."""
from __future__ import annotations

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
