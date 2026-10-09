"""Host enrollment uses an app hash independently of model/project/state paths."""
from types import SimpleNamespace

import pytest

from poolhouse import home
from poolhouse.home import device_id


def test_app_hash_is_stable_across_state_roots_and_distinct_devices(monkeypatch, tmp_path):
    host = ["a" * 64]
    monkeypatch.setattr(home, "import_module", lambda name: SimpleNamespace(hashed_id=lambda app: host[0] if app == "poolhouse" else ""))
    monkeypatch.setenv("POOLHOUSE_HOME", str(tmp_path / "one"))
    first = device_id()
    monkeypatch.setenv("POOLHOUSE_HOME", str(tmp_path / "two"))
    monkeypatch.setenv("POOLHOUSE_WORKSPACE_HOME", str(tmp_path / "project"))
    assert device_id() == first
    host[0] = "b" * 64
    assert device_id() != first


def test_unavailable_provider_does_not_mint_random_account(monkeypatch):
    def missing(name):
        raise ModuleNotFoundError(name)
    monkeypatch.setattr(home, "import_module", missing)
    with pytest.raises(RuntimeError, match="identity unavailable"):
        device_id()
