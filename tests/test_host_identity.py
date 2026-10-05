"""Host enrollment uses an app hash independently of model/project/state paths."""
from types import SimpleNamespace

import pytest

from ml_stack import home
from ml_stack.home import device_id


def test_app_hash_is_stable_across_state_roots_and_distinct_devices(monkeypatch, tmp_path):
    host = ["a" * 64]
    monkeypatch.setattr(home, "machineid", SimpleNamespace(hashed_id=lambda app: host[0] if app == "ml-stack" else ""))
    monkeypatch.setenv("ML_STACK_HOME", str(tmp_path / "one"))
    first = device_id()
    monkeypatch.setenv("ML_STACK_HOME", str(tmp_path / "two"))
    monkeypatch.setenv("ML_STACK_WORKSPACE_HOME", str(tmp_path / "project"))
    assert device_id() == first
    host[0] = "b" * 64
    assert device_id() != first


def test_unavailable_provider_does_not_mint_random_account(monkeypatch):
    monkeypatch.setattr(home, "machineid", None)
    with pytest.raises(RuntimeError, match="identity unavailable"):
        device_id()
