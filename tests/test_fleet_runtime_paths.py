from ml_stack import home
from ml_stack.fleet.runtime_paths import configure


def test_custom_runtime_routes_process_state_and_cache(tmp_path, monkeypatch):
    monkeypatch.delenv(home.ROOT_ENV, raising=False)
    monkeypatch.delenv(home.CACHE_ENV, raising=False)
    monkeypatch.setattr(home, "user_home", lambda: tmp_path / "account")
    root = tmp_path / "preview"
    configure(root)
    assert home.home() == root / "state"
    assert home.state("sentinel") == root / "state" / "sentinel"
    assert home.state("keystore") == root / "state" / "keystore"
    assert home.cache() == root / "state" / "cache"


def test_default_runtime_preserves_machine_identity(tmp_path, monkeypatch):
    monkeypatch.delenv(home.ROOT_ENV, raising=False)
    monkeypatch.delenv(home.CACHE_ENV, raising=False)
    monkeypatch.setattr(home, "user_home", lambda: tmp_path / "account")
    expected = home.home()
    configure(expected / "traind")
    assert home.home() == expected
    assert home.cache() == tmp_path / "account" / ".cache" / "ml_stack"


def test_explicit_state_and_cache_roots_are_preserved(tmp_path, monkeypatch):
    state, cache = tmp_path / "state", tmp_path / "cache"
    monkeypatch.setenv(home.ROOT_ENV, str(state))
    monkeypatch.setenv(home.CACHE_ENV, str(cache))
    configure(tmp_path / "preview")
    assert home.home() == state
    assert home.cache() == cache


def test_custom_runtime_preserves_explicit_cache(tmp_path, monkeypatch):
    monkeypatch.delenv(home.ROOT_ENV, raising=False)
    monkeypatch.setattr(home, "user_home", lambda: tmp_path / "account")
    monkeypatch.setenv(home.CACHE_ENV, str(tmp_path / "cache"))
    configure(tmp_path / "preview")
    assert home.home() == tmp_path / "preview" / "state"
    assert home.cache() == tmp_path / "cache"
