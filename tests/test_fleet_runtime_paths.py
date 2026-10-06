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


def test_empty_root_overrides_route_to_custom_runtime(tmp_path, monkeypatch):
    monkeypatch.setenv(home.ROOT_ENV, "")
    monkeypatch.setenv(home.CACHE_ENV, "")
    monkeypatch.setattr(home, "user_home", lambda: tmp_path / "account")
    configure(tmp_path / "preview")
    assert home.home() == tmp_path / "preview" / "state"
    assert home.cache() == tmp_path / "preview" / "state" / "cache"


def test_daemon_routes_roots_before_startup_services(tmp_path, monkeypatch):
    import pytest

    from ml_stack import keystore, sentinel
    from ml_stack.fleet import daemon
    from ml_stack.sentinel.honey import Honey
    from ml_stack.serve.leases import lease_file

    monkeypatch.delenv(home.ROOT_ENV, raising=False)
    monkeypatch.delenv(home.CACHE_ENV, raising=False)
    monkeypatch.setattr(home, "user_home", lambda: tmp_path / "account")
    root = tmp_path / "preview"

    class StartupObserved(Exception):
        pass

    def inspect_startup(_path):
        assert sentinel.sentinel_dir() == root / "state" / "sentinel"
        assert Honey().directory == root / "state"
        assert keystore.Keystore().directory == root / "state" / "keystore"
        assert lease_file() == root / "state" / "servers.json"
        raise StartupObserved

    monkeypatch.setattr(daemon, "load_cluster_key", inspect_startup)
    with pytest.raises(StartupObserved):
        daemon.serve_forever(root=root, announce=False, web=False)


def test_background_startup_does_not_display_token(monkeypatch, capsys):
    from ml_stack import person
    from ml_stack.fleet.runtime_paths import announce_token

    def refuse(_action):
        raise person.HumanRequired("background process")

    monkeypatch.setattr(person, "require_person", refuse)
    announce_token("fixture-token")
    assert capsys.readouterr().out == ""


def test_person_startup_displays_token(monkeypatch, capsys):
    from ml_stack import person
    from ml_stack.fleet.runtime_paths import announce_token

    monkeypatch.setattr(person, "require_person", lambda _action: None)
    announce_token("fixture-token")
    assert capsys.readouterr().out == "  token fixture-token\n"
