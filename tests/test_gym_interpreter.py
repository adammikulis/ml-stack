"""Explicit simulator interpreter selection survives UI requests."""

from types import SimpleNamespace

from ml_stack.fleet import daemon, gym_routes
from ml_stack.fleet.settings import Settings


def test_explicit_worker_python_precedes_managed_environment(monkeypatch):
    monkeypatch.setenv("ML_STACK_GYM_PYTHON", "/explicit/python")
    chosen = []
    monkeypatch.setattr(gym_routes.manager, "configure", chosen.append)
    monkeypatch.setattr(gym_routes, "catalogue", lambda: [])
    request = SimpleNamespace(path="/ui/gym/catalogue", method="GET",
                              ui=SimpleNamespace(environment=SimpleNamespace(exists=True, python="/managed/python")),
                              send=lambda *_: None)
    assert gym_routes.GymRoutes.route(request)
    assert chosen == []


def test_managed_worker_python_is_default(monkeypatch):
    monkeypatch.delenv("ML_STACK_GYM_PYTHON", raising=False)
    chosen = []
    monkeypatch.setattr(gym_routes.manager, "configure", chosen.append)
    monkeypatch.setattr(gym_routes, "catalogue", lambda: [])
    request = SimpleNamespace(path="/ui/gym/catalogue", method="GET",
                              ui=SimpleNamespace(environment=SimpleNamespace(exists=True, python="/managed/python")),
                              send=lambda *_: None)
    assert gym_routes.GymRoutes.route(request)
    assert chosen == ["/managed/python"]


def test_saved_worker_python_is_reused_after_restart(monkeypatch, tmp_path):
    monkeypatch.delenv("ML_STACK_GYM_PYTHON", raising=False)
    python = tmp_path / "python"
    python.write_text("selected interpreter")
    path = tmp_path / "settings.json"
    Settings(gym_python=str(python), setup_done=True).save(path)
    settings = Settings.load(path)
    chosen = []
    monkeypatch.setattr(gym_routes.manager, "configure", chosen.append)
    monkeypatch.setattr(gym_routes, "catalogue", lambda: [])
    request = SimpleNamespace(path="/ui/gym/catalogue", method="GET",
                              ui=SimpleNamespace(settings=settings,
                                                 environment=SimpleNamespace(exists=True, python="/managed/python")),
                              send=lambda *_: None)
    assert gym_routes.GymRoutes.route(request)
    assert chosen == [str(python)]
    assert settings.setup_done


def test_missing_saved_interpreter_reports_reselection(monkeypatch, tmp_path):
    monkeypatch.delenv("ML_STACK_GYM_PYTHON", raising=False)
    responses = []
    request = SimpleNamespace(path="/ui/gym/catalogue", method="GET",
                              ui=SimpleNamespace(settings=Settings(gym_python=str(tmp_path / "missing")),
                                                 environment=None), send=lambda *args: responses.append(args))
    assert gym_routes.GymRoutes.route(request)
    assert responses[0][0] == 400
    assert "--gym-python" in responses[0][1]["error"]


def test_launcher_remembers_venv_path_without_resetting_setup(monkeypatch, tmp_path):
    base = tmp_path / "base-python"
    base.write_text("Python executable")
    python = tmp_path / "venv-python"
    python.symlink_to(base)
    root = tmp_path / "studio"
    root.mkdir()
    Settings(setup_done=True).save(root / "settings.json")
    monkeypatch.setattr(daemon, "serve", lambda *_args, **_kwargs: None)
    assert daemon.run(["--root", str(root), "--gym-python", str(python), "--no-announce"]) == 0
    saved = Settings.load(root / "settings.json")
    assert saved.gym_python == str(python)
    assert saved.setup_done
