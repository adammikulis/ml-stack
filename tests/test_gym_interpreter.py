"""Explicit simulator interpreter selection survives UI requests."""

from types import SimpleNamespace

from ml_stack.fleet import gym_routes


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
