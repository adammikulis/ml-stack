"""Harness context profile admission."""
from types import SimpleNamespace

import pytest

from ml_stack import harness, harnessing
from ml_stack.serve.serving import Config, Serving


def fit(**over):
    return SimpleNamespace(**({"context": 8192, "verdict": "yellow", "n_gpu_layers": "auto",
                               "kv_cache_type": "q4_0", "flash_attn": True, "batch": 1024} | over))


@pytest.fixture
def configured(monkeypatch):
    monkeypatch.setattr(harnessing.chat_template, "trained_context", lambda _: 32768)
    monkeypatch.setattr(harnessing.profile, "profile_for", lambda _: None)
    monkeypatch.setattr(harnessing, "drafted", lambda config, *a, **k: config)
    return lambda: harnessing.config_for("fixture.gguf", harnessing.Want(ctx=0), lambda _: None)


@pytest.mark.parametrize("rating", ["red", "none"])
def test_positive_context_does_not_admit_an_unsafe_verdict(monkeypatch, configured, rating):
    monkeypatch.setattr(harnessing.suggest, "suggest", lambda *a, **k: fit(verdict=rating))
    with pytest.raises(ValueError, match="no context that fits"):
        configured()


def test_automatic_context_preserves_memory_profile(monkeypatch, configured):
    monkeypatch.setattr(harnessing.suggest, "suggest", lambda *a, **k: fit())
    selected = configured().serving
    assert (selected.context, selected.cache_type, selected.flash_attn, selected.extra_args) == (
        8192, "q4_0", True, ("-ub", "1024"))


def test_partial_offload_fit_is_refused(monkeypatch, configured):
    monkeypatch.setattr(harnessing.suggest, "suggest", lambda *a, **k: fit(n_gpu_layers=12))
    with pytest.raises(ValueError, match="full GPU offload"):
        configured()


def test_sdk_checks_admission_before_serving(monkeypatch):
    monkeypatch.setattr(harness.hub, "located", lambda *a, **k: "fixture.gguf")
    monkeypatch.setattr("ml_stack.serve.recent.note", lambda *a, **k: None)
    monkeypatch.setattr(harnessing.leases, "already_up", lambda *a: None)
    config = Config(serving=Serving(model="fixture.gguf", cache_type="q4_0"))
    monkeypatch.setattr(harnessing, "config_for", lambda *a: config)
    calls = []
    monkeypatch.setattr(harnessing, "admitted", lambda *a, **k: calls.append(k) or False)
    with pytest.raises(ValueError, match="wired-memory"), harness.session("fixture.gguf"):
        pytest.fail("unsafe server admitted")
    assert calls == [{"kv": "q4_0"}]


def test_explicit_context_is_preserved_above_training_limit(monkeypatch, configured):
    selected = harnessing.config_for("fixture.gguf", harnessing.Want(ctx=65536), lambda _: None)
    assert selected.serving.context == 65536


@pytest.mark.parametrize("context,slots", [(8193, 2), (1, 2)])
def test_explicit_context_cannot_be_silently_rounded(monkeypatch, configured, context, slots):
    with pytest.raises(ValueError, match="divide evenly"):
        harnessing.config_for("fixture.gguf", harnessing.Want(ctx=context, slots=slots), lambda _: None)


def test_measured_memory_settings_are_replaced_by_the_fit(monkeypatch, configured):
    measured = Config(serving=Serving(model="fixture.gguf", cache_type="f16", flash_attn=False,
                                     extra_args=("-ub", "4096")))
    monkeypatch.setattr(harnessing.profile, "profile_for", lambda _: SimpleNamespace(config=lambda **k: measured))
    monkeypatch.setattr(harnessing.profile, "said", lambda _: "fixture")
    monkeypatch.setattr(harnessing.suggest, "suggest", lambda *a, **k: fit())
    selected = configured().serving
    assert (selected.cache_type, selected.flash_attn, selected.extra_args) == (
        "q4_0", True, ("-ub", "1024"))


def test_automatic_fit_requires_the_estimated_single_slot(monkeypatch, configured):
    with pytest.raises(ValueError, match="supports one slot"):
        harnessing.config_for("fixture.gguf", harnessing.Want(ctx=0, slots=2), lambda _: None)


def test_automatic_fit_refuses_unestimated_profile_flags(monkeypatch, configured):
    measured = Config(serving=Serving(model="fixture.gguf", extra_args=("-b", "8192")))
    monkeypatch.setattr(harnessing.profile, "profile_for", lambda _: SimpleNamespace(config=lambda **k: measured))
    monkeypatch.setattr(harnessing.profile, "said", lambda _: "fixture")
    monkeypatch.setattr(harnessing.suggest, "suggest", lambda *a, **k: fit())
    with pytest.raises(ValueError, match="cannot estimate measured server flags"):
        configured()
