"""Decision as a kind of model: one list of kinds, labels from real GGUF headers and the
decider registry, the filter on discovery and on the listing command, and the decide package
asking the library instead of a path."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from conftest import write_gguf
from test_hub_discover import META, model

from ml_stack import hub
from ml_stack.decide import library, registry, router
from ml_stack.decide.sources import CONFIG, FORMAT
from ml_stack.decide.types import DecideError
from ml_stack.hub import kinds
from ml_stack.serve import models_cli

_real_discover = hub.discover


def decider_dir(root: Path, name: str = "mine") -> Path:
    """What `ml_stack.train.decider` writes, as far as the registry reads it."""
    root.mkdir(parents=True, exist_ok=True)
    (root / CONFIG).write_text(json.dumps({"format": FORMAT, "name": name,
                                           "base": {"repo": "maker/base"}}))
    return root


def by_name(found):
    return {m.name: m for m in found}


def only_in(monkeypatch, where: Path) -> None:
    """Make the library's own discovery look in ``where`` instead of the machine's folders."""
    monkeypatch.setattr(hub, "discover", lambda roots=None, **kw: _real_discover([where], **kw))


def test_the_kinds_are_defined_once_and_include_decision():
    assert kinds.KINDS == hub.KINDS
    assert kinds.DECISION in kinds.KINDS
    with pytest.raises(ValueError, match="decision"):
        kinds.valid("oracle")


def test_headers_label_the_ordinary_kinds(tmp_path):
    model(tmp_path / "chatty-Q4_K_M.gguf", {"general.name": "Chatty"})
    write_gguf(tmp_path / "emb-Q4_K_M.gguf", {**META, "general.architecture": "nomic-bert",
                                              "general.name": "Emb"})
    write_gguf(tmp_path / "ears-Q4_K_M.gguf", {**META, "general.architecture": "whisper",
                                               "general.name": "Ears"})
    got = by_name(hub.discover([tmp_path]))
    assert {n: m.kind for n, m in got.items()} == {
        "Chatty": "chat", "Emb": "embedding", "Ears": "speech"}


def test_a_projector_beside_a_model_makes_it_vision(tmp_path):
    model(tmp_path / "pic-Q4_K_M.gguf", pad=300)
    model(tmp_path / "mmproj-F16.gguf")
    (one,) = hub.discover([tmp_path])
    assert one.kind == "vision"


def test_a_model_that_calls_itself_a_decider_is_labelled_decision(tmp_path):
    model(tmp_path / "pick-Q4_K_M.gguf", {"general.name": "Strands Decision 2B"}, pad=10)
    model(tmp_path / "plain-Q4_K_M.gguf", {"general.name": "Indecisive 2B"}, pad=20)
    got = by_name(hub.discover([tmp_path]))
    assert got["Strands Decision 2B"].kind == "decision"
    assert got["Indecisive 2B"].kind == "chat", "a word inside another word is not the word"


def test_a_registered_decider_directory_labels_the_models_inside_it(tmp_path):
    root = decider_dir(tmp_path / "deciders" / "mine")
    model(root / "head-Q4_K_M.gguf", {"general.name": "Plain Name"}, pad=10)
    model(tmp_path / "elsewhere" / "other-Q4_K_M.gguf", {"general.name": "Other"}, pad=20)
    places = [tmp_path / "deciders", tmp_path / "elsewhere"]
    assert {m.kind for m in hub.discover(places)} == {"chat"}, "not registered yet"
    registry.register(root)
    got = by_name(hub.discover(places))
    assert got["Plain Name"].kind == "decision"
    assert got["Other"].kind == "chat"


def test_a_hub_snapshot_link_inside_a_registered_directory_still_matches(tmp_path):
    blob = model(tmp_path / "deciders" / "blobs" / "abc", {"general.name": "Linked"})
    root = decider_dir(tmp_path / "deciders" / "mine")
    (root / "linked-Q4_K_M.gguf").symlink_to(blob)
    registry.register(root)
    (one,) = hub.discover([tmp_path / "deciders"])
    assert one.kind == "decision"


def test_discover_filters_by_kind_and_refuses_an_unknown_one(tmp_path):
    model(tmp_path / "a-Q4_K_M.gguf", {"general.name": "Alpha"}, pad=10)
    model(tmp_path / "b-Q4_K_M.gguf", {"general.name": "Beta Decider"}, pad=20)
    assert [m.name for m in hub.discover([tmp_path], kind="decision")] == ["Beta Decider"]
    assert [m.name for m in hub.discover([tmp_path], kind="chat")] == ["Alpha"]
    assert hub.discover([tmp_path], kind="speech") == []
    with pytest.raises(ValueError):
        hub.discover([tmp_path], kind="oracle")


def test_the_listing_command_filters_and_shows_the_kind(tmp_path, monkeypatch, capsys):
    model(tmp_path / "a-Q4_K_M.gguf", {"general.name": "Alpha"}, pad=10)
    model(tmp_path / "b-Q4_K_M.gguf", {"general.name": "Beta Decider"}, pad=20)
    only_in(monkeypatch, tmp_path)
    assert models_cli.run(["list", "--kind", "decision", "--json"]) == 0
    rows = json.loads(capsys.readouterr().out)
    assert [(r["name"], r["kind"]) for r in rows] == [("Beta Decider", "decision")]
    assert models_cli.run(["list"]) == 0
    text = capsys.readouterr().out
    assert "KIND" in text and "decision" in text and "chat" in text


def test_decide_resolves_a_decider_through_the_library(tmp_path, monkeypatch):
    root = decider_dir(tmp_path / "deciders" / "mine", "mine")
    model(root / "head-Q4_K_M.gguf", {"general.name": "Head"})
    registry.register(root)
    only_in(monkeypatch, tmp_path / "deciders")
    assert [m.name for m in library.models()] == ["Head"]
    assert library.decider_dir("Head") == root.resolve()
    assert router._trained("mine") == root.resolve()
    assert router._trained("Head") == root.resolve()
    with pytest.raises(DecideError):
        router._trained("nobody")
