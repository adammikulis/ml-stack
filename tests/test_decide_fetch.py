"""Pinned decider files come down through the net pipeline, laid out as a loader expects."""

import hashlib

import pytest

from ml_stack.decide.fetch import Pin, locate
from ml_stack.decide.types import BackendUnavailable, DecideError
from tests.web_site import Site

REV = "a" * 40
BODY = b'{"head_type": "pointer"}'


def pin(name="lora/adapter_config.json", body=BODY):
    return Pin("owner/repo", REV, name, hashlib.sha256(body).hexdigest(), len(body))


@pytest.fixture
def hub(monkeypatch, loopback_net):
    site = Site()
    monkeypatch.setenv("HF_ENDPOINT", site.base)
    site.file(f"/owner/repo/resolve/{REV}/lora/adapter_config.json", BODY, "application/json")
    yield site
    site.close()


def test_a_file_is_not_fetched_unless_asked(hub):
    with pytest.raises(BackendUnavailable, match="ml-stack decide fetch"):
        locate(pin())
    assert hub.hits == []


def test_a_pinned_file_lands_in_the_repositorys_own_layout_and_is_not_fetched_twice(hub):
    got = locate(pin(), download=True)
    assert got.read_bytes() == BODY
    assert got.parent.name == "lora" and got.parent.parent.name == REV
    assert locate(pin(), download=True) == got
    assert len(hub.hits) == 1


def test_a_file_that_is_not_the_pinned_one_is_held_and_never_kept(hub):
    hub.file(f"/owner/repo/resolve/{REV}/lora/adapter_config.json", b'{"head_type": "evil!!"}',
             "application/json")
    with pytest.raises(DecideError, match="pinned"):
        locate(pin(), download=True)
    with pytest.raises(BackendUnavailable):
        locate(pin())


def test_the_hub_is_reached_only_through_the_pipeline_allow_list(monkeypatch):
    """With the test seal in force (no host admitted) the download is refused by name."""
    monkeypatch.delenv("HF_ENDPOINT", raising=False)
    with pytest.raises(DecideError, match=r"huggingface\.co is not on the allow-list"):
        locate(pin(), download=True)


def test_the_injection_classifier_comes_through_the_pipeline_with_the_hubs_own_hashes(
        monkeypatch, loopback_net, tmp_path):
    import json

    from ml_stack.guard import classifier

    model, config = b"\x08\x07onnx-bytes" * 8, b'{"model_type": "deberta"}'
    tree = [{"type": "file", "path": "onnx/model.onnx", "size": len(model),
             "lfs": {"oid": hashlib.sha256(model).hexdigest(), "size": len(model)}},
            {"type": "file", "path": "config.json", "size": len(config)},
            {"type": "file", "path": "pytorch_model.bin", "size": 5}]
    site = Site()
    try:
        monkeypatch.setenv("HF_ENDPOINT", site.base)
        base = f"/{classifier.MODEL}"
        site.file(f"/api/models/{classifier.MODEL}/tree/main", json.dumps(tree).encode(),
                  "application/json")
        site.file(f"{base}/resolve/main/onnx/model.onnx", model, "application/octet-stream")
        site.file(f"{base}/resolve/main/config.json", config, "application/json")
        assert classifier.cached(tmp_path) is None
        folder = classifier.fetch(tmp_path)
        assert (folder / "model.onnx").read_bytes() == model
        assert classifier.cached(tmp_path) == folder
        assert not any("pytorch_model" in hit for hit in site.hits)
        site.file(f"{base}/resolve/main/onnx/model.onnx", model[::-1], "application/octet-stream")
        (folder / "model.onnx").unlink()
        with pytest.raises(Exception, match="sha256 differs"):
            classifier.fetch(tmp_path)
        assert classifier.cached(tmp_path) is None
    finally:
        site.close()
