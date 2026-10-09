"""Optional components, verified linkage and full integrated replacement costs."""

import struct
from unittest.mock import Mock

import pytest

from poolhouse.fleet import component_routes, component_store, model_components as components
from poolhouse.fleet.models import Downloads, Getting, Model, Models
from poolhouse.fleet.weights import ModelError, is_beside
from poolhouse.serve import mtp

SOURCE = "hf:publisher/Example-27B-GGUF/Example-27B-IQ3_S.gguf"


def string(value):
    encoded = value.encode()
    return struct.pack("<Q", len(encoded)) + encoded


def gguf(path, *, arch="qwen3_5", head=False, width=8):
    fields = {"general.architecture": arch, f"{arch}.embedding_length": width,
              f"{arch}.block_count": 4, f"{arch}.nextn_predict_layers": int(head),
              "tokenizer.ggml.pre": "fixture", "tokenizer.ggml.model": "gpt2"}
    table = b""
    for name, value in fields.items():
        encoded = struct.pack("<I", 8) + string(value) if isinstance(value, str) else struct.pack("<II", 4, value)
        table += string(name) + encoded
    tensors = ["token_embd.weight", *( ["blk.3.nextn.eh_proj.weight"] if head else [])]
    raw = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(fields)) + table
    for number, name in enumerate(tensors):
        raw += string(name) + struct.pack("<IQQIQ", 2, 2, 2, 0, number * 32)
    raw += bytes((-len(raw)) % 32) + bytes(32 * len(tensors))
    path.write_bytes(raw)
    return path


def offer(kind="mtp", packaging="integrated"):
    name = "Example-27B-IQ3_S-mtp.gguf" if packaging == "integrated" else "mtp-Example-27B-Q4_0.gguf"
    return {"kind": kind, "packaging": packaging, "name": name, "ref": "hf:publisher/repo/" + name,
            "size_bytes": 12100000000, "download_bytes": 12100000000, "storage_bytes": 12100000000}


def test_integrated_mtp_is_a_full_model_and_full_download():
    rows = components.discover(SOURCE, [{"path": "Example-27B-IQ3_S-mtp.gguf", "size": 12100000000},
                                       {"path": "mmproj-Example-27B-BF16.gguf", "size": 900000000}])
    assert not is_beside(rows[0]["name"])
    assert rows[0]["packaging"] == "integrated"
    assert rows[0]["download_bytes"] == rows[0]["storage_bytes"] == 12100000000
    assert rows[1]["kind"] == "vision"


@pytest.mark.parametrize("row", [{"path": "../mtp-Example-27B-Q4_0.gguf", "size": 10},
                                {"path": "/mtp-Example-27B-Q4_0.gguf", "size": 10}, 2])
def test_hostile_component_paths_are_refused(row):
    with pytest.raises(ModelError):
        components.discover(SOURCE, [row])


def test_ambiguous_or_different_model_heads_are_unavailable():
    rows = components.discover(SOURCE, [{"path": "mtp-Other-27B-Q4_0.gguf", "size": 10},
                                       {"path": "mtp-Example-27B-Q4_0.gguf", "size": 10},
                                       {"path": "mtp-Example-27B-Q5_0.gguf", "size": 10}])
    assert rows[0]["status"] == "unavailable"


def test_explicit_optout_never_queries_the_network():
    models = Mock()
    assert component_routes.selected(models, {"name": "model.gguf", "mtp": False, "vision": False,
                                              "draft": "hf:publisher/repo/head.gguf"}) == []
    models.sources.assert_not_called()


def test_replacement_keeps_base_and_is_used_without_a_separate_draft(tmp_path, monkeypatch):
    base = gguf(tmp_path / "Example-27B-IQ3_S.gguf")
    replacement = gguf(tmp_path / offer()["name"], head=True)
    original = base.read_bytes()
    components.link(base, replacement, offer())
    assert base.read_bytes() == original and components.effective(base) == replacement
    monkeypatch.setattr(mtp, "help_of", lambda _: "--spec-type none,draft-mtp\n")
    from poolhouse.serve.backend import ServerSpec
    planned = mtp.plan(ServerSpec(model=str(replacement)), binary="fixture")
    assert planned.spec_type == "draft-mtp" and planned.draft == ""


@pytest.mark.parametrize("arch,width", [("llama", 8), ("qwen3_5", 16)])
def test_incompatible_head_never_changes_linkage(tmp_path, arch, width):
    base = gguf(tmp_path / "model.gguf")
    head = gguf(tmp_path / "mtp.gguf", arch=arch, head=True, width=width)
    with pytest.raises(ModelError):
        components.link(base, head, offer(packaging="separate"))
    assert components.linked(base, "mtp") is None


def test_failed_replacement_retains_the_original_model(tmp_path):
    base = gguf(tmp_path / "model.gguf")
    original = base.read_bytes()
    models = Mock()
    models.ensure.side_effect = [Model(base.name, base, base.stat().st_size, 0), OSError("transfer stopped")]
    row = Getting("fixture", base.name, components=[offer()])
    Downloads(models)._run(row, None, True)
    assert row.state == "failed" and base.read_bytes() == original
    assert components.effective(base) == base


def test_link_cannot_escape_model_directory(tmp_path):
    model = gguf(tmp_path / "model.gguf")
    component_store.link(model, model, {"kind": "vision", "name": "../secret.gguf"})
    assert components.linked(model, "vision") is None


def test_unselected_and_lan_only_sources_never_request_hub_metadata(tmp_path, monkeypatch):
    model = gguf(tmp_path / "Example-27B-IQ3_S.gguf")
    monkeypatch.setattr("poolhouse.fleet.models.MIN_SIZE", 0)
    components.remember(model, SOURCE)
    monkeypatch.setattr(components.net, "default", lambda: pytest.fail("Unexpected Internet request"))
    for policy in ("", "lan"):
        models = Models([tmp_path], tmp_path, sources=lambda policy=policy: policy)
        assert all(row["status"] == "unavailable" for row in components.catalogue(models, model.name)["components"])


def test_vision_is_linked_and_not_listed_as_a_base_model(tmp_path, monkeypatch):
    model = gguf(tmp_path / "Example-27B-IQ3_S.gguf")
    vision = gguf(tmp_path / "mmproj-Example-27B-BF16.gguf", arch="clip")
    info = {**offer(), "kind": "vision", "packaging": "separate", "name": vision.name}
    components.link(model, vision, info)
    monkeypatch.setattr("poolhouse.fleet.models.MIN_SIZE", 0)
    models = Models([tmp_path], tmp_path)
    assert [row.name for row in models.all()] == [model.name]
    assert models.inventory()[0]["components"][0]["name"] == vision.name


@pytest.mark.parametrize("method", ["GET", "POST"])
def test_addon_route_refuses_untrusted_requests_before_component_lookup(method):
    from unittest.mock import patch

    from poolhouse.fleet import routes

    handler = Mock(path="/ui/models/addons?name=../../outside&source=file:///outside",
                   command=method, client_address=("192.0.2.1", 1000))
    handler.headers = {}
    ui = Mock()
    ui.authed.return_value = False
    ui.may_setup.return_value = "remote setup refused"
    with patch.object(components, "catalogue") as catalogue, patch.object(
            routes, "in_cluster", return_value=False), patch.object(routes.Router, "send") as send:
        routes.Router(ui, handler).run()
        assert send.call_args.args[0] == 403
        catalogue.assert_not_called()
        handler.headers = {routes.UI_HEADER: "1"}
        routes.Router(ui, handler).run()
        assert send.call_args.args[0] == 403
        catalogue.assert_not_called()


@pytest.mark.parametrize("choice", ["../../outside", "true", 1, {}, []])
def test_component_flags_refuse_untyped_selection_before_source_lookup(choice):
    models = Mock()
    with pytest.raises(ModelError):
        component_routes.selected(models, {"name": "model.gguf", "mtp": choice})
    models.sources.assert_not_called()


def test_component_graph_retains_source_and_relationship_after_reopen(tmp_path):
    model = gguf(tmp_path / "model.gguf")
    head = gguf(tmp_path / "mtp-head.gguf", head=True)
    components.remember(model, SOURCE)
    components.link(model, head, offer(packaging="separate"))
    assert components.record(model)["source"] == SOURCE
    assert components.linked(model, "mtp") == head
    assert not list(tmp_path.glob("*.json"))
