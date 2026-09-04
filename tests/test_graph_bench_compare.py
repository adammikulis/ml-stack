"""``ml-stack-bench compare``: three configurations of one model as one document.

Every fixture here is invented. Nothing reads a real store, a real graph, or a real server.
"""

from __future__ import annotations

import json

import pytest

from ml_stack.graph.bench import invented_digest, runs, save
from ml_stack.graph.bench.comparison import assemble, read_standards, stem_of, write
from ml_stack.graph.bench.speed import KIND as SPEED

from conftest import scored_rows

G = 2**30


@pytest.fixture
def store(tmp_path):
    kept = tmp_path / "runs.ladybug"
    mine = invented_digest()
    # the drafted llama.cpp run, its speed grid, and a standard set to go with it
    rows = scored_rows("flash-plain", questions=20, hits=16, seconds=200.0, tokens=(300, 40),
                       draft=(100, 78))
    save(kept, rows, held={"graph": mine, "context": 32768, "slots": 1,
                           "binary": "/opt/builds/unsloth/bin/llama-server", "build": "unsloth",
                           "draft_model": "mtp-thornfell.gguf", "load_s": 41.0,
                           "resident_peak": 70 * G, "weights_bytes": 104 * G,
                           "served_by": {"program": "llama.cpp", "format": "gguf",
                                         "quant": "Q4_K_XL", "model": "thornfell.gguf"}})
    save(kept, [{"prompt_tokens": 512, "streams": 1, "prefill_tps": 900.0, "decode_tps": 30.0,
                 "decode_tps_per_stream": 30.0, "ttft_s": 0.6, "ttft_from": "prompt_ms"}],
         held={"resident_peak": 74 * G, "served_by": {"program": "llama.cpp", "format": "gguf",
                                                       "quant": "Q4_K_XL"}, "build": "unsloth"},
         kind=SPEED, label="flash-speed")
    # the same without the head: a graph run only
    rows = scored_rows("flash-nodraft-plain", questions=20, hits=16, seconds=260.0,
                       tokens=(300, 40))
    save(kept, rows, held={"graph": mine, "context": 32768, "slots": 1,
                           "binary": "/opt/builds/unsloth/bin/llama-server", "build": "unsloth",
                           "load_s": 39.0, "resident_peak": 69 * G,
                           "served_by": {"program": "llama.cpp", "format": "gguf",
                                         "quant": "Q4_K_XL"}})
    # Ollama: a graph run whose program reports no draft, and a speed grid
    rows = scored_rows("flash-ollama-plain", questions=20, hits=15, seconds=300.0,
                       tokens=(300, 40))
    for r in rows:
        r.draft_tokens = r.draft_taken = r.cached_tokens = None
    save(kept, rows, held={"graph": mine, "context": 32768, "resident_peak": 80 * G,
                           "weights_bytes": 66 * G,
                           "served_by": {"program": "ollama", "version": "0.33.3",
                                         "format": "safetensors", "runtime": "mlx",
                                         "quant": "nvfp4", "model": "thornfell:125b-mlx"}})
    save(kept, [{"prompt_tokens": 512, "streams": 1, "prefill_tps": 700.0, "decode_tps": 25.0,
                 "decode_tps_per_stream": 25.0, "ttft_s": 0.8, "ttft_from": "prompt_ms"}],
         held={"served_by": {"program": "ollama", "version": "0.33.3", "format": "safetensors",
                             "runtime": "mlx", "quant": "nvfp4"}},
         kind=SPEED, label="flash-ollama-speed")
    # a graph run over some other community, newer, which must not be taken
    rows = scored_rows("flash-plain", questions=20, hits=20, seconds=10.0)
    save(kept, rows, held={"graph": "somebody-elses", "context": 32768})
    return kept


def test_a_label_stem_drops_the_way_and_what_follows_it():
    assert stem_of("flash-plain") == "flash"
    assert stem_of("flash-nodraft-plain-batch") == "flash-nodraft"
    assert stem_of("flash-ollama-shortlist") == "flash-ollama"
    assert stem_of("flash") == "flash"
    assert stem_of("plain") == "plain", "a label that is only a way is its own stem"


def test_the_document_has_one_entry_per_label_with_nothing_absent_as_zero(store, tmp_path):
    standard = tmp_path / "flash-standard.json"
    standard.write_text(json.dumps({"label": "flash", "sets": {
        "gsm8k": {"score": 0.91, "metric": "exact", "n": 200, "seconds": 900.0}}}))
    got = assemble(runs(store), ["flash-plain", "flash-nodraft-plain", "flash-ollama-plain",
                                 "nothing-plain"],
                   standards=read_standards([standard]), title="Flash three ways",
                   machine_name="Mac (128 GB)")
    assert got["title"] == "Flash three ways" and got["machine"] == "Mac (128 GB)"
    assert got["made_at"]
    drafted, bare, ollama, nothing = got["configs"]

    assert drafted["label"] == "llama.cpp (unsloth) · gguf · Q4_K_XL"
    assert drafted["run"] == "flash-plain"
    assert (drafted["program"], drafted["format"], drafted["quant"]) == ("llama.cpp", "gguf",
                                                                         "Q4_K_XL")
    assert drafted["draft"] is True
    assert drafted["graph"]["f1"] == pytest.approx(0.8) and drafted["graph"]["questions"] == 20
    assert drafted["graph"]["seconds_per_question"] == pytest.approx(10.0)
    assert drafted["graph"]["calls_per_question"] == pytest.approx(3.0)
    assert drafted["speed"] == [{"prompt_tokens": 512, "streams": 1, "prefill_tps": 900.0,
                                 "decode_tps": 30.0, "decode_tps_per_stream": 30.0,
                                 "ttft_s": 0.6, "ttft_from": "prompt_ms"}]
    assert drafted["memory"] == {"peak_gb": 74.0, "load_s": 41.0, "disk_gb": 104.0}, \
        "the peak from whichever run held most, the load and the disk from the graph run"
    assert drafted["standard"] == {"gsm8k": {"score": 0.91, "n": 200, "metric": "exact",
                                             "seconds": 900.0}}
    assert drafted["acceptance"] == pytest.approx(0.78)

    assert bare["draft"] is False, "a llama-server served without a head"
    assert bare["speed"] is None, "not measured for speed is null, not an empty grid"
    assert bare["acceptance"] is None
    assert bare["memory"] == {"peak_gb": 69.0, "load_s": 39.0, "disk_gb": None}
    assert bare["standard"] is None

    assert ollama["label"] == "ollama 0.33.3 · mlx · nvfp4"
    assert ollama["draft"] is None, "a program that does not say has no head to speak of"
    assert ollama["acceptance"] is None
    assert ollama["graph"]["f1"] == pytest.approx(0.75)
    assert ollama["speed"][0]["decode_tps"] == 25.0
    assert ollama["memory"]["peak_gb"] == 80.0 and ollama["memory"]["load_s"] is None

    assert nothing["label"] == "nothing-plain" and nothing["graph"] is None
    assert nothing["speed"] is None and nothing["memory"] is None
    assert nothing["draft"] is None


def test_a_graph_run_over_another_community_is_not_taken_unless_anyway(store):
    got = assemble(runs(store), ["flash-plain"])
    assert got["configs"][0]["graph"]["f1"] == pytest.approx(0.8), "not the newer 100% one"
    got = assemble(runs(store), ["flash-plain"], anyway=True)
    assert got["configs"][0]["graph"]["f1"] == pytest.approx(1.0)


def test_the_export_refuses_a_path_inside_a_repository(store, tmp_path, monkeypatch):
    document = assemble(runs(store), ["flash-plain"])
    inside = tmp_path / "repo" / "comparison.json"
    inside.parent.mkdir()
    (inside.parent / ".git").mkdir()
    with pytest.raises(ValueError, match="inside the git repository"):
        write(document, inside)
    assert not inside.exists()
    outside = tmp_path / "comparison.json"
    assert write(document, outside) == str(outside)
    back = json.loads(outside.read_text())
    assert back["configs"][0]["run"] == "flash-plain"


def test_the_subcommand_writes_the_document_and_says_what_each_label_had(store, tmp_path, capsys):
    import ml_stack.graph.bench as bench

    out = tmp_path / "comparison.json"
    code = bench._main(["compare", "--labels", "flash-plain,flash-ollama-plain,nothing",
                        "--kept", str(store), "--export", str(out), "--title", "Flash",
                        "--machine", "Mac (128 GB)"])
    assert code == 0
    said = capsys.readouterr().out
    assert "flash-plain: llama.cpp (unsloth) · gguf · Q4_K_XL -- graph, speed, memory" in said
    assert "nothing: nothing -- nothing kept under this label" in said
    assert str(out) in said
    back = json.loads(out.read_text())
    assert [c["run"] for c in back["configs"]] == ["flash-plain", "flash-ollama-plain", "nothing"]
    assert back["machine"] == "Mac (128 GB)"


def test_the_subcommand_refuses_a_repository_path_with_exit_2(store, tmp_path, capsys):
    import ml_stack.graph.bench as bench

    inside = tmp_path / "repo" / "comparison.json"
    inside.parent.mkdir()
    (inside.parent / ".git").mkdir()
    assert bench._main(["compare", "--labels", "flash-plain", "--kept", str(store),
                        "--export", str(inside)]) == 2
    assert "inside the git repository" in capsys.readouterr().err
    assert not inside.exists()
