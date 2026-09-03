"""A comparison document turned into the tables an animated comparison is drawn from.

Every document here is written by hand into ``tmp_path``; no bench store is read, no model
is served, and only the one ``slow`` test at the end imports manim.
"""

from __future__ import annotations

import json
import pathlib

import pytest

from ml_stack.graph.bench import animate as a


def a_document() -> dict:
    def row(prompt, streams, prefill, decode, ttft):
        return {"prompt_tokens": prompt, "streams": streams, "prefill_tps": prefill,
                "decode_tps": decode, "ttft_s": ttft}

    return {
        "made_at": "2026-09-03T21:00:00", "machine": "Mac (128 GB)",
        "title": "Glimmer: two builds",
        "configs": [
            {"label": "llama.cpp · GGUF Q4_K_XL · draft head", "program": "llama.cpp",
             "format": "gguf", "quant": "Q4_K_XL", "draft": True,
             "graph": {"f1": 0.80, "recall": 0.89, "precision": 0.77,
                       "seconds_per_question": 26.7, "calls_per_question": 7.1,
                       "questions": 100},
             "speed": [row(512, 1, 471.0, 37.5, 1.2), row(4096, 1, 455.0, 35.1, 9.1),
                       row(16384, 1, 402.0, 30.2, 41.0), row(512, 2, 460.0, 61.0, 1.4),
                       row(512, 4, 440.0, 98.0, 1.9)],
             "memory": {"peak_gb": 99.0, "load_s": 29.3, "disk_gb": 71.2},
             "standard": {"gsm8k": {"score": 0.91, "n": 200},
                          "mmlu_pro": {"score": 0.71, "n": 200},
                          "ifeval": {"score": 0.84, "n": 200},
                          "humaneval": {"score": 0.80, "n": 164}},
             "acceptance": 0.78},
            {"label": "ollama · GGUF Q4_K_M", "program": "ollama", "format": "gguf",
             "quant": "Q4_K_M", "draft": False,
             "graph": {"f1": 0.74, "recall": 0.80, "precision": 0.69,
                       "seconds_per_question": 33.0, "calls_per_question": 8.4,
                       "questions": 100},
             "speed": [row(512, 1, 390.0, 29.8, 1.5), row(4096, 1, None, 27.0, None),
                       row(512, 2, 380.0, 50.0, 1.7)],
             "memory": {"peak_gb": 84.0, "load_s": None},
             "standard": {"gsm8k": {"score": 0.90, "n": 200},
                          "mmlu_pro": {"score": 0.70, "n": 200}},
             "acceptance": None},
            {"label": "mlx · 4-bit", "program": "mlx", "format": "mlx", "quant": "4bit",
             "draft": None,
             "graph": None,
             "speed": [row(512, 1, 510.0, 41.2, 1.0)],
             "memory": None,
             "standard": None},
        ]}


def written(tmp_path, doc=None) -> pathlib.Path:
    where = tmp_path / "comparison.json"
    where.write_text(json.dumps(doc or a_document()), encoding="utf-8")
    return where


# -- the title card ---------------------------------------------------------------------------

def test_a_config_is_named_program_format_quant_and_draft_only_when_it_has_one():
    cfgs = a_document()["configs"]
    assert a.config_line(cfgs[0]) == "llama.cpp · gguf · Q4_K_XL · draft"
    assert a.config_line(cfgs[1]) == "ollama · gguf · Q4_K_M"
    assert a.config_line(cfgs[2]) == "mlx · mlx · 4bit"


def test_the_title_card_carries_title_machine_date_and_one_line_per_config():
    card = a.title_card(a_document())
    assert card["title"] == "Glimmer: two builds"
    assert card["machine"] == "Mac (128 GB)"
    assert card["date"] == "2026-09-03"
    assert card["configs"] == ["llama.cpp · gguf · Q4_K_XL · draft", "ollama · gguf · Q4_K_M",
                               "mlx · mlx · 4bit"]


def test_each_config_gets_its_own_colour_from_one_colour_blind_safe_palette():
    colours = a.colours(a_document())
    assert len(colours) == 3 and len(set(colours.values())) == 3
    assert all(c in a.PALETTE for c in colours.values())
    assert set(colours) == {"llama.cpp · GGUF Q4_K_XL · draft head", "ollama · GGUF Q4_K_M",
                            "mlx · 4-bit"}


# -- bars -------------------------------------------------------------------------------------

def test_a_scale_is_a_round_ceiling_above_the_largest_value():
    assert a.scale([37.5, 41.2]) == 50
    assert a.scale([471.0, 380.0]) == 500
    assert a.scale([0.8, 0.74]) == 1
    assert a.scale([99.0]) == 100
    assert a.scale([]) == 1
    assert a.scale([None, 26.7]) == 30


def test_decode_bars_are_one_group_per_prompt_size_at_one_stream():
    panel = a.speed_panels(a_document())[0]
    assert panel["key"] == "decode"
    assert [g["label"] for g in panel["groups"]] == ["512 tokens", "4096 tokens",
                                                     "16384 tokens"]
    first = panel["groups"][0]["bars"]
    assert [b["value"] for b in first] == [37.5, 29.8, 41.2]
    assert [b["shown"] for b in first] == ["37.5", "29.8", "41.2"]
    assert panel["scale"] == 50
    assert panel["unit"] == "tokens/s"


def test_a_prompt_size_a_config_never_ran_is_not_measured_and_never_a_zero():
    panel = a.speed_panels(a_document())[0]
    last = panel["groups"][2]["bars"]
    assert last[0]["value"] == 30.2
    assert last[1]["value"] is None and last[1]["shown"] == a.NOT_MEASURED
    assert last[2]["value"] is None and last[2]["shown"] == a.NOT_MEASURED
    assert all(b["value"] != 0 for g in panel["groups"] for b in g["bars"])


def test_a_null_inside_a_row_is_not_measured_too():
    prefill = a.speed_panels(a_document())[1]
    assert prefill["key"] == "prefill"
    at_4096 = prefill["groups"][1]["bars"]
    assert at_4096[0]["value"] == 455.0
    assert at_4096[1]["value"] is None and at_4096[1]["shown"] == a.NOT_MEASURED


def test_time_to_first_token_is_the_third_speed_panel_in_seconds():
    ttft = a.speed_panels(a_document())[2]
    assert ttft["key"] == "ttft" and ttft["unit"] == "s"
    assert [b["shown"] for b in ttft["groups"][0]["bars"]] == ["1.20", "1.50", "1.00"]
    assert ttft["groups"][2]["bars"][0]["shown"] == "41.0"


def test_concurrency_is_decode_at_one_two_and_four_streams_at_the_smallest_prompt():
    panel = a.speed_panels(a_document())[3]
    assert panel["key"] == "concurrency"
    assert [g["label"] for g in panel["groups"]] == ["1 stream", "2 streams", "4 streams"]
    assert [b["value"] for b in panel["groups"][2]["bars"]] == [98.0, None, None]
    assert panel["note"] == "512-token prompt"


def test_bars_carry_the_config_colour_and_label():
    panel = a.speed_panels(a_document())[0]
    colours = a.colours(a_document())
    for bar in panel["groups"][0]["bars"]:
        assert bar["colour"] == colours[bar["config"]]


# -- memory -----------------------------------------------------------------------------------

def test_the_machine_memory_is_read_out_of_its_name():
    assert a.machine_gb("Mac (128 GB)") == 128
    assert a.machine_gb("a laptop with 36GB") == 36
    assert a.machine_gb("a laptop") is None


def test_peak_memory_bars_stand_against_a_line_at_the_machine_memory():
    peak, load = a.memory_panels(a_document())
    assert peak["key"] == "peak_gb" and peak["unit"] == "GB"
    assert [b["value"] for b in peak["groups"][0]["bars"]] == [99.0, 84.0, None]
    assert peak["line"] == {"value": 128, "label": "128 GB on this machine"}
    assert peak["scale"] == 150
    assert load["key"] == "load_s" and load["unit"] == "s"
    assert [b["shown"] for b in load["groups"][0]["bars"]] == ["29.3", a.NOT_MEASURED,
                                                                a.NOT_MEASURED]


def test_without_a_machine_size_the_peak_panel_has_no_line():
    doc = a_document()
    doc["machine"] = "a desk"
    peak, _ = a.memory_panels(doc)
    assert peak["line"] is None
    assert peak["scale"] == 100


# -- the graph bench ---------------------------------------------------------------------------

def test_the_frontier_is_every_point_nothing_beats_on_both_axes():
    points = [{"x": 26.7, "y": 0.80, "config": "a"}, {"x": 33.0, "y": 0.74, "config": "b"},
              {"x": 20.0, "y": 0.60, "config": "c"}, {"x": 40.0, "y": 0.85, "config": "d"}]
    assert [p["config"] for p in a.frontier(points)] == ["c", "a", "d"]


def test_the_scatter_has_a_labelled_point_per_measured_config_and_lists_the_rest():
    sc = a.graph_scatter(a_document())
    assert [p["config"] for p in sc["points"]] == ["llama.cpp · GGUF Q4_K_XL · draft head",
                                                   "ollama · GGUF Q4_K_M"]
    assert sc["points"][0]["x"] == 26.7 and sc["points"][0]["y"] == 0.80
    assert sc["points"][0]["shown"] == "F1 0.80 · 26.7 s"
    assert [p["config"] for p in sc["frontier"]] == ["llama.cpp · GGUF Q4_K_XL · draft head"]
    assert sc["x_scale"] == 40 and sc["y_scale"] == 1
    assert sc["x_label"] == "seconds per question" and sc["y_label"] == "F1"
    assert sc["not_measured"] == ["mlx · 4-bit"]


def test_calls_per_question_is_a_single_group_of_bars():
    panel = a.calls_panel(a_document())
    assert panel["key"] == "calls"
    assert [b["shown"] for b in panel["groups"][0]["bars"]] == ["7.1", "8.4", a.NOT_MEASURED]
    assert panel["scale"] == 10


# -- standard sets -----------------------------------------------------------------------------

def test_the_standard_grid_is_every_set_any_config_ran_with_blanks_marked():
    grid = a.standard_grid(a_document())
    assert grid["sets"] == ["gsm8k", "mmlu_pro", "ifeval", "humaneval"]
    assert [r["config"] for r in grid["rows"]] == ["llama.cpp · GGUF Q4_K_XL · draft head",
                                                   "ollama · GGUF Q4_K_M", "mlx · 4-bit"]
    assert [c["shown"] for c in grid["rows"][0]["cells"]] == ["91%", "71%", "84%", "80%"]
    assert grid["rows"][0]["cells"][0]["n"] == 200
    assert [c["shown"] for c in grid["rows"][1]["cells"]] == ["90%", "70%", a.NOT_MEASURED,
                                                               a.NOT_MEASURED]
    assert all(c["shown"] == a.NOT_MEASURED for c in grid["rows"][2]["cells"])
    assert grid["rows"][2]["cells"][0]["score"] is None


def test_a_grid_with_no_sets_anywhere_is_empty():
    doc = a_document()
    for cfg in doc["configs"]:
        cfg["standard"] = None
    assert a.standard_grid(doc)["sets"] == []


# -- the closing card --------------------------------------------------------------------------

def test_the_closing_card_has_headline_numbers_per_config_and_where_it_was_measured():
    card = a.closing(a_document())
    assert card["lines"][0] == {
        "config": "llama.cpp · GGUF Q4_K_XL · draft head", "colour": a.PALETTE[0],
        "numbers": "37.5 tok/s · F1 0.80 · 26.7 s/question · 99 GB peak · 78% accepted"}
    assert card["lines"][1]["numbers"] == "29.8 tok/s · F1 0.74 · 33.0 s/question · 84 GB peak"
    assert card["lines"][2]["numbers"] == "41.2 tok/s · graph not measured"
    assert card["footer"] == "measured on Mac (128 GB), 2026-09-03"


# -- the plan ----------------------------------------------------------------------------------

def test_the_plan_is_every_scene_in_order_and_its_seconds_add_up_to_the_cut():
    scenes = a.plan(a_document(), seconds=50)
    assert [s["key"] for s in scenes] == ["title", "decode", "prefill", "ttft", "concurrency",
                                          "memory", "graph", "calls", "standard", "closing"]
    assert abs(sum(s["seconds"] for s in scenes) - 50) < 1e-6
    short = a.plan(a_document(), seconds=20)
    assert abs(sum(s["seconds"] for s in short) - 20) < 1e-6
    assert all(s["seconds"] > 0 for s in short)


def test_a_scene_with_nothing_measured_is_left_out_and_its_seconds_shared_out():
    doc = a_document()
    for cfg in doc["configs"]:
        cfg["standard"] = None
        cfg["memory"] = None
    scenes = a.plan(doc, seconds=50)
    assert "standard" not in [s["key"] for s in scenes]
    assert "memory" not in [s["key"] for s in scenes]
    assert abs(sum(s["seconds"] for s in scenes) - 50) < 1e-6


def test_only_keeps_the_scenes_named_and_gives_them_the_whole_cut():
    scenes = a.plan(a_document(), seconds=6, only=["title"])
    assert [s["key"] for s in scenes] == ["title"]
    assert scenes[0]["seconds"] == 6


# -- the command -------------------------------------------------------------------------------

def test_dry_run_prints_the_plan_and_writes_nothing(tmp_path, capsys):
    out = tmp_path / "out" / "comparison.mp4"
    code = a.main([str(written(tmp_path)), "--out", str(out), "--png", str(tmp_path / "x.png"),
                   "--seconds", "40", "--dry-run"])
    assert code == 0
    text = capsys.readouterr().out
    assert "Glimmer: two builds" in text and "3 configs" in text
    for key in ("title", "decode", "prefill", "ttft", "concurrency", "memory", "graph",
                "calls", "standard", "closing"):
        assert key in text
    assert "40.0 s" in text
    assert not out.exists() and not (tmp_path / "x.png").exists()
    assert list(tmp_path.iterdir()) == [tmp_path / "comparison.json"]


def test_a_document_without_configs_is_refused(tmp_path, capsys):
    where = written(tmp_path, {"title": "nothing", "configs": []})
    assert a.main([str(where), "--out", str(tmp_path / "o.mp4"), "--dry-run"]) == 2
    assert "no configs" in capsys.readouterr().err


@pytest.mark.slow
def test_the_title_card_renders_to_a_video_and_a_last_frame(tmp_path):
    pytest.importorskip("manim")
    out = tmp_path / "title.mp4"
    png = tmp_path / "title.png"
    a.render(a_document(), out=out, png=png, quality="l", seconds=3, only=["title"],
             work=tmp_path / "work")
    assert out.exists() and out.stat().st_size > 0
    assert png.exists() and png.stat().st_size > 0
