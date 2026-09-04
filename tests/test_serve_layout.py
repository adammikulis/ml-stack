"""`ml-stack-models layout`: the attention layout off a GGUF header, as words and as JSON."""
from __future__ import annotations

import json

import pytest
from ml_stack import hub
from ml_stack.serve.layout import Layout, _ranges, layout, render

from conftest import write_gguf
from test_serve_fit import F16, IQ4_NL, Q4_K, with_tensors


def gemma_shaped(tmp_path):
    """Twelve layers: five slide in six, the last four share, one lookup table."""
    meta = {"general.architecture": "gemma4", "gemma4.block_count": 12,
            "gemma4.context_length": 131072, "gemma4.embedding_length": 1536,
            "gemma4.attention.head_count": 8, "gemma4.attention.head_count_kv": 1,
            "gemma4.attention.key_length": 512, "gemma4.attention.value_length": 512,
            "gemma4.attention.sliding_window": 512,
            "gemma4.attention.sliding_window_pattern": [True, True, True, True, True, False] * 2,
            "gemma4.attention.key_length_swa": 256, "gemma4.attention.value_length_swa": 256,
            "gemma4.attention.shared_kv_layers": 4}
    return with_tensors(tmp_path / "thornfield-E4B-UD-Q4_K_XL.gguf", meta, [
        ("per_layer_token_embd.weight", Q4_K, (3072, 262144)),
        ("blk.0.attn_k.weight", F16, (1536, 512)),
        ("token_embd.weight", Q4_K, (1536, 262144)),
    ])


def gpt_oss_shaped(tmp_path):
    """Eight layers, a 128-token window and no pattern key, 32 experts with 4 used."""
    meta = {"general.architecture": "gpt-oss", "gpt-oss.block_count": 8,
            "gpt-oss.context_length": 131072, "gpt-oss.embedding_length": 2880,
            "gpt-oss.attention.head_count": 64, "gpt-oss.attention.head_count_kv": 8,
            "gpt-oss.attention.key_length": 64, "gpt-oss.attention.value_length": 64,
            "gpt-oss.attention.sliding_window": 128, "gpt-oss.expert_count": 32,
            "gpt-oss.expert_used_count": 4, "gpt-oss.expert_feed_forward_length": 2880}
    return with_tensors(tmp_path / "marrowgate-20b-mxfp4.gguf", meta, [
        ("blk.0.ffn_down_exps.weight", Q4_K, (2880, 2880, 32)),
        ("blk.0.attn_k.weight", F16, (2880, 512)),
    ])


def flash_next_shaped(tmp_path):
    """Eight layers, attention on every fourth with a 4:1 indexer, the rest recurrent,
    512 experts and a 3-gram table in one tensor."""
    meta = {"general.architecture": "qwen4exp", "qwen4exp.block_count": 8,
            "qwen4exp.context_length": 262144, "qwen4exp.embedding_length": 2560,
            "qwen4exp.attention.head_count": 24, "qwen4exp.attention.head_count_kv": 2,
            "qwen4exp.attention.key_length": 256, "qwen4exp.attention.value_length": 256,
            "qwen4exp.full_attention_interval": 4,
            "qwen4exp.attention.compress_ratios": [0, 0, 0, 4, 0, 0, 0, 4],
            "qwen4exp.attention.indexer.head_count": 4,
            "qwen4exp.attention.indexer.key_length": 128,
            "qwen4exp.attention.indexer.top_k": 2048,
            "qwen4exp.ssm.inner_size": 6144, "qwen4exp.ssm.state_size": 128,
            "qwen4exp.ssm.group_count": 16, "qwen4exp.ssm.conv_kernel": 4,
            "qwen4exp.expert_count": 512, "qwen4exp.expert_used_count": 10,
            "qwen4exp.expert_feed_forward_length": 640,
            "qwen4exp.expert_shared_feed_forward_length": 640}
    return with_tensors(tmp_path / "flash-UD-Q4_K_XL.gguf", meta, [
        ("per_layer_token_embd.weight", IQ4_NL, (160, 320001536)),
        ("blk.3.ffn_down_exps.weight", Q4_K, (640, 2560, 512)),
        ("blk.3.attn_k.weight", F16, (2560, 512)),
        ("blk.3.indexer.attn_k.weight", F16, (2560, 128)),
    ])


def test_gemma_shaped_header_names_the_sliding_full_and_shared_layers(tmp_path):
    lay = layout(gemma_shaped(tmp_path))
    assert lay.arch == "gemma4" and lay.n_layer == 12
    assert lay.layers("full") == [5]
    assert lay.layers("sliding") == [0, 1, 2, 3, 4, 6, 7]
    assert lay.layers("shared") == [8, 9, 10, 11]
    assert lay.layers("recurrent") == []
    assert lay.window == 512 and lay.pattern == "one bool per layer"
    assert (lay.key_length_swa, lay.value_length_swa) == (256, 256)
    assert lay.tables == (("per_layer_token_embd.weight", "q4_K", (3072, 262144),
                           3072 * 262144 // 256 * 144),)
    text = render(lay)
    assert text.startswith("thornfield-E4B (Q4_K_XL) is gemma4: 12 layers, 8 heads, 1 KV head, "
                           "head size 512, 131,072 context. 1 layer holds a full cache; "
                           "7 slide over a 512-token window, key 256; the last 4 read the "
                           "cache of the layers before them and own none.")
    assert "- full cache: 1 -- 5" in text
    assert ("- sliding: 7 -- 0-4, 6-7; window 512; pattern: one bool per layer; "
            "key_length_swa 256, value_length_swa 256") in text
    assert "- shared KV: 4 -- 8-11; shared_kv_layers 4" in text
    assert "- table: per_layer_token_embd.weight  q4_K  (3,072 x 262,144)  " in text
    assert "table 1 (" in text and "attention 1 (" in text and "embedding 1 (" in text


def test_gpt_oss_shaped_header_alternates_with_period_two_and_counts_experts(tmp_path):
    lay = layout(gpt_oss_shaped(tmp_path))
    assert lay.layers("sliding") == [0, 2, 4, 6]
    assert lay.layers("full") == [1, 3, 5, 7]
    assert lay.pattern == "no pattern named; period 2"
    assert (lay.expert_count, lay.expert_used_count) == (32, 4)
    text = render(lay)
    assert "4 layers hold a full cache; 4 slide over a 128-token window. " in text
    assert "32 experts, 4 used, 2880 wide." in text
    assert "- sliding: 4 -- 0, 2, 4, 6; window 128; pattern: no pattern named; period 2" in text
    assert "- experts: 32, 4 used, 2880 wide" in text
    assert "experts 1 (" in text


def test_flash_next_shaped_header_names_recurrent_layers_ratios_indexer_and_table(tmp_path):
    lay = layout(flash_next_shaped(tmp_path))
    assert lay.layers("full") == [3, 7]
    assert lay.layers("recurrent") == [0, 1, 2, 4, 5, 6]
    assert lay.compress_ratios == (0, 0, 0, 4, 0, 0, 0, 4)
    assert lay.indexer == {"head_count": 4, "key_length": 128, "top_k": 2048}
    assert lay.ssm == {"inner_size": 6144, "state_size": 128, "group_count": 16,
                       "conv_kernel": 4}
    text = render(lay)
    assert "2 layers hold a full cache; 6 are recurrent. " in text
    assert ("The attention layers are sparse: an indexer (4 heads, key 128) scores blocks "
            "of 4 tokens and each token attends to the top 2,048. "
            "512 experts, 10 used, 640 wide plus a shared expert of 640. "
            "A lookup table in one tensor: per_layer_token_embd.weight, "
            "160 x 320,001,536, iq4_nl, 26.8G, gathered a row at a time.") in text
    assert "- full cache: 2 -- 3, 7" in text
    assert ("- recurrent: 6 -- 0-2, 4-6; ssm inner_size 6144, state_size 128, "
            "group_count 16, conv_kernel 4") in text
    assert "- compress ratios: 4 on 2 layers (3, 7)" in text
    assert "- indexer: head_count 4, key_length 128, top_k 2,048" in text
    assert "- experts: 512, 10 used, 640 wide, shared expert 640" in text


def test_a_header_missing_every_attention_key_reads_as_not_named(tmp_path):
    path = write_gguf(tmp_path / "bare.gguf", {"general.architecture": "x", "x.block_count": 3})
    lay = layout(path)
    assert lay.kinds == ("full", "full", "full") and lay.shared == (False,) * 3
    assert lay.compress_ratios == () and lay.indexer == {} and lay.tables == ()
    assert "bare is x: 3 layers, 0 heads, 0 KV heads, head size 0, 0 context." in render(lay)


def test_json_round_trips(tmp_path):
    lay = layout(flash_next_shaped(tmp_path))
    assert Layout.from_json(lay.to_json()) == lay
    assert json.loads(lay.to_json())["compress_ratios"] == [0, 0, 0, 4, 0, 0, 0, 4]


def test_ranges_collapse_runs():
    assert _ranges([]) == "none"
    assert _ranges([0, 1, 2, 3, 5, 7, 8, 9]) == "0-3, 5, 7-9"
    assert _ranges([3, 7]) == "3, 7"


class TestCommand:
    def test_layout_prints_the_paragraph_and_the_bullets(self, tmp_path, capsys):
        path = gemma_shaped(tmp_path)
        assert hub.main(["layout", str(path)]) == 0
        out = capsys.readouterr().out
        assert out.startswith("thornfield-E4B (Q4_K_XL) is gemma4:")
        assert "- shared KV: 4 -- 8-11; shared_kv_layers 4" in out

    def test_json_is_the_same_layout(self, tmp_path, capsys):
        path = flash_next_shaped(tmp_path)
        assert hub.main(["layout", str(path), "--json"]) == 0
        assert Layout.from_json(capsys.readouterr().out) == layout(path)

    def test_a_file_that_is_not_a_gguf_is_refused_with_its_name(self, tmp_path, capsys):
        path = tmp_path / "notes.gguf"
        path.write_bytes(b"not a gguf at all")
        assert hub.main(["layout", str(path)]) == 2
        assert "cannot read" in capsys.readouterr().err

    def test_layout_is_listed_in_the_help(self, capsys):
        with pytest.raises(SystemExit):
            hub.main(["--help"])
        assert "layout" in capsys.readouterr().out
