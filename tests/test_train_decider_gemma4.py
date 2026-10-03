"""The pointer torso on a Gemma 4 base: the text weights load from a multimodal checkpoint,
hidden states at the option and answer positions are position-specific and independent of
padding, and one whole training run writes a directory that loads and decides. Runs on a tiny
random Gemma 4 on the CPU."""

from __future__ import annotations

import json

import pytest

from ml_stack.decide import pointer_prompt
from ml_stack.decide.pins import BASES, GEMMA4_E2B
from tests.test_train_decider import OPTS, run_tiny, tiny_cases

TEXT = dict(vocab_size=300, vocab_size_per_layer_input=300, hidden_size=32, intermediate_size=64,
            num_hidden_layers=6, num_attention_heads=4, num_key_value_heads=1, head_dim=8,
            global_head_dim=16, hidden_size_per_layer_input=8, num_kv_shared_layers=2,
            layer_types=["sliding_attention"] * 2 + ["full_attention"]
            + ["sliding_attention"] * 2 + ["full_attention"],
            sliding_window=8, max_position_embeddings=512, use_double_wide_mlp=True)


@pytest.fixture
def tiny_gemma(tmp_path):
    """A random 6-layer multimodal Gemma 4 and a word-level tokenizer, saved as a base."""
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("peft")
    tokenizers = pytest.importorskip("tokenizers")
    words = {"[UNK]": 0}
    for case in tiny_cases():
        text = pointer_prompt.render(case.question, str(case.state), case.options).text
        for tok in tokenizers.pre_tokenizers.Whitespace().pre_tokenize_str(text):
            words.setdefault(tok[0], len(words))
    tk = tokenizers.Tokenizer(tokenizers.models.WordLevel(words, unk_token="[UNK]"))
    tk.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    base = tmp_path / "gemma"
    base.mkdir()
    tk.save(str(base / "tokenizer.json"))
    (base / "tokenizer_config.json").write_text(json.dumps(
        {"tokenizer_class": "PreTrainedTokenizerFast", "unk_token": "[UNK]"}))
    text = {**TEXT, "vocab_size": len(words), "vocab_size_per_layer_input": len(words)}
    config = transformers.Gemma4Config(
        text_config=text,
        vision_config=dict(hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                           num_attention_heads=2, num_key_value_heads=2, head_dim=8,
                           global_head_dim=8),
        audio_config=dict(hidden_size=16, num_hidden_layers=1, num_attention_heads=2,
                          output_proj_dims=16, conv_kernel_size=5))
    transformers.Gemma4ForConditionalGeneration(config).save_pretrained(str(base))
    return base


def test_the_pinned_gemma_base_is_selectable_and_fully_pinned():
    assert BASES["gemma-4-e2b"] is GEMMA4_E2B
    assert {p.filename for p in GEMMA4_E2B.files} == {
        "config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json"}
    assert all(p.repo == "google/gemma-4-E2B" and len(p.revision) == 40 and len(p.sha256) == 64
               for p in GEMMA4_E2B.files)


def test_the_torso_holds_the_checkpoints_text_weights_not_fresh_ones(tiny_gemma):
    torch = pytest.importorskip("torch")
    from safetensors import safe_open

    from ml_stack.decide.pointer import load_torso
    torso = load_torso(tiny_gemma, None, "float32", "cpu")
    with safe_open(str(tiny_gemma / "model.safetensors"), "pt") as f:
        want = f.get_tensor("model.language_model.layers.0.self_attn.q_proj.weight")
        emb = f.get_tensor("model.language_model.embed_tokens.weight")
    assert torch.equal(torso.layers[0].self_attn.q_proj.weight, want)
    assert torch.equal(torso.embed_tokens.weight, emb)
    assert not hasattr(torso, "vision_tower")


def _hidden(torso, ids, mask):
    import torch
    with torch.inference_mode():
        return torso(input_ids=ids, attention_mask=mask, use_cache=False).last_hidden_state


def test_hidden_states_differ_by_position_and_ignore_right_padding(tiny_gemma):
    torch = pytest.importorskip("torch")
    from ml_stack.decide.pointer import load_torso
    torso = load_torso(tiny_gemma, None, "float32", "cpu")
    torch.manual_seed(0)
    vocab = torso.embed_tokens.weight.shape[0]
    ids = torch.randint(5, vocab, (1, 20))
    alone = _hidden(torso, ids, torch.ones_like(ids))[0]
    assert all(not torch.allclose(alone[i], alone[j], atol=1e-4)
               for i, j in ((3, 7), (7, 12), (12, 19), (0, 19)))
    padded_ids = torch.cat([ids, torch.zeros(1, 9, dtype=torch.long)], dim=1)
    mask = torch.cat([torch.ones_like(ids), torch.zeros(1, 9, dtype=torch.long)], dim=1)
    padded = _hidden(torso, padded_ids, mask)[0, :20]
    assert torch.allclose(alone, padded, atol=1e-4)
    changed = ids.clone()
    changed[0, 10] = 5 + (int(ids[0, 10]) - 4) % (vocab - 5)
    after = _hidden(torso, changed, torch.ones_like(changed))[0]
    assert torch.allclose(alone[:10], after[:10], atol=1e-4)
    assert not torch.allclose(alone[10:], after[10:], atol=1e-4)


def test_a_whole_run_on_a_gemma_base_loads_and_decides(tiny_gemma, tmp_path):
    from ml_stack.decide.pointer import PointerDecider
    result, out = run_tiny(tiny_gemma, tmp_path, steps=4)
    assert len(result.losses) == 4 and (out / "lora" / "adapter_model.safetensors").is_file()
    got = PointerDecider(out, device="cpu").decide("Is the number even?", "the number is 4", OPTS)
    assert sum(got.scores.values()) == pytest.approx(1.0) and got.choice in ("yes", "no")


def test_the_net_scores_each_option_against_the_answer_position_with_a_padded_batch(tiny_gemma):
    torch = pytest.importorskip("torch")
    from transformers import AutoTokenizer

    from ml_stack.decide.pointer import build_head, load_torso
    from ml_stack.train.decider import build_net, collate, encode
    torso = load_torso(tiny_gemma, None, "float32", "cpu")
    tok = AutoTokenizer.from_pretrained(tiny_gemma, local_files_only=True)
    head = build_head(torch, 32, 16)
    net = build_net(torch, torso, head).eval()
    cases = [tiny_cases(2)[0], tiny_cases(2)[1].__class__(
        "Is the number even?", "the number is 1000 and a few more words to pad the batch out",
        OPTS, "no")]
    rows = [encode(tok, c, [0, 1], 4096) for c in cases]
    assert len(rows[0]["ids"]) != len(rows[1]["ids"])
    with torch.inference_mode():
        batched = net(collate(torch, rows, [0, 0], "cpu"))
        for i, r in enumerate(rows):
            ids = torch.tensor([r["ids"]])
            hidden = _hidden(torso, ids, torch.ones_like(ids))[0]
            want = head(hidden[-1:], hidden[r["spots"]].unsqueeze(0))[0]
            assert torch.allclose(batched[i], want, atol=1e-4)
