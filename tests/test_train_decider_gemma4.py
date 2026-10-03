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

TEXT = {
    "vocab_size": 300, "vocab_size_per_layer_input": 300, "hidden_size": 32,
    "intermediate_size": 64, "num_hidden_layers": 6, "num_attention_heads": 4,
    "num_key_value_heads": 1, "head_dim": 8, "global_head_dim": 16,
    "hidden_size_per_layer_input": 8, "num_kv_shared_layers": 2,
    "layer_types": ["sliding_attention"] * 2 + ["full_attention"]
    + ["sliding_attention"] * 2 + ["full_attention"],
    "sliding_window": 8, "max_position_embeddings": 512, "use_double_wide_mlp": True}


@pytest.fixture(autouse=True)
def _never_the_real_home():
    """Fails a test that would train into the real state root: it runs outside tests/conftest.py,
    which moves the root."""
    from ml_stack import home
    if home.home() == home.user_home() / ".ml-stack":
        pytest.fail(f"{home.home()} is the real state root; run this under tests/conftest.py")


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
    words.setdefault("<bos>", len(words))
    tk = tokenizers.Tokenizer(tokenizers.models.WordLevel(words, unk_token="[UNK]"))
    tk.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    tk.post_processor = tokenizers.processors.TemplateProcessing(
        single="<bos> $A", special_tokens=[("<bos>", words["<bos>"])])
    base = tmp_path / "gemma"
    base.mkdir()
    tk.save(str(base / "tokenizer.json"))
    (base / "tokenizer_config.json").write_text(json.dumps(
        {"tokenizer_class": "PreTrainedTokenizerFast", "unk_token": "[UNK]"}))
    text = {**TEXT, "vocab_size": len(words), "vocab_size_per_layer_input": len(words)}
    config = transformers.Gemma4Config(
        text_config=text,
        vision_config={"hidden_size": 16, "intermediate_size": 32, "num_hidden_layers": 1,
                       "num_attention_heads": 2, "num_key_value_heads": 2, "head_dim": 8,
                       "global_head_dim": 8},
        audio_config={"hidden_size": 16, "num_hidden_layers": 1, "num_attention_heads": 2,
                      "output_proj_dims": 16, "conv_kernel_size": 5})
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


def _step_grads(tiny_gemma, micros, *, checkpoint):
    """Adapter and head gradients of one optimiser step over ``micros`` (rate 0, nothing moves)."""
    import torch

    from ml_stack.decide.pointer import build_head, load_torso
    from ml_stack.train.decider import (
        DeciderStep,
        _loss,
        attach_features,
        build_net,
        checkpoint_activations,
    )
    from ml_stack.train.lora import Lora
    torch.manual_seed(0)
    plain = load_torso(tiny_gemma, None, "float32", "cpu")
    from ml_stack.train.decider import Settings
    assert checkpoint_activations(plain, Settings(grad_checkpoint=checkpoint), "cpu") == checkpoint
    torso = attach_features(plain, Lora(4, 8, 0.0, ("q_proj", "v_proj", "gate_proj")))
    for name, p in torso.named_parameters():
        if "lora_B" in name:
            torch.nn.init.normal_(p, std=0.1)
    net = build_net(torch, torso, build_head(torch, 32, 16))
    opt = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad], lr=0.0)
    step = DeciderStep(net, opt, _loss)
    first = torso.base_model.model.layers[0]
    original, calls = type(first).forward, []
    type(first).forward = lambda self, *a, **k: (calls.append(self is first),
                                                  original(self, *a, **k))[1]
    try:
        loss, applied = step(micros)
    finally:
        type(first).forward = original
    calls = [c for c in calls if c]
    assert applied
    assert len(calls) == len(micros) * (2 if checkpoint else 1)
    grads = {n: p.grad.clone() for n, p in net.named_parameters() if p.grad is not None}
    return loss, grads, step


def _rows(tiny_gemma):
    from transformers import AutoTokenizer

    from ml_stack.decide.cases import Case
    from ml_stack.train.decider import encode
    tok = AutoTokenizer.from_pretrained(tiny_gemma, local_files_only=True)
    out = []
    for i in range(4):
        case = Case("Is the number even?", "the number is " + "1000 " * (i * 3) + str(i), OPTS,
                    "yes" if i % 2 == 0 else "no")
        out.append((encode(tok, case, [0, 1], 4096), case.label_index))
    return out


def _micros(rows, size):
    import torch

    from ml_stack.train.decider import collate
    return [collate(torch, [r for r, _ in rows[k:k + size]], [y for _, y in rows[k:k + size]],
                    "cpu") for k in range(0, len(rows), size)]


def test_checkpointed_gradients_equal_plain_ones_through_shared_kv_layers(tiny_gemma):
    import torch
    rows = _rows(tiny_gemma)
    plain_loss, plain, _ = _step_grads(tiny_gemma, _micros(rows, 4), checkpoint=False)
    ckpt_loss, ckpt, _ = _step_grads(tiny_gemma, _micros(rows, 4), checkpoint=True)
    assert plain_loss == pytest.approx(ckpt_loss, abs=1e-5)
    assert plain.keys() == ckpt.keys() and any(float(g.abs().sum()) > 0 for g in plain.values())
    for name in plain:
        assert torch.allclose(plain[name], ckpt[name], atol=1e-5, rtol=1e-4), name


def test_accumulating_micro_batches_equals_one_larger_batch(tiny_gemma):
    import torch
    rows = _rows(tiny_gemma)
    whole_loss, whole, _ = _step_grads(tiny_gemma, _micros(rows, 4), checkpoint=False)
    for size in (2, 1):
        loss, got, _ = _step_grads(tiny_gemma, _micros(rows, size), checkpoint=False)
        assert loss == pytest.approx(whole_loss, abs=1e-5)
        for name in whole:
            assert torch.allclose(whole[name], got[name], atol=1e-4, rtol=1e-3), (size, name)


def test_a_step_draws_batch_times_accum_cases_ordered_by_length(tiny_gemma):
    from transformers import AutoTokenizer

    from ml_stack.decide.cases import Case
    from ml_stack.train.decider import Settings, draw
    tok = AutoTokenizer.from_pretrained(tiny_gemma, local_files_only=True)
    s = Settings(batch_size=2, accum=3, max_tokens=4096)
    cases = [Case("Is the number even?", "the number is " + "1000 " * (i % 7), OPTS, "yes")
             for i in range(24)]
    micros = draw(cases, tok, s, "cpu", 0)
    assert [m["ids"].shape[0] for m in micros] == [2, 2, 2]
    lengths = [int(n) for m in micros for n in m["mask"].sum(1)]
    assert lengths == sorted(lengths)
    assert draw(cases, tok, s, "cpu", 0)[0]["ids"].tolist() == micros[0]["ids"].tolist()


def test_the_step_reports_its_memory_for_the_metrics_log(tiny_gemma):
    *_, step = _step_grads(tiny_gemma, _micros(_rows(tiny_gemma), 2), checkpoint=False)
    assert set(step.metrics()) == {"peak_gb", "driver_gb"}


def test_checkpointing_is_on_by_default_for_an_accelerator_and_off_on_the_cpu(tiny_gemma):
    from ml_stack.decide.pointer import load_torso
    from ml_stack.train.decider import Settings, checkpoint_activations
    torso = load_torso(tiny_gemma, None, "float32", "cpu")
    assert checkpoint_activations(torso, Settings(), "cpu") is False
    assert torso.gradient_checkpointing is False
    assert checkpoint_activations(torso, Settings(), "mps") is True
    assert torso.gradient_checkpointing is True


def test_the_whole_run_test_refuses_to_run_where_it_would_pin_into_the_real_home(tmp_path):
    import os
    import subprocess
    import sys
    from pathlib import Path
    copy = tmp_path / "outside" / "test_gemma_run_copy.py"
    copy.parent.mkdir()
    copy.write_text(Path(__file__).read_text())
    account = tmp_path / "account"
    account.mkdir()
    root = Path(__file__).resolve().parents[1]
    env = {k: v for k, v in os.environ.items() if k not in ("ML_STACK_HOME", "ML_STACK_CACHE")}
    env.update(HOME=str(account), PYTHONPATH=f"{root}:{root / 'src'}")
    got = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-p", "no:xdist",
         "--rootdir", str(copy.parent), str(copy), "-k", "whole_run_on_a_gemma"],
        cwd=copy.parent, env=env, capture_output=True, text=True, timeout=300, check=False)
    assert got.returncode != 0 and "is the real state root" in got.stdout
    assert list(account.rglob("*")) == []


def test_gemma_prompts_start_with_the_bos_token_and_the_positions_still_find_the_options(tiny_gemma):
    from transformers import AutoTokenizer

    from ml_stack.train.decider import encode
    tok = AutoTokenizer.from_pretrained(tiny_gemma, local_files_only=True)
    bos = tok.convert_tokens_to_ids("<bos>")
    row = encode(tok, tiny_cases(2)[0], [0, 1], 4096)
    assert row["ids"][0] == bos and row["ids"].count(bos) == 1
    assert [tok.convert_ids_to_tokens(row["ids"][i]) for i in row["spots"]] == ["is", "not"]


def test_the_cached_strands_tokenizer_gives_the_same_ids_with_and_without_special_tokens():
    from transformers import AutoTokenizer

    from ml_stack.decide.fetch import locate
    from ml_stack.decide.pins import STRANDS_V19
    from ml_stack.decide.types import BackendUnavailable
    pin = next(p for p in STRANDS_V19.files if p.filename == "tokenizer.json")
    try:
        path = locate(pin)
    except BackendUnavailable:
        pytest.skip("the Strands tokenizer is not downloaded")
    tok = AutoTokenizer.from_pretrained(path.parent, local_files_only=True)
    text = pointer_prompt.render("Is it?", "state text", tiny_cases(2)[0].options).text
    on = tok(text, add_special_tokens=True, return_offsets_mapping=True)
    off = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    assert on["input_ids"] == off["input_ids"] and on["offset_mapping"] == off["offset_mapping"]
