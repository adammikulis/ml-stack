"""Training a pointer-head decider: the splits, the batches, and one whole run on a tiny
random model with a tokenizer built in the test."""

from __future__ import annotations

import json
import re

import pytest

from ml_stack.decide import registry
from ml_stack.decide.cases import Case
from ml_stack.decide.guards import guard_cases
from ml_stack.decide.sources import local_source
from ml_stack.decide.types import DecideError, options_of
from ml_stack.train.decider import Settings, collate, encode, metrics_of, splits

OPTS = options_of({"yes": "it is", "no": "it is not"})


def tiny_cases(n: int = 48) -> list[Case]:
    out = []
    for i in range(n):
        label = "yes" if i % 2 == 0 else "no"
        out.append(Case("Is the number even?", f"the number is {i}", OPTS, label, id=f"c{i}",
                        group=f"g{i // 2}"))
    return out


def test_splits_never_put_a_group_in_two_places_and_cover_every_case():
    cases = guard_cases()
    fit, cal, test = splits(cases, Settings())
    seen = [{c.group for c in part} for part in (fit, cal, test)]
    assert not (seen[0] & seen[1] or seen[0] & seen[2] or seen[1] & seen[2])
    assert len(fit) + len(cal) + len(test) == len(cases)
    assert min(len(fit), len(cal), len(test)) > 0


def test_the_splits_are_the_same_for_the_same_seed_and_differ_for_another():
    cases = guard_cases()
    a = splits(cases, Settings(seed=1))
    assert [c.id for c in a[2]] == [c.id for c in splits(cases, Settings(seed=1))[2]]
    assert [c.id for c in a[2]] != [c.id for c in splits(cases, Settings(seed=2))[2]]


def test_metrics_scale_the_logits_by_the_temperature():
    cases = tiny_cases(2)
    logits = [[4.0, 0.0], [0.0, 4.0]]
    sharp, flat = metrics_of(logits, cases, 1.0), metrics_of(logits, cases, 4.0)
    assert sharp["accuracy"] == flat["accuracy"] == 1.0
    assert sharp["confidence"] > flat["confidence"]
    assert sharp["brier"] < flat["brier"]


class Words:
    """A tokenizer that makes each run of non-space characters one token."""

    def __call__(self, text, return_offsets_mapping=True, add_special_tokens=False):
        found = list(re.finditer(r"\S+", text))
        return {"input_ids": [len(m.group()) for m in found],
                "offset_mapping": [m.span() for m in found]}


def test_encode_finds_the_last_word_of_each_option_line_in_the_chosen_order():
    case = Case("q", "s", options_of({"alpha": "", "beta": ""}), "beta")
    straight, flipped = (encode(Words(), case, order, 1000) for order in ([0, 1], [1, 0]))
    assert straight["order"] == [0, 1] and flipped["order"] == [1, 0]
    assert len(straight["spots"]) == 2 and straight["spots"][0] < straight["spots"][1]
    assert straight["ids"][straight["spots"][0]] == len("alpha")
    assert flipped["ids"][flipped["spots"][0]] == len("beta")


def test_a_case_over_the_token_limit_is_refused_by_name():
    case = Case("q", "word " * 50, OPTS, "yes", id="long-one")
    with pytest.raises(DecideError, match="long-one"):
        encode(Words(), case, [0, 1], 20)


def test_collate_right_pads_and_marks_the_query_and_option_positions():
    torch = pytest.importorskip("torch")
    rows = [{"ids": [5, 6, 7], "spots": [1, 2]}, {"ids": [8, 9], "spots": [0]}]
    got = collate(torch, rows, [1, 0], "cpu")
    assert got["ids"].tolist() == [[5, 6, 7], [8, 9, 0]]
    assert got["mask"].tolist() == [[1, 1, 1], [1, 1, 0]]
    assert got["last"].tolist() == [2, 1]
    assert got["valid"].tolist() == [[True, True], [True, False]]


def test_each_parameter_group_trains_at_its_own_multiple_of_the_rate():
    torch = pytest.importorskip("torch")
    from ml_stack.train.decider import DeciderStep
    a, b = torch.nn.Parameter(torch.zeros(1)), torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.AdamW([{"params": [a], "scale": 1.0}, {"params": [b], "scale": 5.0}])
    DeciderStep(torch.nn.Module(), opt, lambda m, x: x).learning_rate(0.01)
    assert [g["lr"] for g in opt.param_groups] == [0.01, 0.05]


@pytest.fixture
def tiny_base(tmp_path):
    """A random 2-layer Qwen3 and a word-level tokenizer, saved as a base directory."""
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("peft")
    tokenizers = pytest.importorskip("tokenizers")
    from ml_stack.decide import pointer_prompt
    words = {"[UNK]": 0}
    for case in tiny_cases():
        text = pointer_prompt.render(case.question, str(case.state), case.options).text
        for tok in tokenizers.pre_tokenizers.Whitespace().pre_tokenize_str(text):
            words.setdefault(tok[0], len(words))
    tk = tokenizers.Tokenizer(tokenizers.models.WordLevel(words, unk_token="[UNK]"))
    tk.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    base = tmp_path / "base"
    base.mkdir()
    tk.save(str(base / "tokenizer.json"))
    (base / "tokenizer_config.json").write_text(json.dumps(
        {"tokenizer_class": "PreTrainedTokenizerFast", "unk_token": "[UNK]"}))
    config = transformers.Qwen3Config(vocab_size=len(words), hidden_size=32, intermediate_size=64,
                                      num_hidden_layers=2, num_attention_heads=4,
                                      num_key_value_heads=2, head_dim=8)
    transformers.Qwen3Model(config).save_pretrained(str(base))
    return base


def run_tiny(tiny_base, tmp_path, steps=6):
    from ml_stack.train.decider import train
    from ml_stack.train.lora import Lora
    out = tmp_path / "decider"
    settings = Settings(name="tiny", base=tiny_base, steps=steps, batch_size=4, device="cpu",
                        dtype="float32", lora=Lora(4, 8, 0.0, ("q_proj", "v_proj")), lr=1e-3,
                        test_fraction=0.25, calibrate_fraction=0.25, baseline="none")
    return train(tiny_cases(), out, settings), out


def test_a_whole_run_writes_a_directory_that_loads_and_decides(tiny_base, tmp_path):
    from ml_stack.decide.pointer import PointerDecider
    result, out = run_tiny(tiny_base, tmp_path)
    for name in ("decider.json", "head.safetensors", "lora/adapter_model.safetensors",
                 "manifest.json", "model_card.md", "tokenizer.json", "train_log.jsonl"):
        assert (out / name).is_file(), name
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["data_hash"] == result.data_hash and len(manifest["losses"]) == 6
    assert result.n_train + result.n_calibrate + result.n_test == 48
    assert 0 < result.temperature <= 20
    assert "Pipeline check, not a quality claim" in (out / "model_card.md").read_text()
    got = PointerDecider(out, device="cpu").decide("Is the number even?", "the number is 4", OPTS)
    assert sum(got.scores.values()) == pytest.approx(1.0) and got.choice in ("yes", "no")
    assert registry.find("tiny") == out.resolve()


def test_the_trained_directory_is_refused_when_a_file_is_changed_or_pickled(tiny_base, tmp_path):
    _, out = run_tiny(tiny_base, tmp_path, steps=2)
    local_source(out)
    (out / "pickled.pt").write_bytes(b"x")
    with pytest.raises(DecideError, match="not loaded"):
        local_source(out)
    (out / "pickled.pt").unlink()
    with (out / "head.safetensors").open("ab") as f:
        f.write(b"0")
    with pytest.raises(DecideError, match="hash recorded"):
        local_source(out)


def test_the_loss_falls_when_the_model_can_learn_the_rule(tiny_base, tmp_path):
    result, _ = run_tiny(tiny_base, tmp_path, steps=40)
    early, late = result.losses[:8], result.losses[-8:]
    assert sum(late) / 8 < sum(early) / 8
