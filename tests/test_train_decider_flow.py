"""The path from a labelled file to a registered decider: data checks, leakage, the replace
and baseline guards, per-kind temperatures, the pin, path safety and the commands. The model is
a random 2-layer Qwen3 on CPU, so none of it needs a GPU or a download."""

from __future__ import annotations

import json
import re

import pytest

from poolhouse import sentinel
from poolhouse.decide import dataset, metrics, registry
from poolhouse.decide.cases import Case, write_cases
from poolhouse.decide.types import DecideError, options_of
from poolhouse.train import gpu
from poolhouse.train.decider import Settings, fit_temperatures, question_kinds, train
from poolhouse.train.lora import Lora

OPTS = options_of({"yes": "it is", "no": "it is not"})


def numbers(start: int, n: int, *, group: str = "g", skew: int = 2) -> list[Case]:
    """Cases whose label is yes for the numbers divisible by ``skew`` (every other, by default)."""
    return [Case("Is the number even?", f"the number is {i}", OPTS,
                 "yes" if i % skew == 0 else "no", id=f"{group}{i}", group=f"{group}{i // 2}")
            for i in range(start, start + n)]


@pytest.fixture
def tiny_base(tmp_path):
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("peft")
    tokenizers = pytest.importorskip("tokenizers")
    from poolhouse.decide import pointer_prompt
    words = {"[UNK]": 0}
    for case in numbers(0, 200) + numbers(1000, 100):
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


def settings(base, name="flow", **changes):
    fields = {"name": name, "base": base, "steps": 4, "batch_size": 4, "device": "cpu",
              "dtype": "float32", "lora": Lora(4, 8, 0.0, ("q_proj", "v_proj")), "lr": 1e-3,
              "baseline": "none"}
    fields.update(changes)
    return Settings(**fields)


# --- the data -------------------------------------------------------------------------------


def test_a_clean_file_has_no_errors_and_a_small_one_is_warned_about():
    found = dataset.check(numbers(0, 60))
    assert not found.errors and any("pipeline" in w for w in found.warnings)
    assert found.labels == {"yes": 30, "no": 30} and found.kinds == {"noul": 60}


def test_too_few_cases_one_label_and_conflicting_labels_are_errors():
    assert any("at least" in e for e in dataset.check(numbers(0, 10)).errors)
    only_yes = [c for c in numbers(0, 100) if c.label == "yes"]
    assert any("nothing to learn" in e for e in dataset.check(only_yes).errors)
    flipped = Case("Is the number even?", "  THE number is 4", OPTS, "no", id="flip")
    errors = dataset.check([*numbers(0, 60), flipped]).errors
    assert any("two labels" in e for e in errors)


def test_an_unbalanced_label_and_repeated_cases_are_warnings():
    skewed = numbers(0, 100, skew=20)
    assert any("always answering" in w for w in dataset.check(skewed).warnings)
    repeated = [*numbers(0, 60), *numbers(0, 5)]
    assert any("repeat an earlier one" in w for w in dataset.check(repeated).warnings)


def test_a_bad_line_names_the_file_and_the_case(tmp_path):
    path = tmp_path / "d.jsonl"
    path.write_text('{"question": "q", "state": "s", "options": ["a", "b"], "label": "c"}\n')
    with pytest.raises(dataset.DataError, match=r"d\.jsonl, case 1.*label 'c'"):
        dataset.load(path)
    path.write_text("[1, 2]\n")
    with pytest.raises(dataset.DataError, match="JSON object"):
        dataset.load(path)


def test_the_same_case_or_group_in_training_and_evaluation_is_a_leak():
    train_cases = numbers(0, 60)
    assert dataset.leaks(train_cases, numbers(1000, 20)) == []
    same = [Case("is the number even?", "the   number is 4", OPTS, "yes", id="other")]
    assert any("also in the training" in x for x in dataset.leaks(train_cases, same))
    regrouped = [Case("q", "new", OPTS, "yes", group="g3")]
    assert any("groups are in both" in x for x in dataset.leaks(train_cases, regrouped))
    assert dataset.key_of(same[0]) == dataset.key_of(train_cases[4])


def test_a_leaking_evaluation_file_stops_the_run_before_any_work(tiny_base, tmp_path):
    out = tmp_path / "never"
    with pytest.raises(dataset.DataError, match="also in the training"):
        train(numbers(0, 60), out, settings(tiny_base), eval_cases=numbers(40, 20, group="h"))
    assert not out.exists() and not registry.taken("flow")


# --- calibration -----------------------------------------------------------------------------


def test_each_kind_gets_its_own_temperature_and_a_small_kind_gets_none():
    def case(kind, label):
        return Case(f"{kind}?", "s", OPTS, label, kind=kind)

    cases = [case("noul", "yes" if i % 5 else "no") for i in range(60)]
    cases += [case("choice", "yes") for i in range(60)]
    cases += [case("score", "yes") for _ in range(5)]
    logits = [[6.0, 0.0]] * 60 + [[0.1, 0.0]] * 60 + [[1.0, 0.0]] * 5
    overall, by_kind = fit_temperatures(logits, cases)
    assert set(by_kind) == {"noul", "choice"} and 0 < overall <= 20
    overconfident = [[8.0, 0.0]] * 40
    flat_all, none = fit_temperatures(overconfident, [case("noul", "yes" if i % 4 else "no")
                                                      for i in range(40)])
    assert flat_all > 2.0 and set(none) == {"noul"}
    assert by_kind["noul"] > 1.0 > by_kind["choice"]
    assert question_kinds(cases) == {"choice?": "choice", "noul?": "noul", "score?": "score"}


def test_metrics_report_the_abstain_rate_at_the_floor():
    rows = [[0.95, 0.05], [0.6, 0.4], [0.5, 0.5], [0.1, 0.9]]
    got = metrics.of_rows(rows, [0, 1, 0, 1], floor=0.8)
    assert got.abstain_rate == pytest.approx(0.5)
    assert got.answered_accuracy == pytest.approx(1.0)
    assert got.accuracy == pytest.approx(0.75)
    assert metrics.worse_than(got, got) == []
    worse = metrics.of_rows([[0.6, 0.4]] * 4, [1, 1, 1, 1], floor=0.8)
    assert len(metrics.worse_than(worse, got)) == 2


# --- a whole run -----------------------------------------------------------------------------


def run(tiny_base, tmp_path, name="flow", cases=None, **changes):
    out = tmp_path / name
    return train(cases or numbers(0, 120), out, settings(tiny_base, name, **changes)), out


def test_a_run_registers_pins_and_writes_the_metrics_and_per_kind_temperatures(tiny_base,
                                                                              tmp_path):
    from poolhouse.decide.pointer import PointerDecider
    result, out = run(tiny_base, tmp_path)
    assert result.registered and registry.find("flow") == out.resolve()
    config = json.loads((out / "decider.json").read_text())
    assert config["question_kinds"] == {"is the number even?": "noul"}
    assert set(config["temperature_by_kind"]) <= {"noul"}
    for key in ("accuracy", "brier", "ece", "abstain_rate", "answered_accuracy"):
        assert key in result.metrics
    card = (out / "model_card.md").read_text()
    assert "abstain rate at 0.8" in card and "this decider" in card
    got = PointerDecider(out, device="cpu").decide("Is the number even?", "the number is 4", OPTS)
    assert got.choice in ("yes", "no")
    pins = sentinel.default().manifest.pins()
    head = str(out / "head.safetensors")
    assert head in pins and pins[head].source == "trained:flow"


def test_a_trained_file_changed_afterwards_fails_the_sentinel_check(tiny_base, tmp_path):
    _, out = run(tiny_base, tmp_path)
    node = sentinel.default()
    assert node.verify_before_load(out / "head.safetensors")
    with (out / "head.safetensors").open("ab") as f:
        f.write(b"0")
    assert not node.verify_before_load(out / "head.safetensors")


def test_a_registered_name_is_not_replaced_without_asking_and_the_old_one_is_kept(tiny_base,
                                                                                  tmp_path):
    _, out1 = run(tiny_base, tmp_path)
    before = (out1 / "decider.json").read_bytes()
    with pytest.raises(DecideError, match="already registered"):
        train(numbers(0, 120), tmp_path / "second", settings(tiny_base))
    assert registry.find("flow") == out1.resolve() and not (tmp_path / "second").exists()
    second = tmp_path / "second"
    train(numbers(0, 120), second, settings(tiny_base, replace=True, seed=1))
    assert registry.find("flow") == second.resolve()
    assert registry.find("flow.prev") == out1.resolve()
    assert (out1 / "decider.json").read_bytes() == before


def test_the_registry_refuses_a_second_directory_under_a_taken_name_by_itself(tmp_path):
    from poolhouse.decide.sources import FORMAT
    dirs = []
    for n in ("a", "b"):
        d = tmp_path / n
        d.mkdir()
        (d / "decider.json").write_text(json.dumps(
            {"format": FORMAT, "name": "same", "base": {"repo": "r"}}))
        dirs.append(d)
    registry.register(dirs[0])
    registry.register(dirs[0])
    with pytest.raises(DecideError, match="already registered"):
        registry.register(dirs[1])
    assert registry.find("same") == dirs[0].resolve()
    registry.register(dirs[1], replace=True)
    assert registry.find("same.prev") == dirs[0].resolve()


def test_a_run_never_writes_into_a_directory_that_has_files(tiny_base, tmp_path):
    out = tmp_path / "used"
    out.mkdir()
    (out / "keep.txt").write_text("mine")
    with pytest.raises(DecideError, match="not empty"):
        train(numbers(0, 120), out, settings(tiny_base))
    assert (out / "keep.txt").read_text() == "mine"


def test_a_decider_worse_than_the_baseline_is_written_but_not_registered(tiny_base, tmp_path):
    train_cases = numbers(0, 100, skew=3)
    held = [Case("Is the number even?", f"the number is {i}", OPTS, "no", id=f"e{i}",
                 group=f"e{i}") for i in range(1001, 1040, 3)]
    with pytest.raises(DecideError, match="worse than the baseline"):
        train(train_cases, tmp_path / "weak",
              settings(tiny_base, "weak", baseline="majority", lr=1e-9, steps=1),
              eval_cases=held)
    assert (tmp_path / "weak" / "decider.json").is_file() and not registry.taken("weak")
    manifest = json.loads((tmp_path / "weak" / "manifest.json").read_text())
    assert manifest["registered"] is False and manifest["refusal"]
    got = train(train_cases, tmp_path / "weak2",
                settings(tiny_base, "weak", baseline="majority", lr=1e-9, steps=1,
                         allow_worse=True), eval_cases=held)
    assert got.registered and registry.taken("weak")


def test_a_poisoned_dataset_writes_nothing_outside_the_decider_directory(tiny_base, tmp_path):
    evil = "../../../escape `x` | <script>\n# injected"
    cases = [Case(f"Is the number even? {evil}", f"the number is {i} {evil}",
                  options_of({"yes": evil, "no": "no"}), "yes" if i % 2 == 0 else "no",
                  id=evil, group=f"g{i // 2}", tags=(evil,)) for i in range(120)]
    sandbox = tmp_path / "work"
    sandbox.mkdir()
    before = set(tmp_path.rglob("*"))
    out = sandbox / "decider"
    train(cases, out, settings(tiny_base, "poison"))
    written = set(tmp_path.rglob("*")) - before
    state = tmp_path / "machine-state"
    stray = [p for p in written if out not in p.parents and p not in (out, state) and state not in p.parents]
    assert not stray
    card = (out / "model_card.md").read_text()
    assert "<script>" not in card and "# injected" not in card


@pytest.mark.parametrize("name", ["../up", "a/b", "/abs", "UPPER", "x.prev", "", "a" * 64,
                                  "..", "-lead"])
def test_a_name_that_is_not_plain_is_refused(name):
    with pytest.raises(DecideError):
        registry.check_name(name)


def test_a_config_edited_to_a_hostile_temperature_is_refused(tiny_base, tmp_path):
    from poolhouse.decide.sources import local_source
    _, out = run(tiny_base, tmp_path)
    config = json.loads((out / "decider.json").read_text())
    for bad in (0, -1, 1e9, "nan", None):
        (out / "decider.json").write_text(json.dumps({**config, "temperature": bad}))
        with pytest.raises(DecideError, match="temperature"):
            local_source(out)


def test_a_run_refuses_to_write_inside_a_git_work_tree(tiny_base, tmp_path):
    (tmp_path / "repo" / ".git").mkdir(parents=True)
    out = tmp_path / "repo" / "sub" / "decider"
    with pytest.raises(DecideError, match="git work tree"):
        train(numbers(0, 120), out, settings(tiny_base))
    assert not out.exists()
    train(numbers(0, 120), out, settings(tiny_base, allow_repo=True))
    assert out.is_dir()


# --- the GPU hold ----------------------------------------------------------------------------


class Wire:
    """Answers the three Broker calls `gpu.hold` makes and records them."""

    def __init__(self, servers=(), granted=True):
        self.servers, self.granted, self.calls = list(servers), granted, []

    def status(self, *, start=False):
        return {"servers": self.servers}

    def claim(self, name, info, *, timeout=0.0):
        self.calls.append(("claim", name))
        return {"granted": self.granted, "pid": 7, "info": {"purpose": "other run"}}

    def unclaim(self, name):
        self.calls.append(("unclaim", name))
        return True


def test_the_gpu_is_claimed_for_the_block_and_given_back_even_on_error():
    wire = Wire()
    with pytest.raises(RuntimeError), gpu.hold("t", wire=wire):
        raise RuntimeError
    assert wire.calls == [("claim", gpu.CLAIM), ("unclaim", gpu.CLAIM)]


def test_a_held_server_or_a_second_run_refuses_the_hold():
    busy = Wire([{"model": "big", "port": 1, "loading": False,
                  "holders": [{"label": "poolhouse-agent", "pid": 3}]}])
    with pytest.raises(DecideError, match=r"in use.*big"), gpu.hold("t", wire=busy):
        pass
    assert busy.calls == []
    second = Wire(granted=False)
    with pytest.raises(DecideError, match="another training run"), gpu.hold("t", wire=second):
        pass


def test_waiting_gpu_hold_delegates_busy_server_admission_to_broker():
    busy = Wire([{"model": "big", "port": 1, "loading": False,
                  "holders": [{"label": "model holder", "pid": 3}]}])
    with gpu.hold("queued model", wait_s=10, wire=busy):
        assert busy.calls == [("claim", gpu.CLAIM)]
    assert busy.calls[-1] == ("unclaim", gpu.CLAIM)


# --- the commands ----------------------------------------------------------------------------


def test_the_train_command_dry_runs_trains_registers_and_the_eval_command_scores_it(
        tiny_base, tmp_path, capsys):
    from poolhouse.decide_cli import main
    data, held = tmp_path / "d.jsonl", tmp_path / "e.jsonl"
    write_cases(data, numbers(0, 120))
    write_cases(held, numbers(1000, 40, group="h"))
    common = ["train", "--data", str(data), "--name", "cli", "--base", str(tiny_base),
              "--device", "cpu", "--steps", "3", "--baseline", "none", "--rank", "4"]
    assert main([*common, "--dry-run", "--out", str(tmp_path / "o")]) == 0
    assert not registry.taken("cli") and not (tmp_path / "o").exists()
    assert main([*common, "--eval", str(held), "--out", str(tmp_path / "o")]) == 0
    assert registry.taken("cli")
    assert main([*common, "--out", str(tmp_path / "o2")]) == 2
    out = capsys.readouterr()
    assert "already registered" in out.err
    assert main(["eval", str(held), "--decider", "cli", "--floor", "0.7", "--out",
                 str(tmp_path / "r.json")]) == 0
    report = json.loads((tmp_path / "r.json").read_text())[0]
    assert report["n"] == 40 and report["floor"] == 0.7 and 0 <= report["abstain_rate"] <= 1
    assert re.search(r"abstain@0\.7=", capsys.readouterr().out)


def test_a_missing_or_bad_base_is_an_error_not_a_traceback(tmp_path, capsys):
    from poolhouse.decide_cli import main
    data = tmp_path / "d.jsonl"
    write_cases(data, numbers(0, 120))
    assert main(["train", "--data", str(data), "--name", "n", "--base", str(tmp_path / "nope"),
                 "--dry-run"]) == 2
    assert main(["train", "--data", str(data), "--name", "Bad/Name", "--dry-run"]) == 2
