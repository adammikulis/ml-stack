"""A LoRA over the tool-calls recipe: adapter, merge, GGUF, and what it says it will cost.

The point of this path is a model too big to full-fine-tune here -- a gemma-4 E4B tool
caller -- so nothing in these tests may load one. Everything trains the same two-layer
Gemma3 built in ``tmp_path`` that ``test_train_tools`` builds, on the CPU, and the export
runs against a fake converter that writes a real (tiny) GGUF and a fake quantiser that
copies it, so the path checkpoint -> adapter -> merged -> GGUF -> preflight runs end to end
without llama.cpp, without the Hub and without a GPU. The E4B numbers are only ever
*planned* here: an estimate that refuses itself is the thing under test, not a run.
"""

from __future__ import annotations

import json
import stat
import struct
import sys
from pathlib import Path

import pytest
from ml_stack.train import lora as lora_mod
from ml_stack.train.lora import CEILING_MIN, Lora, OverCeiling, fingerprint
from ml_stack.train.recipes import validate
from ml_stack.train.run import main, plan_for, run
from ml_stack.train.tools import synthesise, write_dataset
from test_train_tools import PROMPTS, TOOLS, make_tiny_base

# One invented base, named the way the recipe's e4b size names it, for the plans that must
# never load anything.
E4B = "unsloth/gemma-4-E4B-it"

GEMMA_META = {
    "general.architecture": "gemma3",
    "gemma3.block_count": 2,
    "gemma3.attention.head_count_kv": 1,
    "gemma3.attention.key_length": 16,
}


@pytest.fixture(autouse=True)
def on_the_cpu(monkeypatch):
    monkeypatch.setenv("ML_STACK_DEVICE", "cpu")
    monkeypatch.delenv("MLSTACK_TRAIN_CEILING", raising=False)


def write_gguf(path: Path, metadata: dict) -> Path:
    """A real, minimal GGUF v3 file: magic, version, counts, and string/int metadata."""

    def kv(name: str, value: object) -> bytes:
        head = struct.pack("<Q", len(name.encode())) + name.encode()
        if isinstance(value, int):
            return head + struct.pack("<I", 4) + struct.pack("<I", value)
        encoded = str(value).encode()
        return head + struct.pack("<I", 8) + struct.pack("<Q", len(encoded)) + encoded

    body = b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", 0) + struct.pack(
        "<Q", len(metadata))
    for name, value in metadata.items():
        body += kv(name, value)
    path.write_bytes(body)
    return path


@pytest.fixture
def dataset(tmp_path):
    """Conversations that end in a tool call, and the tiny base they were rendered for."""
    rows = synthesise(TOOLS, prompts=PROMPTS, per_tool=12)
    texts = [m["content"] for r in rows for m in r["messages"] if m.get("content")]
    texts += [json.dumps(m["tool_calls"][0]["function"]["arguments"])
              for r in rows for m in r["messages"] if m.get("tool_calls")]
    texts += ["call:find_recipe", "call:list_shelf", "call:open_page"]
    base = make_tiny_base(tmp_path / "base", texts)
    data = tmp_path / "data"
    write_dataset(data, rows, base=str(base))
    return data, base


@pytest.fixture
def llama_cpp(tmp_path, monkeypatch):
    """A converter that writes a real tiny GGUF and a quantiser that copies it."""
    template = write_gguf(tmp_path / "template.gguf", GEMMA_META)
    llama = tmp_path / "llama.cpp"
    (llama / "build" / "bin").mkdir(parents=True)
    (llama / "convert_hf_to_gguf.py").write_text(
        "import sys\n"
        "argv = sys.argv[1:]\n"
        f"open(argv[argv.index('--outfile') + 1], 'wb').write(open({str(template)!r}, 'rb').read())\n")
    quantize = llama / "build" / "bin" / "llama-quantize"
    quantize.write_text('#!/bin/sh\ncp "$1" "$2"\n')
    quantize.chmod(quantize.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("LLAMA_CPP_ROOT", str(llama))
    monkeypatch.setenv("LLAMA_CPP_DIR", str(llama))
    return llama


def e4b_data(tmp_path: Path) -> Path:
    """A data directory whose manifest names the real E4B base -- so a plan is about E4B --
    with two rows, which is enough for a refusal that happens before anything is read."""
    data = tmp_path / "e4b-data"
    data.mkdir(exist_ok=True)
    row = {"messages": [{"role": "user", "content": "who works on compilers?"},
                        {"role": "assistant", "content": "look_up"}], "tools": []}
    (data / "train.jsonl").write_text("\n".join(json.dumps(row) for _ in range(4000)))
    (data / "holdout.jsonl").write_text(json.dumps(row) + "\n")
    (data / "manifest.json").write_text(json.dumps({"base": E4B, "rows": 4001}))
    return data


class TestSettings:
    def test_the_recipe_now_carries_the_lora_fields_and_an_e4b_size(self):
        got = validate("tool-calls", {})
        assert got["lora"] is False, "a plain run is still the full fine-tune"
        assert got["lora_rank"] == 16 and got["lora_alpha"] == 32

    def test_the_e4b_size_fills_in_the_defaults_that_fit_this_machine(self):
        """A size's defaults are what suits *that* model: one set of numbers cannot serve a
        270m and an 8B at once."""
        got = validate("tool-calls", {"size": "e4b"})
        assert got["lora"] is True
        assert (got["batch_size"], got["context"], got["steps"]) == (4, 2048, 1000)
        assert got["learning_rate"] == 0.0001

    def test_what_the_caller_says_beats_the_size_default(self):
        got = validate("tool-calls", {"size": "e4b", "steps": 40, "batch_size": 1})
        assert got["steps"] == 40 and got["batch_size"] == 1 and got["context"] == 2048

    def test_the_targets_are_attention_and_mlp_unless_named(self):
        assert Lora.of(validate("tool-calls", {})).targets == lora_mod.DEFAULT_TARGETS
        named = Lora.of(validate("tool-calls", {"lora_targets": "q_proj, v_proj"}))
        assert named.targets == ("q_proj", "v_proj")

    def test_an_out_of_range_rank_is_refused(self):
        with pytest.raises(ValueError, match="lora_rank"):
            validate("tool-calls", {"lora_rank": 9999})


class TestWithoutPeft:
    def test_the_recipe_names_the_extra_rather_than_failing_deeper(self, dataset,
                                                                   monkeypatch):
        """Without peft there is no LoRA path, and the refusal is the first thing that
        happens rather than an AttributeError inside a builder."""
        from ml_stack.train.recipes import build

        data, _ = dataset
        monkeypatch.setitem(sys.modules, "peft", None)
        with pytest.raises(ValueError, match="train-lora"):
            build("tool-calls", {"lora": True, "steps": 20, "context": 128,
                                 "batch_size": 2}, data)


class TestThePlan:
    """Every plan here is for the machine the product is for -- a 128 GB unified-memory
    Mac, where a fine-tune runs on MPS. The tests themselves never leave the CPU."""

    @pytest.fixture(autouse=True)
    def on_mps(self, monkeypatch):
        from ml_stack.train.recipes import tool_calls

        monkeypatch.setattr(tool_calls, "device_for", lambda: "mps")

    def test_e4b_says_what_fits_and_what_it_will_take(self, tmp_path):
        config = validate("tool-calls", {"size": "e4b"})
        fit = plan_for("tool-calls", config, e4b_data(tmp_path))

        assert fit.base == E4B
        assert 7.5 < fit.params_b < 8.5 and fit.active_b == 4.0
        # 16G of frozen bf16 base, an adapter and its moments, and the activations of one
        # batch: comfortably inside 128G, which is the point of the LoRA.
        assert 16 < fit.resident_gb < 40
        assert fit.tokens_per_step == 4 * 2048
        assert fit.epochs == pytest.approx(1.0, abs=0.01)
        said = "\n".join(fit.lines())
        assert "estimate" in said and "s/step" in said
        assert "--dry-run" in said, "an estimate must say how to replace itself"

    def test_a_run_of_hours_is_refused_over_the_same_ceiling_the_bench_uses(self, tmp_path):
        config = validate("tool-calls", {"size": "e4b"})
        fit = plan_for("tool-calls", config, e4b_data(tmp_path))

        assert fit.ceiling_min == CEILING_MIN and fit.over
        with pytest.raises(OverCeiling, match="ceiling"):
            lora_mod.refuse_over_ceiling(fit, yes=False)
        lora_mod.refuse_over_ceiling(fit, yes=True)
        assert "--yes" in fit.refusal() and "--dry-run" in fit.refusal()

    def test_the_command_refuses_before_it_loads_anything(self, tmp_path, capsys):
        """Exit 5, the bench's own code for a refused estimate -- and no download: an 8B
        base is 16G, and finding out it was too slow after fetching it is the failure."""
        code = main(["--recipe", "tool-calls", "--size", "e4b", "--lora",
                     "--data", str(e4b_data(tmp_path)), "--out", str(tmp_path / "run")])
        assert code == 5
        err = capsys.readouterr().err
        assert "over the 30 min ceiling" in err
        assert not (tmp_path / "run").exists()

    def test_a_shorter_run_is_not_refused(self, tmp_path):
        config = validate("tool-calls", {"size": "e4b", "steps": 20})
        fit = plan_for("tool-calls", config, e4b_data(tmp_path))
        assert not fit.over
        lora_mod.refuse_over_ceiling(fit, yes=False)

    def test_a_measurement_replaces_the_estimate(self, tmp_path):
        config = validate("tool-calls", {"size": "e4b"})
        guessed = plan_for("tool-calls", config, e4b_data(tmp_path))
        measured = plan_for("tool-calls", config, e4b_data(tmp_path), seconds_per_step=2.0)
        assert measured.measured and measured.seconds == 2000
        assert not guessed.measured and guessed.seconds != measured.seconds
        assert "an estimate" not in "\n".join(measured.lines())


class TestTheAdapter:
    def test_a_checkpoint_holds_the_adapter_and_not_the_frozen_base(self, dataset, tmp_path):
        """E4B's base is 16G and identical at every step; writing it every checkpoint
        would spend the disk on a copy of something already on this machine."""
        pytest.importorskip("peft")
        from safetensors.torch import load_file

        data, base = dataset
        out = tmp_path / "run"
        got = run("tool-calls", {"lora": True, "steps": 20, "context": 256, "batch_size": 2,
                                 "learning_rate": 0.001}, data, out)

        assert 0 < got["lora"]["trainable_parameters"] < got["parameters"]
        written = Path(got["checkpoint"]) / "model.safetensors"
        tensors = load_file(str(written))
        assert tensors and all("lora" in name for name in tensors), sorted(tensors)[:3]
        assert written.stat().st_size < (base / "model.safetensors").stat().st_size
        # And the manifest's own record of it, which is what a later run is compared to.
        manifest = json.loads((out / "manifest.json").read_text())
        assert manifest["lora"]["trainable_parameters"] == \
            got["lora"]["trainable_parameters"]

    def test_it_resumes_from_its_own_adapter(self, dataset, tmp_path):
        pytest.importorskip("peft")
        data, base = dataset
        out = tmp_path / "run"
        config = {"lora": True, "steps": 20, "context": 256, "batch_size": 2,
                  "learning_rate": 0.001}
        run("tool-calls", config, data, out)
        again = run("tool-calls", config, data, out)
        assert again["steps"] == 20, "a finished run is not trained twice"

    def test_a_checkpoint_of_another_rank_does_not_fit(self, dataset, tmp_path):
        """A LoRA checkpoint only fits the rank and modules it was trained with, and a
        partial restore of an adapter is a silently different model."""
        pytest.importorskip("peft")
        from ml_stack.train.checkpoint import CheckpointError

        data, _ = dataset
        out = tmp_path / "run"
        run("tool-calls", {"lora": True, "steps": 20, "context": 256, "batch_size": 2,
                           "learning_rate": 0.001}, data, out)
        with pytest.raises(CheckpointError, match="rank"):
            run("tool-calls", {"lora": True, "lora_rank": 8, "steps": 40, "context": 256,
                               "batch_size": 2, "learning_rate": 0.001}, data, out)

    def test_a_target_this_architecture_does_not_have_says_so(self, dataset, tmp_path):
        pytest.importorskip("peft")
        from ml_stack.train.recipes import build

        data, _ = dataset
        with pytest.raises(ValueError, match="nonesuch_proj"):
            build("tool-calls", {"lora": True, "lora_targets": "nonesuch_proj",
                                 "steps": 20, "context": 256, "batch_size": 2}, data)

    def test_the_full_fine_tune_export_refuses_a_lora_run_and_names_the_merge(self, dataset,
                                                                             tmp_path):
        pytest.importorskip("peft")
        from ml_stack.train.recipes.tool_calls import save_pretrained

        data, base = dataset
        out = tmp_path / "run"
        run("tool-calls", {"lora": True, "steps": 20, "context": 256, "batch_size": 2,
                           "learning_rate": 0.001}, data, out)
        with pytest.raises(ValueError, match="merge"):
            save_pretrained(out, str(base), tmp_path / "model")


class TestTheTools:
    def test_the_managed_builds_own_converter_is_preferred(self, tmp_path, monkeypatch):
        """Converting through the same checkout the served binary was built from is how a
        GGUF is written by the code that will read it."""
        monkeypatch.delenv("LLAMA_CPP_ROOT", raising=False)
        monkeypatch.delenv("LLAMA_CPP_DIR", raising=False)
        managed = tmp_path / "managed" / "src"
        managed.mkdir(parents=True)
        (managed / "convert_hf_to_gguf.py").write_text("# the managed build's own\n")
        monkeypatch.setattr(lora_mod, "managed_source", lambda: managed)

        assert lora_mod.converter() == (managed / "convert_hf_to_gguf.py").resolve()

    def test_an_explicit_checkout_still_wins(self, tmp_path, monkeypatch):
        managed = tmp_path / "managed" / "src"
        managed.mkdir(parents=True)
        (managed / "convert_hf_to_gguf.py").write_text("# the managed build's own\n")
        monkeypatch.setattr(lora_mod, "managed_source", lambda: managed)
        mine = tmp_path / "mine"
        mine.mkdir()
        (mine / "convert_hf_to_gguf.py").write_text("# a clone somebody asked for\n")
        monkeypatch.setenv("LLAMA_CPP_ROOT", str(mine))

        assert lora_mod.converter() == (mine / "convert_hf_to_gguf.py").resolve()


class TestEndToEnd:
    def test_the_command_trains_merges_exports_and_records_what_it_trained_on(
            self, dataset, llama_cpp, tmp_path, capsys):
        """One command, the whole path: an adapter, the base with it folded in, a GGUF the
        serve path can read, and a manifest that identifies the data by hash."""
        pytest.importorskip("peft")
        data, base = dataset
        out = tmp_path / "out"

        code = main(["--recipe", "tool-calls", "--data", str(data), "--out", str(out),
                     "--lora", "--lora-rank", "8", "--lora-targets", "q_proj,v_proj",
                     "--export-gguf", "--set", "steps=20", "--set", "batch_size=2",
                     "--set", "context=256", "--set", "learning_rate=0.001"])
        printed = capsys.readouterr().out
        assert code == 0, printed

        adapter = out / "adapter"
        assert (adapter / "adapter_model.safetensors").is_file()
        assert json.loads((adapter / "adapter_config.json").read_text())["r"] == 8
        assert (out / "merged" / "config.json").is_file()
        gguf = out / f"{base.name}-tools-Q8_0.gguf"
        assert gguf.is_file() and gguf.read_bytes()[:4] == b"GGUF"

        manifest = json.loads((out / "manifest.json").read_text())
        assert manifest["data"]["examples"] == sum(
            1 for f in ("train.jsonl", "holdout.jsonl")
            for line in (data / f).read_text().splitlines() if line.strip())
        assert len(manifest["data"]["sha256"]) == 64
        assert manifest["lora"]["settings"]["rank"] == 8
        assert manifest["lora"]["settings"]["targets"] == ["q_proj", "v_proj"]
        assert manifest["lora"]["gguf"] == str(gguf)
        assert manifest["steps"] == 20 and manifest["base"] == str(base)

    def test_the_manifest_changes_when_the_data_does(self, dataset, tmp_path):
        data, _ = dataset
        before = fingerprint(data)
        (data / "train.jsonl").write_text(
            (data / "train.jsonl").read_text() + json.dumps(
                {"messages": [{"role": "user", "content": "one more"},
                              {"role": "assistant", "content": "chat"}]}) + "\n")
        after = fingerprint(data)
        assert after["sha256"] != before["sha256"]
        assert after["examples"] == before["examples"] + 1
        assert {f["file"] for f in after["files"]} == {"train.jsonl", "holdout.jsonl"}

    def test_the_exported_file_passes_the_serve_paths_own_preflight(self, tmp_path):
        """The smoke: the file that was just written is one `ml-stack-serve up` would
        agree to load -- asked with the preflight's own seams, so no server starts and no
        tensor is read."""
        gguf = write_gguf(tmp_path / "caller-tools-Q8_0.gguf", GEMMA_META)
        binary = tmp_path / "llama-server"
        binary.write_text("#!/bin/sh\necho '-m, --model FNAME  model path'\n")
        binary.chmod(binary.stat().st_mode | stat.S_IEXEC)

        report = lora_mod.preflight_export(
            gguf, binary=binary, arches=lambda _b: {"gemma3"},
            flags=lambda _b: frozenset(), limit_bytes=64 * 10 ** 9)
        assert report.ok, report.said()
        assert "gemma3" in report.said()
        assert lora_mod.summarise(report).endswith("all ok")

    def test_an_architecture_the_build_cannot_read_is_caught_before_the_load(self, tmp_path):
        gguf = write_gguf(tmp_path / "caller-tools-Q8_0.gguf",
                          {**GEMMA_META, "general.architecture": "invented-arch"})
        report = lora_mod.preflight_export(
            gguf, binary=tmp_path / "llama-server", arches=lambda _b: {"gemma3"},
            flags=lambda _b: frozenset(), limit_bytes=64 * 10 ** 9)
        assert not report.ok
        assert "invented-arch" in lora_mod.summarise(report)

    def test_a_dry_run_measures_a_step_and_writes_nothing(self, dataset, tmp_path, capsys):
        pytest.importorskip("peft")
        data, _ = dataset
        out = tmp_path / "dry"
        code = main(["--recipe", "tool-calls", "--data", str(data), "--out", str(out),
                     "--lora", "--dry-run", "--set", "steps=4000", "--set", "batch_size=2",
                     "--set", "context=256"])
        printed = capsys.readouterr().out
        assert code == 0, printed
        assert "measured:" in printed and "s/step" in printed
        assert not (out / "adapter").exists()
        assert not list(out.glob("*.gguf"))
