"""A Qwen4-Exp checkpoint on disk, loaded the way `Engine` loads Flash-Next.

The checkpoint is a tiny random ``qwen4_exp`` with a 4-bit PLE n-gram table, written as a
sharded Hub download is: safetensors, an index and a config. The load reads the table from
disk through a hard-linked view instead of holding it, and decodes as the resident model does.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

mx = pytest.importorskip("mlx.core", reason="ml-stack[spec]")
pytest.importorskip("mlx_vlm.models.qwen4_exp", reason="ml-stack[spec]")
pytest.importorskip("tokenizers", reason="pip install tokenizers")

import mlx.nn as nn  # noqa: E402
from mlx.utils import tree_flatten  # noqa: E402
from mlx_vlm.models.qwen4_exp import Model, ModelConfig  # noqa: E402
from mlx_vlm.utils import load_model  # noqa: E402
from test_spec_qwen4 import plain_greedy, tiny_flash  # noqa: E402
from tokenizers import Tokenizer, decoders, models, pre_tokenizers  # noqa: E402

from ml_stack.serve.backend import ServerSpec  # noqa: E402
from ml_stack.serve.mlx_tree import report_for  # noqa: E402
from ml_stack.spec.decode import Asked, decode  # noqa: E402
from ml_stack.spec.engine import Engine, EngineConfig, load_target  # noqa: E402

EOS = 96
PLE = ".ple.ple_embedding.ngram_embedding."
TEMPLATE = ("{% for m in messages %}<|im_start|> {{ m['content'] }} <|im_end|> {% endfor %}"
            "{% if add_generation_prompt %}<|im_start|>{% endif %}")
VISION = {"depth": 1, "hidden_size": 16, "intermediate_size": 32, "num_heads": 2,
          "out_hidden_size": 64, "patch_size": 2, "spatial_merge_size": 1,
          "temporal_patch_size": 1, "num_position_embeddings": 16,
          "deepstack_visual_indexes": []}


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory: pytest.TempPathFactory) -> Path:
    where = tmp_path_factory.mktemp("tiny-flash")
    text = {**vars(tiny_flash().args), "ple_embed_dim": 512}
    raw = {"model_type": "qwen4_exp", "text_config": text, "vision_config": VISION,
           "eos_token_id": EOS}
    model = Model(ModelConfig.from_dict(raw))
    model.set_dtype(mx.bfloat16)
    nn.quantize(model, group_size=32, bits=4,
                class_predicate=lambda path, module: PLE[:-1] in path + ".")
    mx.eval(model.parameters())
    params = {name: value if PLE in name or not mx.issubdtype(value.dtype, mx.floating)
              else value.astype(mx.float32)
              for name, value in tree_flatten(model.parameters())}
    mx.save_safetensors(str(where / "model.safetensors"), params, metadata={"format": "mlx"})
    (where / "model.safetensors.index.json").write_text(json.dumps(
        {"metadata": {}, "weight_map": dict.fromkeys(params, "model.safetensors")}))
    raw["quantization"] = {"group_size": 32, "bits": 4, "mode": "affine"}
    (where / "config.json").write_text(json.dumps(raw))
    vocab = {f"w{i}": i for i in range(94)}
    vocab.update({"<|im_start|>": 94, "<|im_end|>": 95, "</think>": 96})
    tokenizer = Tokenizer(models.WordLevel(vocab, unk_token="w0"))
    tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer.decoder = decoders.WordPiece(prefix="##")
    tokenizer.save(str(where / "tokenizer.json"))
    (where / "tokenizer_config.json").write_text(json.dumps({
        "tokenizer_class": "PreTrainedTokenizerFast", "eos_token": "<|im_end|>",
        "chat_template": TEMPLATE}))
    return where


@pytest.fixture
def cache_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("ML_STACK_CACHE", str(tmp_path / "cache"))
    monkeypatch.setenv("ML_STACK_HOME", str(tmp_path / "home"))
    return tmp_path


def test_the_loaded_model_reads_its_ngram_table_from_disk_and_decodes_as_the_resident_one(
        checkpoint: Path, cache_home: Path) -> None:
    resident = load_model(checkpoint).language_model
    model, tokenizer = load_target(checkpoint)
    text = model.language_model
    table = text.model.layers[1].ple.ple_embedding.ngram_embedding
    assert not any(PLE in name for name, _ in tree_flatten(text.parameters()))
    prompt = [(7 * i + 2) % 90 for i in range(24)]
    assert table(mx.array([[3, 5]])).shape == (1, 2, 32)
    assert plain_greedy(text, prompt, 24) == plain_greedy(resident, prompt, 24)
    assert EOS in tokenizer.eos_token_ids


def test_a_tree_engine_on_the_checkpoint_writes_what_plain_greedy_decoding_writes(
        checkpoint: Path, cache_home: Path) -> None:
    engine = Engine(EngineConfig(model=str(checkpoint), drafter="ngram", max_nodes=8))
    prompt = [(5 * i + 3) % 90 for i in range(30)]
    out = decode(engine.session, prompt, Asked(32))
    assert out.tokens == plain_greedy(engine.model.language_model, prompt, 32)


def test_the_fit_charges_the_weights_a_load_holds_and_not_the_mapped_table(
        checkpoint: Path, cache_home: Path) -> None:
    report = report_for(ServerSpec(model=f"mlx:{checkpoint}", draft="ngram"))
    held = sum(v.nbytes for k, v in mx.load(str(checkpoint / "model.safetensors")).items()
               if PLE not in k)
    assert report.ok
    assert report.weights_bytes == held
