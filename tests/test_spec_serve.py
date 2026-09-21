"""An MLX model served with tree decoding, asked the way every client asks: over HTTP.

The model is a tiny randomly initialised ``qwen3_5`` written to a directory with a word-level
tokenizer, so `mlx_lm.load` reads it the way it reads a downloaded one.
"""

from __future__ import annotations

import json
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

mx = pytest.importorskip("mlx.core", reason="ml-stack[spec]")
pytest.importorskip("tokenizers", reason="pip install tokenizers")
pytest.importorskip("mlx_lm.models.qwen3_5",
                    reason="qwen3_5 arrived in mlx-lm 0.31.3: pip install 'ml-stack[spec]'")

from mlx.utils import tree_flatten  # noqa: E402
from mlx_lm import load  # noqa: E402
from test_spec_decode import plain_greedy, tiny_hybrid  # noqa: E402
from tokenizers import Tokenizer, decoders, models, pre_tokenizers  # noqa: E402

from ml_stack.client import Client, Request  # noqa: E402
from ml_stack.graph.serve import Handler  # noqa: E402
from ml_stack.http import ServerError  # noqa: E402
from ml_stack.serve.backend import ServerSpec  # noqa: E402
from ml_stack.serve.manager import ServerManager  # noqa: E402
from ml_stack.serve.mlx_tree_server import TreeCompleter  # noqa: E402
from ml_stack.serve.ports import free_port  # noqa: E402
from ml_stack.spec.engine import Engine, EngineConfig  # noqa: E402

TEMPLATE = ("{% for m in messages %}<|im_start|> {{ m['content'] }} <|im_end|> {% endfor %}"
            "{% if add_generation_prompt %}<|im_start|>{% endif %}")
ASKED = [{"role": "user", "content": "w5 w7 w11 w13 w17 w19"}]


@pytest.fixture(scope="module")
def weights(tmp_path_factory: pytest.TempPathFactory) -> Path:
    where = tmp_path_factory.mktemp("tiny-mlx")
    model = tiny_hybrid()
    vocab = {f"w{i}": i for i in range(94)}
    vocab.update({"<|im_start|>": 94, "<|im_end|>": 95, "</think>": 96})
    tokenizer = Tokenizer(models.WordLevel(vocab, unk_token="w0"))
    tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer.decoder = decoders.WordPiece(prefix="##")
    tokenizer.save(str(where / "tokenizer.json"))
    (where / "tokenizer_config.json").write_text(json.dumps({
        "tokenizer_class": "PreTrainedTokenizerFast", "eos_token": "<|im_end|>",
        "chat_template": TEMPLATE}))
    text = vars(model.language_model.args)
    (where / "config.json").write_text(json.dumps({"model_type": "qwen3_5",
                                                   "text_config": dict(text)}))
    mx.save_safetensors(str(where / "model.safetensors"),
                        dict(tree_flatten(model.parameters())))
    return where


@pytest.fixture
def cache_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("ML_STACK_CACHE", str(tmp_path / "cache"))
    monkeypatch.setenv("ML_STACK_HOME", str(tmp_path / "home"))
    return tmp_path


def expected_text(weights: Path, count: int) -> str:
    model, tokenizer = load(str(weights))
    prompt = tokenizer.apply_chat_template(ASKED, add_generation_prompt=True)
    written = [t for t in plain_greedy(model, prompt, count) if t not in tokenizer.eos_token_ids]
    return tokenizer.decode(written).strip()


def test_a_served_tree_engine_answers_what_plain_greedy_decoding_writes(weights: Path,
                                                                       cache_home: Path) -> None:
    engine = Engine(EngineConfig(model=str(weights), drafter="ngram", max_nodes=8))
    completer = TreeCompleter(engine, name=weights.name, context=512, rule="lossless")
    server = ThreadingHTTPServer(("127.0.0.1", 0),
                                 Handler.configured(name="TreeTest", completer=completer))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        client = Client(f"http://127.0.0.1:{server.server_address[1]}",
                        request=Request(temperature=0.0, n_predict=16))
        whole = client.chat(ASKED, think=False)
        pieces: list[str] = []
        streamed = client.chat(ASKED, think=False, on_delta=lambda kind, text: pieces.append(text))
    finally:
        server.shutdown()
        server.server_close()
    wanted = expected_text(weights, 16)
    assert whole.content.strip() == wanted
    assert streamed.content.strip() == wanted
    assert "".join(pieces).strip() == wanted
    timings = whole.raw["timings"]
    assert timings["verify_n"] >= 1
    assert timings["predicted_n"] == whole.raw["usage"]["completion_tokens"]


def test_a_streamed_failure_reaches_the_client_as_an_error(weights: Path,
                                                           cache_home: Path) -> None:
    engine = Engine(EngineConfig(model=str(weights), drafter="ngram", max_nodes=8))
    completer = TreeCompleter(engine, name=weights.name, context=512, rule="no-such-rule")
    server = ThreadingHTTPServer(("127.0.0.1", 0),
                                 Handler.configured(name="TreeTest", completer=completer))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        client = Client(f"http://127.0.0.1:{server.server_address[1]}",
                        request=Request(temperature=0.0, n_predict=4))
        with pytest.raises(ServerError, match="no-such-rule"):
            client.chat(ASKED, think=False, on_delta=lambda kind, text: None)
    finally:
        server.shutdown()
        server.server_close()


@pytest.mark.slow
def test_a_lease_on_mlx_weights_starts_the_tree_server_and_it_answers(weights: Path,
                                                                     cache_home: Path) -> None:
    manager = ServerManager(state_file=cache_home / "servers.json")
    spec = ServerSpec(model=str(weights), port=free_port(), context=512, draft="ngram",
                      spec_draft_max=8)
    info = manager.lease(spec, timeout=180.0, roam=False, anyway=True)
    try:
        assert info.backend == "mlx-tree"
        reply = Client(info.base_url,
                       request=Request(temperature=0.0, n_predict=16)).chat(ASKED, think=False)
        assert reply.content.strip() == expected_text(weights, 16)
        assert reply.raw["timings"]["verify_n"] >= 1
    finally:
        manager.release(info)
