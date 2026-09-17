"""Tree verification of Qwen4-Exp (Flash-Next) on a tiny random model, against mlx-vlm's decoding.

Every feature is on: hyper-connections, a PLE n-gram layer with its dilated convolution, MoE
routing, and QSA attention with an indexer budget of four blocks of four, so a node attends
sparsely from position 19 and the prompts below sit before, across and past that point. The
reference is mlx-vlm's own model, prefilled with the prompt and then fed one token at a time.
"""

from __future__ import annotations

from collections.abc import Sequence

import pytest

mx = pytest.importorskip("mlx.core", reason="ml-stack[spec]")
pytest.importorskip("mlx_vlm.models.qwen4_exp", reason="ml-stack[spec]")

from mlx_vlm.models.qwen4_exp.config import ModelConfig  # noqa: E402
from mlx_vlm.models.qwen4_exp.language import LanguageModel  # noqa: E402

from ml_stack.spec.decode import Asked, Session, decode  # noqa: E402
from ml_stack.spec.drafters import Budget  # noqa: E402
from ml_stack.spec.drafters.dflash import DFlashDrafter  # noqa: E402
from ml_stack.spec.drafters.dflash_model import DFlashConfig, DFlashDraftModel  # noqa: E402
from ml_stack.spec.drafters.mtp import Beam, MtpDrafter  # noqa: E402
from ml_stack.spec.drafters.mtp_qwen4 import Qwen4MtpHead  # noqa: E402
from ml_stack.spec.drafters.ngram import NgramDrafter  # noqa: E402
from ml_stack.spec.qwen4 import Qwen4ExpLayout  # noqa: E402
from ml_stack.spec.tree import Tree, chain  # noqa: E402
from ml_stack.spec.verify import TreeVerifier  # noqa: E402

EOS = 96
TREE = Tree([31, 37, EOS, 43, 47, 53, 59, 61, 62, 63], [-1, 0, 0, 1, 1, 2, 3, 6, 7, 8])


def tiny_flash(seed: int = 0) -> LanguageModel:
    mx.random.seed(seed)
    text = {
        "model_type": "qwen4_exp_text", "hidden_size": 64, "num_hidden_layers": 8,
        "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 16, "vocab_size": 97,
        "linear_num_value_heads": 4, "linear_num_key_heads": 2, "linear_key_head_dim": 32,
        "linear_value_head_dim": 12, "linear_conv_kernel_dim": 4, "full_attention_interval": 4,
        "num_experts": 8, "num_experts_per_tok": 2, "moe_intermediate_size": 32,
        "shared_expert_intermediate_size": 32, "rms_norm_eps": 1e-6,
        "max_position_embeddings": 4096, "hc_count": 4, "hc_lowrank": 8,
        "ple_layer_ids": [2], "ple_embed_dim": 64, "ngram_size": 3, "heads_per_ngram": 8,
        "ngram_vocab_size_base": 101, "make_ngram_vocab_size_divisible_by": 8,
        "split_ngram_parts": 4, "indexer_n_heads": 4, "indexer_kv_heads": 1,
        "indexer_head_dim": 16, "indexer_budget": 16, "indexer_compress_ratio": 4,
        "eos_token_id": EOS, "tie_word_embeddings": False,
        "rope_parameters": {"type": "default", "mrope_section": [2, 1, 1],
                            "rope_theta": 10000.0, "partial_rotary_factor": 0.5}}
    config = ModelConfig.from_dict({"model_type": "qwen4_exp", "text_config": text,
                                    "vision_config": {"deepstack_visual_indexes": []},
                                    "eos_token_id": EOS})
    model = LanguageModel(config.text_config, config)
    model.set_dtype(mx.float32)
    mx.eval(model.parameters())
    return model


@pytest.fixture(scope="module")
def model() -> LanguageModel:
    return tiny_flash()


def prompt_of(length: int) -> list[int]:
    return [(5 * i + 3) % 90 for i in range(length)]


def decoded(model: LanguageModel, tokens: Sequence[int], tail: Sequence[int]) -> mx.array:
    """The next-token logits after prefilling ``tokens`` and feeding ``tail`` one at a time."""
    cache = model.make_cache()
    logits = model(mx.array([list(tokens)]), cache=cache).logits[0, -1]
    for token in tail:
        logits = model(mx.array([[token]]), cache=cache).logits[0, -1]
    return logits


class Reference:
    """Drafts the next tokens of a known greedy continuation as one chain."""

    taps: tuple[int, ...] = ()

    def __init__(self, continuation: Sequence[int], depth: int = 6) -> None:
        self.continuation, self.depth, self.kept = list(continuation), depth, [0]

    def prefill(self, tokens: Sequence[int], hidden: object) -> None:
        self.kept = [0]

    def accept(self, tokens: Sequence[int], hidden: object) -> None:
        self.kept.append(self.kept[-1] + len(tokens))

    def state(self) -> object:
        return list(self.kept)

    def restore(self, state: object) -> None:
        self.kept = list(state)  # type: ignore[call-overload]

    def draft(self, root: int, hidden: object) -> Tree:
        done = self.kept[-1]
        return chain([root, *self.continuation[done:done + self.depth]])


def plain_greedy(model: LanguageModel, prompt: Sequence[int], count: int) -> list[int]:
    cache = model.make_cache()
    logits = model(mx.array([list(prompt)]), cache=cache).logits[0, -1]
    out: list[int] = []
    for _ in range(count):
        out.append(int(logits.argmax().item()))
        logits = model(mx.array([[out[-1]]]), cache=cache).logits[0, -1]
    return out


@pytest.mark.parametrize("length", [8, 17, 40])
def test_every_tree_node_scores_as_decoding_its_own_path_one_token_at_a_time(model,
                                                                             length) -> None:
    layout = Qwen4ExpLayout(model)
    prompt = prompt_of(length)
    verifier = TreeVerifier(layout, layout.make_cache())
    verifier.prefill(prompt)
    logits, _ = verifier.forward(TREE)
    for node in range(TREE.n):
        expected = decoded(model, prompt, [TREE.tokens[i] for i in TREE.path_to(node)])
        assert mx.allclose(logits[node], expected, atol=1e-4).item(), node


@pytest.mark.parametrize("length", [14, 40])
def test_a_committed_path_continues_as_if_it_had_been_decoded_one_token_at_a_time(
        model, length) -> None:
    layout = Qwen4ExpLayout(model)
    prompt = prompt_of(length)
    verifier = TreeVerifier(layout, layout.make_cache())
    verifier.prefill(prompt)
    verifier.forward(TREE)
    path = TREE.path_to(9)
    verifier.commit(path)
    kept = prompt + [TREE.tokens[i] for i in path]
    after = Tree([67, 71, 73, 5, 6], [-1, 0, 0, 2, 3])
    logits, _ = verifier.forward(after)
    for node in range(after.n):
        expected = decoded(model, kept, [after.tokens[i] for i in after.path_to(node)])
        assert mx.allclose(logits[node], expected, atol=1e-4).item(), node


def spark(layout: Qwen4ExpLayout, budget: Budget) -> DFlashDrafter:
    config = DFlashConfig(hidden_size=64, num_hidden_layers=2, num_attention_heads=4,
                          num_key_value_heads=2, head_dim=16, intermediate_size=96,
                          vocab_size=97, rms_norm_eps=1e-6, rope_theta=10000.0,
                          max_position_embeddings=4096, block_size=7, target_layer_ids=(1, 5),
                          mask_token_id=95, sample_from_anchor=True, causal=False)
    drafter = DFlashDraftModel(config)
    drafter.set_dtype(mx.float32)
    mx.eval(drafter.parameters())
    return DFlashDrafter(layout, drafter, config, budget)


def mtp(layout: Qwen4ExpLayout, budget: Budget) -> MtpDrafter:
    head = Qwen4MtpHead(layout.args)
    head.set_dtype(mx.float32)
    mx.eval(head.parameters())
    return MtpDrafter(layout, head, budget, Beam(depth=3))


def test_greedy_tree_decoding_is_plain_greedy_decoding_across_the_indexer_budget(model) -> None:
    layout = Qwen4ExpLayout(model)
    prompt = prompt_of(12)
    reference = plain_greedy(model, prompt, 40)
    budget = Budget(max_nodes=16)
    for drafter in (Reference(reference), NgramDrafter(budget), spark(layout, budget),
                    mtp(layout, budget)):
        out = decode(Session(layout, drafter), prompt, Asked(40))
        assert out.tokens == reference, type(drafter).__name__
        if isinstance(drafter, Reference):
            assert out.tokens_per_pass > 4


def test_a_prompt_that_extends_the_last_one_reuses_its_cache_past_the_indexer_budget(
        model) -> None:
    layout = Qwen4ExpLayout(model)
    prompt = prompt_of(12)
    session = Session(layout, NgramDrafter(Budget(16)))
    first = decode(session, prompt, Asked(20))
    longer = prompt + first.tokens + [31, 37]
    again = decode(session, longer, Asked(12))
    assert again.reused >= len(prompt)
    assert again.tokens == plain_greedy(model, longer, 12)
