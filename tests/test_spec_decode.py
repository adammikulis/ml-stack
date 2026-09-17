"""Tree verification on a small randomly initialised hybrid model, against plain mlx-lm decoding.

The model is mlx-lm's own ``qwen3_5`` at a tiny size, in float32, so its sequential forward
is the reference each claim is measured against.
"""

from __future__ import annotations

from collections.abc import Sequence

import pytest

mx = pytest.importorskip("mlx.core", reason="ml-stack[spec]")

from mlx_lm.generate import generate_step  # noqa: E402
from mlx_lm.models import qwen3_5  # noqa: E402
from mlx_lm.models.gated_delta import compute_g, gated_delta_update  # noqa: E402

from ml_stack.spec import deltanet  # noqa: E402
from ml_stack.spec.decode import Asked, Session, decode  # noqa: E402
from ml_stack.spec.drafters import Budget  # noqa: E402
from ml_stack.spec.drafters.dflash import DFlashDrafter  # noqa: E402
from ml_stack.spec.drafters.dflash_model import DFlashConfig, DFlashDraftModel  # noqa: E402
from ml_stack.spec.drafters.mtp import Beam, MtpDrafter, MtpHead  # noqa: E402
from ml_stack.spec.drafters.ngram import NgramDrafter  # noqa: E402
from ml_stack.spec.layout import layout_for  # noqa: E402
from ml_stack.spec.sample import Sampling, sample_rows  # noqa: E402
from ml_stack.spec.tree import Tree  # noqa: E402
from ml_stack.spec.verify import TreeVerifier  # noqa: E402

PROMPT = [5, 7, 11, 13, 17, 19, 23, 29]


def tiny_hybrid(seed: int = 0, layers: int = 8) -> object:
    mx.random.seed(seed)
    text = {"model_type": "qwen3_5_text", "hidden_size": 64, "intermediate_size": 128,
            "num_hidden_layers": layers, "num_attention_heads": 4, "num_key_value_heads": 2,
            "head_dim": 16, "vocab_size": 97, "linear_num_value_heads": 4,
            "linear_num_key_heads": 2, "linear_key_head_dim": 32, "linear_value_head_dim": 12,
            "linear_conv_kernel_dim": 4, "full_attention_interval": 4,
            "rope_parameters": {"rope_type": "default", "rope_theta": 10000.0,
                                "partial_rotary_factor": 0.5}}
    model = qwen3_5.Model(qwen3_5.ModelArgs.from_dict({"model_type": "qwen3_5",
                                                       "text_config": text}))
    model.set_dtype(mx.float32)
    mx.eval(model.parameters())
    return model


@pytest.fixture(scope="module")
def model() -> object:
    return tiny_hybrid()


def plain_greedy(model: object, prompt: Sequence[int], count: int) -> list[int]:
    return [int(token) for token, _ in generate_step(mx.array(list(prompt)), model,
                                                      max_tokens=count,
                                                      sampler=lambda logits: logits.argmax(-1))]


class KnownContinuation:
    """Drafts the continuation it was given along one branch, with a wrong sibling at each level."""

    taps: tuple[int, ...] = ()

    def __init__(self, continuation: Sequence[int], depth: int = 6) -> None:
        self.continuation, self.depth = list(continuation), depth
        self.done = 0

    def prefill(self, tokens: Sequence[int], hidden: object) -> None:
        self.done = 0

    def accept(self, tokens: Sequence[int], hidden: object) -> None:
        self.done += len(tokens)

    def state(self) -> object:
        return self.done

    def restore(self, state: object) -> None:
        self.done = int(state)  # type: ignore[call-overload]

    def draft(self, root: int, hidden: object) -> Tree:
        ahead = self.continuation[self.done:self.done + self.depth]
        tokens, parent, above = [root], [-1], 0
        for token in ahead:
            tokens += [(token + 1) % 97, token]
            parent += [above, above]
            above = len(tokens) - 1
        return Tree(tokens, parent)


def test_every_tree_node_scores_as_the_sequential_forward_of_its_own_path(model) -> None:
    layout = layout_for(model)
    tree = Tree([31, 37, 41, 43, 47, 53, 59, 61], [-1, 0, 0, 1, 1, 2, 3, 6])
    verifier = TreeVerifier(layout, layout.make_cache())
    verifier.prefill(PROMPT)
    logits, _ = verifier.forward(tree)
    for node in range(tree.n):
        path = PROMPT + [tree.tokens[i] for i in tree.path_to(node)]
        expected = model(mx.array(path)[None])[0, -1]
        assert mx.allclose(logits[node], expected, atol=1e-4).item(), node


def test_a_committed_path_continues_as_if_it_had_been_decoded_one_token_at_a_time(model) -> None:
    layout = layout_for(model)
    tree = Tree([31, 37, 41, 43, 47, 53, 59, 61], [-1, 0, 0, 1, 1, 2, 3, 6])
    verifier = TreeVerifier(layout, layout.make_cache())
    verifier.prefill(PROMPT)
    verifier.forward(tree)
    path = tree.path_to(7)
    verifier.commit(path)
    after = Tree([67, 71, 73], [-1, 0, 0])
    logits, _ = verifier.forward(after)
    kept = PROMPT + [tree.tokens[i] for i in path]
    for node, tail in ((0, [67]), (1, [67, 71]), (2, [67, 73])):
        expected = model(mx.array(kept + tail)[None])[0, -1]
        assert mx.allclose(logits[node], expected, atol=1e-4).item(), node


def test_the_deltanet_walk_matches_the_recurrence_along_each_path() -> None:
    mx.random.seed(3)
    tree = Tree([0] * 7, [-1, 0, 0, 1, 2, 4, 4])
    count, key_heads, value_heads, key_dim, value_dim = tree.n, 2, 4, 32, 12
    q = mx.random.normal((count, key_heads, key_dim))
    k = mx.random.normal((count, key_heads, key_dim)) * 0.2
    v = mx.random.normal((count, value_heads, value_dim))
    a = mx.random.normal((count, value_heads))
    b = mx.random.normal((count, value_heads))
    a_log, dt_bias = mx.random.normal((value_heads,)), mx.random.normal((value_heads,))
    beta, alpha = mx.sigmoid(b), compute_g(a_log, a, dt_bias)
    p, y = deltanet.walk(q, k, v, (alpha, beta), mx.array(tree.ancestors, mx.int32))
    state = mx.random.normal((1, value_heads, value_dim, key_dim)) * 0.1
    y = y + mx.matmul(p.transpose(1, 0, 2), state[0].transpose(0, 2, 1)).transpose(1, 0, 2)
    for node in range(tree.n):
        rows = mx.array(tree.path_to(node))
        out, _ = gated_delta_update(q[rows][None], k[rows][None], v[rows][None], a[rows][None],
                                    b[rows][None], a_log, dt_bias, state, None)
        assert mx.allclose(y[node], out[0, -1], atol=1e-4).item(), node


def test_greedy_tree_decoding_is_plain_greedy_decoding_when_the_draft_is_mostly_right(
        model) -> None:
    reference = plain_greedy(model, PROMPT, 40)
    session = Session(layout_for(model), KnownContinuation(reference))
    out = decode(session, PROMPT, Asked(40))
    assert out.tokens == reference
    assert out.tokens_per_pass > 4


def test_greedy_tree_decoding_is_plain_greedy_decoding_with_every_drafter(model) -> None:
    layout = layout_for(model)
    reference = plain_greedy(model, PROMPT, 24)
    budget = Budget(max_nodes=16)
    head = MtpHead(layout.args)
    head.set_dtype(mx.float32)
    config = DFlashConfig(hidden_size=64, num_hidden_layers=2, num_attention_heads=4,
                          num_key_value_heads=2, head_dim=16, intermediate_size=96,
                          vocab_size=97, rms_norm_eps=1e-6, rope_theta=10000.0,
                          max_position_embeddings=4096, block_size=8,
                          target_layer_ids=(1, 5), mask_token_id=96, selector_rank=8,
                          selector_top_k=6, conv_kernel_size=2, conv_group_size=16)
    dflash = DFlashDraftModel(config)
    dflash.set_dtype(mx.float32)
    mx.eval(head.parameters(), dflash.parameters())
    for drafter in (NgramDrafter(budget), MtpDrafter(layout, head, budget, Beam(depth=3)),
                    DFlashDrafter(layout, dflash, config, budget)):
        session = Session(layout, drafter)
        out = decode(session, PROMPT, Asked(24))
        assert out.tokens == reference, type(drafter).__name__


def test_a_prompt_that_extends_the_last_one_reuses_its_cache_and_decodes_the_same(model) -> None:
    layout = layout_for(model)
    first = decode(Session(layout, NgramDrafter(Budget(16))), PROMPT, Asked(12))
    longer = PROMPT + first.tokens + [31, 37]
    session = Session(layout, NgramDrafter(Budget(16)))
    decode(session, PROMPT, Asked(12))
    again = decode(session, longer, Asked(12))
    fresh = decode(Session(layout, NgramDrafter(Budget(16))), longer, Asked(12))
    assert again.reused >= len(PROMPT)
    assert again.tokens == fresh.tokens == plain_greedy(model, longer, 12)


def test_the_sampler_draws_from_the_distribution_it_reports() -> None:
    mx.random.seed(1)
    logits = (mx.random.normal((3, 200)) * 3).astype(mx.bfloat16)
    sampling = Sampling(temperature=0.8, top_k=10, top_p=0.9)
    drawn, ids, probs = sample_rows(logits, sampling, stats=True)
    mx.eval(drawn, ids, probs)
    for row in range(3):
        values = mx.sort(logits[row].astype(mx.float32))[::-1][:10]
        assert logits[row][ids[row, :10]].astype(mx.float32).tolist() == values.tolist()
        top = mx.softmax(values / 0.8)
        kept = int(mx.sum(mx.cumsum(top) - top < 0.9).item())
        assert mx.allclose(probs[row, :kept], top[:kept] / top[:kept].sum(), atol=1e-3).item()
        assert float(probs[row, kept:].sum()) == 0.0
        assert int(drawn[row]) in ids[row, :kept].tolist()
