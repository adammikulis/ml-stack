"""Memory estimates from a header, against real load logs and synthetic machines."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from conftest import write_gguf

from ml_stack.hub.probe import (
    GIB,
    Gpu,
    MachineMemory,
    amd_gpus,
    linux_available,
    macos_available,
    nvidia_gpus,
)
from ml_stack.serve import estimate as est
from ml_stack.serve.loadlog import parse_load_log
from ml_stack.serve.suggest import Candidate, Want, suggest, suggest_meta, suggest_model

FIXTURE = json.loads((Path(__file__).parent / "fixtures" / "estimate_logs.json").read_text())

LAPTOP_8 = MachineMemory(8 * GIB, 5 * GIB, True, int(5.3 * GIB), (), "Darwin")
UNIFIED_16 = MachineMemory(16 * GIB, 11 * GIB, True, 12 * GIB, (), "Darwin")
UNIFIED_64 = MachineMemory(64 * GIB, 50 * GIB, True, 48 * GIB, (), "Darwin")
CARD_24 = MachineMemory(32 * GIB, 24 * GIB, False, 0,
                        (Gpu("card", 24 * GIB, 23 * GIB, "nvidia"),), "Linux")
CPU_ONLY = MachineMemory(16 * GIB, 11 * GIB, False, 0, (), "Linux")


def dense(layers=32, kv_heads=8, dim=128, train=32768, **extra) -> dict:
    return {"general.architecture": "llama", "llama.block_count": layers,
            "llama.context_length": train, "llama.embedding_length": 4096,
            "llama.attention.head_count": 32, "llama.attention.head_count_kv": kv_heads,
            "llama.attention.key_length": dim, "llama.attention.value_length": dim, **extra}


@pytest.mark.parametrize("entry", FIXTURE, ids=lambda e: e["model"][:24])
def test_estimates_match_recorded_llama_cpp_load_logs(entry):
    for run in entry["runs"]:
        got = est.estimate_meta(entry["meta"], entry["file_bytes"],
                                est.Setup(context=run["context"], kv_cache_type=run["kv"],
                                          flash_attn=True))
        seen = parse_load_log(run["log"])
        kv = seen.kv_bytes + seen.swa_bytes
        assert abs(got.kv_cache_bytes - kv) <= 0.1 * kv + 2**20, run
        assert got.state_bytes == pytest.approx(seen.recurrent_bytes, rel=0.01)
        actual = entry["file_bytes"] + kv + seen.recurrent_bytes + seen.compute
        assert got.total_bytes == pytest.approx(actual, rel=0.05), run


def test_q8_0_is_about_half_of_f16_for_the_cache():
    full = est.estimate_meta(dense(), 4 * GIB, est.Setup(kv_cache_type="f16", flash_attn=True))
    half = est.estimate_meta(dense(), 4 * GIB, est.Setup(flash_attn=True))
    assert est.DEFAULT_KV == "q8_0" and half.kv_cache_type == "q8_0"
    assert half.kv_cache_bytes / full.kv_cache_bytes == pytest.approx(34 / 64, rel=0.001)


def test_a_quantised_v_cache_needs_flash_attention():
    got = est.estimate_meta(dense(), GIB, est.Setup(flash_attn=False))
    assert got.kv_cache_type == "q8_0/f16"
    full = est.estimate_meta(dense(), GIB, est.Setup(kv_cache_type="f16", flash_attn=False))
    assert full.kv_cache_bytes > got.kv_cache_bytes > full.kv_cache_bytes * 0.7
    assert any("V cache" in n for n in got.notes)


def test_context_and_slots_multiply_the_cache():
    one = est.estimate_meta(dense(), GIB, est.Setup(context=4096, flash_attn=True))
    four = est.estimate_meta(dense(), GIB, est.Setup(context=4096, parallel=4, flash_attn=True))
    assert four.kv_cache_bytes == 4 * one.kv_cache_bytes
    assert four.context == 4096 and four.parallel == 4


def test_a_recurrent_layer_costs_state_per_slot_and_no_cache_per_token():
    hybrid = dense(layers=8, **{"llama.full_attention_interval": 4, "llama.ssm.conv_kernel": 4,
                                "llama.ssm.inner_size": 1024, "llama.ssm.state_size": 64,
                                "llama.ssm.group_count": 4})
    plain = est.estimate_meta(dense(layers=8), GIB, est.Setup(flash_attn=True))
    mixed = est.estimate_meta(hybrid, GIB, est.Setup(parallel=2, flash_attn=True))
    assert mixed.kv_cache_bytes == plain.kv_cache_bytes * 2 // 4
    assert mixed.state_bytes == 2 * 6 * ((3 * (1024 + 2 * 4 * 64) + 64 * 1024) * 4)


def test_partial_offload_moves_weights_to_the_cpu_side():
    whole = est.estimate_meta(dense(), 8 * GIB, est.Setup(flash_attn=True))
    half = est.estimate_meta(dense(), 8 * GIB, est.Setup(n_gpu_layers=16, flash_attn=True))
    assert half.gpu_bytes < whole.gpu_bytes * 0.6 and half.cpu_bytes > whole.cpu_bytes
    assert half.gpu_bytes + half.cpu_bytes == half.total_bytes


def test_the_breakdown_adds_up_to_the_total():
    got = est.estimate_meta(dense(), 3 * GIB, est.Setup(mmproj_bytes=GIB // 2, draft_bytes=GIB))
    assert sum(got.breakdown.values()) == got.total_bytes
    assert {"Weights", "KV cache", "Compute buffers", "Vision projector",
            "Draft model"} <= set(got.breakdown)
    row = got.as_dict()
    assert row["breakdown"][0] == {"name": "Weights", "bytes": 3 * GIB}


def test_a_header_without_cache_keys_is_marked_approximate():
    got = est.estimate_meta({"general.architecture": "x"}, GIB, est.Setup())
    assert got.kv_cache_bytes == 0 and got.confidence == "approx"
    assert any("lacks" in n for n in got.notes)


def test_a_dense_model_with_flash_attention_is_exact_from_the_header():
    got = est.estimate_meta(dense(), GIB, est.Setup(flash_attn=True))
    assert got.confidence == "exact-from-header"


def test_a_mixture_of_experts_says_all_experts_stay_resident():
    got = est.estimate_meta(dense(**{"llama.expert_count": 128, "llama.expert_used_count": 8}),
                            GIB, est.Setup())
    assert any("8 of 128" in n for n in got.notes)


def test_the_score_matrix_is_counted_without_flash_attention():
    on = est.estimate_meta(dense(), GIB, est.Setup(context=8192, flash_attn=True))
    off = est.estimate_meta(dense(), GIB, est.Setup(context=8192, flash_attn=False))
    assert off.compute_buffer_bytes - on.compute_buffer_bytes == 32 * 512 * 8192 * 4


def est_for(used: float, budget: int = 10 * GIB) -> est.Estimate:
    base = est.estimate_meta(dense(), GIB, est.Setup(flash_attn=True))
    return est.Estimate(**{**{f: getattr(base, f) for f in base.__slots__},
                           "total_bytes": int(used * budget), "gpu_bytes": 0,
                           "cpu_bytes": int(used * budget)})


def test_verdict_is_green_to_seventy_percent_then_yellow_then_red():
    machine = MachineMemory(100 * GIB, 12 * GIB, False, 0, (), "Linux")
    reserve = 2 * GIB
    rate = lambda share: est.verdict(est_for(share), machine, reserve)  # noqa: E731
    assert [rate(0.5), rate(0.7), rate(0.71), rate(1.0), rate(1.01)] == [
        "green", "green", "yellow", "yellow", "red"]


def test_the_default_reserve_is_two_gib_or_a_tenth_of_ram():
    assert est.reserve_default(LAPTOP_8) == 2 * GIB
    assert est.reserve_default(UNIFIED_64) == 6.4 * GIB // 1 or est.reserve_default(
        UNIFIED_64) == 64 * GIB // 10


def test_an_unknown_machine_has_no_verdict():
    assert est.verdict(est_for(0.1), MachineMemory(), None) == "none"


def test_unified_memory_is_limited_by_the_gpu_working_set():
    small = est.estimate_meta(dense(), 7 * GIB, est.Setup(context=2048, flash_attn=True))
    roomy = MachineMemory(64 * GIB, 60 * GIB, True, 6 * GIB, (), "Darwin")
    assert est.verdict(small, roomy) == "red"
    assert est.verdict(small, UNIFIED_64) == "green"


def test_a_discrete_card_rates_the_gpu_part_against_free_vram():
    big = est.estimate_meta(dense(), 26 * GIB, est.Setup(flash_attn=True))
    assert est.verdict(big, CARD_24) == "red"
    split = est.estimate_meta(dense(), 26 * GIB, est.Setup(n_gpu_layers=16, flash_attn=True))
    assert est.verdict(split, CARD_24) in ("yellow", "red", "green")
    assert split.gpu_bytes < 23 * GIB


@pytest.fixture
def small_model(tmp_path):
    path = tmp_path / "small-Q4_K_M.gguf"
    write_gguf(path, dense(layers=16, train=65536))
    with path.open("ab") as f:
        f.write(b"0" * (GIB // 4))
    return path


def test_max_context_is_the_longest_that_holds_the_verdict(small_model):
    machine = MachineMemory(8 * GIB, 4 * GIB, False, 0, (), "Linux")
    green = est.max_context(small_model, machine, max_verdict="green")
    yellow = est.max_context(small_model, machine, max_verdict="yellow")
    assert 0 < green <= yellow <= 65536 and green % 256 == 0
    over = est.estimate(small_model, context=yellow + 256)
    assert est.verdict(over, machine) == "red" or yellow == 65536
    assert est.verdict(est.estimate(small_model, context=green), machine) == "green"


def test_max_context_is_zero_when_the_weights_alone_do_not_fit(small_model):
    tiny = MachineMemory(2 * GIB, 2 * GIB, False, 0, (), "Linux")
    assert est.max_context(small_model, tiny) == 0


def test_estimate_reads_an_installed_file(small_model):
    got = est.estimate(small_model, context=1024)
    assert got.weights_bytes == small_model.stat().st_size
    assert got.trained_context == 65536
    with pytest.raises(FileNotFoundError):
        est.estimate("no-such-model.gguf", context=1024)


@pytest.mark.parametrize("machine", [LAPTOP_8, UNIFIED_16, UNIFIED_64, CARD_24, CPU_ONLY],
                         ids=["8GB", "16GB", "64GB", "24GB-card", "cpu"])
@pytest.mark.parametrize("goal", ["agent", "chat", "long-context", "fast"])
def test_a_suggestion_is_never_worse_than_asked_unless_nothing_fits(machine, goal):
    found = dense(train=131072)
    got = suggest_meta(found, 4 * GIB, machine, Want(goal))
    assert got.context <= 131072 and got.parallel == 1
    assert got.verdict == "green" or "Nothing fits" in got.reasons[0]
    assert got.kv_cache_type in ("q8_0", "q8_0/f16", "q4_0", "q4_0/f16")


def test_agent_on_a_big_machine_aims_for_32k_one_slot_all_layers():
    got = suggest_meta(dense(train=131072), 4 * GIB, UNIFIED_64)
    assert (got.context, got.parallel, got.n_gpu_layers) == (32768, 1, "auto")
    assert got.kv_cache_type == "q8_0" and got.batch == 2048 and got.verdict == "green"
    assert got.reasons[0].startswith("Context 32k: the largest that stays green with ")
    assert any("q8_0" in r for r in got.reasons)


def test_long_context_takes_the_largest_that_fits_and_never_the_trained_limit():
    got = suggest_meta(dense(train=65536), 4 * GIB, UNIFIED_64, Want("long-context"))
    assert got.context == 65536 and "capped" in " ".join(got.reasons)
    tight = suggest_meta(dense(train=1_000_000), 4 * GIB, UNIFIED_16, Want("long-context"))
    assert 8192 <= tight.context < 1_000_000 and tight.verdict == "green"


def test_a_small_machine_gets_a_smaller_context_for_the_same_model():
    small = suggest_meta(dense(train=131072), 2 * GIB, LAPTOP_8)
    big = suggest_meta(dense(train=131072), 2 * GIB, UNIFIED_64)
    assert small.context < big.context


def test_nothing_fitting_returns_the_smallest_option_with_the_reason():
    got = suggest_meta(dense(), 30 * GIB, LAPTOP_8)
    assert got.verdict == "red" and got.context == 512
    assert "Nothing fits" in got.reasons[0] and got.kv_cache_type == "q4_0"


def test_a_card_that_cannot_hold_the_model_offloads_some_layers():
    got = suggest_meta(dense(layers=40), 26 * GIB, CARD_24, Want(max_verdict="yellow"))
    assert isinstance(got.n_gpu_layers, int) and 0 < got.n_gpu_layers < 40


def test_without_a_gpu_no_layers_are_offloaded():
    got = suggest_meta(dense(), 2 * GIB, CPU_ONLY)
    assert got.n_gpu_layers == 0


def test_without_flash_attention_the_v_cache_stays_f16_and_the_reason_says_so():
    odd = dense(dim=72)
    got = suggest_meta(odd, 2 * GIB, UNIFIED_64)
    assert got.flash_attn is False and got.kv_cache_type == "q8_0/f16"
    assert any("Flash attention is not available" in r for r in got.reasons)


def test_alternatives_offer_a_smaller_and_a_bigger_option():
    got = suggest_meta(dense(train=131072), 4 * GIB, UNIFIED_64)
    labels = [o.label for o in got.alternatives]
    assert labels[0].startswith("smaller and faster: 16k")
    assert labels[1].startswith("bigger and tighter: 64k")
    assert got.alternatives[0].verdict == "green"


def test_suggestions_are_deterministic():
    one = suggest_meta(dense(train=131072), 4 * GIB, UNIFIED_16).as_dict()
    two = suggest_meta(dense(train=131072), 4 * GIB, UNIFIED_16).as_dict()
    assert one == two


def test_an_unknown_goal_is_refused():
    with pytest.raises(ValueError, match="goal is one of"):
        suggest_meta(dense(), GIB, UNIFIED_16, Want("nonsense"))


def test_suggest_reads_an_installed_model(small_model):
    got = suggest(small_model, UNIFIED_16, goal="chat")
    assert got.context <= 65536 and got.verdict == "green"
    assert got.setup().kv_cache_type == got.kv_cache_type


def test_models_are_ranked_larger_first_when_they_fit():
    fits = [Candidate("tiny-Q4", 1 * GIB, 1_000_000_000, "Q4_K_M", "llama", 8192),
            Candidate("mid-Q4", 4 * GIB, 7_000_000_000, "Q4_K_M", "llama", 8192),
            Candidate("huge-Q4", 40 * GIB, 70_000_000_000, "Q4_K_M", "llama", 8192),
            Candidate("embed-Q8", 1 * GIB, 300_000_000, "Q8_0", "bert", 512)]
    ranked = suggest_model(fits, UNIFIED_16)
    assert [r.candidate.name for r in ranked] == ["mid-Q4", "tiny-Q4", "huge-Q4"]
    assert [r.verdict for r in ranked][0] == "green" and ranked[-1].verdict == "red"


def test_fast_prefers_the_smaller_model_that_is_still_useful():
    pool = [Candidate("a", 1 * GIB, 1_000_000_000, "Q4_K_M", "llama"),
            Candidate("b", 3 * GIB, 4_000_000_000, "Q4_K_M", "llama"),
            Candidate("c", 8 * GIB, 14_000_000_000, "Q4_K_M", "llama")]
    assert suggest_model(pool, UNIFIED_64, "fast")[0].candidate.name == "b"
    assert suggest_model(pool, UNIFIED_64, "agent")[0].candidate.name == "c"


def test_lower_quantisations_rank_below_higher_at_the_same_size():
    pool = [Candidate("q2", 10 * GIB, 20_000_000_000, "Q2_K", "llama"),
            Candidate("q8", 10 * GIB, 20_000_000_000, "Q8_0", "llama")]
    assert [r.candidate.name for r in suggest_model(pool, UNIFIED_64)] == ["q8", "q2"]


VM_STAT = """Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pages free:                                5220.
Pages active:                              1608855.
Pages inactive:                            1604216.
Pages speculative:                         2405.
Pages purgeable:                           10.
"""


def test_macos_available_adds_free_inactive_speculative_and_purgeable_pages():
    assert macos_available(VM_STAT) == 16384 * (5220 + 1604216 + 2405 + 10)


def test_linux_available_reads_memavailable():
    assert linux_available("MemTotal: 100 kB\nMemAvailable:    2048 kB\n") == 2048 * 1024
    assert linux_available("MemTotal: 100 kB\n") == 0


def test_nvidia_smi_csv_is_read_in_mebibytes():
    got = nvidia_gpus("NVIDIA GeForce RTX 4090, 24564, 23000\nbroken line\n")
    assert got == (Gpu("NVIDIA GeForce RTX 4090", 24564 * 2**20, 23000 * 2**20, "nvidia"),)


def test_rocm_smi_json_is_read_in_bytes():
    text = json.dumps({"card0": {"VRAM Total Memory (B)": "17163091968",
                                 "VRAM Total Used Memory (B)": "1000"}})
    (card,) = amd_gpus(text)
    assert (card.total_bytes, card.free_bytes, card.vendor) == (17163091968, 17163090968, "amd")
    assert amd_gpus("not json") == ()
