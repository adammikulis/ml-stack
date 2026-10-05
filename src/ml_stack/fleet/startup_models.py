"""Model installation choices for the first-run wizard."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from ml_stack.hub.probe import GIB, MachineMemory, machine_memory
from ml_stack.serve.estimate import DEFAULT_KV, Setup, estimate_meta
from ml_stack.serve.fit import records
from ml_stack.serve.preflight import _head_kv_per_token

from .catalogue import SUGGESTED, Suggestion

CONTEXT = 8192
QWEN_27B_HEADER = {
    "general.architecture": "qwen3_5", "qwen3_5.block_count": 64,
    "qwen3_5.attention.head_count": 24, "qwen3_5.attention.head_count_kv": 4,
    "qwen3_5.attention.key_length": 256, "qwen3_5.attention.value_length": 256,
    "qwen3_5.full_attention_interval": 4, "qwen3_5.nextn_predict_layers": 1,
    "qwen3_5.ssm.conv_kernel": 4, "qwen3_5.ssm.inner_size": 6144,
    "qwen3_5.ssm.state_size": 128, "qwen3_5.ssm.group_count": 16,
}


def memory_pool(machine: MachineMemory) -> tuple[str, int, int]:
    """The largest card's capacity and free memory, or the machine's shared pool."""
    if machine.gpus:
        card = max(machine.gpus, key=lambda gpu: gpu.total_bytes)
        return "VRAM", card.total_bytes, card.free_bytes
    if machine.unified:
        capacity = machine.gpu_limit_bytes or machine.total_ram
        return "unified GPU memory", capacity, min(capacity, machine.available_ram)
    return "RAM", machine.total_ram, machine.available_ram


def memory_need(pick: Suggestion, context: int = CONTEXT) -> int:
    """Estimated bytes for weights, prediction head and one chat context."""
    measured = [row for row in records() if row.model == pick.file
                and row.cache_type == "q8_0" and bool(row.spec) == bool(pick.draft_ref)]
    if measured:
        row = measured[-1]
        return row.loaded() + row.cost(context)
    if pick.params_b == 27 and pick.family in {"", "Qwen"}:
        draft_kv = (_head_kv_per_token(QWEN_27B_HEADER, DEFAULT_KV, DEFAULT_KV) * context
                    if pick.draft_ref or pick.mtp_ref else 0)
        return estimate_meta(QWEN_27B_HEADER, int((pick.mtp_gb or pick.gb) * GIB), Setup(
            context=context, draft_bytes=int(pick.draft_gb * GIB),
            draft_kv_bytes=draft_kv)).total_bytes
    return int((pick.gb + pick.draft_gb) * GIB * 1.25) + GIB


def choices(*, machine: MachineMemory | None = None, disk_gb: float = 0,
            installed: Iterable[str] = ()) -> dict[str, Any]:
    """Model offers ranked by card capacity, with current memory reported separately."""
    pool, capacity, free = memory_pool(machine or machine_memory())
    tier = round(capacity / GIB) if pool != "RAM" else 0
    here = set(installed)
    rows = []
    for pick in SUGGESTED:
        if pick.params_b == 27 and tier < 24:
            continue
        need = memory_need(pick)
        if not capacity or need >= capacity * 0.95:
            continue
        if disk_gb and pick.file not in here and (pick.mtp_gb or pick.gb) + pick.draft_gb > disk_gb:
            continue
        rows.append({**pick.public(), "installed": pick.file in here,
                     "memory_gb": round(need / GIB, 1), "fits_now": need < free * 0.95,
                     "estimated_context": max((tokens for tokens in
                        (8192, 16384, 32768, 65536, 131072, 262144)
                        if memory_need(pick, tokens) <= capacity * 0.9), default=0)
                        if pick.params_b == 27 else 0,
                     "recommended": False})
    preferred = next((row for row in rows if row["params_b"] == 27 and "UD-Q4_K_XL" in row["file"]), None)
    if preferred is None:
        preferred = next((row for row in rows if row["params_b"] == 27), None)
    if preferred is None:
        preferred = next((row for row in reversed(rows) if row["family"] == "Qwen"), None)
    if preferred is not None:
        preferred["recommended"] = True
    return {"ok": True, "models": rows, "memory": {"pool": pool, "capacity_gb": round(capacity / GIB, 1),
                                        "free_gb": round(free / GIB, 1)}, "context": CONTEXT}
