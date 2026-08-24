# Copyright 2026 dfi / 0xdfi
# SPDX-License-Identifier: Apache-2.0
"""Source contracts only; these do not claim GPU/runtime correctness."""

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def text(path: str) -> str:
    return (ROOT / path).read_text()


def allocator_math():
    source = text("runtime/overlays/vllm/vllm/utils/mem_utils.py")
    tree = ast.parse(source)
    selected = [
        node
        for node in tree.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef))
        and node.name
        in {
            "UnifiedMemoryBudgetError",
            "UnifiedMemoryAllocatorReceipt",
            "compute_unified_memory_allocator_ceiling",
        }
    ]
    namespace: dict[str, object] = {}
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(
                body=[
                    ast.ImportFrom(
                        module="dataclasses",
                        names=[ast.alias(name="dataclass")],
                        level=0,
                    ),
                    *selected,
                ],
                type_ignores=[],
            )),
            "mem_utils_allocator_contract.py",
            "exec",
        ),
        namespace,
    )
    return namespace["compute_unified_memory_allocator_ceiling"]


def test_builda_v0_only_and_two_real_mla_call_sites():
    kernel = text("runtime/kernels/builda_bmm_v0.py")
    ast.parse(kernel)
    assert "builda_bmm_v1" not in kernel
    mla = text("runtime/overlays/vllm/vllm/model_executor/layers/attention/mla_attention.py")
    assert mla.count("_builda_bmm(") == 2
    assert "mqa_q_nope, W_UK_T, mqa_ql_nope" in mla
    assert "x, self.W_UV, out.transpose(0, 1)" in mla


def test_b12x_uses_int64_addressing_for_offsets_beyond_int32():
    source = text("runtime/overlays/b12x/b12x/attention/indexer/tiled_topk.py")
    assert (1 << 20) * (1 << 12) > (1 << 31)
    assert "input_stride: Int64" in source
    assert "row_base = Int64(data_bid) * input_stride" in source
    assert "tile_size = Int64" in source


def test_dcp_mapping_merge_and_mtp_mapping_are_present():
    indexer = text("runtime/overlays/vllm/vllm/model_executor/layers/sparse_attn_indexer.py")
    topk = text("runtime/overlays/b12x/b12x/attention/indexer/tiled_topk.py")
    mtp = text("runtime/overlays/vllm/vllm/model_executor/models/deepseek_mtp.py")
    assert "_merge_dcp_topk_global" in indexer and "dcp_rank" in indexer
    assert "merge_tiled_topk_candidates" in topk and "dtype=torch.int64" in topk
    assert "packed_modules_mapping" in mtp and "expert_params_mapping" in mtp


def test_nvfp4_full_record_and_page_contract_agree_across_sources():
    kv = text("runtime/overlays/vllm/vllm/v1/kv_cache_interface.py")
    backend = text("runtime/overlays/vllm/vllm/v1/attention/backends/mla/b12x_mla_sparse.py")
    native = text("runtime/patches/native/champion-sm121.patch")
    assert "block_size * 368" in kv
    assert "compact ``nvfp4_ds_mla`` record (368 B/token)" in backend
    assert "kRecordBytes = 368" in native
    assert 64 * 368 == 23552


def test_lm_head_top64_scheduler_and_per_host_allocator_source_contracts():
    logits = text("runtime/overlays/vllm/vllm/model_executor/layers/logits_processor.py")
    scheduler = text("runtime/overlays/vllm/vllm/v1/core/sched/scheduler.py")
    presets = text("presets/large-1m.json")
    assert 'VLLM_LMHEAD_V2_TOPM", "64"' in logits
    assert "decode_prefill_token_budget" in scheduler and "prefill_tokens_scheduled" in scheduler
    assert '"scope": "per-host"' in presets and '"floor": "computed"' in presets


def test_unified_memory_budget_parser_and_worker_wiring_are_published():
    envs = text("runtime/overlays/vllm/vllm/envs.py")
    mem_utils = text("runtime/overlays/vllm/vllm/utils/mem_utils.py")
    gpu_worker = text("runtime/overlays/vllm/vllm/v1/worker/gpu_worker.py")
    for source in (envs, mem_utils, gpu_worker):
        ast.parse(source)

    assert '"VLLM_UNIFIED_MEMORY_ALLOCATOR_BUDGET_BYTES"' in envs
    assert "def compute_unified_memory_allocator_ceiling(" in mem_utils
    assert "def apply_unified_memory_allocator_ceiling(" in mem_utils
    assert (
        "phase_budget_bytes=envs.VLLM_UNIFIED_MEMORY_ALLOCATOR_BUDGET_BYTES"
        in gpu_worker
    )
    assert "set_per_process_memory_fraction" in gpu_worker


def test_explicit_unified_memory_floor_can_raise_conservative_allowance():
    compute = allocator_math()
    gib = 1024**3
    receipt = compute(
        phase="profile",
        total_cuda_bytes=128 * gib,
        total_host_bytes=128 * gib,
        available_bytes=8 * gib,
        reserve_bytes=4 * gib,
        current_allocated_bytes=100 * gib,
        current_reserved_bytes=100 * gib,
        phase_budget_bytes=112 * gib,
    )

    conservative_allowance = 100 * gib + (8 * gib - 4 * gib)
    assert conservative_allowance == 104 * gib
    assert receipt.requested_ceiling_bytes == 112 * gib
    assert receipt.requested_ceiling_bytes > conservative_allowance


def test_total_minus_host_reserve_remains_the_hard_allocator_cap():
    compute = allocator_math()
    gib = 1024**3
    receipt = compute(
        phase="graphs",
        total_cuda_bytes=128 * gib,
        total_host_bytes=128 * gib,
        available_bytes=8 * gib,
        reserve_bytes=4 * gib,
        current_allocated_bytes=100 * gib,
        current_reserved_bytes=100 * gib,
        phase_budget_bytes=127 * gib,
    )

    assert receipt.requested_ceiling_bytes == 124 * gib
    assert receipt.requested_ceiling_bytes == receipt.pool_total_bytes - 4 * gib
    assert receipt.requested_ceiling_bytes < receipt.phase_budget_bytes


def test_accepted_r17_live_floor_selects_exact_observed_receipt():
    compute = allocator_math()
    accepted_floor = 119_605_592_064
    receipt = compute(
        phase="profile",
        total_cuda_bytes=128_849_018_880,
        total_host_bytes=128_849_018_880,
        available_bytes=18_183_303_168,
        reserve_bytes=4_294_967_296,
        current_allocated_bytes=104_014_544_896,
        current_reserved_bytes=104_014_544_896,
        phase_budget_bytes=accepted_floor,
    )

    assert receipt.current_baseline_bytes + (
        receipt.available_bytes - receipt.reserve_bytes
    ) == 117_902_880_768
    assert receipt.requested_ceiling_bytes == accepted_floor
    assert receipt.phase_budget_bytes == accepted_floor
