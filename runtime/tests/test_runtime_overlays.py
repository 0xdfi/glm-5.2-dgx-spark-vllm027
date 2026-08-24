# Copyright 2026 dfi / 0xdfi
# SPDX-License-Identifier: Apache-2.0
"""Source contracts only; these do not claim GPU/runtime correctness."""

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def text(path: str) -> str:
    return (ROOT / path).read_text()


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
    assert '"scope": "per-host"' in presets and '"budget": "computed"' in presets
