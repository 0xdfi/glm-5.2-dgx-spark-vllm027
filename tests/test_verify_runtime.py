# Copyright 2026 dfi / 0xdfi
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from copy import deepcopy

import pytest

from scripts.verify_runtime import RuntimeEvidenceError, verify_runtime_evidence


H = "a" * 64
SUBJECT = {
    "image_digest": "sha256:" + H,
    "build_lock_id": "b" * 64,
    "model_id": "c" * 64,
    "sidecar_id": "d" * 64,
    "preset_id": "e" * 64,
}


def evidence():
    return {
        "schema_version": "1.0.0",
        "subject": SUBJECT,
        "profile": "fast-320k",
        "window_id": "window-1",
        "ranks": [
            {"rank": rank, "host": f"node-{rank}", "image_digest": SUBJECT["image_digest"], "restart_count": 0, "oom_kills": 0}
            for rank in range(4)
        ],
        "engine": {
            "max_model_len": 320000,
            "kv_cache_dtype": "nvfp4_ds_mla",
            "decode_context_parallel_size": 1,
            "adaptive_mtp": [2, 4, 5],
            "graph_mode": "FULL",
            "graph_capture_sizes": [6, 12, 18, 24],
            "fuse_gemm_comms": False,
        },
        "transport": {"nccl_backend": "IB", "socket_fallback": False},
        "first_fire": {
            "window_id": "window-1",
            "build_a": True,
            "sidecar": True,
            "mtp_mapping": True,
        },
        "log_events": [
            {"phase": "pre-readiness", "severity": "warning", "code": "known-duplicate-registration"}
        ],
    }


def expected_engine():
    return {
        "max_model_len": 320000,
        "kv_cache_dtype": "nvfp4_ds_mla",
        "decode_context_parallel_size": 1,
        "adaptive_mtp": [2, 4, 5],
    }


def test_runtime_verifier_requires_exact_four_rank_current_window_contract():
    result = verify_runtime_evidence(evidence(), SUBJECT, expected_engine())
    assert result["result"] == "PROVISIONAL"
    for mutate, match in (
        (lambda doc: doc["ranks"].pop(), "four ranks"),
        (lambda doc: doc["engine"].update(graph_capture_sizes=[6, 12]), "graph capture"),
        (lambda doc: doc["transport"].update(socket_fallback=True), "Socket"),
        (lambda doc: doc["ranks"][0].update(restart_count=1), "restart"),
        (lambda doc: doc["first_fire"].update(window_id="old"), "current window"),
    ):
        doc = deepcopy(evidence())
        mutate(doc)
        with pytest.raises(RuntimeEvidenceError, match=match):
            verify_runtime_evidence(doc, SUBJECT, expected_engine())

def test_runtime_warning_is_only_known_and_pre_readiness():
    for phase, severity, code in (
        ("post-readiness", "warning", "known-duplicate-registration"),
        ("pre-readiness", "fatal", "known-duplicate-registration"),
        ("pre-readiness", "warning", "mystery"),
    ):
        doc = evidence()
        doc["log_events"] = [{"phase": phase, "severity": severity, "code": code}]
        with pytest.raises(RuntimeEvidenceError, match="log event"):
            verify_runtime_evidence(doc, SUBJECT, expected_engine())
