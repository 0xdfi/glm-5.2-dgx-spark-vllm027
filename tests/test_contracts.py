import json
from copy import deepcopy

import pytest

from champion_runtime.contracts import ContractError, load_contract
from champion_runtime.receipts import construct_receipt


SHA256 = "a" * 64
GIT_SHA = "b" * 40


def write_json(tmp_path, payload, name="contract.json"):
    path = tmp_path / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def valid_build_lock():
    return {
        "schema_version": "1.0.0",
        "architecture": "aarch64",
        "python": {"version": "3.12.13"},
        "cuda": {"version": "13.0.2", "architectures": ["12.1a"]},
        "base_image": f"registry.example/runtime@sha256:{SHA256}",
        "sources": [
            {
                "name": "vllm",
                "repository": "https://example.invalid/vllm.git",
                "commit": GIT_SHA,
                "tree": "c" * 40,
            }
        ],
        "artifacts": [
            {
                "name": "torch",
                "url": "https://example.invalid/torch.whl",
                "sha256": "d" * 64,
            }
        ],
        "toolchain": [{"name": "gcc", "version": "14.2.0"}],
        "build_flags": ["TORCH_CUDA_ARCH_LIST=12.1a"],
        "manifests": {
            "python_source_sha256": "e" * 64,
            "native_source_sha256": "f" * 64,
        },
    }


def valid_preset():
    return {
        "schema_version": "1.0.0",
        "name": "fast-320k",
        "profile": "fast",
        "model": {
            "checkpoint_sha256": SHA256,
            "sidecar_sha256": "b" * 64,
        },
        "effective_engine": {
            "tensor_parallel_size": 4,
            "pipeline_parallel_size": 1,
            "prefill_context_parallel_size": 1,
            "decode_context_parallel_size": 1,
            "max_model_len": 320000,
            "max_num_seqs": 4,
            "max_num_batched_tokens": 2048,
            "decode_prefill_tokens": 1024,
            "kv_cache_bytes_per_rank": 10232320000,
            "graph_mode": "FULL",
            "graph_capture_sizes": [6, 12, 18, 24],
            "attention_backend": "nvfp4_ds_mla",
            "enable_prefix_caching": True,
            "async_scheduling": True,
            "fuse_gemm_comms": False,
            "adaptive_mtp": {"k": [2, 4, 5], "window": 32},
            "sidecar_required": True,
        },
    }


def valid_provenance():
    return {
        "schema_version": "1.0.0",
        "records": [
            {
                "path": "runtime/example.py",
                "classification": "project-apache",
                "copyright": "Copyright 2026 dfi / 0xdfi",
                "license": "Apache-2.0",
                "origin_repository": "https://example.invalid/project.git",
                "origin_commit": GIT_SHA,
                "origin_path": "runtime/example.py",
                "base_sha256": None,
                "target_sha256": SHA256,
                "modified_by": "dfi / 0xdfi",
                "evidence": [{"path": "NOTICE", "sha256": "b" * 64}],
            }
        ],
    }


def valid_receipt():
    return construct_receipt({
        "schema_version": "1.0.0",
        "receipt_type": "verify-source",
        "command": "verify-source",
        "created_at": "2026-08-23T12:00:00Z",
        "subject": {"source_release_id": "b" * 64},
        "evidence": [{"path": "evidence/source.json", "sha256": "c" * 64}],
        "result": "PASS",
        "metadata": {"git_commit": GIT_SHA, "run_id": "12345"},
    })


def valid_status():
    return {
        "schema_version": "1.0.0",
        "updated_at": "2026-08-23T12:00:00Z",
        "claims": [
            {
                "claim": "source-complete",
                "status": "pending",
                "subject": {"source_release_id": SHA256},
                "receipt_ids": [],
            }
        ],
    }


@pytest.mark.parametrize(
    ("kind", "payload"),
    [
        ("build-lock", valid_build_lock()),
        ("preset", valid_preset()),
        ("provenance", valid_provenance()),
        ("receipt", valid_receipt()),
        ("status", valid_status()),
    ],
)
def test_loads_valid_contracts(tmp_path, kind, payload):
    assert load_contract(write_json(tmp_path, payload), kind) == payload


def test_preset_rejects_unknown_fields(tmp_path):
    payload = valid_preset() | {"typo_field": True}

    with pytest.raises(ContractError, match=r"\$: Additional properties"):
        load_contract(write_json(tmp_path, payload), "preset")


def test_preset_rejects_unknown_nested_fields_with_path(tmp_path):
    payload = valid_preset()
    payload["effective_engine"]["adaptive_mtp"]["typo_field"] = True

    with pytest.raises(
        ContractError,
        match=r"\$\.effective_engine\.adaptive_mtp: Additional properties",
    ):
        load_contract(write_json(tmp_path, payload), "preset")


def test_build_lock_rejects_mutable_git_tag_without_commit_and_tree(tmp_path):
    payload = valid_build_lock()
    payload["sources"] = [
        {
            "name": "vllm",
            "repository": "https://example.invalid/vllm.git",
            "tag": "v0.27.0",
        }
    ]

    with pytest.raises(ContractError, match=r"\$\.sources\[0\]"):
        load_contract(write_json(tmp_path, payload), "build-lock")


def test_build_lock_rejects_non_sha_image_reference(tmp_path):
    payload = valid_build_lock() | {"base_image": "registry.example/runtime:latest"}

    with pytest.raises(ContractError, match=r"\$\.base_image: .*sha256"):
        load_contract(write_json(tmp_path, payload), "build-lock")


def test_build_lock_rejects_short_or_mixed_case_hashes(tmp_path):
    payload = valid_build_lock()
    payload["artifacts"][0]["sha256"] = "A" * 63

    with pytest.raises(ContractError, match=r"\$\.artifacts\[0\]\.sha256"):
        load_contract(write_json(tmp_path, payload), "build-lock")


def test_preset_requires_effective_engine(tmp_path):
    payload = deepcopy(valid_preset())
    del payload["effective_engine"]

    with pytest.raises(ContractError, match=r"\$: 'effective_engine' is a required property"):
        load_contract(write_json(tmp_path, payload), "preset")


def test_receipt_rejects_comparison_only_subject(tmp_path):
    payload = valid_receipt()
    payload["subject"] = {"run_id": "comparison-only"}

    with pytest.raises(ContractError, match=r"\$\.subject"):
        load_contract(write_json(tmp_path, payload), "receipt")


def test_receipt_rejects_forged_content_id(tmp_path):
    payload = valid_receipt()
    payload["receipt_id"] = "0" * 64

    with pytest.raises(ContractError, match="receipt_id mismatch"):
        load_contract(write_json(tmp_path, payload), "receipt")


@pytest.mark.parametrize("field", ["evidence", "result"])
def test_receipt_rejects_content_mutation(tmp_path, field):
    payload = valid_receipt()
    if field == "evidence":
        payload[field][0]["sha256"] = "d" * 64
    else:
        payload[field] = "FAIL"

    with pytest.raises(ContractError, match="receipt_id mismatch"):
        load_contract(write_json(tmp_path, payload), "receipt")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("command", "Authorization:" + " Bearer " + "canary-secret-value"),
        ("evidence", "https://user:canary-password@example.invalid/evidence"),
    ],
)
def test_receipt_load_rejects_secret_like_values(tmp_path, field, value):
    payload = valid_receipt()
    if field == "command":
        payload[field] = value
    else:
        payload[field][0]["path"] = value

    with pytest.raises(ContractError, match="secret-like value"):
        load_contract(write_json(tmp_path, payload), "receipt")


def test_receipt_command_must_be_a_stable_identifier(tmp_path):
    payload = valid_receipt()
    payload["command"] = "verify-source --verbose"

    with pytest.raises(ContractError, match="stable identifier"):
        load_contract(write_json(tmp_path, payload), "receipt")


def test_json_rejects_duplicate_object_names(tmp_path):
    path = tmp_path / "duplicate.json"
    path.write_text(
        '{"schema_version":"1.0.0","schema_version":"1.0.0"}',
        encoding="utf-8",
    )

    with pytest.raises(ContractError, match="duplicate object name 'schema_version'"):
        load_contract(path, "preset")


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_json_rejects_non_json_numeric_constants(tmp_path, constant):
    path = tmp_path / "constant.json"
    path.write_text(f'{{"schema_version":{constant}}}', encoding="utf-8")

    with pytest.raises(ContractError, match=f"non-JSON constant {constant}"):
        load_contract(path, "preset")


def status_with(status, *, receipt_ids=None, reason=None):
    payload = valid_status()
    claim = payload["claims"][0]
    claim["status"] = status
    claim["receipt_ids"] = [] if receipt_ids is None else receipt_ids
    if reason is not None:
        claim["reason"] = reason
    return payload


def test_passed_status_requires_evidence_receipt(tmp_path):
    with pytest.raises(ContractError, match=r"\$\.claims\[0\]\.receipt_ids"):
        load_contract(write_json(tmp_path, status_with("passed")), "status")


@pytest.mark.parametrize("status", ["failed", "blocked"])
def test_failed_and_blocked_status_require_reason(tmp_path, status):
    with pytest.raises(ContractError, match=r"\$\.claims\[0\]"):
        load_contract(write_json(tmp_path, status_with(status)), "status")


@pytest.mark.parametrize(
    "payload",
    [
        status_with("pending"),
        status_with("passed", receipt_ids=[SHA256]),
        status_with("failed", reason="verification failed"),
        status_with("blocked", reason="missing evidence"),
        status_with("not-applicable", reason="claim does not apply"),
    ],
)
def test_status_state_shapes_are_valid(tmp_path, payload):
    assert load_contract(write_json(tmp_path, payload), "status") == payload


def test_unknown_contract_kind_is_actionable(tmp_path):
    with pytest.raises(ContractError, match="unknown contract kind 'typo'"):
        load_contract(write_json(tmp_path, {}), "typo")


def test_invalid_json_reports_the_contract_path(tmp_path):
    path = tmp_path / "broken.json"
    path.write_text('{"schema_version":', encoding="utf-8")

    with pytest.raises(ContractError, match=r"broken\.json: invalid JSON"):
        load_contract(path, "preset")
