from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from champion_runtime.contracts import load_contract


ROOT = Path(__file__).resolve().parents[1]
RENDER = ROOT / "scripts" / "render_launch.py"
CLUSTER = ROOT / "tests" / "fixtures" / "cluster-fixture.env"

EXPECTED = {
    "fast-320k": (1, 320000, 10232320000),
    "balanced-650k": (2, 650000, 10395482112),
    "large-1m": (4, 1000000, 7995534848),
}


def render(name: str) -> dict:
    result = subprocess.run(
        [sys.executable, str(RENDER), "--preset", name, "--cluster", str(CLUSTER), "--format", "json"],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def test_three_canonical_presets_are_exact_and_schema_valid():
    for name, (dcp, max_len, kv_bytes) in EXPECTED.items():
        preset = load_contract(ROOT / f"presets/{name}.json", "preset")
        engine = preset["effective_engine"]
        assert engine["tensor_parallel_size"] == 4
        assert engine["pipeline_parallel_size"] == 1
        assert engine["prefill_context_parallel_size"] == 1
        assert engine["decode_context_parallel_size"] == dcp
        assert engine["max_model_len"] == max_len
        assert engine["kv_cache_bytes_per_rank"] == kv_bytes
        assert engine["max_num_seqs"] == 4
        assert engine["max_num_batched_tokens"] == 2048
        assert engine["decode_prefill_tokens"] == 1024
        assert engine["graph_mode"] == "FULL"
        assert engine["graph_capture_sizes"] == [6, 12, 18, 24]
        assert engine["adaptive_mtp"] == {"k": [2, 4, 5], "max_tokens": 5, "window": 32}
        assert engine["kv_cache_dtype"] == "nvfp4_ds_mla"
        assert engine["attention_backend"] == "B12X_MLA_SPARSE"
        assert engine["moe_backend"] == "marlin"
        assert engine["enable_prefix_caching"] is True
        assert engine["async_scheduling"] is True
        assert engine["sidecar_required"] is True
        assert engine["fuse_gemm_comms"] is False


def test_profile_specific_dcp_allocator_and_pinning_contracts():
    fast = load_contract(ROOT / "presets/fast-320k.json", "preset")
    assert fast["effective_engine"]["dcp_communication"] == "none"
    assert fast["effective_engine"]["dcp_kv_interleave"] == 0
    assert fast["effective_engine"]["allocator"] == {
        "scope": "fleet-uniform",
        "floor": "computed",
        "profile_drafter_cap": 1,
    }
    assert fast["container_env"] == {
        "VLLM_R17_PROFILE_DRAFTER_CAP": "1",
        "VLLM_UNIFIED_MEMORY_HOST_RESERVE_BYTES": "4294967296",
        "VLLM_UNIFIED_MEMORY_LOAD_CEILING": "0",
    }
    assert fast["host_actions"] == {
        "boot_cpuset": None,
        "post_health_cpuset": "5-9,15-19",
        "post_health_memory_node": "0",
        "rollback_cpuset": "0-19",
        "rollback_memory_node": "0",
    }

    balanced = load_contract(ROOT / "presets/balanced-650k.json", "preset")
    assert balanced["effective_engine"]["dcp_communication"] == "a2a"
    assert balanced["effective_engine"]["dcp_kv_interleave"] == 1
    assert balanced["host_actions"]["post_health_cpuset"] is None
    assert "VLLM_UNIFIED_MEMORY_HOST_RESERVE_BYTES" not in balanced["container_env"]
    assert "VLLM_UNIFIED_MEMORY_LOAD_CEILING" not in balanced["container_env"]

    large = load_contract(ROOT / "presets/large-1m.json", "preset")
    assert large["effective_engine"]["allocator"] == {
        "scope": "per-host",
        "floor": "computed",
        "profile_drafter_cap": 1,
    }
    assert "VLLM_UNIFIED_MEMORY_ALLOCATOR_BUDGET_BYTES" not in large["container_env"]
    assert "VLLM_UNIFIED_MEMORY_HOST_RESERVE_BYTES" not in large["container_env"]
    assert "VLLM_UNIFIED_MEMORY_LOAD_CEILING" not in large["container_env"]


def test_rendered_views_have_champion_baseline_without_experiment_arms():
    for name in EXPECTED:
        output = render(name)
        argv = output["ordered_argv"]
        env = dict(output["container_env"])
        compilation = argv[argv.index("--compilation-config") + 1]

        assert "--cpuset-cpus" not in argv
        assert json.loads(compilation)["pass_config"]["fuse_gemm_comms"] is False
        assert env["NCCL_BUFFSIZE"] == "4194304"
        assert env["VLLM_B12X_MLA_SPEC_EXTEND_AS_DECODE"] == "0"
        assert env["VLLM_MOE_MARLIN_ATOMIC_ADD"] == "1"
        assert env["VLLM_ADAPTIVE_SPEC_DEPTHS"] == "2,4,5"
        assert env["PYTORCH_CUDA_ALLOC_CONF"] == "expandable_segments:True"
        if name == "fast-320k":
            assert env["VLLM_UNIFIED_MEMORY_HOST_RESERVE_BYTES"] == "4294967296"
            assert env["VLLM_UNIFIED_MEMORY_LOAD_CEILING"] == "0"
        else:
            assert "VLLM_UNIFIED_MEMORY_HOST_RESERVE_BYTES" not in env
            assert "VLLM_UNIFIED_MEMORY_LOAD_CEILING" not in env
        assert output["expected_engine_core"]["decode_prefill_tokens"] == 1024
        assert output["model_claim"] == "blocked-until-complete-hashes-and-rights"
        assert all("192.168." not in value for value in output["site"].values() if isinstance(value, str))


def test_rendered_snapshots_are_deterministic():
    for name in EXPECTED:
        assert render(name) == json.loads((ROOT / f"tests/fixtures/{name}.rendered.json").read_text())


def test_renderer_rejects_secret_values(tmp_path: Path):
    cluster = tmp_path / "cluster.env"
    canary = "CHAMPION_TEST_" + "SECRET_9f34"
    cluster.write_text(CLUSTER.read_text() + f"REGISTRY_TOKEN={canary}\n")
    result = subprocess.run(
        [sys.executable, str(RENDER), "--preset", "fast-320k", "--cluster", str(cluster)],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    assert result.returncode != 0
    assert "secret values are forbidden" in result.stderr
    assert canary not in result.stderr
