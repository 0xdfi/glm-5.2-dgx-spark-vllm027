#!/usr/bin/env python3
"""Render a canonical champion preset into argv, environment, and EngineCore views."""

from __future__ import annotations

import argparse
import json
import re
import shlex
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from champion_runtime.contracts import ContractError, load_contract


PRESET_NAMES = {
    "fast": "fast-320k",
    "balanced": "balanced-650k",
    "large": "large-1m",
    "fast-320k": "fast-320k",
    "balanced-650k": "balanced-650k",
    "large-1m": "large-1m",
}
SECRET_NAME = re.compile(r"(?:TOKEN|PASSWORD|SECRET|PRIVATE_KEY|CREDENTIAL)(?!.*(?:FILE|PATH|REF)$)")
ENV_NAME = re.compile(r"^[A-Z][A-Z0-9_]*$")
RUNTIME_ONLY_CLUSTER_SETTINGS = {
    "CHAMPION_APPROVE_MEMORY_COMPACTION",
}
REQUIRED_SITE = {
    "CHAMPION_HOSTS",
    "CHAMPION_FABRIC_IPS",
    "CHAMPION_RDMA_IFNAME",
    "CHAMPION_IB_HCA",
    "CHAMPION_RDMA_UVERBS",
    "CHAMPION_RDMA_CM",
    "CHAMPION_MODEL_PATH",
    "CHAMPION_CHECKPOINT_LOCK",
    "CHAMPION_SIDECAR_LOCK",
    "CHAMPION_IMAGE_REF",
    "CHAMPION_IMAGE_ID",
    "CHAMPION_API_HOST",
    "CHAMPION_API_PORT",
    "CHAMPION_FLEET_LOCK",
    "CHAMPION_RUN_ID",
}


class RenderError(ValueError):
    pass


def load_cluster(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise RenderError(f"cannot read cluster configuration {path}: {exc}") from exc
    for number, raw in enumerate(lines, 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise RenderError(f"{path}:{number}: expected NAME=value")
        name, value = line.split("=", 1)
        if not ENV_NAME.fullmatch(name) or name in values:
            raise RenderError(f"{path}:{number}: invalid or duplicate setting")
        if name in RUNTIME_ONLY_CLUSTER_SETTINGS:
            raise RenderError(
                f"{path}:{number}: {name} is a runtime-only setting and must not be persisted"
            )
        if SECRET_NAME.search(name):
            raise RenderError("secret values are forbidden in cluster.env; use a mode-0600 credential reference")
        value = value.strip()
        if len(value) >= 2 and value[:1] == value[-1:] and value[0] in {'"', "'"}:
            value = value[1:-1]
        if not value or any(token in value for token in ("`", "$(", "\n", "\r")):
            raise RenderError(f"{path}:{number}: unsafe or empty setting")
        values[name] = value
    missing = sorted(REQUIRED_SITE - set(values))
    if missing:
        raise RenderError(f"{path}: missing settings: {', '.join(missing)}")
    hosts = values["CHAMPION_HOSTS"].split(",")
    fabric = values["CHAMPION_FABRIC_IPS"].split(",")
    if len(hosts) != 4 or len(set(hosts)) != 4:
        raise RenderError("CHAMPION_HOSTS must contain exactly four unique hosts")
    if len(fabric) != 4 or len(set(fabric)) != 4:
        raise RenderError("CHAMPION_FABRIC_IPS must contain exactly four unique addresses")
    return values


def load_common() -> dict[str, Any]:
    payload = json.loads((ROOT / "presets/common.json").read_text(encoding="utf-8"))
    if payload.get("schema_version") != "1.0.0":
        raise RenderError("presets/common.json has unsupported schema_version")
    return payload


def verify_common(preset: dict[str, Any], common: dict[str, Any]) -> None:
    for key, expected in common["effective_engine"].items():
        if preset["effective_engine"].get(key) != expected:
            raise RenderError(f"{preset['name']}: common engine value drifted: {key}")


def ordered_argv(engine: dict[str, Any], api_port: str) -> list[str]:
    spec = {
        "adaptive_speculative_tokens_window": engine["adaptive_mtp"]["window"],
        "attention_backend": engine["attention_backend"],
        "draft_sample_method": "probabilistic",
        "method": "mtp",
        "moe_backend": engine["moe_backend"],
        "num_speculative_tokens": engine["adaptive_mtp"]["max_tokens"],
        "quantization": "compressed-tensors",
        "rejection_sample_method": "block",
    }
    compilation = {
        "cudagraph_mode": engine["graph_mode"],
        "cudagraph_capture_sizes": engine["graph_capture_sizes"],
        "pass_config": {"fuse_gemm_comms": False},
    }
    argv = [
        "python3", "-m", "vllm.entrypoints.openai.api_server",
        "--model", "/models",
        "--tokenizer", "/models",
        "--served-model-name", "glm-5.2",
        "--trust-remote-code",
        "--load-format", "auto",
        "--quantization", "compressed-tensors",
        "--distributed-executor-backend", "ray",
        "--tensor-parallel-size", str(engine["tensor_parallel_size"]),
        "--pipeline-parallel-size", str(engine["pipeline_parallel_size"]),
        "--prefill-context-parallel-size", str(engine["prefill_context_parallel_size"]),
        "--decode-context-parallel-size", str(engine["decode_context_parallel_size"]),
    ]
    if engine["dcp_communication"] == "a2a":
        argv.extend([
            "--dcp-comm-backend", "a2a",
            "--dcp-kv-cache-interleave-size", str(engine["dcp_kv_interleave"]),
        ])
    argv.extend([
        "--gpu-memory-utilization", "0.90",
        "--max-model-len", str(engine["max_model_len"]),
        "--max-num-seqs", str(engine["max_num_seqs"]),
        "--max-num-batched-tokens", str(engine["max_num_batched_tokens"]),
        "--generation-config", "vllm",
        "--port", api_port,
        "--host", "0.0.0.0",
        "--no-enable-flashinfer-autotune",
        "--no-enable-log-requests",
        "--kv-cache-memory-bytes", str(engine["kv_cache_bytes_per_rank"]),
        "--kv-cache-dtype", engine["kv_cache_dtype"],
        "--attention-backend", engine["attention_backend"],
        "--moe-backend", engine["moe_backend"],
        "--reasoning-parser", "glm45",
        "--tool-call-parser", "glm47",
        "--enable-auto-tool-choice",
        "--speculative-config", json.dumps(spec, separators=(",", ":"), sort_keys=True),
        "--long-prefill-token-threshold", "2048",
        "--async-scheduling",
        "--enable-decode-aware-prefill",
        "--decode-prefill-token-budget", str(engine["decode_prefill_tokens"]),
        "--idle-prefill-token-budget", "2048",
        "--max-long-prefills-per-step", "1",
        "--compilation-config", json.dumps(compilation, separators=(",", ":")),
        "--enable-prefix-caching",
    ])
    return argv


def render(preset_name: str, cluster_path: Path) -> dict[str, Any]:
    canonical_name = PRESET_NAMES.get(preset_name)
    if canonical_name is None:
        raise RenderError(f"unknown preset {preset_name!r}")
    preset = load_contract(ROOT / f"presets/{canonical_name}.json", "preset")
    common = load_common()
    verify_common(preset, common)
    cluster = load_cluster(cluster_path)
    env = dict(common["container_env"])
    env.update(preset.get("container_env", {}))
    env["NCCL_SOCKET_IFNAME"] = cluster["CHAMPION_RDMA_IFNAME"]
    env["GLOO_SOCKET_IFNAME"] = cluster["CHAMPION_RDMA_IFNAME"]
    env["NCCL_IB_HCA"] = cluster["CHAMPION_IB_HCA"]
    engine = preset["effective_engine"]
    output = {
        "schema_version": "1.0.0",
        "preset": canonical_name,
        "profile": preset["profile"],
        "ordered_argv": ordered_argv(engine, cluster["CHAMPION_API_PORT"]),
        "container_env": [[name, env[name]] for name in sorted(env)],
        "expected_engine_core": engine,
        "host_actions": preset["host_actions"],
        "model": preset["model"],
        "model_claim": "blocked-until-complete-hashes-and-rights",
        "site": {
            "hosts": cluster["CHAMPION_HOSTS"],
            "fabric_ips": cluster["CHAMPION_FABRIC_IPS"],
            "rdma_interface": cluster["CHAMPION_RDMA_IFNAME"],
            "ib_hca": cluster["CHAMPION_IB_HCA"],
            "model_mount": cluster["CHAMPION_MODEL_PATH"],
            "api_host": cluster["CHAMPION_API_HOST"],
            "api_port": cluster["CHAMPION_API_PORT"],
            "image_ref": cluster["CHAMPION_IMAGE_REF"],
            "run_id": cluster["CHAMPION_RUN_ID"],
        },
    }
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", required=True)
    parser.add_argument("--cluster", type=Path, required=True)
    parser.add_argument("--format", choices=("json", "argv", "env", "engine"), default="json")
    args = parser.parse_args(argv)
    try:
        output = render(args.preset, args.cluster)
        if args.format == "json":
            print(json.dumps(output, indent=2, sort_keys=True))
        elif args.format == "argv":
            print(shlex.join(output["ordered_argv"]))
        elif args.format == "env":
            for name, value in output["container_env"]:
                print(f"{name}={shlex.quote(value)}")
        else:
            print(json.dumps(output["expected_engine_core"], indent=2, sort_keys=True))
        return 0
    except (OSError, json.JSONDecodeError, ContractError, RenderError) as exc:
        print(f"render-launch: FAIL: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
