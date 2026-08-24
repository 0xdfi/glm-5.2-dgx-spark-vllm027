from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
CHAMPION = ROOT / "bin" / "champion"
FIXTURE_BIN = ROOT / "tests" / "fixtures" / "bin"
CLUSTER = ROOT / "tests" / "fixtures" / "cluster-fixture.env"
VALID_COMPACTION_NONCE = "0123456789abcdef0123456789abcdef"


def invocation_env(
    *args: str,
    scenario: str,
    tmp_path: Path,
    cluster_env: Path,
    log: Path,
) -> dict[str, str]:
    return os.environ | {
        "PATH": f"{FIXTURE_BIN}:{os.environ['PATH']}",
        "CHAMPION_CLUSTER_ENV": str(cluster_env),
        "CHAMPION_SSH_BIN": str(FIXTURE_BIN / "ssh"),
        "CHAMPION_DOCKER_BIN": str(FIXTURE_BIN / "docker"),
        "CHAMPION_CURL_BIN": str(FIXTURE_BIN / "curl"),
        "CHAMPION_FAKE_LOG": str(log),
        "CHAMPION_FAKE_SCENARIO": scenario,
        "CHAMPION_FAKE_PROFILE": args[1] if len(args) > 1 else "fast",
        "CHAMPION_FAKE_REMOTE_LOCK": str(tmp_path / "canonical-remote-lock"),
        "CHAMPION_FAKE_RECEIPT_DIR": str(tmp_path / "compaction-receipts"),
        "CHAMPION_FAKE_PREFLIGHT_GATE": str(tmp_path / "preflight-gate"),
        "CHAMPION_FAKE_PREFLIGHT_WAITING": str(tmp_path / "preflight-waiting"),
        "CHAMPION_IDLE_SECONDS": "0",
        "CHAMPION_READY_ATTEMPTS": "2",
        "CHAMPION_LOCK_READY_ATTEMPTS": "20",
        "CHAMPION_LOCK_READY_DELAY_SECONDS": "0.01",
        "CHAMPION_LOCK_RELEASE_ATTEMPTS": "20",
    }


def invoke(
    *args: str,
    scenario: str = "healthy",
    tmp_path: Path | None = None,
    compaction_nonce: str | None = VALID_COMPACTION_NONCE,
    cluster_env: Path = CLUSTER,
    extra_env: dict[str, str] | None = None,
):
    test_dir = tmp_path or ROOT
    log = test_dir / "fake-commands.log"
    log.unlink(missing_ok=True)
    command_args = list(args)
    if (
        len(command_args) >= 2
        and command_args[0] == "launch"
        and command_args[1] in {"fast", "fast-320k"}
        and compaction_nonce is not None
        and "--compaction-nonce" not in command_args
    ):
        command_args.extend(("--compaction-nonce", compaction_nonce))
    env = invocation_env(
        *command_args,
        scenario=scenario,
        tmp_path=test_dir,
        cluster_env=cluster_env,
        log=log,
    )
    if extra_env:
        env.update(extra_env)
    result = subprocess.run(
        [str(CHAMPION), *command_args],
        cwd=ROOT,
        text=True,
        capture_output=True,
        env=env,
    )
    commands = log.read_text() if log.exists() else ""
    return result, commands


def test_front_door_surface_and_bare_gate_commands_fail_closed(tmp_path: Path):
    help_result, _ = invoke("help", tmp_path=tmp_path)
    for command in ("bootstrap", "configure", "verify-hosts", "build", "pull", "render", "launch", "verify", "battery", "status", "stage-image", "stop"):
        assert command in help_result.stdout

    for command in ("verify", "battery"):
        result, _ = invoke(command, "fast", tmp_path=tmp_path)
        assert result.returncode == 78
        assert "not implicit" in result.stderr


def test_static_verify_runs_only_source_contracts_without_host_tools(tmp_path: Path):
    result, commands = invoke("verify", "fast", "--static", tmp_path=tmp_path)
    assert result.returncode == 0, result.stderr
    assert "9 passed" in result.stdout
    assert commands == ""


def test_verify_hosts_enforces_capabilities_and_reports_drift(tmp_path: Path):
    result, commands = invoke("verify-hosts", scenario="healthy", tmp_path=tmp_path)
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["result"] == "PASS"
    assert len(report["hosts"]) == 4
    assert all(item["architecture"] == "aarch64" for item in report["hosts"])
    assert all(item["gpu"] == "NVIDIA GB10" for item in report["hosts"])
    assert report["comparison_environment"] == "drift-recorded"
    assert report["lock_scope"] == "canonical-rank0-required-for-mutation"
    assert "ssh node-a" in commands

    failed, _ = invoke("verify-hosts", scenario="low-memory", tmp_path=tmp_path)
    assert failed.returncode != 0
    assert "MemAvailable" in failed.stderr


def test_launcher_refuses_protected_gpu_and_port_states(tmp_path: Path):
    protected, protected_commands = invoke("launch", "fast", scenario="protected", tmp_path=tmp_path)
    assert protected.returncode != 0
    assert "protected container" in protected.stderr
    assert "docker stop" not in protected_commands
    assert "docker rm" not in protected_commands

    gpu, _ = invoke("launch", "fast", scenario="live-gpu", tmp_path=tmp_path)
    assert gpu.returncode != 0
    assert "GPU compute application" in gpu.stderr

    port, _ = invoke("launch", "fast", scenario="busy-port", tmp_path=tmp_path)
    assert port.returncode != 0
    assert "target port" in port.stderr


def test_healthy_fast_launch_boots_unpinned_then_pins_and_soaks(tmp_path: Path):
    result, commands = invoke("launch", "fast", tmp_path=tmp_path)
    assert result.returncode == 0, result.stderr
    assert "launch-result=PASS profile=fast pin-state=pinned" in result.stdout
    run_lines = [line for line in commands.splitlines() if "docker run" in line]
    assert len(run_lines) == 4
    assert all("--cpuset" not in line for line in run_lines)
    assert all("PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True" in line for line in run_lines)
    assert all("VLLM_R17_PROFILE_DRAFTER_CAP=1" in line for line in run_lines)
    assert all(
        "VLLM_UNIFIED_MEMORY_HOST_RESERVE_BYTES=4294967296" in line
        for line in run_lines
    )
    assert all("VLLM_UNIFIED_MEMORY_LOAD_CEILING=0" in line for line in run_lines)
    assert all(
        "VLLM_UNIFIED_MEMORY_ALLOCATOR_BUDGET_BYTES=122406567936" in line
        for line in run_lines
    )
    assert commands.count("printf 3 > /proc/sys/vm/drop_caches") == 4
    assert commands.index("/proc/sys/vm/drop_caches") < commands.index("docker run")
    assert commands.index("allocator-env-readback") < commands.index("ray start --head")
    assert commands.count("docker update --cpuset-cpus 5-9,15-19 --cpuset-mems 0") == 4
    assert commands.count("chat-probe") >= 2
    assert commands.count("cache-busted-sparse-indexer-probe") == 1
    assert "idle-soak 0" in commands
    assert "Socket" not in result.stdout


def test_fast_launch_requires_one_use_compaction_nonce_before_remote_work(
    tmp_path: Path,
):
    result, commands = invoke(
        "launch", "fast", tmp_path=tmp_path, compaction_nonce=None
    )

    assert result.returncode != 0
    assert "--compaction-nonce" in result.stderr
    assert "ssh " not in commands
    assert "drop_caches" not in commands
    assert "docker run" not in commands


def test_fast_launch_rejects_persisted_compaction_approval(tmp_path: Path):
    cluster = tmp_path / "cluster.env"
    cluster.write_text(
        CLUSTER.read_text() + "CHAMPION_APPROVE_MEMORY_COMPACTION=1\n",
        encoding="utf-8",
    )

    result, commands = invoke(
        "launch",
        "fast",
        tmp_path=tmp_path,
        cluster_env=cluster,
    )

    assert result.returncode != 0
    assert "runtime-only setting" in result.stderr
    assert "drop_caches" not in commands
    assert "docker run" not in commands


def test_legacy_environment_approval_cannot_authorize_compaction(tmp_path: Path):
    result, commands = invoke(
        "launch",
        "fast",
        tmp_path=tmp_path,
        compaction_nonce=None,
        extra_env={"CHAMPION_APPROVE_MEMORY_COMPACTION": "1"},
    )

    assert result.returncode != 0
    assert "--compaction-nonce" in result.stderr
    assert "ssh " not in commands
    assert "drop_caches" not in commands


@pytest.mark.parametrize(
    "nonce",
    ("short", "g" * 32, "0123456789abcdef0123456789abcde/"),
)
def test_invalid_compaction_nonce_is_rejected_before_remote_work(
    tmp_path: Path, nonce: str
):
    result, commands = invoke(
        "launch", "fast", tmp_path=tmp_path, compaction_nonce=nonce
    )

    assert result.returncode == 64
    assert "32 hexadecimal" in result.stderr
    assert "ssh " not in commands
    assert "drop_caches" not in commands
    assert "docker run" not in commands


def test_compaction_nonce_is_consumed_once_and_never_logged_raw(tmp_path: Path):
    nonce = "fedcba9876543210fedcba9876543210"
    first, first_commands = invoke(
        "launch", "fast", tmp_path=tmp_path, compaction_nonce=nonce
    )
    assert first.returncode == 0, first.stderr
    assert nonce not in first.stdout + first.stderr + first_commands

    replay, replay_commands = invoke(
        "launch", "fast", tmp_path=tmp_path, compaction_nonce=nonce
    )
    assert replay.returncode != 0
    assert "already consumed" in replay.stderr
    assert nonce not in replay.stdout + replay.stderr + replay_commands
    assert "drop_caches" not in replay_commands
    assert "docker run" not in replay_commands


def test_invalid_protected_container_regex_blocks_before_remote_work(tmp_path: Path):
    cluster = tmp_path / "invalid-regex.env"
    cluster.write_text(
        CLUSTER.read_text()
        + "CHAMPION_PROTECTED_CONTAINER_PATTERN=([unterminated\n",
        encoding="utf-8",
    )

    result, commands = invoke(
        "launch", "fast", tmp_path=tmp_path, cluster_env=cluster
    )

    assert result.returncode != 0
    assert "protected container pattern is not valid ERE" in result.stderr
    assert "ssh " not in commands
    assert "drop_caches" not in commands
    assert "docker run" not in commands


def test_protected_container_grep_error_fails_closed_before_compaction(
    tmp_path: Path,
):
    grep_fixture = tmp_path / "protected-grep"
    grep_fixture.write_text(
        "#!/usr/bin/env bash\n"
        "payload=$(cat)\n"
        "[[ -z $payload ]] && exit 1\n"
        "exit 2\n",
        encoding="utf-8",
    )
    grep_fixture.chmod(0o755)
    result, commands = invoke(
        "launch",
        "fast",
        scenario="protected-grep-error",
        tmp_path=tmp_path,
        extra_env={"CHAMPION_PATTERN_GREP_BIN": str(grep_fixture)},
    )

    assert result.returncode != 0
    assert "protected container pattern evaluation failed" in result.stderr
    assert "drop_caches" not in commands
    assert "docker run" not in commands


def test_remote_fleet_lock_blocks_second_launcher_and_stop_coordinator(
    tmp_path: Path,
):
    first_log = tmp_path / "first.log"
    first_args = [
        "launch",
        "fast",
        "--compaction-nonce",
        "11111111111111111111111111111111",
    ]
    first_env = invocation_env(
        *first_args,
        scenario="hold-preflight",
        tmp_path=tmp_path,
        cluster_env=CLUSTER,
        log=first_log,
    )
    first = subprocess.Popen(
        [str(CHAMPION), *first_args],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=first_env,
    )
    try:
        for _ in range(200):
            if (tmp_path / "preflight-waiting").exists():
                break
            if first.poll() is not None:
                break
            import time

            time.sleep(0.01)
        assert (tmp_path / "preflight-waiting").exists(), first.communicate()

        second, second_commands = invoke(
            "launch",
            "fast",
            tmp_path=tmp_path,
            compaction_nonce="22222222222222222222222222222222",
        )
        assert second.returncode != 0
        assert "canonical fleet lock is busy" in second.stderr
        assert "CHAMPION_FLEET_LOCK_PROTOCOL_V1" in second_commands
        assert "drop_caches" not in second_commands
        assert "docker run" not in second_commands

        stopped, stop_commands = invoke("stop", tmp_path=tmp_path)
        assert stopped.returncode != 0
        assert "canonical fleet lock is busy" in stopped.stderr
        assert "CHAMPION_FLEET_LOCK_PROTOCOL_V1" in stop_commands
        assert "docker stop" not in stop_commands
    finally:
        (tmp_path / "preflight-gate").touch()
        stdout, stderr = first.communicate(timeout=10)
    assert first.returncode == 0, stderr
    first_commands = first_log.read_text(encoding="utf-8")
    assert "CHAMPION_FLEET_LOCK_PROTOCOL_V1" in first_commands
    assert first_commands.index("remote-lock-acquired") < first_commands.index(
        "uname -m"
    )
    assert first_commands.index("docker update --cpuset-cpus 5-9,15-19") < first_commands.index(
        "remote-lock-release"
    )


def test_remote_lock_liveness_loss_blocks_all_mutation(tmp_path: Path):
    result, commands = invoke(
        "launch", "fast", scenario="lock-dies-after-ready", tmp_path=tmp_path
    )

    assert result.returncode != 0
    assert "canonical fleet lock" in result.stderr
    assert "drop_caches" not in commands
    assert "docker run" not in commands


@pytest.mark.parametrize(
    ("scenario", "message"),
    (
        ("gpu-query-error", "GPU application query failed"),
        ("port-query-error", "target port query failed"),
        ("protected-query-error", "protected container query failed"),
    ),
)
def test_fast_launch_fails_closed_when_precompaction_query_fails(
    tmp_path: Path, scenario: str, message: str
):
    result, commands = invoke("launch", "fast", scenario=scenario, tmp_path=tmp_path)

    assert result.returncode != 0
    assert message in result.stderr
    assert "drop_caches" not in commands
    assert "docker run" not in commands


def test_fast_launch_checks_target_identity_before_compaction(tmp_path: Path):
    result, commands = invoke(
        "launch", "fast", scenario="existing-target", tmp_path=tmp_path
    )

    assert result.returncode != 0
    assert "existing target container" in result.stderr
    assert "drop_caches" not in commands
    assert "docker run" not in commands


def test_allocator_readback_failure_blocks_ray_start(tmp_path: Path):
    result, commands = invoke(
        "launch", "fast", scenario="allocator-readback-mismatch", tmp_path=tmp_path
    )

    assert result.returncode != 0
    assert "allocator floor environment mismatch" in result.stderr
    assert "ray start" not in commands


def test_cache_busted_sparse_probe_requires_more_than_2048_prompt_tokens(tmp_path: Path):
    result, commands = invoke(
        "launch", "fast", scenario="short-indexer-probe", tmp_path=tmp_path
    )

    assert result.returncode != 0
    assert "cache-busted sparse-indexer semantic probe failed" in result.stderr
    assert "cache-busted-sparse-indexer-probe" in commands


def test_cache_busted_sparse_probe_requires_semantic_marker(tmp_path: Path):
    result, commands = invoke(
        "launch", "fast", scenario="wrong-indexer-response", tmp_path=tmp_path
    )

    assert result.returncode != 0
    assert "cache-busted sparse-indexer semantic probe failed" in result.stderr
    assert "cache-busted-sparse-indexer-probe" in commands


def invoke_sparse_probe_fixture(tmp_path: Path, content: str):
    log = tmp_path / "curl-commands.log"
    payload = json.dumps(
        {
            "model": "glm-5.2",
            "messages": [{"role": "user", "content": content}],
            "temperature": 1.0,
            "max_tokens": 16,
            "stream": False,
        },
        separators=(",", ":"),
    )
    result = subprocess.run(
        [
            str(FIXTURE_BIN / "curl"),
            "-fsS",
            "-H",
            "Content-Type: application/json",
            "-H",
            "X-Champion-Probe: cache-busted-sparse-indexer",
            "--data-binary",
            payload,
            "http://fixture/v1/chat/completions",
        ],
        text=True,
        capture_output=True,
        env=os.environ | {"CHAMPION_FAKE_LOG": str(log)},
    )
    return result


@pytest.mark.parametrize(
    "content",
    (
        "probe " * 4096 + "\nReply briefly with SPARK_INDEX_OK.",
        "cache-bust=fixture-run-deadbeef "
        + "probe " * 16
        + "\nReply briefly with SPARK_INDEX_OK.",
    ),
    ids=("missing-cache-bust", "short-probe-body"),
)
def test_sparse_probe_fixture_rejects_header_only_false_greens(
    tmp_path: Path, content: str
):
    result = invoke_sparse_probe_fixture(tmp_path, content)

    assert result.returncode != 0
    assert "invalid cache-busted sparse-indexer payload" in result.stderr


def test_sparse_probe_fixture_derives_usage_from_valid_payload(tmp_path: Path):
    content = (
        "cache-bust=fixture-run-0123456789abcdef0123456789abcdef "
        + "probe " * 4096
        + "\nReply briefly with SPARK_INDEX_OK."
    )
    result = invoke_sparse_probe_fixture(tmp_path, content)

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["usage"]["prompt_tokens"] == 4119


def test_non_fast_allocator_contracts_remain_profile_specific(tmp_path: Path):
    balanced, balanced_commands = invoke("launch", "balanced", tmp_path=tmp_path)
    assert balanced.returncode == 0, balanced.stderr
    assert "VLLM_UNIFIED_MEMORY_ALLOCATOR_BUDGET_BYTES=" not in balanced_commands
    assert "drop_caches" not in balanced_commands

    large, large_commands = invoke("launch", "large", tmp_path=tmp_path)
    assert large.returncode == 0, large.stderr
    run_lines = [line for line in large_commands.splitlines() if "docker run" in line]
    assert len(run_lines) == 4
    assert len(
        {
            part
            for line in run_lines
            for part in line.split()
            if part.startswith("VLLM_UNIFIED_MEMORY_ALLOCATOR_BUDGET_BYTES=")
        }
    ) == 4
    assert "drop_caches" not in large_commands


def test_pin_failure_rolls_all_exact_ranks_back_and_preserves_fleet(tmp_path: Path):
    result, commands = invoke("launch", "fast", scenario="pin-fail", tmp_path=tmp_path)
    assert result.returncode != 0
    assert "unpinned-rollback" in result.stderr
    assert commands.count("docker update --cpuset-cpus 0-19 --cpuset-mems 0") == 4
    assert "docker rm" not in commands
    assert "docker rmi" not in commands


def test_stop_requires_exact_identity_and_orders_workers_before_coordinator(tmp_path: Path):
    result, commands = invoke("stop", scenario="owned-running", tmp_path=tmp_path)
    assert result.returncode == 0, result.stderr
    stopped = [line for line in commands.splitlines() if "docker stop" in line]
    assert [line.split()[-1] for line in stopped] == [
        "fixture-run-rank3",
        "fixture-run-rank2",
        "fixture-run-rank1",
        "fixture-run-rank0",
    ]
    assert "docker rm" not in commands

    unknown, unknown_commands = invoke("stop", scenario="unknown-identity", tmp_path=tmp_path)
    assert unknown.returncode != 0
    assert "UNKNOWN" in unknown.stderr
    assert "docker stop" not in unknown_commands


def test_stage_image_requires_authenticated_status_and_never_touches_containers(tmp_path: Path):
    result, commands = invoke("stage-image", tmp_path=tmp_path)
    assert result.returncode != 0
    assert "authenticated accepted build status" in result.stderr
    assert "docker load" not in commands
    assert "docker stop" not in commands
    assert "docker rm" not in commands


def test_authenticated_image_staging_holds_canonical_lock_through_all_loads(
    tmp_path: Path,
):
    status = tmp_path / "accepted-status.json"
    status.write_text(
        json.dumps(
            {
                "accepted_build": {
                    "status": "passed",
                    "authenticated": True,
                    "build_subject_id": "fixture-build-subject",
                    "image_id": "sha256:"
                    + "a" * 64,
                    "builder_host": "node-a",
                }
            }
        ),
        encoding="utf-8",
    )
    cluster = tmp_path / "stage-cluster.env"
    cluster.write_text(
        CLUSTER.read_text().replace(
            "CHAMPION_STATUS_FILE=tests/fixtures/status-pending.json",
            f"CHAMPION_STATUS_FILE={status}",
        ),
        encoding="utf-8",
    )

    result, commands = invoke("stage-image", tmp_path=tmp_path, cluster_env=cluster)

    assert result.returncode == 0, result.stderr
    assert commands.count("docker load") == 4
    assert "CHAMPION_FLEET_LOCK_PROTOCOL_V1" in commands
    assert commands.index("remote-lock-acquired") < commands.index("docker load")
    assert commands.rindex("docker load") < commands.index("remote-lock-release")


def test_legacy_launcher_is_only_a_deprecation_wrapper():
    text = (ROOT / "recipes/launch.sh").read_text()
    assert "deprecated" in text.lower()
    assert "bin/champion" in text
    assert "docker run" not in text
