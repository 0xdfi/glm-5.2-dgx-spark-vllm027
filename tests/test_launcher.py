from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CHAMPION = ROOT / "bin" / "champion"
FIXTURE_BIN = ROOT / "tests" / "fixtures" / "bin"
CLUSTER = ROOT / "tests" / "fixtures" / "cluster-fixture.env"


def invoke(*args: str, scenario: str = "healthy", tmp_path: Path | None = None):
    log = (tmp_path or ROOT) / "fake-commands.log"
    log.unlink(missing_ok=True)
    env = os.environ | {
        "PATH": f"{FIXTURE_BIN}:{os.environ['PATH']}",
        "CHAMPION_CLUSTER_ENV": str(CLUSTER),
        "CHAMPION_SSH_BIN": str(FIXTURE_BIN / "ssh"),
        "CHAMPION_DOCKER_BIN": str(FIXTURE_BIN / "docker"),
        "CHAMPION_CURL_BIN": str(FIXTURE_BIN / "curl"),
        "CHAMPION_FAKE_LOG": str(log),
        "CHAMPION_FAKE_SCENARIO": scenario,
        "CHAMPION_IDLE_SECONDS": "0",
        "CHAMPION_READY_ATTEMPTS": "2",
    }
    result = subprocess.run(
        [str(CHAMPION), *args], cwd=ROOT, text=True, capture_output=True, env=env
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
    assert "5 passed" in result.stdout
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
    assert commands.count("docker update --cpuset-cpus 5-9,15-19 --cpuset-mems 0") == 4
    assert commands.count("chat-probe") >= 2
    assert "idle-soak 0" in commands
    assert "Socket" not in result.stdout


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


def test_legacy_launcher_is_only_a_deprecation_wrapper():
    text = (ROOT / "recipes/launch.sh").read_text()
    assert "deprecated" in text.lower()
    assert "bin/champion" in text
    assert "docker run" not in text
