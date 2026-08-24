from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VERIFY = ROOT / "scripts" / "verify_model.py"
VERIFY_LICENSE = ROOT / "scripts" / "verify_model_license.py"
FETCH = ROOT / "scripts" / "fetch_model_artifacts.sh"
FIXTURE = ROOT / "tests" / "fixtures" / "model-small"


def run(*args: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [*args], cwd=ROOT, text=True, capture_output=True, env=env
    )


def test_published_locks_are_honest_and_separate():
    checkpoint = json.loads((ROOT / "models/checkpoint.lock.json").read_text())
    sidecar = json.loads((ROOT / "models/sidecar.lock.json").read_text())

    assert checkpoint["artifact_id"] == "glm52-dfi-hybrid-corrected-v4-lmheadw8v2"
    assert checkpoint["base_model"]["revision"] == "1d3bcfe5ec549ecd000fd80b37f191183842e983"
    assert checkpoint["rights"]["status"] == "unknown"
    assert checkpoint["distribution"]["method"] == "external-required"
    assert checkpoint["integrity"]["status"] == "unknown"
    assert checkpoint["integrity"]["files"] == []
    assert sidecar["artifact_id"] == "lmhead_w8v2_sidecar.safetensors"
    assert sidecar["rights"]["status"] == "unknown"
    assert sidecar["integrity"]["status"] == "unknown"
    assert sidecar["integrity"]["files"] == []
    assert "shard_count" not in checkpoint


def test_fixture_verifies_only_complete_file_checksums():
    result = run(
        sys.executable,
        str(VERIFY),
        "--checkpoint-lock",
        str(FIXTURE / "checkpoint.lock.json"),
        "--sidecar-lock",
        str(FIXTURE / "sidecar.lock.json"),
        "--model-dir",
        str(FIXTURE / "artifacts"),
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["result"] == "PASS"
    assert payload["verified_files"] == 3


def test_fixture_rejects_wrong_sidecar_hash(tmp_path: Path):
    copied = tmp_path / "artifacts"
    copied.mkdir()
    for source in (FIXTURE / "artifacts").iterdir():
        (copied / source.name).write_bytes(source.read_bytes())
    original = (copied / "sidecar.fixture").read_bytes()
    (copied / "sidecar.fixture").write_bytes(b"X" + original[1:])

    result = run(
        sys.executable,
        str(VERIFY),
        "--checkpoint-lock",
        str(FIXTURE / "checkpoint.lock.json"),
        "--sidecar-lock",
        str(FIXTURE / "sidecar.lock.json"),
        "--model-dir",
        str(copied),
    )
    assert result.returncode != 0
    assert "sidecar.fixture" in result.stderr
    assert "SHA-256 mismatch" in result.stderr


def test_unknown_rights_fail_closed_without_pass_receipts(tmp_path: Path):
    checkpoint_receipt = tmp_path / "checkpoint.json"
    sidecar_receipt = tmp_path / "sidecar.json"
    result = run(
        sys.executable,
        str(VERIFY_LICENSE),
        "--checkpoint",
        "models/checkpoint.lock.json",
        "--checkpoint-receipt",
        str(checkpoint_receipt),
        "--sidecar",
        "models/sidecar.lock.json",
        "--sidecar-receipt",
        str(sidecar_receipt),
    )
    assert result.returncode != 0
    assert "unknown rights" in result.stderr
    assert not checkpoint_receipt.exists()
    assert not sidecar_receipt.exists()
    assert "PASS" not in result.stdout + result.stderr


def test_fetch_is_staged_only_and_never_echoes_credentials():
    canary = "CHAMPION_TEST_" + "SECRET_9f34"
    env = os.environ | {"MODEL_TOKEN": canary}
    result = run(
        "bash",
        str(FETCH),
        "--model-dir",
        str(FIXTURE / "artifacts"),
        "--checkpoint-lock",
        str(FIXTURE / "checkpoint.lock.json"),
        "--sidecar-lock",
        str(FIXTURE / "sidecar.lock.json"),
        env=env,
    )
    assert result.returncode == 0, result.stderr
    assert canary not in result.stdout + result.stderr
    assert "already-staged" in result.stdout

    refused = run("bash", str(FETCH), "--remote", "https://example.invalid/model")
    assert refused.returncode != 0
    assert "no approved remote" in refused.stderr


def test_fixture_hashes_are_complete_not_prefixes():
    for name in ("checkpoint.lock.json", "sidecar.lock.json"):
        lock = json.loads((FIXTURE / name).read_text())
        for entry in lock["integrity"]["files"]:
            data = (FIXTURE / "artifacts" / entry["path"]).read_bytes()
            assert entry["size"] == len(data)
            assert entry["sha256"] == hashlib.sha256(data).hexdigest()
