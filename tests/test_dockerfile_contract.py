from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

from champion_runtime.contracts import load_contract


ROOT = Path(__file__).resolve().parents[1]
DOCKERFILE = (ROOT / "docker/Dockerfile.champion").read_text()


def test_build_lock_validates_and_base_is_arm64_digest():
    lock = load_contract(ROOT / "runtime/manifests/build-lock.json", "build-lock")
    assert lock["base_image"] == "docker.io/nvidia/cuda@sha256:450d11555d20ac8ebbbc13ebf17589c2bd42869171a90179ce7098b4a5e64c6a"
    assert lock["cuda"]["architectures"] == ["12.1a"]
    assert lock["base_image"] in DOCKERFILE


def test_every_source_root_is_commit_and_tree_locked_in_dockerfile():
    roots = json.loads((ROOT / "runtime/manifests/upstream-lock.json").read_text())["sources"]
    for root in roots:
        assert root["commit"] in DOCKERFILE or root["name"] == "triton"
        assert root["tree"] in DOCKERFILE or root["name"] == "triton"
    assert "triton==3.5.1" in (ROOT / "runtime/requirements.lock").read_text()


def test_dockerfile_has_no_historical_binary_build_roots():
    forbidden = {
        "02dba246bc7e9e4b189ce1cb43b25792176ca4389352d31ebecd5b27651e16bb",
        "b8ed2f89a59c810be5a6b13c54ab2ebb956770837119cef762a46e0607fc6941",
        "be9feccb01c0532d849fa30aab6a0edc809773be22ba8bb4a734040a9e4f95b8",
    }
    assert not any(value in DOCKERFILE for value in forbidden)
    assert "O14_RELEASE" not in DOCKERFILE


def test_sm121a_and_effective_overlay_contract_are_explicit():
    for required in ("TORCH_CUDA_ARCH_LIST=12.1a", "VLLM_CUDA_ARCH_LIST=12.1a", "CMAKE_CUDA_ARCHITECTURES=121a"):
        assert required in DOCKERFILE
    assert "runtime/kernels/builda_bmm_v0.py" in DOCKERFILE
    assert not re.search(r"^COPY .*builda_bmm_v1", DOCKERFILE, re.MULTILINE)


def test_apt_and_python_requirements_are_exact_versions():
    apt_block = re.search(r"apt-get install -y --no-install-recommends(.*?)&& rm", DOCKERFILE, re.S)
    assert apt_block
    for token in re.findall(r"\\\n\s+([^\\\s]+)", apt_block.group(1)):
        assert "=" in token
    for line in (ROOT / "runtime/requirements.lock").read_text().splitlines():
        if line and not line.startswith(("#", "--")):
            assert re.fullmatch(r"[a-zA-Z0-9_.-]+==[^\s]+", line)


def test_only_source_projection_is_embedded():
    assert "compliance/source-provenance.json" in DOCKERFILE
    assert "compliance/provenance.json" not in DOCKERFILE
    assert 'com.0xdfi.champion.live-b4-parity="not-verified"' in DOCKERFILE


def test_build_produced_receipt_matches_shared_contract(tmp_path):
    receipt = tmp_path / "build.json"
    result = subprocess.run(
        [
            str(ROOT / ".venv/bin/python"),
            str(ROOT / "scripts/build_subject.py"),
            "--root",
            str(ROOT),
            "--image-digest",
            "sha256:" + "a" * 64,
            "--receipt",
            str(receipt),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    payload = load_contract(receipt, "receipt")
    assert payload["receipt_type"] == "build-produced"
    assert payload["result"] == "PROVISIONAL"
