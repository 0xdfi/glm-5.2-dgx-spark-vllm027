# Copyright 2026 dfi / 0xdfi
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import hashlib
import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / ".github/workflows"
PINS = {
    "actions/checkout": "3d3c42e5aac5ba805825da76410c181273ba90b1",
    "actions/setup-python": "5fda3b95a4ea91299a34e894583c3862153e4b97",
    "actions/attest": "1e69f48acb82d1966a394da916b4c1698aa569d6",
    "actions/upload-artifact": "043fb46d1a93c77aae656e7c1c64a875d1fc6a0",
}


def workflow(name: str) -> str:
    return (WORKFLOWS / name).read_text(encoding="utf-8")


def test_every_action_is_from_the_fixed_allowlist_and_immutable():
    for path in sorted(WORKFLOWS.glob("*.yml")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if "uses:" not in line:
                continue
            value = line.split("uses:", 1)[1].strip()
            action, separator, revision = value.partition("@")
            assert separator and PINS.get(action) == revision, (path, value)


def test_ci_is_read_only_and_runs_all_source_gates_on_prs_and_main():
    text = workflow("ci.yml")
    assert "pull_request:" in text and "push:" in text and "main" in text
    assert "contents: read" in text
    for forbidden in ("pull_request_target", "id-token: write", "attestations: write", "packages: write", "self-hosted", "${{ secrets.", "upload-artifact"):
        assert forbidden not in text
    for required in (
        "fetch-depth: 0", "persist-credentials: false", "3.12.3", "bootstrap_dev.sh",
        "make test", "make contracts", "make lint-manifests", "reuse lint",
        "register_provenance.py --staged --check", "audit_licenses.py --strict",
        "audit_container_licenses.py", "scan_secrets.sh", "scan_large_files.py",
        "bash -n", "test_docs.py", "tools/import_runtime.py", "9ba54db8f7c852eabb90372f7dcf47f87eafbe20",
    ):
        assert required in text


def test_evidence_workflow_has_one_fixed_mode_and_blocks_non_source_modes():
    text = workflow("attest-evidence.yml")
    assert "workflow_dispatch:" in text
    assert all(f"- {mode}" in text for mode in ("source", "build", "live", "registry-readback"))
    assert "source-evidence" in text
    assert "refs/heads/main" in text and "0xdfi/glm-5.2-dgx-spark-vllm027" in text
    for unsafe in ("self-hosted", "packages: write", "bundle:", "path_input", "ref_input", "url_input", "sha_input", "${{ secrets."):
        assert unsafe not in text
    assert "id-token: write" in text and "attestations: write" in text and "contents: read" in text
    assert "actions/attest@" in text and "actions/upload-artifact@" in text
    assert "source-complete-candidate.receipt.json" in text
    assert "BLOCKED" in text and "inputs.mode != 'source'" in text


def test_release_is_manual_guarded_and_can_only_create_a_draft_source_snapshot():
    text = workflow("release.yml")
    assert "workflow_dispatch:" in text and "inputs:" not in text
    assert "source-release" in text and "ubuntu-24.04" in text
    assert "refs/heads/main" in text and "0xdfi/glm-5.2-dgx-spark-vllm027" in text
    assert "render_release_status.py --check" in text
    assert "verify_attestation.py" in text
    assert "package_release.sh" in text
    assert "gh release create" in text and "--draft" in text and "--prerelease" in text and "SOURCE SNAPSHOT" in text
    for forbidden in ("packages: write", "ghcr.io", "download-artifact", "actions/cache", "champion-runtime-v1.0", "--latest"):
        assert forbidden not in text


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_source_package_is_deterministic_across_directories_and_umasks(tmp_path: Path):
    outputs = []
    for name, mask in (("one", "022"), ("two", "077")):
        output = tmp_path / name
        command = f"umask {mask}; ./scripts/package_release.sh --output '{output}'"
        result = subprocess.run(["bash", "-c", command], cwd=ROOT, text=True, capture_output=True)
        assert result.returncode == 0, result.stderr
        outputs.append(output)
    names = {path.name for path in outputs[0].iterdir()}
    assert names == {"glm52-champion-source.tar.gz", "source-manifest.json", "SHA256SUMS"}
    for name in sorted(names):
        assert _digest(outputs[0] / name) == _digest(outputs[1] / name)
        assert (outputs[0] / name).stat().st_mode & 0o777 == 0o644
    listing = subprocess.run(["tar", "-tzf", str(outputs[0] / "glm52-champion-source.tar.gz")], check=True, capture_output=True, text=True).stdout
    assert "LICENSES/Apache-2.0.txt" in listing
    assert "publication/status.json" in listing
    assert "runtime/manifests/build-lock.json" in listing
    assert ".git/" not in listing and "cluster.env\n" not in listing
    assert not any(suffix in listing for suffix in (".safetensors", ".whl", ".so\n", ".tar\n"))
