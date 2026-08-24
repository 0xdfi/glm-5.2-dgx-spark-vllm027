from __future__ import annotations

import json
from pathlib import Path

from scripts.verify_image import verify_rootfs, verify_source


ROOT = Path(__file__).resolve().parents[1]


def test_repository_source_closure_verifies():
    verify_source(ROOT)


def test_rootfs_requires_manifest_legal_bundle_and_rebuilt_native(tmp_path):
    bundle = tmp_path / "opt/champion"
    for relative in ("manifests/build-lock.json", "manifests/overlay-files.json", "legal/LICENSE", "legal/NOTICE", "legal/source-provenance.json"):
        path = bundle / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n")
    native = tmp_path / "opt/vllm-extra/nvfp4_ds_mla/_C_stable_libtorch.abi3.so"
    native.parent.mkdir(parents=True)
    native.write_bytes(b"ELF-test")
    verify_rootfs(tmp_path)


def test_rootfs_rejects_mutable_full_ledger(tmp_path):
    bundle = tmp_path / "opt/champion"
    for relative in ("manifests/build-lock.json", "manifests/overlay-files.json", "legal/LICENSE", "legal/NOTICE", "legal/source-provenance.json", "legal/provenance.json"):
        path = bundle / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n")
    native = tmp_path / "opt/vllm-extra/nvfp4_ds_mla/_C_stable_libtorch.abi3.so"
    native.parent.mkdir(parents=True)
    native.write_bytes(b"ELF-test")
    try:
        verify_rootfs(tmp_path)
    except ValueError as exc:
        assert "mutable full provenance" in str(exc)
    else:
        raise AssertionError("mutable ledger should be rejected")


def test_native_reference_is_informational_not_a_bit_identity_gate():
    native = json.loads((ROOT / "runtime/manifests/native-abi.json").read_text())
    reference = native["historical_reference_object"]
    assert reference["sha256"] == "be9feccb01c0532d849fa30aab6a0edc809773be22ba8bb4a734040a9e4f95b8"
    assert reference["require_bit_identity"] is False
