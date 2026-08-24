from __future__ import annotations

import ast
import hashlib
import json
import shutil
from pathlib import Path

import pytest

from scripts.verify_image import verify_source as verify_image_source
from scripts.verify_source import _verify_runtime_manifest


ROOT = Path(__file__).resolve().parents[1]

EXPECTED_LATER_CHAMPION_OVERLAYS = {
    "b12x": {
        "attention/indexer/paged.py": (
            "d654783235800f50a9434c0463cffe7f6c50f5efe53586ca2ef388f26adb5a66"
        ),
        "attention/indexer/scratch.py": (
            "c3eb82ab405685130e4d6092960bcaf01e8e4a58e9c8f731deb0e9c1ea51c453"
        ),
    },
    "vllm": {
        "model_executor/layers/sparse_attn_indexer.py": (
            "113d85a97d7f4ef1047ebd2381e09b0a83bc7346f35bd2265100fabc9f3580fc"
        ),
        "model_executor/models/deepseek_mtp.py": (
            "c40240abf3acd15cc18376744cd8046673c8ec2b3b18c61813ba029b65cbc010"
        ),
        "model_executor/models/deepseek_v2.py": (
            "c83ad34fa2477c425606effb6816621379029886c1fa81ea8bdcbe76ffd948ea"
        ),
        "v1/attention/backends/mla/sparse_utils.py": (
            "7e1ca1c76b81c09f6198dd51e4649b4cd1ae0314311d1604464c3566ff624adf"
        ),
    },
}


def source(package: str, relative: str) -> str:
    path = ROOT / "runtime" / "overlays" / package / package / relative
    return path.read_text(encoding="utf-8")


def test_exact_later_champion_sources_are_present_and_manifested():
    manifest = json.loads(
        (ROOT / "runtime/manifests/overlay-files.json").read_text(encoding="utf-8")
    )
    for package, expected in EXPECTED_LATER_CHAMPION_OVERLAYS.items():
        entries = {entry["path"]: entry for entry in manifest[package]["files"]}
        for relative, digest in expected.items():
            path = ROOT / "runtime" / "overlays" / package / package / relative
            assert path.is_file(), f"missing later champion source: {relative}"
            assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
            assert entries[relative]["sha256"] == digest


@pytest.mark.parametrize(
    "verifier",
    (verify_image_source, _verify_runtime_manifest),
    ids=("image-verifier", "source-verifier"),
)
@pytest.mark.parametrize(
    "reanchor_build_lock",
    (False, True),
    ids=("manifest-hash", "effective-count"),
)
def test_verifiers_reject_coherently_shrunken_overlay(
    tmp_path, verifier, reanchor_build_lock
):
    shutil.copytree(ROOT / "runtime", tmp_path / "runtime")
    manifest_path = tmp_path / "runtime/manifests/overlay-files.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    removed = manifest["vllm"]["files"].pop()
    manifest["vllm"]["file_count"] = len(manifest["vllm"]["files"])
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (tmp_path / "runtime/overlays/vllm/vllm" / removed["path"]).unlink()
    if reanchor_build_lock:
        build_path = tmp_path / "runtime/manifests/build-lock.json"
        build = json.loads(build_path.read_text(encoding="utf-8"))
        build["manifests"]["python_source_sha256"] = hashlib.sha256(
            manifest_path.read_bytes()
        ).hexdigest()
        build_path.write_text(
            json.dumps(build, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    with pytest.raises(ValueError, match="overlay manifest|overlay count"):
        verifier(tmp_path)


def test_b12x_global_topk_accepts_only_dcp2_and_dcp4():
    indexer = source("vllm", "model_executor/layers/sparse_attn_indexer.py")
    ast.parse(indexer)
    assert "_B12X_DCP_SUPPORTED_WORLD_SIZES = (2, 4)" in indexer
    assert "dcp_world_size not in _B12X_DCP_SUPPORTED_WORLD_SIZES" in indexer
    assert "dcp_rank not in range(dcp_world_size)" in indexer
    assert "_merge_b12x_dcp_topk" in indexer
    assert "_dcp_all_gather_first_dim_into" in indexer
    assert "_assert_dcp1_for_b12x" not in indexer


def test_dcp_score_scratch_is_owned_by_both_model_callers():
    main = source("vllm", "model_executor/models/deepseek_v2.py")
    mtp = source("vllm", "model_executor/models/deepseek_mtp.py")
    ast.parse(main)
    ast.parse(mtp)
    assert "def allocate_sparse_indexer_buffers(" in main
    assert "decode_context_parallel_size) > 1" in main
    assert "topk_scores_buffer = torch.empty(" in main
    assert "topk_scores_buffer=self.topk_scores_buffer" in main
    assert "topk_indices_buffer, topk_scores_buffer = allocate_sparse_indexer_buffers(" in mtp
    assert "topk_scores_buffer=topk_scores_buffer" in mtp


def test_dcp4_local_to_global_remap_is_fail_closed():
    sparse_utils = source("vllm", "v1/attention/backends/mla/sparse_utils.py")
    ast.parse(sparse_utils)
    assert "dcp_world_size not in (2, 4)" in sparse_utils
    assert "dcp_rank not in range(dcp_world_size)" in sparse_utils
    assert "DCP_WORLD_SIZE=dcp_world_size" in sparse_utils
    assert "DCP_RANK=dcp_rank" in sparse_utils


def test_large_profile_cap_is_wired_to_profile_geometry_only():
    indexer = source("vllm", "model_executor/layers/sparse_attn_indexer.py")
    large = json.loads((ROOT / "presets/large-1m.json").read_text(encoding="utf-8"))
    common = json.loads((ROOT / "presets/common.json").read_text(encoding="utf-8"))
    assert common["container_env"]["VLLM_USE_B12X_SPARSE_INDEXER"] == "1"
    assert large["effective_engine"]["decode_context_parallel_size"] == 4
    assert large["container_env"]["VLLM_R17_PROFILE_DRAFTER_CAP"] == "1"
    assert 'os.environ.get("VLLM_R17_PROFILE_DRAFTER_CAP", "0")' in indexer
    assert "if _r17_profile_pagetable_cap_enabled():" in indexer
    assert "_r17_capped_profile_k_rows(" in indexer


def test_two_level_fold_uses_fixed_caller_owned_scratch():
    paged = source("b12x", "attention/indexer/paged.py")
    scratch = source("b12x", "attention/indexer/scratch.py")
    ast.parse(paged)
    ast.parse(scratch)
    assert "scratch.get_indexer_two_level_fold_buffers(" in paged
    assert "output_row_stride=fold_capacity_slices" in paged
    assert "fold_values = torch.empty(" not in paged
    assert "def get_indexer_two_level_fold_buffers(" in scratch
    assert "_PAGED_INDEX_TWO_LEVEL_MAX_SLICES = 32" in scratch
    assert "two_level_fold_values_offset_bytes" in scratch
