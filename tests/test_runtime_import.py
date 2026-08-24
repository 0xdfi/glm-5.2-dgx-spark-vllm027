from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
def load_importer():
    spec = importlib.util.spec_from_file_location("import_runtime", ROOT / "tools/import_runtime.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_pinned_o14_identity_and_public_inventory():
    module = load_importer()
    assert module.O14_COMMIT == "9ba54db8f7c852eabb90372f7dcf47f87eafbe20"
    assert module.O14_TREE == "46e1d7837747c080d7300496abe9aa5f7c0e127d"
    manifest = json.loads((ROOT / "runtime/manifests/o14-import.json").read_text())
    assert manifest["source"]["commit"] == module.O14_COMMIT
    assert manifest["public_source_inventory"]["vllm_file_count"] == 74
    assert manifest["live_b4_reconciliation"]["status"] == "not-verified"
    assert manifest["live_b4_reconciliation"]["observed_overlay_files"] == {
        "envs.py": "a1f1c8a0b73441658d22148cef086a766186b299ee74be4ab01a7249266fdd6c",
        "utils/mem_utils.py": "132a62eba1a633b8295acd48ecdf99bc3f982a78655ac6b6a5939365b47daf6b",
        "v1/worker/gpu_worker.py": "654f08d0352fabc126d9515e838ba18d7190fd1f39edcc8655f52440913cef13",
    }


def test_effective_champion_includes_later_runtime_closure_and_excludes_v1():
    module = load_importer()
    manifest = json.loads((ROOT / "runtime/manifests/overlay-files.json").read_text())
    assert manifest["vllm"]["file_count"] == 77
    assert manifest["b12x"]["file_count"] == 5
    assert set(module.LATER_CHAMPION_OVERRIDES) == {"vllm", "b12x"}
    assert set(module.LATER_CHAMPION_OVERRIDES["vllm"]) == {
        "envs.py",
        "model_executor/layers/sparse_attn_indexer.py",
        "model_executor/models/deepseek_mtp.py",
        "model_executor/models/deepseek_v2.py",
        "utils/mem_utils.py",
        "v1/attention/backends/mla/sparse_utils.py",
        "v1/worker/gpu_worker.py",
    }
    assert set(module.LATER_CHAMPION_OVERRIDES["b12x"]) == {
        "attention/indexer/paged.py",
        "attention/indexer/scratch.py",
    }
    assert manifest["build_a"] == {
        "active": "v0",
        "sha256": "965b6aeaf3a0c41abefec89a31592f6b8061a053053831be7c5e6f7560857515",
        "source": "runtime/kernels/builda_bmm_v0.py",
        "target": "vllm/model_executor/layers/attention/builda_bmm_v0.py",
        "v1_included": False,
    }
    assert not list((ROOT / "runtime").rglob("*builda_bmm_v1*"))


def test_later_champion_override_reader_rejects_tampered_source(tmp_path):
    module = load_importer()
    relative = "model_executor/layers/sparse_attn_indexer.py"
    target = tmp_path / "runtime/overlays/vllm/vllm" / relative
    target.parent.mkdir(parents=True)
    target.write_text("# tampered\n")
    try:
        module._later_champion_content(tmp_path, "vllm", relative)
    except module.ImportError as exc:
        assert "later champion source hash mismatch" in str(exc)
    else:
        raise AssertionError("tampered later champion source should be rejected")


def test_donor_roots_are_first_class_and_licensed():
    lock = json.loads((ROOT / "runtime/manifests/upstream-lock.json").read_text())
    roots = {entry["name"]: entry for entry in lock["sources"]}
    assert roots["flash-attention"]["commit"] == "28e862d21806bc3580207aa0ad4e2759151e9827"
    assert roots["flash-attention"]["license"] == "BSD-3-Clause"
    assert roots["msa"]["license"] == "MIT"
    assert roots["flashmla"]["license"] == "MIT"
    assert {"flashkda", "tml-fa4", "deepgemm", "qutlass", "cutlass", "triton"} <= roots.keys()


def test_verify_only_rejects_tampered_overlay(tmp_path):
    module = load_importer()
    target = tmp_path / "runtime/overlays/vllm/vllm/config/cache.py"
    target.parent.mkdir(parents=True)
    original = b"original\n"
    target.write_bytes(original + b"# tampered\n")
    try:
        module._verify_exact(tmp_path, {target.relative_to(tmp_path).as_posix(): original})
    except module.ImportError as exc:
        assert "differs from pinned O14 blob" in str(exc)
    else:
        raise AssertionError("tampered overlay should be rejected")


def test_importer_rejects_wrong_source_repository(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path / "wrong")], check=True)
    result = subprocess.run(
        ["python3", str(ROOT / "tools/import_runtime.py"), "--root", str(ROOT), "--source", str(tmp_path / "wrong"), "--verify-only"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert "runtime-source-closure: FAIL" in result.stderr
