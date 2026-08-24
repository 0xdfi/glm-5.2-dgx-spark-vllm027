#!/usr/bin/env python3
"""Import and verify the public champion source closure.

This tool deliberately has no SSH, Docker, or inference code. It combines the
pinned O14 Git object with hash-pinned later champion sources already published
in this tree, so importing cannot interact with a serving cluster.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path, PurePosixPath
from typing import Any


O14_REPOSITORY = "https://github.com/0xdfi/GLM-5.2-Harness-O14-4x-DGX-Spark.git"
O14_COMMIT = "9ba54db8f7c852eabb90372f7dcf47f87eafbe20"
O14_TREE = "46e1d7837747c080d7300496abe9aa5f7c0e127d"
BUILD_A_V0_SHA256 = "965b6aeaf3a0c41abefec89a31592f6b8061a053053831be7c5e6f7560857515"
NATIVE_PATCH_SHA256 = "55344cbfd51765ba1ad77f4bf4c12724617c123e5d0c8d32f795b6203b5c30dc"
HISTORICAL_NATIVE_OBJECT_SHA256 = (
    "be9feccb01c0532d849fa30aab6a0edc809773be22ba8bb4a734040a9e4f95b8"
)
LATER_CHAMPION_OVERRIDES: dict[str, dict[str, dict[str, str]]] = {
    "vllm": {
        "model_executor/layers/sparse_attn_indexer.py": {
            "sha256": "113d85a97d7f4ef1047ebd2381e09b0a83bc7346f35bd2265100fabc9f3580fc",
            "base_sha256": "a6386326379dcf94439a0b020cd51a71ebfd0a36eaeb563417c821456b82a475",
            "stage": "r17-dcp2-v12l+dcp4-b4",
        },
        "model_executor/models/deepseek_mtp.py": {
            "sha256": "c40240abf3acd15cc18376744cd8046673c8ec2b3b18c61813ba029b65cbc010",
            "base_sha256": "83a52a68c0e2c3cd032cf0aff2c58570d87cf094f41f3a031208d1a74f6a2ede",
            "stage": "r17-dcp2-v12l",
        },
        "model_executor/models/deepseek_v2.py": {
            "sha256": "c83ad34fa2477c425606effb6816621379029886c1fa81ea8bdcbe76ffd948ea",
            "base_sha256": "bcbd9cde689ffe3f5f271aff3a5e01351b05a68371192cac9d65cb5ffc6b4ba9",
            "stage": "r17-dcp2-late-dcp-bind",
        },
        "v1/attention/backends/mla/sparse_utils.py": {
            "sha256": "7e1ca1c76b81c09f6198dd51e4649b4cd1ae0314311d1604464c3566ff624adf",
            "base_sha256": "a15a5767a6afe1427dccc395d7125d3f8f2b701b859c31eec91918d04d5761b4",
            "stage": "r17-dcp2-v3+dcp4",
        },
    },
    "b12x": {
        "attention/indexer/paged.py": {
            "sha256": "d654783235800f50a9434c0463cffe7f6c50f5efe53586ca2ef388f26adb5a66",
            "base_sha256": "78275bb3d1534f4c4321f610ce9c3548689a74bb3c6590ef9b067e782f0dcf76",
            "stage": "r17-b12x-two-level-fold-repair",
        },
        "attention/indexer/scratch.py": {
            "sha256": "c3eb82ab405685130e4d6092960bcaf01e8e4a58e9c8f731deb0e9c1ea51c453",
            "base_sha256": "a19a29b80daeab710a3dea8aa3194fe931e0eee698adc8b2b4148e2b399040c2",
            "stage": "r17-b12x-two-level-fold-repair",
        },
    },
}


class ImportError(RuntimeError):
    """Raised when the pinned source closure cannot be proven."""


def _git(repo: Path, *args: str) -> bytes:
    result = subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, check=False
    )
    if result.returncode:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise ImportError(f"git {' '.join(args)} failed: {detail}")
    return result.stdout


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _safe_relative(value: str) -> str:
    path = PurePosixPath(value)
    if not value or path.is_absolute() or ".." in path.parts or "\\" in value:
        raise ImportError(f"unsafe manifest path: {value!r}")
    return path.as_posix()


def _blob(repo: Path, path: str) -> bytes:
    safe = _safe_relative(path)
    return _git(repo, "show", f"{O14_COMMIT}:{safe}")


def _later_champion_content(root: Path, package: str, relative: str) -> bytes:
    try:
        expected = LATER_CHAMPION_OVERRIDES[package][relative]["sha256"]
    except KeyError as exc:
        raise ImportError(
            f"unknown later champion source: {package}/{relative}"
        ) from exc
    path = root / "runtime" / "overlays" / package / package / relative
    if not path.is_file() or path.is_symlink():
        raise ImportError(f"missing regular later champion source: {package}/{relative}")
    content = path.read_bytes()
    actual = _sha256(content)
    if actual != expected:
        raise ImportError(
            f"later champion source hash mismatch: {package}/{relative}: "
            f"expected {expected}, got {actual}"
        )
    return content


def _manifest(repo: Path, name: str) -> dict[str, Any]:
    path = f"reproducibility/manifests/{name}.json"
    try:
        value = json.loads(_blob(repo, path))
    except json.JSONDecodeError as exc:
        raise ImportError(f"{path}: invalid JSON: {exc}") from exc
    if not isinstance(value, dict) or not isinstance(value.get("files"), list):
        raise ImportError(f"{path}: expected object with files array")
    return value


def _donor(path: str) -> dict[str, str]:
    if path.startswith("vllm_flash_attn/cute/"):
        return {
            "name": "flash-attention",
            "repository": "https://github.com/vllm-project/flash-attention.git",
            "commit": "28e862d21806bc3580207aa0ad4e2759151e9827",
            "license": "BSD-3-Clause",
        }
    if path.startswith("third_party/fmha_sm100/cute/"):
        return {
            "name": "MSA",
            "repository": "https://github.com/vllm-project/MSA.git",
            "commit": "087c161814d4d9c735b46c21212a09e5f8eb92fa",
            "license": "MIT",
        }
    if path == "third_party/flashmla/flash_mla_interface.py":
        return {
            "name": "FlashMLA",
            "repository": "https://github.com/vllm-project/FlashMLA.git",
            "commit": "a8f794d1251cbfd88a5011445dd5582289c727e4",
            "license": "MIT",
        }
    return {
        "name": "vLLM",
        "repository": "https://github.com/vllm-project/vllm.git",
        "commit": "6e448d0ea9bf3d88d898b65449ca6dc2aec170ac",
        "license": "Apache-2.0",
    }


def _load_reference(repo: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    commit = _git(repo, "rev-parse", f"{O14_COMMIT}^{{commit}}").decode().strip()
    tree = _git(repo, "rev-parse", f"{O14_COMMIT}^{{tree}}").decode().strip()
    if commit != O14_COMMIT or tree != O14_TREE:
        raise ImportError(
            f"O14 identity mismatch: expected {O14_COMMIT}/{O14_TREE}, got {commit}/{tree}"
        )
    vllm = _manifest(repo, "vllm-overlay")
    b12x = _manifest(repo, "b12x-overlay")
    native = _manifest(repo, "native-patch")
    if len(vllm["files"]) != 74 or len(b12x["files"]) != 3 or len(native["files"]) != 14:
        raise ImportError("O14 source inventory is not the required 74/3/14 closure")
    return vllm, b12x, native


def _expected_files(
    root: Path, repo: Path, vllm: dict[str, Any], b12x: dict[str, Any]
) -> tuple[dict[str, bytes], dict[str, Any]]:
    output: dict[str, bytes] = {}
    vllm_entries: list[dict[str, Any]] = []
    source_inventory: list[dict[str, Any]] = []
    for entry in vllm["files"]:
        path = _safe_relative(entry["path"])
        source_path = f"reproducibility/overlays/vllm/vllm/{path}"
        content = _blob(repo, source_path)
        digest = _sha256(content)
        expected = entry["publication_sha256"]
        if digest != expected:
            raise ImportError(f"O14 vLLM blob hash mismatch: {path}")
        enabled = not path.endswith("builda_bmm_v0.py") and not path.endswith(
            "builda_bmm_v1.py"
        )
        source_inventory.append(
            {
                "path": path,
                "sha256": digest,
                "effective_champion_overlay": enabled,
            }
        )
        if not enabled:
            continue
        destination = f"runtime/overlays/vllm/vllm/{path}"
        override = LATER_CHAMPION_OVERRIDES["vllm"].get(path)
        target_content = (
            _later_champion_content(root, "vllm", path) if override else content
        )
        output[destination] = target_content
        donor = _donor(path)
        target_entry = {
            "path": path,
            "target": f"vllm/{path}",
            "sha256": _sha256(target_content),
            "base_sha256": entry.get("base_wheel_sha256"),
            "classification": entry.get("classification"),
            "donor": donor,
        }
        if override:
            target_entry["later_champion_source"] = {
                "authority": "sealed-r17-source-lineage",
                "stage": override["stage"],
            }
            target_entry["o14_predecessor_path"] = source_path
        else:
            target_entry["o14_origin_path"] = source_path
        vllm_entries.append(target_entry)

    o14_vllm_paths = {entry["path"] for entry in vllm["files"]}
    for path, metadata in LATER_CHAMPION_OVERRIDES["vllm"].items():
        if path in o14_vllm_paths:
            continue
        content = _later_champion_content(root, "vllm", path)
        output[f"runtime/overlays/vllm/vllm/{path}"] = content
        vllm_entries.append(
            {
                "path": path,
                "target": f"vllm/{path}",
                "sha256": metadata["sha256"],
                "base_sha256": metadata["base_sha256"],
                "classification": "modified",
                "donor": _donor(path),
                "later_champion_source": {
                    "authority": "sealed-r17-source-lineage",
                    "stage": metadata["stage"],
                },
            }
        )
    vllm_entries.sort(key=lambda item: item["path"])
    if len(vllm_entries) != 74:
        raise ImportError(f"effective vLLM overlay must contain 74 files, got {len(vllm_entries)}")

    b12x_entries: list[dict[str, Any]] = []
    for entry in b12x["files"]:
        path = _safe_relative(entry["path"])
        source_path = f"reproducibility/overlays/b12x/b12x/{path}"
        content = _blob(repo, source_path)
        digest = _sha256(content)
        if digest != entry["target_sha256"]:
            raise ImportError(f"O14 B12X blob hash mismatch: {path}")
        destination = f"runtime/overlays/b12x/b12x/{path}"
        output[destination] = content
        b12x_entries.append(
            {
                "path": path,
                "target": f"b12x/{path}",
                "sha256": digest,
                "base_sha256": entry["base_sha256"],
                "donor": {
                    "name": "B12X",
                    "repository": "https://github.com/lukealonso/b12x.git",
                    "commit": "334a2d75d166becea0aa640b402d521ea0a290eb",
                    "license": "Apache-2.0",
                },
                "o14_origin_path": source_path,
            }
        )

    for path, metadata in LATER_CHAMPION_OVERRIDES["b12x"].items():
        content = _later_champion_content(root, "b12x", path)
        output[f"runtime/overlays/b12x/b12x/{path}"] = content
        b12x_entries.append(
            {
                "path": path,
                "target": f"b12x/{path}",
                "sha256": metadata["sha256"],
                "base_sha256": metadata["base_sha256"],
                "classification": "modified",
                "donor": {
                    "name": "B12X",
                    "repository": "https://github.com/lukealonso/b12x.git",
                    "commit": "334a2d75d166becea0aa640b402d521ea0a290eb",
                    "license": "Apache-2.0",
                },
                "later_champion_source": {
                    "authority": "sealed-r17-source-lineage",
                    "stage": metadata["stage"],
                },
            }
        )
    b12x_entries.sort(key=lambda item: item["path"])

    kernel_source = "kernels/builda_bmm_v0.py"
    kernel = _blob(repo, kernel_source)
    if _sha256(kernel) != BUILD_A_V0_SHA256:
        raise ImportError("Build A v0 hash mismatch")
    output["runtime/kernels/builda_bmm_v0.py"] = kernel

    overlay = {
        "schema_version": "1.0.0",
        "source": {"repository": O14_REPOSITORY, "commit": O14_COMMIT, "tree": O14_TREE},
        "vllm": {"file_count": 74, "files": vllm_entries},
        "b12x": {"file_count": 5, "files": b12x_entries},
        "build_a": {
            "active": "v0",
            "source": "runtime/kernels/builda_bmm_v0.py",
            "target": "vllm/model_executor/layers/attention/builda_bmm_v0.py",
            "sha256": BUILD_A_V0_SHA256,
            "v1_included": False,
        },
    }
    o14_import = {
        "schema_version": "1.0.0",
        "source": {"repository": O14_REPOSITORY, "commit": O14_COMMIT, "tree": O14_TREE},
        "public_source_inventory": {
            "vllm_file_count": 74,
            "vllm_files": source_inventory,
            "b12x_file_count": 3,
            "native_path_count": 14,
        },
        "effective_champion": {
            "vllm_overlay_file_count": 74,
            "b12x_overlay_file_count": 5,
            "build_a_kernel": "v0",
            "build_a_v1_excluded": True,
            "later_champion_source_overrides": {
                package: {
                    path: {
                        "sha256": metadata["sha256"],
                        "stage": metadata["stage"],
                    }
                    for path, metadata in paths.items()
                }
                for package, paths in LATER_CHAMPION_OVERRIDES.items()
            },
        },
        "live_b4_reconciliation": {
            "status": "not-verified",
            "reason": "No live serving host or image was inspected for this publication batch.",
        },
    }
    output["runtime/manifests/overlay-files.json"] = _json_bytes(overlay)
    output["runtime/manifests/o14-import.json"] = _json_bytes(o14_import)
    return output, overlay


def _static_manifests(native: dict[str, Any], overlay: dict[str, Any]) -> dict[str, bytes]:
    roots = [
        ("vllm-python", "https://github.com/vllm-project/vllm.git", "6e448d0ea9bf3d88d898b65449ca6dc2aec170ac", "1fd074d072a3f052b5202d9df51a663f30cb4693", "Apache-2.0", "build"),
        ("vllm-native", "https://github.com/vllm-project/vllm.git", "e232d262369b8c918cf478a7a96a0fcf8127cf65", "e33103c3231203e5c4ad6d70815bd3df163d5ce8", "Apache-2.0", "build"),
        ("b12x", "https://github.com/lukealonso/b12x.git", "334a2d75d166becea0aa640b402d521ea0a290eb", "4936c7767f26967be28f1394fc5aa1d03171b34a", "Apache-2.0", "build"),
        ("flash-attention", "https://github.com/vllm-project/flash-attention.git", "28e862d21806bc3580207aa0ad4e2759151e9827", "ea5138c1ecbf25bd986115ee8bd0b4d1a62368c5", "BSD-3-Clause", "donor-and-build"),
        ("msa", "https://github.com/vllm-project/MSA.git", "087c161814d4d9c735b46c21212a09e5f8eb92fa", "88d4889ebe7619b7185a08df6e0aee038ed865c1", "MIT", "donor-and-build"),
        ("flashmla", "https://github.com/vllm-project/FlashMLA.git", "a8f794d1251cbfd88a5011445dd5582289c727e4", "7d788825942afb6e863a6f576adc6014dc5fe300", "MIT", "donor-and-build"),
        ("flashkda", "https://github.com/vllm-project/FlashKDA.git", "a3e42bbbece3bb38f7c426b880315294a336e82f", "47629d456b0d13492a1af847a002287604a25347", "Apache-2.0", "build"),
        ("tml-fa4", "https://github.com/vllm-project/tml-fa4.git", "b206834606ed5b5f21f8eed6b0683f528ea9cf7d", "4c9b977eeefca9ad45072ce9c44056a323ad8da5", "BSD-3-Clause", "build"),
        ("deepgemm", "https://github.com/vllm-project/DeepGEMM.git", "e21c821f39a2056d68067a466c64ddc942200106", "f45f2a8e8e3cc017cb5aa524f4ac77be5f130db5", "MIT", "build"),
        ("qutlass", "https://github.com/IST-DASLab/qutlass.git", "e74319e3405ce6d71965732880f5dc1f52371f64", "7c522b19e1341e36c63c86bbd664981dda49cd5e", "Apache-2.0", "build"),
        ("cutlass", "https://github.com/NVIDIA/cutlass.git", "da5e086dab31d63815acafdac9a9c5893b1c69e2", "e41cda383c529729667cab3511a850d1f501de46", "BSD-3-Clause", "build"),
        ("triton", "https://github.com/triton-lang/triton.git", "0add68262ab0a2e33b84524346cb27cbb2787356", "de055ee6adc5cfdd4891ccc3b50cf0a5cbd96163", "MIT", "runtime-version-lock"),
    ]
    upstream = {
        "schema_version": "1.0.0",
        "sources": [
            {"name": n, "repository": r, "commit": c, "tree": t, "license": lic, "role": role}
            for n, r, c, t, lic, role in roots
        ],
    }
    native_abi = {
        "schema_version": "1.0.0",
        "upstream": {
            "repository": "https://github.com/vllm-project/vllm.git",
            "commit": "e232d262369b8c918cf478a7a96a0fcf8127cf65",
            "tree": "e33103c3231203e5c4ad6d70815bd3df163d5ce8",
        },
        "patch": "runtime/patches/native/champion-sm121.patch",
        "patch_sha256": NATIVE_PATCH_SHA256,
        "target_source_manifest_sha256": native["target_source_manifest_sha256"],
        "architecture": "12.1a",
        "expected_library": "vllm/_C_stable_libtorch.abi3.so",
        "required_symbols": [
            "torch.ops._C_cache_ops.concat_and_cache_nvfp4_mla",
        ],
        "historical_reference_object": {
            "sha256": HISTORICAL_NATIVE_OBJECT_SHA256,
            "require_bit_identity": False,
            "purpose": "informational comparison only",
        },
    }
    upstream_bytes = _json_bytes(upstream)
    native_bytes = _json_bytes(native_abi)
    overlay_bytes = _json_bytes(overlay)
    build_lock = {
        "schema_version": "1.0.0",
        "architecture": "aarch64",
        "python": {"version": "3.12.3"},
        "cuda": {"version": "13.0.2", "architectures": ["12.1a"]},
        "base_image": "docker.io/nvidia/cuda@sha256:450d11555d20ac8ebbbc13ebf17589c2bd42869171a90179ce7098b4a5e64c6a",
        "sources": [
            {"name": n, "repository": r, "commit": c, "tree": t}
            for n, r, c, t, _lic, _role in roots
        ],
        "artifacts": [
            {
                "name": "rust-1.95.0-aarch64-unknown-linux-gnu",
                "url": "https://static.rust-lang.org/dist/2026-04-16/rust-1.95.0-aarch64-unknown-linux-gnu.tar.gz",
                "sha256": "3b9385d3144ac57616befa0ccbac524f857ba1b4ab074226e73a24d43568a98e",
            },
            {
                "name": "torch-2.11.0+cu130-cp312-aarch64",
                "url": "https://download-r2.pytorch.org/whl/cu130/torch-2.11.0%2Bcu130-cp312-cp312-manylinux_2_28_aarch64.whl",
                "sha256": "252f237d417fac3ba59b1635815c1f035a8241f2af038f2c076ed430932d89f1",
            },
            {
                "name": "flashinfer-python-0.6.15",
                "url": "https://files.pythonhosted.org/packages/66/a6/d3a1bf32b97f6c2c32aedb0b49fcd45259d9fb8d1131e86327b5672dd1b0/flashinfer_python-0.6.15-py3-none-any.whl",
                "sha256": "da6c339e14db4831ade0d593324f02907d44bd4b86b640faad4727d9e089477b",
            },
            {
                "name": "flashinfer-cubin-0.6.15",
                "url": "https://github.com/flashinfer-ai/flashinfer/releases/download/v0.6.15/flashinfer_cubin-0.6.15-py3-none-any.whl",
                "sha256": "3870958b02a1752b44121d6842c531e1be2a29202f623a5fd5f08efe3eb70186",
            },
            {
                "name": "flashinfer-jit-cache-0.6.15-cu130-aarch64",
                "url": "https://github.com/flashinfer-ai/flashinfer/releases/download/v0.6.15/flashinfer_jit_cache-0.6.15+cu130-cp39-abi3-manylinux_2_28_aarch64.whl",
                "sha256": "39e4f52378a34107ee277907bdec8bf8528c13a84143de8469d6da42ad937db3",
            },
        ],
        "toolchain": [
            {"name": "ubuntu", "version": "24.04@base-image-digest"},
            {"name": "gcc", "version": "13"},
            {"name": "cmake", "version": "3.28"},
            {"name": "ninja", "version": "1.11"},
            {"name": "rust", "version": "1.95.0"},
        ],
        "build_flags": [
            "VLLM_TARGET_DEVICE=cuda",
            "TORCH_CUDA_ARCH_LIST=12.1a",
            "VLLM_CUDA_ARCH_LIST=12.1a",
            "CMAKE_CUDA_ARCHITECTURES=121a",
            "MAX_JOBS=4",
        ],
        "manifests": {
            "python_source_sha256": _sha256(overlay_bytes),
            "native_source_sha256": _sha256(native_bytes),
        },
    }
    return {
        "runtime/manifests/upstream-lock.json": upstream_bytes,
        "runtime/manifests/native-abi.json": native_bytes,
        "runtime/manifests/build-lock.json": _json_bytes(build_lock),
    }


def _native_patch(repo: Path, native: dict[str, Any]) -> bytes:
    content = _blob(repo, "reproducibility/patches/native/exp1-r4-native.patch")
    if _sha256(content) != NATIVE_PATCH_SHA256 or native["patch_sha256"] != NATIVE_PATCH_SHA256:
        raise ImportError("native source patch hash mismatch")
    return content


def _runtime_requirements(repo: Path) -> bytes:
    """Materialize the observed O14 package snapshot without binary build roots."""

    source = _blob(repo, "reproducibility/observed-runtime-requirements.txt")
    lines = source.decode("utf-8").splitlines()
    locked: list[str] = []
    for line in lines:
        normalized = line.casefold()
        if normalized.startswith(("b12x==", "vllm==")):
            continue
        if normalized.startswith("triton=="):
            line = "triton==3.5.1"
        locked.append(line)
    header = [
        "# Copyright 2026 dfi / 0xdfi",
        "# SPDX-License-Identifier: Apache-2.0",
        "# Exact observed O14 package snapshot; vLLM/B12X are built from source.",
        "--extra-index-url https://download.pytorch.org/whl/cu130",
    ]
    return ("\n".join(header + locked) + "\n").encode("utf-8")


def _verify_exact(root: Path, expected: dict[str, bytes]) -> None:
    managed_roots = [
        root / "runtime/overlays/vllm",
        root / "runtime/overlays/b12x",
        root / "runtime/kernels",
        root / "runtime/patches/native",
    ]
    expected_paths = set(expected)
    for relative, content in expected.items():
        path = root / relative
        if not path.is_file() or path.is_symlink():
            raise ImportError(f"missing regular imported file: {relative}")
        if path.read_bytes() != content:
            raise ImportError(f"imported file differs from pinned O14 blob: {relative}")
    actual = {
        path.relative_to(root).as_posix()
        for directory in managed_roots
        if directory.exists()
        for path in directory.rglob("*")
        if path.is_file()
    }
    managed_expected = {
        path
        for path in expected_paths
        if any(path == base.relative_to(root).as_posix() or path.startswith(base.relative_to(root).as_posix() + "/") for base in managed_roots)
    }
    if actual != managed_expected:
        raise ImportError(
            f"managed runtime member mismatch: missing={sorted(managed_expected-actual)} "
            f"extras={sorted(actual-managed_expected)}"
        )
    if any("builda_bmm_v1" in path for path in actual):
        raise ImportError("Build A v1 is forbidden from the effective champion closure")


def import_runtime(root: Path, repo: Path, verify_only: bool) -> None:
    vllm, b12x, native = _load_reference(repo)
    expected, overlay = _expected_files(root, repo, vllm, b12x)
    expected["runtime/patches/native/champion-sm121.patch"] = _native_patch(repo, native)
    expected["runtime/requirements.lock"] = _runtime_requirements(repo)
    expected.update(_static_manifests(native, overlay))
    if not verify_only:
        for relative, content in expected.items():
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
    _verify_exact(root, expected)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--source",
        type=Path,
        help="local clone containing the pinned O14 commit (or set O14_SOURCE)",
    )
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args(argv)
    source = args.source
    if source is None and os.environ.get("O14_SOURCE"):
        source = Path(os.environ["O14_SOURCE"])
    if source is None:
        candidate = args.root.resolve().parent / "GLM-5.2-Harness-O14-4x-DGX-Spark"
        if candidate.is_dir():
            source = candidate
    if source is None:
        parser.error("--source is required unless O14_SOURCE or a sibling O14 clone exists")
    try:
        import_runtime(args.root.resolve(), source.resolve(), args.verify_only)
    except ImportError as exc:
        print(f"runtime-source-closure: FAIL: {exc}", file=sys.stderr)
        return 1
    print("runtime-source-closure: PASS (O14=74 vLLM source, effective=74+Build-A-v0; B12X=5; native=14)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
