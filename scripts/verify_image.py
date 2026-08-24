#!/usr/bin/env python3
"""Verify source-context or extracted-image champion runtime invariants."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path, PurePosixPath


PRIVATE = re.compile(
    rb"/(?:Users|home)/[^/\s]+(?:/|\b)|BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def safe(root: Path, relative: str) -> Path:
    value = PurePosixPath(relative)
    if value.is_absolute() or ".." in value.parts:
        raise ValueError(f"unsafe path: {relative}")
    return root.joinpath(*value.parts)


def verify_source(root: Path) -> None:
    manifests = root / "manifests" if (root / "manifests").is_dir() else root / "runtime/manifests"
    prefix = root if (root / "overlays").is_dir() else root / "runtime"
    overlay_path = manifests / "overlay-files.json"
    overlay = json.loads(overlay_path.read_text())
    build = json.loads((manifests / "build-lock.json").read_text())
    if sha256(overlay_path) != build["manifests"]["python_source_sha256"]:
        raise ValueError("overlay manifest hash does not match build lock")
    o14 = json.loads((manifests / "o14-import.json").read_text())
    effective = o14["effective_champion"]
    expected_counts = {
        "vllm": effective["vllm_overlay_file_count"],
        "b12x": effective["b12x_overlay_file_count"],
    }
    for package in ("vllm", "b12x"):
        if overlay[package]["file_count"] != expected_counts[package]:
            raise ValueError(
                f"{package} overlay count does not match pinned effective champion"
            )
        if overlay[package]["file_count"] != len(overlay[package]["files"]):
            raise ValueError(f"{package} overlay count does not match its file list")
        for entry in overlay[package]["files"]:
            path = safe(prefix / f"overlays/{package}/{package}", entry["path"])
            if not path.is_file() or sha256(path) != entry["sha256"]:
                raise ValueError(f"{package} overlay mismatch: {entry['path']}")
    kernel = prefix / "kernels/builda_bmm_v0.py"
    if sha256(kernel) != overlay["build_a"]["sha256"]:
        raise ValueError("Build A v0 mismatch")
    if list(prefix.rglob("*builda_bmm_v1*")):
        raise ValueError("Build A v1 is forbidden")
    native = json.loads((manifests / "native-abi.json").read_text())
    patch = root / native["patch"] if (root / "runtime").is_dir() else prefix / "patches/native/champion-sm121.patch"
    if sha256(patch) != native["patch_sha256"]:
        raise ValueError("native patch mismatch")
    for path in prefix.rglob("*"):
        if "__pycache__" in path.parts or path.suffix in {".pyc", ".pyo"}:
            continue
        if path.is_file() and PRIVATE.search(path.read_bytes()):
            raise ValueError(f"private path or credential pattern: {path.relative_to(root)}")


def verify_rootfs(root: Path) -> None:
    bundle = root / "opt/champion"
    for relative in ("manifests/build-lock.json", "manifests/overlay-files.json", "legal/LICENSE", "legal/NOTICE", "legal/source-provenance.json"):
        if not (bundle / relative).is_file():
            raise ValueError(f"image is missing {relative}")
    native = root / "opt/vllm-extra/nvfp4_ds_mla/_C_stable_libtorch.abi3.so"
    if not native.is_file() or native.stat().st_size == 0:
        raise ValueError("image is missing the rebuilt NVFP4 native object")
    if (bundle / "legal/provenance.json").exists():
        raise ValueError("mutable full provenance ledger must not be embedded")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--source-tree", type=Path)
    mode.add_argument("--rootfs", type=Path)
    args = parser.parse_args()
    try:
        if args.source_tree:
            verify_source(args.source_tree.resolve())
        else:
            verify_rootfs(args.rootfs.resolve())
    except (OSError, KeyError, ValueError, json.JSONDecodeError) as exc:
        print(f"image-verify: FAIL: {exc}")
        return 1
    print("image-verify: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
