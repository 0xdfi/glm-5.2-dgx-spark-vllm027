#!/usr/bin/env python3
# Copyright 2026 dfi / 0xdfi
# SPDX-License-Identifier: Apache-2.0
"""Verify the tracked, rebuild-relevant source closure and emit one leaf receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from champion_runtime.contracts import load_contract
from champion_runtime.receipts import construct_receipt, source_release_id
from scripts.render_launch import render


class SourceVerificationError(ValueError):
    """Raised when a source release is incomplete or inconsistent."""


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _clean(root: Path) -> bool:
    for args in (("diff", "--quiet"), ("diff", "--cached", "--quiet")):
        if subprocess.run(["git", "-C", str(root), *args]).returncode:
            return False
    return True


def _verify_runtime_manifest(root: Path) -> None:
    manifests = root / "runtime/manifests"
    overlay_path = manifests / "overlay-files.json"
    overlay = json.loads(overlay_path.read_text())
    build = json.loads((manifests / "build-lock.json").read_text())
    if _sha(overlay_path) != build["manifests"]["python_source_sha256"]:
        raise SourceVerificationError("overlay manifest hash does not match build lock")
    o14 = json.loads((manifests / "o14-import.json").read_text())
    effective = o14["effective_champion"]
    expected_counts = {
        "vllm": effective["vllm_overlay_file_count"],
        "b12x": effective["b12x_overlay_file_count"],
    }
    for package in ("vllm", "b12x"):
        if overlay[package]["file_count"] != expected_counts[package]:
            raise SourceVerificationError(
                f"{package} overlay count does not match pinned effective champion"
            )
        if overlay[package]["file_count"] != len(overlay[package]["files"]):
            raise SourceVerificationError(
                f"{package} overlay count does not match its file list"
            )
        base = root / f"runtime/overlays/{package}/{package}"
        for entry in overlay[package]["files"]:
            path = base / entry["path"]
            if not path.is_file() or _sha(path) != entry["sha256"]:
                raise SourceVerificationError(f"{package} overlay mismatch: {entry['path']}")
    kernel = root / "runtime/kernels/builda_bmm_v0.py"
    if _sha(kernel) != overlay["build_a"]["sha256"]:
        raise SourceVerificationError("Build A v0 mismatch")
    if any(path.is_file() for path in (root / "runtime").rglob("*builda_bmm_v1*")):
        raise SourceVerificationError("Build A v1 is forbidden")


def verify_source_tree(root: Path, *, require_clean: bool = True, run_license_audit: bool = True) -> dict[str, Any]:
    root = root.resolve()
    if require_clean and not _clean(root):
        raise SourceVerificationError("source checkout is dirty")
    try:
        _verify_runtime_manifest(root)
        build = load_contract(root / "runtime/manifests/build-lock.json", "build-lock")
        o14 = json.loads((root / "runtime/manifests/o14-import.json").read_text())
        overlay = json.loads((root / "runtime/manifests/overlay-files.json").read_text())
        native = json.loads((root / "runtime/manifests/native-abi.json").read_text())
    except Exception as exc:
        raise SourceVerificationError(str(exc)) from exc
    dockerfile = (root / "docker/Dockerfile.champion").read_text()
    if build["base_image"] not in dockerfile:
        raise SourceVerificationError("Dockerfile base digest does not match build lock")
    for source in build["sources"]:
        if source["commit"] not in dockerfile or source["tree"] not in dockerfile:
            if source["name"] == "triton":
                continue  # locked as installed runtime input, not cloned by Dockerfile
            raise SourceVerificationError(f"Dockerfile omits locked source identity: {source['name']}")
    if o14.get("live_b4_reconciliation", {}).get("status") != "not-verified":
        raise SourceVerificationError("live-b4 parity must remain explicitly not-verified")
    if overlay.get("build_a", {}).get("active") != "v0" or overlay["build_a"].get("v1_included") is not False:
        raise SourceVerificationError("Build A effective selection drifted")
    if _sha(root / native["patch"]) != native["patch_sha256"]:
        raise SourceVerificationError("native patch hash drifted")
    for lock in ("models/checkpoint.lock.json", "models/sidecar.lock.json"):
        value = json.loads((root / lock).read_text())
        if value["distribution"]["method"] != "external-required":
            raise SourceVerificationError(f"{lock}: source release must not bundle model artifacts")
    for name in ("fast-320k", "balanced-650k", "large-1m"):
        expected = json.loads((root / f"tests/fixtures/{name}.rendered.json").read_text())
        if render(name, root / "tests/fixtures/cluster.env") != expected:
            raise SourceVerificationError(f"rendered preset snapshot drifted: {name}")
    if run_license_audit:
        result = subprocess.run([sys.executable, str(root / "scripts/audit_licenses.py"), "--strict"], cwd=root, capture_output=True, text=True)
        if result.returncode:
            raise SourceVerificationError("license/provenance/REUSE audit failed: " + result.stderr.strip())
    return {
        "schema_version": "1.0.0", "result": "PASS", "source_release_id": source_release_id(root),
        "live_b4_parity": "not-verified", "overlay_manifest_sha256": _sha(root / "runtime/manifests/overlay-files.json"),
        "build_lock_sha256": _sha(root / "runtime/manifests/build-lock.json"), "model_artifacts": "external-required",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    try:
        report = verify_source_tree(args.root, require_clean=not args.allow_dirty)
    except SourceVerificationError as exc:
        print(f"source-closure: FAIL: {exc}", file=sys.stderr)
        return 1
    rendered = (json.dumps(report, indent=2, sort_keys=True) + "\n").encode()
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_bytes(rendered)
    if args.receipt:
        if not args.report:
            parser.error("--receipt requires --report")
        receipt = construct_receipt({
            "schema_version": "1.0.0", "receipt_type": "source-closure", "command": "source-closure",
            "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "subject": {"source_release_id": report["source_release_id"]},
            "evidence": [
                {"path": args.report.name, "sha256": hashlib.sha256(rendered).hexdigest()},
                {"path": "runtime/manifests/build-lock.json", "sha256": report["build_lock_sha256"]},
                {"path": "runtime/manifests/overlay-files.json", "sha256": report["overlay_manifest_sha256"]},
            ],
            "result": "PASS", "metadata": {},
        })
        args.receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
