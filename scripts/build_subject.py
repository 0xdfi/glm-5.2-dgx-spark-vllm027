#!/usr/bin/env python3
"""Compute the stable champion image build subject and optional build receipt."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from champion_runtime.receipts import (
    build_subject_id,
    construct_receipt,
    source_release_id,
)
from champion_runtime.contracts import load_contract


CONSUMED = [
    "docker/Dockerfile.champion",
    "runtime/requirements.lock",
    "runtime/overlays",
    "runtime/kernels/builda_bmm_v0.py",
    "runtime/patches/native/champion-sm121.patch",
    "runtime/manifests",
    "scripts/verify_image.py",
    "LICENSE",
    "LICENSES",
    "NOTICE",
    "CREDITS.md",
    "compliance/source-provenance.json",
]
IMAGE = re.compile(r"^sha256:[0-9a-f]{64}$")


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def identities(root: Path) -> dict[str, str]:
    return {
        "build_subject_id": build_subject_id(root, CONSUMED),
        "source_release_id": source_release_id(root),
        "build_lock_id": file_sha256(root / "runtime/manifests/build-lock.json"),
        "source_manifest_id": file_sha256(root / "runtime/manifests/overlay-files.json"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--image-digest")
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    values = identities(root)
    if args.receipt:
        if not args.image_digest or not IMAGE.fullmatch(args.image_digest):
            parser.error("--receipt requires --image-digest sha256:<64 lowercase hex>")
        body = {
            "schema_version": "1.0.0",
            "receipt_type": "build-produced",
            "command": "build-produced",
            "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "subject": values | {"image_digest": args.image_digest},
            "evidence": [
                {"path": "runtime/manifests/build-lock.json", "sha256": values["build_lock_id"]},
                {"path": "runtime/manifests/overlay-files.json", "sha256": values["source_manifest_id"]},
                {
                    "path": "compliance/source-provenance.json",
                    "sha256": file_sha256(root / "compliance/source-provenance.json"),
                },
            ],
            "result": "PROVISIONAL",
            "metadata": {},
        }
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        args.receipt.write_text(
            json.dumps(construct_receipt(body), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        load_contract(args.receipt, "receipt")
    print(json.dumps(values, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
