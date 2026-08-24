#!/usr/bin/env python3
"""Fail closed unless checkpoint and sidecar rights are independently verified."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from champion_runtime.receipts import construct_receipt


VERIFIED = {"verified-redistributable", "verified-external-use"}


class LicenseVerificationError(ValueError):
    pass


def load_lock(path: Path, expected_type: str) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    payload = json.loads(raw)
    if payload.get("artifact_type") != expected_type:
        raise LicenseVerificationError(f"{path}: expected {expected_type} lock")
    rights = payload.get("rights", {})
    if rights.get("status") not in VERIFIED:
        raise LicenseVerificationError(f"{expected_type} has unknown rights")
    if not rights.get("license") or not rights.get("evidence"):
        raise LicenseVerificationError(f"{expected_type} verified rights require cited evidence")
    return payload, hashlib.sha256(raw).hexdigest()


def write_receipt(path: Path, artifact_type: str, lock: dict[str, Any], lock_id: str) -> None:
    subject_key = "model_id" if artifact_type == "checkpoint" else "sidecar_id"
    receipt = construct_receipt(
        {
            "schema_version": "1.0.0",
            "receipt_type": f"{artifact_type}-license-candidate",
            "command": "verify-model-license",
            "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "subject": {subject_key: lock_id},
            "evidence": [{"path": f"models/{artifact_type}.lock.json", "sha256": lock_id}],
            "result": "PROVISIONAL",
            "metadata": {
                "workflow": f"local-unattested-{artifact_type}-rights-check",
            },
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(receipt, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, 0o644)
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-receipt", type=Path, required=True)
    parser.add_argument("--sidecar", type=Path, required=True)
    parser.add_argument("--sidecar-receipt", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        checkpoint, checkpoint_id = load_lock(args.checkpoint, "checkpoint")
        sidecar, sidecar_id = load_lock(args.sidecar, "sidecar")
        write_receipt(args.checkpoint_receipt, "checkpoint", checkpoint, checkpoint_id)
        write_receipt(args.sidecar_receipt, "sidecar", sidecar, sidecar_id)
        print("model-license: PROVISIONAL (local evidence is not authenticated)")
        return 0
    except (OSError, json.JSONDecodeError, LicenseVerificationError) as exc:
        print(f"model-license: BLOCKED: {exc}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
