#!/usr/bin/env python3
# Copyright 2026 dfi / 0xdfi
# SPDX-License-Identifier: Apache-2.0
"""Compose exactly five source leaf receipts into a provisional candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from champion_runtime.receipts import ReceiptError, construct_receipt, verify_receipt


REQUIRED_TYPES = (
    "source-closure", "provenance-license", "source-component-license",
    "source-secret-scan", "large-file-scan",
)


class PublicationSourceError(ValueError):
    """Raised when the source receipt bundle is incomplete or unsafe."""


def _contained(root: Path, path: Path) -> str:
    if path.is_symlink():
        raise PublicationSourceError(f"receipt path must be a regular non-symlink file: {path}")
    try:
        relative = path.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise PublicationSourceError(f"receipt path is not contained in evidence root: {path}") from exc
    if not path.is_file() or relative.is_absolute() or ".." in relative.parts:
        raise PublicationSourceError(f"unsafe receipt path: {path}")
    return relative.as_posix()


def compose_source_candidate(paths: list[Path], evidence_root: Path) -> dict[str, Any]:
    if len(paths) != len(REQUIRED_TYPES):
        raise PublicationSourceError("expected exactly one receipt for each of five source gates")
    by_type: dict[str, tuple[Path, dict[str, Any]]] = {}
    for path in paths:
        relative = _contained(evidence_root, path)
        try:
            receipt = verify_receipt(json.loads(path.read_text()))
        except (OSError, json.JSONDecodeError, ReceiptError) as exc:
            raise PublicationSourceError(f"receipt_id validation failed for {relative}: {exc}") from exc
        kind = receipt["receipt_type"]
        if kind not in REQUIRED_TYPES or kind in by_type:
            raise PublicationSourceError(f"expected exactly one {kind} receipt")
        if receipt["result"] != "PASS":
            raise PublicationSourceError(f"{kind} did not pass")
        for item in receipt["evidence"]:
            raw = PurePosixPath(item["path"])
            if raw.is_absolute() or ".." in raw.parts:
                raise PublicationSourceError(f"unsafe evidence path in {kind}")
            evidence = evidence_root.joinpath(*raw.parts)
            if not evidence.is_file() or evidence.is_symlink():
                # Source-manifest evidence may intentionally name a repository
                # path outside the detached evidence bundle; those are checked
                # by the leaf producer. Detached report evidence must exist.
                if raw.parts and raw.parts[0] not in {"runtime", "compliance", "LICENSES"}:
                    raise PublicationSourceError(f"missing contained evidence for {kind}: {raw}")
                continue
            if hashlib.sha256(evidence.read_bytes()).hexdigest() != item["sha256"]:
                raise PublicationSourceError(f"evidence hash mismatch for {kind}: {raw}")
        by_type[kind] = (path, receipt)
    if set(by_type) != set(REQUIRED_TYPES):
        raise PublicationSourceError("expected exactly one receipt of each required type")
    sources = {receipt["subject"].get("source_release_id") for _, receipt in by_type.values()}
    if len(sources) != 1 or None in sources:
        raise PublicationSourceError("source_release_id mismatch across receipts")
    evidence = []
    for kind in REQUIRED_TYPES:
        path, _ = by_type[kind]
        evidence.append({"path": _contained(evidence_root, path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    return construct_receipt({
        "schema_version": "1.0.0", "receipt_type": "source-complete-candidate", "command": "source-complete-candidate",
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "subject": {"source_release_id": sources.pop()}, "evidence": evidence,
        "result": "PROVISIONAL", "metadata": {"claim": "requires-trusted-ci-attestation"},
    })


def _atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--receipt", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = compose_source_candidate(args.receipt, args.evidence_root)
    except PublicationSourceError as exc:
        print(f"source-publication-candidate: FAIL: {exc}", file=sys.stderr)
        return 1
    _atomic(args.output, result)
    print("source-publication-candidate: PROVISIONAL")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
