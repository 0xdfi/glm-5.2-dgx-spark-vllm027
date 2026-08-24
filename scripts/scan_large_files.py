#!/usr/bin/env python3
# Copyright 2026 dfi / 0xdfi
# SPDX-License-Identifier: Apache-2.0
"""Scan the Git index and reachable release history for large/binary blobs."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from champion_runtime.receipts import construct_receipt, source_release_id


MAX_BYTES = 10 * 1024 * 1024
PROHIBITED = (
    ".safetensors", ".bin", ".pt", ".pth", ".whl", ".so", ".tar",
    ".tar.gz", ".tgz", ".zip", ".ckpt", ".gguf", ".onnx", ".dylib",
)


class LargeFileError(RuntimeError):
    """Raised when Git object enumeration cannot be completed."""


def _git(root: Path, *args: str, input_bytes: bytes | None = None, check: bool = True) -> bytes:
    result = subprocess.run(["git", "-C", str(root), *args], input=input_bytes, capture_output=True)
    if check and result.returncode:
        raise LargeFileError(result.stderr.decode(errors="replace").strip() or f"git {' '.join(args)} failed")
    return result.stdout


def _objects(root: Path, revision: str) -> list[tuple[str, str, str]]:
    candidates: list[tuple[str, str]] = []
    if subprocess.run(["git", "-C", str(root), "rev-parse", "--verify", revision], capture_output=True).returncode == 0:
        for raw in _git(root, "rev-list", "--objects", revision).decode(errors="surrogateescape").splitlines():
            oid, _, path = raw.partition(" ")
            if path:
                candidates.append((oid, path))
    for entry in _git(root, "ls-files", "--stage", "-z").split(b"\0"):
        if not entry:
            continue
        metadata, _, raw_path = entry.partition(b"\t")
        parts = metadata.split()
        if len(parts) == 3 and parts[2] == b"0":
            candidates.append((parts[1].decode(), raw_path.decode(errors="surrogateescape")))
    if not candidates:
        return []
    unique_oids = sorted({oid for oid, _ in candidates})
    query = b"".join((oid + "\n").encode() for oid in unique_oids)
    output = _git(root, "cat-file", "--batch-check=%(objectname) %(objecttype) %(objectsize)", input_bytes=query)
    metadata: dict[str, tuple[str, str]] = {}
    for line in output.decode().splitlines():
        oid, kind, size = line.split()
        metadata[oid] = (kind, size)
    return [(oid, path, metadata[oid][1]) for oid, path in candidates if metadata.get(oid, (None,))[0] == "blob"]


def scan_repository(root: Path, revision: str = "HEAD") -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for oid, path, size_text in _objects(root.resolve(), revision):
        size = int(size_text)
        rules = []
        if size > MAX_BYTES:
            rules.append("blob-over-10mib")
        if path.casefold().endswith(PROHIBITED):
            rules.append("prohibited-suffix")
        for rule in rules:
            key = (oid, path, rule)
            if key not in seen:
                findings.append({"rule": rule, "object": oid, "path": path, "size": size})
                seen.add(key)
    return sorted(findings, key=lambda item: (item["path"], item["rule"], item["object"]))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--revision", default="HEAD")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    findings = scan_repository(root, args.revision)
    report = {"schema_version": "1.0.0", "scope": "git-index-and-reachable-history", "max_bytes": MAX_BYTES, "result": "FAIL" if findings else "PASS", "findings": findings}
    rendered = (json.dumps(report, indent=2, sort_keys=True) + "\n").encode()
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_bytes(rendered)
    if args.receipt:
        if not args.report:
            parser.error("--receipt requires --report")
        evidence_path = args.report.name
        receipt = construct_receipt({
            "schema_version": "1.0.0", "receipt_type": "large-file-scan", "command": "large-file-scan",
            "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "subject": {"source_release_id": source_release_id(root)},
            "evidence": [{"path": evidence_path, "sha256": hashlib.sha256(rendered).hexdigest()}],
            "result": report["result"], "metadata": {},
        })
        args.receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return int(bool(findings))


if __name__ == "__main__":
    raise SystemExit(main())
