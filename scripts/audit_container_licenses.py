#!/usr/bin/env python3
# Copyright 2026 dfi / 0xdfi
# SPDX-License-Identifier: Apache-2.0
"""Fail-closed source/image/registry component license evaluator."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from champion_runtime.receipts import construct_receipt, source_release_id


class ComponentLicenseError(ValueError):
    """Raised when the requested distribution scope lacks license evidence."""


def audit_components(matrix: dict[str, Any], root: Path, *, scope: str, venue: str | None = None, sbom: dict[str, Any] | None = None) -> dict[str, Any]:
    components = matrix.get("components")
    if matrix.get("schema_version") != "1.0.0" or not isinstance(components, list):
        raise ComponentLicenseError("invalid component matrix")
    if scope == "source":
        checked, excluded = [], []
        for item in components:
            if item.get("distribution") == "external-required":
                excluded.append(item.get("name"))
                continue
            if item.get("distribution") != "source":
                continue
            if item.get("license") in {None, "unknown", "NOASSERTION"}:
                raise ComponentLicenseError(f"{item.get('name')}: source license is unknown/NOASSERTION")
            license_text = item.get("license_text")
            if not isinstance(license_text, str) or not (root / license_text).is_file():
                raise ComponentLicenseError(f"{item.get('name')}: governing license text is missing")
            if item.get("releasable") is not True:
                raise ComponentLicenseError(f"{item.get('name')}: source is not releasable")
            checked.append(item["name"])
        return {"result": "PASS", "scope": "source", "checked": checked, "excluded_external": excluded}
    if scope == "image":
        if sbom is None:
            raise ComponentLicenseError("exact image SBOM and matching build receipt are required")
        raise ComponentLicenseError("image license mapping is blocked until the exact built-image SBOM is reconciled")
    if scope == "registry" and venue == "ghcr":
        uncleared = [item.get("name") for item in components if item.get("distribution") in {"image", "external-required"} and item.get("redistribution", {}).get("ghcr") != "allowed"]
        if uncleared:
            raise ComponentLicenseError("GHCR publication is blocked: " + ", ".join(str(name) for name in uncleared))
        return {"result": "PASS", "scope": "registry", "venue": "ghcr"}
    raise ComponentLicenseError(f"unsupported scope/venue: {scope}/{venue}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, default=ROOT / "compliance/container-components.json")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--scope", choices=("source", "image", "registry"), required=True)
    parser.add_argument("--venue", choices=("ghcr",))
    parser.add_argument("--sbom", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    try:
        report = audit_components(json.loads(args.matrix.read_text()), root, scope=args.scope, venue=args.venue, sbom=json.loads(args.sbom.read_text()) if args.sbom else None)
    except (OSError, json.JSONDecodeError, ComponentLicenseError) as exc:
        print(f"component-license-audit: BLOCKED: {exc}", file=sys.stderr)
        return 1
    rendered = (json.dumps(report, indent=2, sort_keys=True) + "\n").encode()
    if args.report:
        args.report.write_bytes(rendered)
    if args.receipt:
        if args.scope != "source" or not args.report:
            parser.error("only --scope source can emit a source-only receipt, and --report is required")
        receipt = construct_receipt({
            "schema_version": "1.0.0", "receipt_type": "source-component-license", "command": "source-component-license",
            "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "subject": {"source_release_id": source_release_id(root)},
            "evidence": [{"path": args.report.name, "sha256": hashlib.sha256(rendered).hexdigest()}],
            "result": "PASS", "metadata": {},
        })
        args.receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
