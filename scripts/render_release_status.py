#!/usr/bin/env python3
# Copyright 2026 dfi / 0xdfi
# SPDX-License-Identifier: Apache-2.0
"""Render the machine-readable publication status as deterministic Markdown."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from champion_runtime.contracts import ContractError, load_contract


def _cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def render_release_status(status: dict[str, Any]) -> str:
    lines = [
        "# Release status",
        "",
        "This file is generated from `publication/status.json`; edit the JSON only through the evidence gate tooling, then regenerate this page.",
        "",
        f"Machine status updated: `{status['updated_at']}`.",
        "",
        "A `pending` or `blocked` row is not a PASS. Historical measurements and source-build inputs do not promote any row.",
        "",
        "| Claim | Profile | Status | Evidence / reason |",
        "|---|---|---|---|",
    ]
    for claim in status["claims"]:
        profile = claim.get("profile", "—")
        if claim["status"] == "passed":
            detail = ", ".join(f"`{receipt}`" for receipt in claim["receipt_ids"])
        else:
            detail = claim.get("reason", "Trusted evidence has not promoted this claim.")
        lines.append(
            f"| {_cell(claim['claim'])} | {_cell(profile)} | {_cell(claim['status'])} | {_cell(detail)} |"
        )
    lines.extend([
        "",
        "See [Reproducibility](reproducibility.md) for the evidence ladder and [Licensing](licensing.md) for the source, model, and OCI boundaries.",
        "",
    ])
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--status", type=Path, default=ROOT / "publication/status.json")
    parser.add_argument("--output", type=Path, default=ROOT / "docs/release-status.md")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    try:
        payload = load_contract(args.status, "status")
        rendered = render_release_status(payload)
        if args.check:
            if not args.output.is_file() or args.output.read_text(encoding="utf-8") != rendered:
                print("release-status: FAIL: generated documentation drifted", file=sys.stderr)
                return 1
        else:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered, encoding="utf-8")
    except (OSError, json.JSONDecodeError, ContractError, KeyError, TypeError) as exc:
        print(f"release-status: FAIL: {exc}", file=sys.stderr)
        return 1
    print("release-status: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
