#!/usr/bin/env python3
# Copyright 2026 dfi / 0xdfi
# SPDX-License-Identifier: Apache-2.0
"""Pure evaluator for first-three-valid-in-five benchmark attempts."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any


INVALIDATIONS = {
    "idle-preflight", "foreign-traffic", "unexpected-api-connection",
    "runtime-health-failure", "gpu-clock-cap-changed", "client-failure-before-acceptance",
}


class BatteryError(ValueError):
    """Raised when attempts are invalid, inconsistent, or fail thresholds."""


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evaluate_attempts(attempts: list[dict[str, Any]], baselines: dict[str, Any], *, expected_baseline_hash: str) -> dict[str, Any]:
    policy = baselines["threshold_policy"]
    maximum = policy["max_attempts"]
    required = policy["required_valid_batteries"]
    if not attempts or len(attempts) > maximum:
        raise BatteryError(f"battery requires one through {maximum} attempts")
    ordered = sorted(attempts, key=lambda item: (item.get("created_at", ""), item.get("attempt", 0)))
    numbers = [item.get("attempt") for item in ordered]
    if numbers != list(range(1, len(ordered) + 1)):
        raise BatteryError("attempts must be complete and chronological from one")
    baseline_hashes = {item.get("baselines_sha256") for item in ordered}
    if baseline_hashes != {expected_baseline_hash}:
        raise BatteryError("baselines hash mismatch")
    subjects = [item.get("subject") for item in ordered]
    if any(subject != subjects[0] for subject in subjects[1:]):
        raise BatteryError("attempt subject mismatch")
    profile = ordered[0].get("profile")
    if any(item.get("profile") != profile for item in ordered):
        raise BatteryError("profile mismatch")
    if profile == "fast-320k" and not subjects[0].get("pin_id"):
        raise BatteryError("Fast battery requires the authenticated pin identity")
    raw_hashes = []
    valid = []
    for item in ordered:
        raw = item.get("raw_sha256")
        if not isinstance(raw, str) or len(raw) != 64:
            raise BatteryError("raw attempt hash is required")
        raw_hashes.append(raw)
        if item.get("valid") is True:
            if item.get("invalidation") is not None:
                raise BatteryError("valid attempt cannot have an invalidation")
            if item.get("completed") is not True:
                raise BatteryError("valid attempt must be complete")
            valid.append(item)
        elif item.get("invalidation") not in INVALIDATIONS:
            raise BatteryError("unknown invalidation class")
    selected = valid[:required]
    if len(selected) < required:
        if len(ordered) == maximum:
            raise BatteryError("fewer than three valid batteries after five attempts")
        return {"result": "PROVISIONAL", "complete": False, "valid_attempts": len(selected), "raw_sha256": raw_hashes, "subject": subjects[0]}
    metric_policy = baselines["profiles"][profile]["metrics"]
    medians: dict[str, float] = {}
    for metric, limits in metric_policy.items():
        values = []
        for item in selected:
            value = item.get("metrics", {}).get(metric)
            if not isinstance(value, (int, float)):
                raise BatteryError(f"missing numeric metric {metric}")
            if value < limits["individual_floor"]:
                raise BatteryError(f"individual threshold failed for {metric}")
            values.append(float(value))
        median = statistics.median(values)
        if median < limits["median_floor"]:
            raise BatteryError(f"median threshold failed for {metric}")
        medians[metric] = median
    return {
        "result": "PROVISIONAL", "claim": "battery-suite", "complete": True,
        "profile": profile, "subject": subjects[0], "baselines_sha256": expected_baseline_hash,
        "selected_attempts": [item["attempt"] for item in selected], "raw_sha256": raw_hashes,
        "medians": medians,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt-dir", type=Path, required=True)
    parser.add_argument("--baselines", type=Path, default=Path(__file__).resolve().parents[1] / "benchmarks/baselines.json")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    attempts = [json.loads(path.read_text()) for path in sorted(args.attempt_dir.glob("*.json"))]
    result = evaluate_attempts(attempts, json.loads(args.baselines.read_text()), expected_baseline_hash=_sha(args.baselines))
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered)
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
