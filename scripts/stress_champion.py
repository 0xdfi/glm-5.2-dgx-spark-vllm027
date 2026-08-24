#!/usr/bin/env python3
# Copyright 2026 dfi / 0xdfi
# SPDX-License-Identifier: Apache-2.0
"""Profile-aware stress plan and opt-in HTTP executor."""

from __future__ import annotations

import argparse
import json
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


class StressError(ValueError):
    """Raised when a stress execution is incomplete or contaminated."""


def stress_plan(profile: str) -> list[dict[str, Any]]:
    contexts = {
        "fast-320k": [("immediate", 0), ("c4", 0), ("context-8k", 8192), ("context-65k", 65536), ("context-240k", 240000), ("context-capacity", 299000)],
        "balanced-650k": [("immediate", 0), ("c4", 0), ("context-capacity", 600000)],
        "large-1m": [("immediate", 0), ("c4", 0), ("context-capacity", 900000)],
    }
    if profile not in contexts:
        raise StressError(f"unknown profile: {profile}")
    plan = [{"name": name, "context_tokens": tokens, "idle_before_seconds": 0} for name, tokens in contexts[profile]]
    plan.append({"name": "post-idle", "context_tokens": 0, "idle_before_seconds": 300})
    return plan


def validate_executor_contract(report: dict[str, Any]) -> dict[str, Any]:
    if report.get("tokenizer_endpoint") != "/tokenize":
        raise StressError("exact /tokenize endpoint is required")
    if report.get("post_idle_generation") is not True:
        raise StressError("post-idle generation is required")
    for leg in report.get("legs", []):
        if not isinstance(leg.get("token_count"), int) or leg["token_count"] < 0:
            raise StressError("every leg requires an exact token count")
        if not isinstance(leg.get("output"), str) or not leg["output"]:
            raise StressError("every leg requires a nonempty output")
        if leg.get("health") is not True or leg.get("idle") is not True:
            raise StressError("every leg requires post-health and post-idle")
    return {"result": "PROVISIONAL", "claim": "offline-stress-validation", "legs": len(report.get("legs", []))}


def _post(base: str, path: str, payload: dict[str, Any]) -> dict[str, Any]:
    url = urllib.parse.urljoin(base.rstrip("/") + "/", path.lstrip("/"))
    request = urllib.request.Request(url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=1800) as response:
        return json.loads(response.read())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", choices=("fast-320k", "balanced-650k", "large-1m"))
    parser.add_argument("--validate-report", type=Path)
    args = parser.parse_args()
    if args.plan:
        print(json.dumps(stress_plan(args.plan), indent=2, sort_keys=True))
        return 0
    if args.validate_report:
        print(json.dumps(validate_executor_contract(json.loads(args.validate_report.read_text())), indent=2, sort_keys=True))
        return 0
    parser.error("choose --plan or --validate-report; live execution requires an explicit future command")


if __name__ == "__main__":
    raise SystemExit(main())
