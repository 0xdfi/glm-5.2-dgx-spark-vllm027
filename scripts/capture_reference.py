#!/usr/bin/env python3
# Copyright 2026 dfi / 0xdfi
# SPDX-License-Identifier: Apache-2.0
"""Fail-closed live-reference capture client (never invoked by static CI)."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


REQUIRED_SUBJECT = {
    "image_digest", "build_lock_id", "model_id", "sidecar_id", "preset_id",
    "tokenizer_id", "sampling_id",
}


class CaptureError(RuntimeError):
    """Raised before or during an unsafe/incomplete reference capture."""


def capture_plan(prompt_suite: dict[str, Any]) -> list[dict[str, Any]]:
    """Expand the fixed prompt suite into the exact retained capture matrix."""

    prompts = prompt_suite.get("prompts")
    if prompt_suite.get("schema_version") != "1.0.0" or not isinstance(prompts, list):
        raise CaptureError("invalid prompt suite")
    plan: list[dict[str, Any]] = []
    for prompt in prompts:
        sampling = prompt.get("sampling", {})
        prompt_id = prompt.get("id")
        if sampling.get("teacher_forced") is not True:
            raise CaptureError(f"{prompt_id}: teacher-forced capture is required")
        seeds = sampling.get("seeds")
        if sampling.get("free_runs") != 5 or not isinstance(seeds, list) or len(seeds) != 5 or len(set(seeds)) != 5:
            raise CaptureError(f"{prompt_id}: exactly five distinct seeded free runs are required")
        if sampling.get("temperature_zero_repeats") != 10:
            raise CaptureError(f"{prompt_id}: exactly ten temperature-zero repeats are required")
        plan.append({"mode": "teacher-forced", "prompt_id": prompt_id})
        plan.extend({"mode": "free-generation", "prompt_id": prompt_id, "seed": seed} for seed in seeds)
        plan.extend({"mode": "temperature-zero", "prompt_id": prompt_id, "run": run} for run in range(1, 11))
    return plan


def validate_capture_preconditions(
    output: Path,
    subject: dict[str, Any],
    *,
    idle: bool,
    complete_identity: bool,
    counters_isolated: bool,
) -> None:
    if output.exists():
        raise CaptureError(f"refusing to overwrite reference output: {output}")
    if set(subject) != REQUIRED_SUBJECT or not complete_identity:
        raise CaptureError("complete identity is required for image/build/model/sidecar/preset/tokenizer/sampling")
    if not idle:
        raise CaptureError("idle server evidence is required")
    if not counters_isolated:
        raise CaptureError("counter isolation is required")


def _request(base: str, path: str, payload: dict[str, Any] | None = None) -> Any:
    url = urllib.parse.urljoin(base.rstrip("/") + "/", path.lstrip("/"))
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=300) as response:
        body = response.read()
    return body.decode() if path in {"/health", "/metrics"} else json.loads(body)


def atomic_publish(output: Path, files: dict[str, bytes]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        for name, content in files.items():
            target = temporary / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
        os.replace(temporary, output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-base", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--subject", type=Path, required=True)
    parser.add_argument("--complete-identity", action="store_true")
    parser.add_argument("--idle-confirmed", action="store_true")
    parser.add_argument("--counters-isolated", action="store_true")
    args = parser.parse_args()
    subject = json.loads(args.subject.read_text())
    validate_capture_preconditions(
        args.output, subject, idle=args.idle_confirmed,
        complete_identity=args.complete_identity, counters_isolated=args.counters_isolated,
    )
    # The capture path intentionally requires both exact tokenization and live
    # health before any artifact is published. Detailed vectors are retained in
    # a temporary directory until every request and counter reconciliation ends.
    health = _request(args.api_base, "/health")
    metrics_before = _request(args.api_base, "/metrics")
    tokenize = _request(args.api_base, "/tokenize", {"messages": [{"role": "user", "content": "identity probe"}], "add_special_tokens": True})
    if not health or not isinstance(tokenize, dict) or not (isinstance(tokenize.get("count"), int) or isinstance(tokenize.get("tokens"), list)):
        raise CaptureError("health/tokenizer precondition failed")
    metrics_after = _request(args.api_base, "/metrics")
    if metrics_before != metrics_after:
        raise CaptureError("counter isolation failed during capture preflight")
    manifest = {"schema_version": "1.0.0", "subject": subject, "status": "capture-client-ready", "note": "no inference vectors were requested by this preflight-only invocation"}
    atomic_publish(args.output, {"manifest.json": (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
