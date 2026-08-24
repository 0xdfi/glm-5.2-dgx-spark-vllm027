#!/usr/bin/env python3
# Copyright 2026 dfi / 0xdfi
# SPDX-License-Identifier: Apache-2.0
"""Pure parser/evaluator for captured four-rank runtime evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator


ROOT = Path(__file__).resolve().parents[1]
KNOWN_PRE_READY_WARNINGS = {"known-duplicate-registration"}


class RuntimeEvidenceError(ValueError):
    """Raised when captured runtime evidence cannot support the contract."""


def verify_runtime_evidence(document: dict[str, Any], subject: dict[str, str], expected_engine: dict[str, Any]) -> dict[str, Any]:
    schema = json.loads((ROOT / "schemas/runtime-evidence.schema.json").read_text())
    errors = list(Draft202012Validator(schema).iter_errors(document))
    if errors:
        detail = "; ".join(item.message for item in errors)
        labels = []
        if len(document.get("ranks", [])) != 4:
            labels.append("exact four ranks")
        if document.get("engine", {}).get("graph_capture_sizes") != [6, 12, 18, 24]:
            labels.append("graph capture census")
        if document.get("transport", {}).get("socket_fallback") is not False:
            labels.append("Socket fallback")
        if any(row.get("restart_count") or row.get("oom_kills") for row in document.get("ranks", [])):
            labels.append("restart/OOM state")
        raise RuntimeEvidenceError((", ".join(labels) + ": " if labels else "") + detail)
    if document["subject"] != subject:
        raise RuntimeEvidenceError("runtime subject mismatch")
    if [row["rank"] for row in document["ranks"]] != [0, 1, 2, 3]:
        raise RuntimeEvidenceError("runtime requires exact four ranks ordered 0..3")
    if any(row["image_digest"] != subject["image_digest"] for row in document["ranks"]):
        raise RuntimeEvidenceError("rank image identity mismatch")
    if any(row["restart_count"] != 0 for row in document["ranks"]):
        raise RuntimeEvidenceError("runtime restart count is nonzero")
    if any(row["oom_kills"] != 0 for row in document["ranks"]):
        raise RuntimeEvidenceError("runtime OOM kill count is nonzero")
    for name, value in expected_engine.items():
        if document["engine"].get(name) != value:
            raise RuntimeEvidenceError(f"engine {name} mismatch")
    if document["first_fire"]["window_id"] != document["window_id"]:
        raise RuntimeEvidenceError("first-fire evidence is not from the current window")
    for event in document["log_events"]:
        allowed = event["phase"] == "pre-readiness" and event["severity"] == "warning" and event["code"] in KNOWN_PRE_READY_WARNINGS
        if not allowed:
            raise RuntimeEvidenceError(f"disallowed log event: {event['phase']}/{event['severity']}/{event['code']}")
    return {"result": "PROVISIONAL", "claim": "offline-runtime-verification", "subject": dict(subject), "ranks": 4}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--subject", type=Path, required=True)
    parser.add_argument("--preset", type=Path, required=True)
    args = parser.parse_args()
    preset = json.loads(args.preset.read_text())
    engine = preset["effective_engine"]
    expected = {name: engine[name] for name in ("max_model_len", "kv_cache_dtype", "decode_context_parallel_size")}
    expected["adaptive_mtp"] = engine["adaptive_mtp"]["k"]
    result = verify_runtime_evidence(json.loads(args.evidence.read_text()), json.loads(args.subject.read_text()), expected)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
