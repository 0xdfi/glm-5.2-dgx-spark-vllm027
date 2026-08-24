#!/usr/bin/env python3
# Copyright 2026 dfi / 0xdfi
# SPDX-License-Identifier: Apache-2.0
"""Pure offline differential-correctness evaluator.

Offline results are deliberately PROVISIONAL. Trusted CI may attest their
inputs later; this tool never edits publication status.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator


ROOT = Path(__file__).resolve().parents[1]
TOLERANCE = 0.05
SUBJECT_KEYS = (
    "image_digest", "build_lock_id", "model_id", "sidecar_id", "preset_id",
    "tokenizer_id", "sampling_id",
)


class CorrectnessError(ValueError):
    """Raised when candidate output violates the differential contract."""


def _validate(document: dict[str, Any]) -> None:
    schema = json.loads((ROOT / "correctness/teacher_forced_reference.schema.json").read_text())
    errors = sorted(Draft202012Validator(schema).iter_errors(document), key=lambda item: list(item.absolute_path))
    if errors:
        raise CorrectnessError("invalid correctness document: " + "; ".join(item.message for item in errors))
    for row in document["teacher_forced"]:
        if len(row["topk_token_ids"]) != len(row["topk_logprobs"]):
            raise CorrectnessError("top-k token and logprob lengths differ")


def _key(row: dict[str, Any]) -> tuple[str, int]:
    return row["prompt_id"], row["position"]


def compare_correctness(reference: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    """Compare two already-captured documents without any host access."""

    reference_subject = reference.get("subject") if isinstance(reference, dict) else None
    candidate_subject = candidate.get("subject") if isinstance(candidate, dict) else None
    if not isinstance(reference_subject, dict) or not isinstance(candidate_subject, dict) or any(
        reference_subject.get(key) != candidate_subject.get(key) for key in SUBJECT_KEYS
    ):
        raise CorrectnessError("subject mismatch between reference and candidate")
    _validate(reference)
    _validate(candidate)

    left = {_key(row): row for row in reference["teacher_forced"]}
    right = {_key(row): row for row in candidate["teacher_forced"]}
    if set(left) != set(right):
        raise CorrectnessError("teacher-forced positions do not match")
    swaps = 0
    for key, expected in left.items():
        actual = right[key]
        if actual["prefix"] != expected["prefix"]:
            raise CorrectnessError(f"teacher-forced prefix mismatch at {key}")
        if set(actual["topk_token_ids"]) != set(expected["topk_token_ids"]):
            raise CorrectnessError(f"top-k candidate set mismatch at {key}")
        expected_probs = dict(zip(expected["topk_token_ids"], expected["topk_logprobs"], strict=True))
        actual_probs = dict(zip(actual["topk_token_ids"], actual["topk_logprobs"], strict=True))
        if any(abs(actual_probs[token] - value) > TOLERANCE + 1e-12 for token, value in expected_probs.items()):
            raise CorrectnessError(f"logprob tolerance exceeded at {key}")
        if actual["topk_token_ids"][0] != expected["topk_token_ids"][0]:
            runner_up = expected["topk_token_ids"][1]
            gap = expected["topk_logprobs"][0] - expected["topk_logprobs"][1]
            if actual["topk_token_ids"][0] != runner_up or gap > TOLERANCE + 1e-12:
                raise CorrectnessError(f"top-1 swap is not the exact near-tie runner-up at {key}")
            swaps += 1

    free_left = {(row["prompt_id"], row["seed"]): row for row in reference["free_generation"]}
    free_right = {(row["prompt_id"], row["seed"]): row for row in candidate["free_generation"]}
    if set(free_left) != set(free_right):
        raise CorrectnessError("free-generation run identities differ")
    divergent = 0
    for key, expected in free_left.items():
        actual = free_right[key]
        prefix = 0
        for left_token, right_token in zip(expected["token_ids"], actual["token_ids"]):
            if left_token != right_token:
                break
            prefix += 1
        diverged = expected["token_ids"] != actual["token_ids"]
        divergent += int(diverged)
        if diverged and prefix == 0:
            raise CorrectnessError(f"free-generation diverged before a shared prefix at {key}")
        if not actual["semantic_valid"] or not actual["schema_valid"]:
            raise CorrectnessError(f"free-generation semantic/schema validation failed at {key}")
        if not diverged and actual["text"] != expected["text"]:
            raise CorrectnessError(f"free-generation text changed without token divergence at {key}")

    zero = Counter(row["output_sha256"] for row in candidate["temperature_zero"])
    modal_hash, modal_count = zero.most_common(1)[0] if zero else (None, 0)
    return {
        "result": "PROVISIONAL",
        "claim": "offline-differential-correctness",
        "subject": dict(candidate["subject"]),
        "teacher_forced": {"positions": len(left), "allowed_top1_swaps": swaps},
        "free_generation": {"runs": len(free_right), "semantic_only_after_divergence": divergent},
        "temperature_zero": {"runs": len(candidate["temperature_zero"]), "modal_output_sha256": modal_hash, "modal_count": modal_count, "threshold": None},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = compare_correctness(json.loads(args.reference.read_text()), json.loads(args.candidate.read_text()))
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered)
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
