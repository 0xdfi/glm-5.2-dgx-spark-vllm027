# Copyright 2026 dfi / 0xdfi
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from scripts.capture_reference import CaptureError, capture_plan, validate_capture_preconditions
from scripts.verify_correctness import CorrectnessError, compare_correctness


ROOT = Path(__file__).resolve().parents[1]
HEX = "a" * 64
SUBJECT = {
    "image_digest": "sha256:" + HEX,
    "build_lock_id": "b" * 64,
    "model_id": "c" * 64,
    "sidecar_id": "d" * 64,
    "preset_id": "e" * 64,
    "tokenizer_id": "f" * 64,
    "sampling_id": "1" * 64,
}


def _record(*, top=(10, 11), probs=(-0.1, -0.13), text="ok", schema_valid=True):
    return {
        "schema_version": "1.0.0",
        "subject": dict(SUBJECT),
        "teacher_forced": [
            {"prompt_id": "prose", "position": 3, "prefix": [1, 2, 3], "topk_token_ids": list(top), "topk_logprobs": list(probs)}
        ],
        "free_generation": [
            {"prompt_id": "prose", "seed": 7, "token_ids": [5, 6], "text": text, "semantic_valid": True, "schema_valid": schema_valid}
        ],
        "temperature_zero": [
            {"prompt_id": "prose", "run": index, "output_sha256": HEX}
            for index in range(1, 11)
        ],
    }


def test_prompt_suite_is_fixed_and_covers_five_behaviors():
    payload = json.loads((ROOT / "correctness/prompts.json").read_text())
    assert [item["id"] for item in payload["prompts"]] == [
        "prose", "code", "reasoning", "json", "tool-call"
    ]
    assert {item["kind"] for item in payload["prompts"]} == {
        "prose", "code", "reasoning", "json", "tool-call"
    }
    assert all(item["sampling"]["free_runs"] == 5 for item in payload["prompts"])
    assert all(item["sampling"]["temperature_zero_repeats"] == 10 for item in payload["prompts"])
    plan = capture_plan(payload)
    assert len([item for item in plan if item["mode"] == "teacher-forced"]) == 5
    assert len([item for item in plan if item["mode"] == "free-generation"]) == 25
    assert len([item for item in plan if item["mode"] == "temperature-zero"]) == 50


def test_reference_schema_is_strict_and_subject_complete():
    schema = json.loads((ROOT / "correctness/teacher_forced_reference.schema.json").read_text())
    valid = _record()
    assert not list(Draft202012Validator(schema).iter_errors(valid))
    invalid = valid | {"unexpected": True}
    assert list(Draft202012Validator(schema).iter_errors(invalid))
    missing = _record()
    del missing["subject"]["sidecar_id"]
    assert list(Draft202012Validator(schema).iter_errors(missing))


def test_subject_mismatch_is_reported_before_numeric_comparison():
    candidate = _record(probs=(-9.0, -9.0))
    candidate["subject"] = SUBJECT | {"preset_id": "2" * 64}
    with pytest.raises(CorrectnessError, match="subject mismatch"):
        compare_correctness(_record(), candidate)


def test_teacher_forced_tolerance_and_exact_runner_up_near_tie():
    assert compare_correctness(_record(), _record(probs=(-0.14, -0.16)))["result"] == "PROVISIONAL"
    near_tie = _record(top=(11, 10), probs=(-0.11, -0.14))
    assert compare_correctness(_record(), near_tie)["teacher_forced"]["allowed_top1_swaps"] == 1
    with pytest.raises(CorrectnessError, match="runner-up"):
        compare_correctness(_record(probs=(-0.1, -0.3)), _record(top=(11, 10), probs=(-0.25, -0.1)))
    with pytest.raises(CorrectnessError, match="logprob"):
        compare_correctness(_record(), _record(probs=(-0.151, -0.13)))


def test_after_allowed_divergence_only_semantic_and_schema_are_compared():
    candidate = _record(text="different but valid")
    candidate["free_generation"][0]["token_ids"] = [5, 99, 100]
    assert compare_correctness(_record(), candidate)["free_generation"]["semantic_only_after_divergence"] == 1
    candidate["free_generation"][0]["schema_valid"] = False
    with pytest.raises(CorrectnessError, match="semantic/schema"):
        compare_correctness(_record(), candidate)


def test_capture_preconditions_fail_closed_without_touching_output(tmp_path: Path):
    out = tmp_path / "reference"
    out.mkdir()
    marker = out / "preserve"
    marker.write_text("keep")
    with pytest.raises(CaptureError, match="refusing to overwrite"):
        validate_capture_preconditions(out, SUBJECT, idle=True, complete_identity=True, counters_isolated=True)
    assert marker.read_text() == "keep"
    for changed, message in ((False, "complete identity"), (None, "idle"), ("counter", "counter")):
        kwargs = {"idle": True, "complete_identity": True, "counters_isolated": True}
        if changed is False:
            kwargs["complete_identity"] = False
        elif changed is None:
            kwargs["idle"] = False
        else:
            kwargs["counters_isolated"] = False
        with pytest.raises(CaptureError, match=message):
            validate_capture_preconditions(tmp_path / "new", SUBJECT, **kwargs)
