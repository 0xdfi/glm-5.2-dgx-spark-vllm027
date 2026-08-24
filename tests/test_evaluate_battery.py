# Copyright 2026 dfi / 0xdfi
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from copy import deepcopy

import pytest

from scripts.evaluate_battery import BatteryError, evaluate_attempts


H = "a" * 64
SUBJECT = {
    "source_release_id": H,
    "build_subject_id": "b" * 64,
    "build_lock_id": "c" * 64,
    "source_manifest_id": "d" * 64,
    "image_digest": "sha256:" + "e" * 64,
    "model_id": "f" * 64,
    "sidecar_id": "1" * 64,
    "preset_id": "2" * 64,
    "pin_id": "3" * 64,
}
BASELINE_HASH = "4" * 64


def attempt(number, *, valid=True, value=100.0, invalidation=None, pin=True):
    return {
        "schema_version": "1.0.0",
        "attempt": number,
        "created_at": f"2026-08-23T12:0{number}:00Z",
        "profile": "fast-320k",
        "subject": SUBJECT if pin else {k: v for k, v in SUBJECT.items() if k != "pin_id"},
        "baselines_sha256": BASELINE_HASH,
        "raw_sha256": str(number) * 64,
        "valid": valid,
        "invalidation": invalidation,
        "metrics": {name: value for name in ("peak_c1", "peak_c4", "prefill", "prose_c1", "prose_c4")},
        "completed": True,
    }


BASELINES = {
    "threshold_policy": {"max_attempts": 5, "required_valid_batteries": 3, "individual_fraction": .9, "median_fraction": .95},
    "profiles": {"fast-320k": {"metrics": {name: {"reference": 100.0, "individual_floor": 90.0, "median_floor": 95.0} for name in ("peak_c1", "peak_c4", "prefill", "prose_c1", "prose_c4")}}},
}


def test_first_three_valid_chronological_attempts_pass_provisionally():
    attempts = [attempt(1, valid=False, invalidation="idle-preflight"), attempt(2), attempt(3), attempt(4), attempt(5, value=1)]
    result = evaluate_attempts(attempts, BASELINES, expected_baseline_hash=BASELINE_HASH)
    assert result["result"] == "PROVISIONAL"
    assert result["selected_attempts"] == [2, 3, 4]


def test_battery_uses_first_three_without_cherry_picking_and_rejects_drift_and_slow_runs():
    attempts = [attempt(1), attempt(2), attempt(3), attempt(4)]
    assert evaluate_attempts(attempts, BASELINES, expected_baseline_hash=BASELINE_HASH)["selected_attempts"] == [1, 2, 3]
    drift = [attempt(1), attempt(2), attempt(3)]
    drift[2]["subject"] = SUBJECT | {"preset_id": "9" * 64}
    with pytest.raises(BatteryError, match="subject"):
        evaluate_attempts(drift, BASELINES, expected_baseline_hash=BASELINE_HASH)
    slow = [attempt(1), attempt(2), attempt(3, value=89.99)]
    with pytest.raises(BatteryError, match="individual"):
        evaluate_attempts(slow, BASELINES, expected_baseline_hash=BASELINE_HASH)


def test_less_than_three_is_incomplete_until_fifth_then_failure():
    early = evaluate_attempts([attempt(1), attempt(2)], BASELINES, expected_baseline_hash=BASELINE_HASH)
    assert early["result"] == "PROVISIONAL" and early["complete"] is False
    exhausted = [attempt(i, valid=i < 3, invalidation=None if i < 3 else "foreign-traffic") for i in range(1, 6)]
    with pytest.raises(BatteryError, match="five attempts"):
        evaluate_attempts(exhausted, BASELINES, expected_baseline_hash=BASELINE_HASH)


def test_only_six_invalidation_classes_and_fast_pin_are_accepted():
    with pytest.raises(BatteryError, match="invalidation"):
        evaluate_attempts([attempt(1, valid=False, invalidation="timeout")], BASELINES, expected_baseline_hash=BASELINE_HASH)
    with pytest.raises(BatteryError, match="pin"):
        evaluate_attempts([attempt(1, pin=False)], BASELINES, expected_baseline_hash=BASELINE_HASH)
