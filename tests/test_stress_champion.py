# Copyright 2026 dfi / 0xdfi
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest

from scripts.stress_champion import StressError, stress_plan, validate_executor_contract


def test_stress_plans_have_exact_context_legs():
    assert [leg["context_tokens"] for leg in stress_plan("fast-320k")] == [0, 0, 8192, 65536, 240000, 299000, 0]
    assert [leg["context_tokens"] for leg in stress_plan("balanced-650k")] == [0, 0, 600000, 0]
    assert [leg["context_tokens"] for leg in stress_plan("large-1m")] == [0, 0, 900000, 0]
    for profile in ("fast-320k", "balanced-650k", "large-1m"):
        plan = stress_plan(profile)
        assert plan[0]["name"] == "immediate"
        assert plan[1]["name"] == "c4"
        assert plan[-1] == {"name": "post-idle", "context_tokens": 0, "idle_before_seconds": 300}


def test_http_executor_requires_tokenize_health_idle_and_nonempty_outputs():
    valid = {
        "tokenizer_endpoint": "/tokenize",
        "legs": [{"name": "x", "token_count": 10, "output": "ok", "health": True, "idle": True}],
        "post_idle_generation": True,
    }
    assert validate_executor_contract(valid)["result"] == "PROVISIONAL"
    for key, value, match in (
        ("tokenizer_endpoint", "/approx", "/tokenize"),
        ("post_idle_generation", False, "post-idle"),
    ):
        bad = valid | {key: value}
        with pytest.raises(StressError, match=match):
            validate_executor_contract(bad)
    bad = valid | {"legs": [{"name": "x", "token_count": 10, "output": "", "health": True, "idle": True}]}
    with pytest.raises(StressError, match="nonempty"):
        validate_executor_contract(bad)
