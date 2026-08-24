from __future__ import annotations

import threading
import json
from pathlib import Path

import pytest

from benchmarks import probe


ROOT = Path(__file__).resolve().parents[1]


def metric_text(**values):
    return "\n".join(f"{name} {value}" for name, value in values.items()) + "\n"


def base_metrics(gen=0, req=0, prompt=0, running=0, waiting=0):
    return {
        "vllm:generation_tokens_total": gen,
        "vllm:request_success_total": req,
        "vllm:prompt_tokens_total": prompt,
        "vllm:num_requests_running": running,
        "vllm:num_requests_waiting": waiting,
    }


def test_prometheus_aggregates_all_label_sets_and_general_names():
    text = (
        'vllm:generation_tokens_total{model="a",finished_reason="stop"} 10\n'
        'vllm:generation_tokens_total{model="b",finished_reason="length"} 12\n'
        'custom_metric_1{worker="0"} 3\n'
    )
    assert probe.parse_prometheus(text)["vllm:generation_tokens_total"] == 22
    assert probe.parse_prometheus(text)["custom_metric_1"] == 3


def test_missing_counter_and_counter_reset_fail_closed():
    with pytest.raises(probe.ProbeError, match="missing mandatory counter"):
        probe.require_counters({"vllm:num_requests_running": 0})
    with pytest.raises(probe.ProbeError, match="counter reset"):
        probe.counter_delta({"x": 3}, {"x": 2}, "x")


def test_idle_requires_running_and_waiting_zero():
    with pytest.raises(probe.IdleRefusal, match="waiting"):
        probe.assert_idle(base_metrics(waiting=1))
    with pytest.raises(probe.IdleRefusal, match="running"):
        probe.assert_idle(base_metrics(running=1))


def test_reconcile_requires_exact_request_and_completion_token_deltas():
    before = base_metrics(gen=10, req=2)
    after = base_metrics(gen=42, req=4)
    probe.reconcile(before, after, expected_requests=2, client_completion_tokens=32)
    with pytest.raises(probe.Contamination, match="foreign request"):
        probe.reconcile(before, base_metrics(gen=42, req=5), 2, 32)
    with pytest.raises(probe.Contamination, match="completion token"):
        probe.reconcile(before, base_metrics(gen=43, req=4), 2, 32)


class FakeClient:
    def __init__(self):
        self.requests = []

    def post_json(self, path, payload, timeout):
        self.requests.append((path, payload, timeout))
        if path == "/tokenize":
            prompt = payload.get("prompt") or payload["messages"][0]["content"]
            return {"count": len(prompt.split())}
        return {
            "usage": {"prompt_tokens": 4, "completion_tokens": 2},
            "choices": [{"message": {"content": "ok", "reasoning_content": ""}}],
        }


def test_tokenize_exact_sizing_and_explicit_fallback_label():
    client = FakeClient()
    prompt, info = probe.size_prefill_prompt(client, 12, nonce_value="unique")
    assert client.post_json("/tokenize", {"model": "glm-5.2", "prompt": prompt}, 30)["count"] == 12
    assert info == {"sizing_method": "server-tokenize-exact", "exact_sizing": True}

    class NoTokenizer(FakeClient):
        def post_json(self, path, payload, timeout):
            if path == "/tokenize":
                raise probe.HttpFailure("unsupported", accepted=False)
            return super().post_json(path, payload, timeout)

    _, fallback = probe.size_prefill_prompt(NoTokenizer(), 12, nonce_value="unique", allow_estimated=True)
    assert fallback == {"sizing_method": "estimated-4-chars-per-token", "exact_sizing": False}


def test_prefill_timing_prefers_ttft_histogram_and_labels_fallback():
    before = base_metrics(prompt=0) | {
        "vllm:time_to_first_token_seconds_sum": 2,
        "vllm:time_to_first_token_seconds_count": 1,
    }
    after = base_metrics(prompt=100) | {
        "vllm:time_to_first_token_seconds_sum": 2.5,
        "vllm:time_to_first_token_seconds_count": 2,
    }
    rate, method = probe.prefill_rate(before, after, elapsed=5)
    assert (rate, method) == (200.0, "server-ttft-histogram")
    rate, method = probe.prefill_rate(base_metrics(prompt=0), base_metrics(prompt=100), elapsed=5)
    assert (rate, method) == (20.0, "end-to-end-fallback-includes-decode")


def test_determinism_exact_trio_and_chat_disables_thinking():
    assert probe.DETERMINISM_PROMPTS == (
        "The four moons of", "def fibonacci(n):", "The capital of France is"
    )
    client = FakeClient()
    probe.chat(client, "prompt", 32, temperature=0.0)
    payload = client.requests[-1][1]
    assert payload["max_tokens"] == 32 and payload["temperature"] == 0.0
    assert payload["chat_template_kwargs"]["enable_thinking"] is False


def test_unique_nonce_values():
    assert len({probe.nonce() for _ in range(100)}) == 100


def test_worker_error_surfaces_and_sampler_stops(monkeypatch):
    stopped = threading.Event()

    class Client:
        def metrics(self):
            return base_metrics()

        def health(self):
            return True

    monkeypatch.setattr(probe, "chat", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    with pytest.raises(probe.WorkerFailure, match="boom"):
        probe.run_concurrent(Client(), lambda: "p", 1, 32, sampler_stopped=stopped)
    assert stopped.is_set()


def test_completed_slow_response_is_valid_and_post_health_idle_checked(monkeypatch):
    snapshots = iter([
        base_metrics(), base_metrics(),
        base_metrics(gen=2, req=1), base_metrics(gen=2, req=1),
    ])

    class Client:
        def metrics(self):
            return next(snapshots)

        def health(self):
            return True

    monkeypatch.setattr(probe, "chat", lambda *a, **k: {
        "started_at": 0.0, "finished_at": 100.0, "completion_tokens": 2,
        "prompt_tokens": 4, "text": "ok"
    })
    result = probe.run_concurrent(Client(), lambda: "p", 1, 32, sample=False)
    assert result["valid"] is True
    assert result["wall_tok_s"] < 1
    assert result["post_health"] is True and result["post_idle"] is True


def test_failure_before_acceptance_is_invalid_but_after_acceptance_is_valid_failure():
    assert probe.classify_worker_failure(0) == {
        "valid": False, "invalidation": "client-failure-before-acceptance"
    }
    assert probe.classify_worker_failure(1) == {
        "valid": True, "gate": "FAIL", "failure": "client-failure-after-acceptance"
    }


def test_peak_contract_uses_900_tokens_temperature_zero():
    assert probe.PEAK_MAX_TOKENS == 900
    assert probe.PEAK_TEMPERATURE == 0.0


def test_locked_baselines_use_unrounded_references_and_exact_floors():
    payload = json.loads((ROOT / "benchmarks/baselines.json").read_text(encoding="utf-8"))
    expected = {
        "fast-320k": {"prose_c1": 26.565, "prose_c4": 55.95, "peak_c1": 40.47, "peak_c4": 81.02, "prefill": 694.1},
        "balanced-650k": {"prose_c1": 23.78, "prose_c4": 52.56, "peak_c1": 34.965, "peak_c4": 72.59, "prefill": 585.3},
        "large-1m": {"prose_c1": 22.25, "prose_c4": 48.775, "peak_c1": 36.60, "peak_c4": 73.23, "prefill": 422.8},
    }
    for profile, metrics in expected.items():
        for name, reference in metrics.items():
            actual = payload["profiles"][profile]["metrics"][name]
            assert actual["reference"] == reference
            assert actual["median_floor"] == pytest.approx(reference * 0.95)
            assert actual["individual_floor"] == pytest.approx(reference * 0.90)
