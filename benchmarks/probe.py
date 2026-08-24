#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Jordan Miller
# SPDX-License-Identifier: MIT
"""Reproducible GLM-5.2 speed probe, integrated from external PR #2.

Original contribution: Jordan Miller (ecohash-co / jdmpike), commit
525b210dc100b75792ac5a0818d4eab0cdddc5a8, with Claude Fable 5 credited as
co-author.  The prose, peak, and cold-prefill prompt text is retained from that
contribution.  Champion integration hardens counter reconciliation, failures,
token sizing, timing labels, and evidence persistence.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import secrets
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


MANDATORY_COUNTERS = (
    "vllm:generation_tokens_total",
    "vllm:request_success_total",
    "vllm:prompt_tokens_total",
    "vllm:num_requests_running",
    "vllm:num_requests_waiting",
)
DETERMINISM_PROMPTS = (
    "The four moons of",
    "def fibonacci(n):",
    "The capital of France is",
)
PEAK_MAX_TOKENS = 900
PEAK_TEMPERATURE = 0.0
INVALIDATION_OUTCOMES = frozenset(
    {
        "idle-preflight",
        "foreign-traffic",
        "unexpected-api-connection",
        "runtime-health-failure",
        "gpu-clock-cap-changed",
        "client-failure-before-acceptance",
    }
)
PROFILES = {
    "fast-320k": 187022,
    "balanced-650k": 209022,
    "large-1m": 187022,
}
PROSE_TOPICS = (
    "the history of wooden boat building in the Pacific Northwest",
    "how mountain weather systems form and dissipate",
    "the economics of medieval wool trading between England and Flanders",
    "the sensory experience of walking through a tropical greenhouse in winter",
    "how cartographers handled unexplored regions before satellite imagery",
    "the acoustics of concert halls and why some rooms sound alive",
    "the social life of crows and what it suggests about intelligence",
    "how sourdough cultures differ from city to city and why",
)
PREFILL_UNIT = (
    "In the year of the survey the river ran higher than the oldest keeper "
    "could remember, and the ledgers kept in the stone house recorded each "
    "barge, its draft, its cargo and the toll assessed at the lower gate. "
)
PREFILL_SUFFIX = "\nIn one sentence: what did the ledgers record?"
METRIC = re.compile(
    r"^([A-Za-z_:][A-Za-z0-9_:]*)(?:\{[^}]*\})?\s+"
    r"([-+]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][-+]?[0-9]+)?)$"
)


class ProbeError(RuntimeError):
    """Base class for a probe that cannot produce clean measurements."""


class IdleRefusal(ProbeError):
    invalidation = "idle-preflight"


class Contamination(ProbeError):
    invalidation = "foreign-traffic"


class HealthFailure(ProbeError):
    invalidation = "runtime-health-failure"


class CounterReset(HealthFailure):
    pass


class WorkerFailure(ProbeError):
    def __init__(self, message: str, outcome: dict[str, Any]):
        super().__init__(message)
        self.outcome = outcome


class HttpFailure(ProbeError):
    def __init__(self, message: str, *, accepted: bool):
        super().__init__(message)
        self.accepted = accepted


def nonce() -> str:
    """Return an unpredictable cache-busting value."""

    return secrets.token_urlsafe(18)


def parse_prometheus(text: str) -> dict[str, float]:
    """Sum every label-set series into its Prometheus metric family."""

    totals: dict[str, float] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = METRIC.fullmatch(line)
        if match is None:
            continue
        value = float(match.group(2))
        if not math.isfinite(value):
            raise ProbeError(f"non-finite Prometheus value for {match.group(1)}")
        totals[match.group(1)] = totals.get(match.group(1), 0.0) + value
    return totals


def require_counters(metrics: dict[str, float]) -> None:
    missing = [name for name in MANDATORY_COUNTERS if name not in metrics]
    if missing:
        raise ProbeError(f"missing mandatory counter(s): {', '.join(missing)}")


def counter_delta(before: dict[str, float], after: dict[str, float], name: str) -> float:
    if name not in before or name not in after:
        raise ProbeError(f"missing mandatory counter: {name}")
    delta = after[name] - before[name]
    if delta < 0:
        raise CounterReset(f"counter reset detected for {name}")
    return delta


def assert_idle(metrics: dict[str, float]) -> None:
    require_counters(metrics)
    running = metrics["vllm:num_requests_running"]
    waiting = metrics["vllm:num_requests_waiting"]
    if running != 0:
        raise IdleRefusal(f"server is not idle: running={running}")
    if waiting != 0:
        raise IdleRefusal(f"server is not idle: waiting={waiting}")


def assert_post_idle(metrics: dict[str, float]) -> None:
    try:
        assert_idle(metrics)
    except IdleRefusal as exc:
        raise HealthFailure(f"post-probe idle check failed: {exc}") from exc


def reconcile(
    before: dict[str, float],
    after: dict[str, float],
    expected_requests: int,
    client_completion_tokens: int,
) -> None:
    requests = counter_delta(before, after, "vllm:request_success_total")
    if requests != expected_requests:
        raise Contamination(
            f"foreign request detected: expected delta {expected_requests}, got {requests:g}"
        )
    generated = counter_delta(before, after, "vllm:generation_tokens_total")
    if generated != client_completion_tokens:
        raise Contamination(
            "completion token reconciliation failed: "
            f"server delta {generated:g}, clients {client_completion_tokens}"
        )


@dataclass
class HttpClient:
    endpoint: str
    model: str = "glm-5.2"

    def __post_init__(self) -> None:
        self.endpoint = self.endpoint.rstrip("/")

    def _request(self, path: str, payload: dict[str, Any] | None, timeout: float) -> bytes:
        request = urllib.request.Request(self.endpoint + path)
        if payload is not None:
            request.add_header("Content-Type", "application/json")
            request.data = json.dumps(payload).encode("utf-8")
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            raise HttpFailure(str(exc), accepted=True) from exc
        except (OSError, urllib.error.URLError) as exc:
            raise HttpFailure(str(exc), accepted=False) from exc

    def get_text(self, path: str, timeout: float = 120) -> str:
        return self._request(path, None, timeout).decode("utf-8")

    def post_json(self, path: str, payload: dict[str, Any], timeout: float = 120) -> dict[str, Any]:
        try:
            result = json.loads(self._request(path, payload, timeout))
        except json.JSONDecodeError as exc:
            raise ProbeError(f"{path} returned invalid JSON") from exc
        if not isinstance(result, dict):
            raise ProbeError(f"{path} returned a non-object response")
        return result

    def metrics(self) -> dict[str, float]:
        result = parse_prometheus(self.get_text("/metrics"))
        require_counters(result)
        return result

    def health(self) -> bool:
        try:
            self.get_text("/health", timeout=10)
            return True
        except ProbeError:
            return False


def require_idle(client: Any, *, checks: int = 1, interval: float = 0.0) -> None:
    for index in range(checks):
        assert_idle(client.metrics())
        if index + 1 < checks:
            time.sleep(interval)


def prose_prompt() -> str:
    topic = secrets.choice(PROSE_TOPICS)
    return (
        f"[session {nonce()}] Write a vivid, meandering, unpredictable essay about "
        f"{topic}. Avoid lists and repetition; vary sentence length; digress freely."
    )


def peak_prompt() -> str:
    return (
        f"# task id {nonce()}\n"
        "Write a single Python file that defines one small dataclass per US state "
        "(name, capital, statehood year), then builds a dict keyed by state name, "
        "then prints each entry on its own line. Plain, repetitive, conventional "
        "code; no commentary, code only."
    )


def chat(
    client: Any,
    prompt: str,
    max_tokens: int,
    *,
    temperature: float = 1.0,
    top_p: float = 0.95,
    top_k: int = 40,
    timeout: float = 600,
) -> dict[str, Any]:
    started = time.monotonic()
    response = client.post_json(
        "/v1/chat/completions",
        {
            "model": getattr(client, "model", "glm-5.2"),
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "chat_template_kwargs": {"enable_thinking": False},
        },
        timeout,
    )
    finished = time.monotonic()
    try:
        usage = response["usage"]
        message = response["choices"][0]["message"]
        completion_tokens = int(usage["completion_tokens"])
        prompt_tokens = int(usage["prompt_tokens"])
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise ProbeError("completion response is missing exact usage or message fields") from exc
    text = (message.get("reasoning_content") or "") + (message.get("content") or "")
    return {
        "started_at": started,
        "finished_at": finished,
        "completion_tokens": completion_tokens,
        "prompt_tokens": prompt_tokens,
        "text": text,
    }


def classify_worker_failure(accepted_requests: int) -> dict[str, Any]:
    if accepted_requests == 0:
        return {"valid": False, "invalidation": "client-failure-before-acceptance"}
    return {"valid": True, "gate": "FAIL", "failure": "client-failure-after-acceptance"}


def run_concurrent(
    client: Any,
    prompt_factory: Callable[[], str],
    concurrency: int,
    max_tokens: int,
    *,
    temperature: float = 1.0,
    sample: bool = True,
    sampler_stopped: threading.Event | None = None,
) -> dict[str, Any]:
    """Run one clean concurrent pass and reconcile exact server/client deltas."""

    require_idle(client)
    before = client.metrics()
    results: list[dict[str, Any] | None] = [None] * concurrency
    worker_errors: list[Exception] = []
    sampler_errors: list[Exception] = []
    samples: list[tuple[float, float]] = []
    stop = threading.Event()

    def sampler() -> None:
        try:
            while not stop.is_set():
                current = client.metrics()
                samples.append((time.monotonic(), current["vllm:generation_tokens_total"]))
                stop.wait(1.0)
        except Exception as exc:  # propagated after every worker is joined
            sampler_errors.append(exc)

    def worker(index: int) -> None:
        try:
            results[index] = chat(
                client,
                prompt_factory(),
                max_tokens,
                temperature=temperature,
            )
        except Exception as exc:  # propagated on the caller thread
            worker_errors.append(exc)

    sampler_thread = threading.Thread(target=sampler, daemon=True)
    workers = [threading.Thread(target=worker, args=(index,)) for index in range(concurrency)]
    try:
        if sample:
            sampler_thread.start()
        for thread in workers:
            thread.start()
        for thread in workers:
            thread.join()
    finally:
        stop.set()
        if sample:
            sampler_thread.join(timeout=3)
        if sampler_stopped is not None:
            sampler_stopped.set()

    after = client.metrics()
    accepted = int(counter_delta(before, after, "vllm:request_success_total"))
    if worker_errors:
        transmitted = any(
            isinstance(exc, HttpFailure) and exc.accepted for exc in worker_errors
        )
        outcome = classify_worker_failure(max(accepted, int(transmitted)))
        raise WorkerFailure(str(worker_errors[0]), outcome) from worker_errors[0]
    if sampler_errors:
        raise WorkerFailure(f"metrics sampler failed: {sampler_errors[0]}", classify_worker_failure(accepted))
    completed = [item for item in results if item is not None]
    if len(completed) != concurrency:
        raise WorkerFailure("worker exited without a result", classify_worker_failure(accepted))
    client_tokens = sum(item["completion_tokens"] for item in completed)
    reconcile(before, after, concurrency, client_tokens)

    started = min(item["started_at"] for item in completed)
    finished = max(item["finished_at"] for item in completed)
    elapsed = finished - started
    if elapsed <= 0:
        raise ProbeError("non-positive monotonic request window")
    generated = counter_delta(before, after, "vllm:generation_tokens_total")
    wall = generated / elapsed

    saturation = None
    first_finish = min(item["finished_at"] for item in completed)
    window = [(stamp, value) for stamp, value in samples if started + 3 <= stamp <= first_finish - 1]
    if len(window) >= 4:
        (left_t, left_v), (right_t, right_v) = window[0], window[-1]
        if right_t > left_t:
            saturation = (right_v - left_v) / (right_t - left_t)

    post_health = client.health()
    if not post_health:
        raise HealthFailure("post-probe health check failed")
    post_metrics = client.metrics()
    assert_post_idle(post_metrics)
    drafts = accepted_tokens = None
    if (
        "vllm:spec_decode_num_drafts_total" in before
        and "vllm:spec_decode_num_drafts_total" in after
        and "vllm:spec_decode_num_accepted_tokens_total" in before
        and "vllm:spec_decode_num_accepted_tokens_total" in after
    ):
        drafts = counter_delta(before, after, "vllm:spec_decode_num_drafts_total")
        accepted_tokens = counter_delta(before, after, "vllm:spec_decode_num_accepted_tokens_total")
    return {
        "valid": True,
        "concurrency": concurrency,
        "wall_tok_s": round(wall, 3),
        "sat_est_tok_s": round(saturation, 3) if saturation is not None else None,
        "per_request_tok_s": round(wall / concurrency, 3),
        "generation_tokens": int(generated),
        "request_delta": accepted,
        "accepted_per_draft": round(accepted_tokens / drafts, 4) if drafts else None,
        "window_s": round(elapsed, 6),
        "post_health": True,
        "post_idle": True,
    }


def _token_count(client: Any, prompt: str) -> int:
    response = client.post_json(
        "/tokenize",
        {
            "model": getattr(client, "model", "glm-5.2"),
            "messages": [{"role": "user", "content": prompt}],
            "add_special_tokens": True,
            "chat_template_kwargs": {"enable_thinking": False},
        },
        30,
    )
    count = response.get("count")
    if count is None and isinstance(response.get("tokens"), list):
        count = len(response["tokens"])
    if not isinstance(count, int) or count < 0:
        raise ProbeError("/tokenize response lacks an exact token count")
    return count


def size_prefill_prompt(
    client: Any,
    target_tokens: int,
    *,
    nonce_value: str | None = None,
    allow_estimated: bool = False,
) -> tuple[str, dict[str, Any]]:
    """Use server tokenization feedback to construct an exact-size prompt."""

    if target_tokens <= 0:
        raise ProbeError("target token count must be positive")
    marker = nonce_value or nonce()
    seed = f"[cold prefill {marker}]" + PREFILL_SUFFIX
    try:
        seed_tokens = _token_count(client, seed)
        if seed_tokens > target_tokens:
            raise ProbeError("target token count is smaller than the cold-prefill envelope")
        unit_tokens = _token_count(client, seed + " " + PREFILL_UNIT) - seed_tokens
        if unit_tokens <= 0:
            raise ProbeError("tokenizer did not advance for the retained prefill unit")
        repetitions = max(0, (target_tokens - seed_tokens) // unit_tokens)
        base_body = f"[cold prefill {marker}] " + PREFILL_UNIT * repetitions
        base = base_body + PREFILL_SUFFIX
        while repetitions and _token_count(client, base) > target_tokens:
            repetitions -= 1
            base_body = f"[cold prefill {marker}] " + PREFILL_UNIT * repetitions
            base = base_body + PREFILL_SUFFIX
        base_count = _token_count(client, base)
        if base_count == target_tokens:
            return base, {"sizing_method": "server-tokenize-exact", "exact_sizing": True}
        remaining = target_tokens - base_count
        low, high = 0, max(remaining * 2 + 16, 32)
        while _token_count(client, base_body + " filler" * high + PREFILL_SUFFIX) < target_tokens:
            high *= 2
            if high > target_tokens * 4 + 1024:
                raise ProbeError("could not bracket exact prefill token target")
        while low <= high:
            middle = (low + high) // 2
            candidate = base_body + " filler" * middle + PREFILL_SUFFIX
            count = _token_count(client, candidate)
            if count == target_tokens:
                return candidate, {"sizing_method": "server-tokenize-exact", "exact_sizing": True}
            if count < target_tokens:
                low = middle + 1
            else:
                high = middle - 1
        for padding in range(max(0, high - 64), low + 65):
            candidate = base_body + " filler" * padding + PREFILL_SUFFIX
            if _token_count(client, candidate) == target_tokens:
                return candidate, {"sizing_method": "server-tokenize-exact", "exact_sizing": True}
        raise ProbeError("server tokenizer could not construct the exact prefill size")
    except (HttpFailure, ProbeError):
        if not allow_estimated:
            raise
        approximate = (f"[cold prefill {marker}] " + PREFILL_UNIT * (target_tokens + 1))[: target_tokens * 4]
        return approximate + PREFILL_SUFFIX, {
            "sizing_method": "estimated-4-chars-per-token",
            "exact_sizing": False,
        }


def prefill_rate(
    before: dict[str, float], after: dict[str, float], *, elapsed: float
) -> tuple[float, str]:
    prompt_tokens = counter_delta(before, after, "vllm:prompt_tokens_total")
    sum_name = "vllm:time_to_first_token_seconds_sum"
    count_name = "vllm:time_to_first_token_seconds_count"
    if all(name in before and name in after for name in (sum_name, count_name)):
        count = counter_delta(before, after, count_name)
        total = counter_delta(before, after, sum_name)
        if count == 1 and total > 0:
            return prompt_tokens / total, "server-ttft-histogram"
    if elapsed <= 0:
        raise ProbeError("non-positive prefill timing window")
    return prompt_tokens / elapsed, "end-to-end-fallback-includes-decode"


def probe_determinism(client: Any) -> dict[str, Any]:
    require_idle(client)
    before = client.metrics()
    diverged = 0
    completion_tokens = 0
    for prompt in DETERMINISM_PROMPTS:
        cache_busted = f"[session {nonce()}] {prompt}"
        left = chat(client, cache_busted, 32, temperature=0.0)
        right = chat(client, cache_busted, 32, temperature=0.0)
        completion_tokens += left["completion_tokens"] + right["completion_tokens"]
        diverged += left["text"] != right["text"]
    after = client.metrics()
    reconcile(before, after, len(DETERMINISM_PROMPTS) * 2, completion_tokens)
    if not client.health():
        raise HealthFailure("post-determinism health check failed")
    assert_post_idle(client.metrics())
    return {
        "probe": "determinism",
        "valid": True,
        "diverged": diverged,
        "of": len(DETERMINISM_PROMPTS),
        "prompts": list(DETERMINISM_PROMPTS),
        "max_tokens": 32,
        "temperature": 0.0,
        "thinking": False,
        "post_health": True,
        "post_idle": True,
    }


def probe_prefill(
    client: Any,
    target_tokens: int,
    *,
    timeout: int,
    allow_estimated: bool = False,
) -> dict[str, Any]:
    require_idle(client)
    prompt, sizing = size_prefill_prompt(
        client, target_tokens, allow_estimated=allow_estimated
    )
    before = client.metrics()
    started = time.monotonic()
    result = chat(client, prompt, 32, temperature=0.0, timeout=timeout)
    elapsed = time.monotonic() - started
    after = client.metrics()
    reconcile(before, after, 1, result["completion_tokens"])
    prompt_delta = counter_delta(before, after, "vllm:prompt_tokens_total")
    if prompt_delta != result["prompt_tokens"]:
        raise Contamination(
            f"prompt token reconciliation failed: server {prompt_delta:g}, client {result['prompt_tokens']}"
        )
    if sizing["exact_sizing"] and prompt_delta != target_tokens:
        raise Contamination(
            f"exact prefill target mismatch: expected {target_tokens}, got {prompt_delta:g}"
        )
    rate, timing_method = prefill_rate(before, after, elapsed=elapsed)
    if not client.health():
        raise HealthFailure("post-prefill health check failed")
    assert_post_idle(client.metrics())
    return {
        "probe": "prefill",
        "valid": True,
        "prompt_tokens": int(prompt_delta),
        "target_tokens": target_tokens,
        "prefill_tok_s": round(rate, 3),
        "window_s": round(elapsed, 6),
        "timing_method": timing_method,
        **sizing,
        "post_health": True,
        "post_idle": True,
    }


def emit(path: Path, payload: dict[str, Any], *, label: str, endpoint: str) -> None:
    record = dict(payload)
    record.update(
        {
            "label": label,
            "endpoint": endpoint,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps(record, indent=2, sort_keys=True))


def _failure_record(exc: BaseException) -> dict[str, Any]:
    if isinstance(exc, WorkerFailure):
        return {"probe": "attempt", **exc.outcome, "error": str(exc)[:500]}
    invalidation = getattr(exc, "invalidation", "runtime-health-failure")
    if invalidation not in INVALIDATION_OUTCOMES:
        invalidation = "runtime-health-failure"
    return {"probe": "attempt", "valid": False, "invalidation": invalidation, "error": str(exc)[:500]}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model", default="glm-5.2")
    parser.add_argument("--out", type=Path, default=Path("probe-results.jsonl"))
    parser.add_argument("--label", default="")
    parser.add_argument("--profile", choices=tuple(PROFILES), default="fast-320k")
    subcommands = parser.add_subparsers(dest="probe", required=True)
    subcommands.add_parser("determinism")
    for name in ("prose", "peak"):
        command = subcommands.add_parser(name)
        command.add_argument("--concurrency", type=int, default=1)
        command.add_argument("--repeat", type=int, default=1)
    prefill = subcommands.add_parser("prefill")
    prefill.add_argument("--target-tokens", type=int)
    prefill.add_argument("--timeout", type=int, default=1800)
    prefill.add_argument("--allow-estimated-sizing", action="store_true")
    battery = subcommands.add_parser("battery")
    battery.add_argument("--skip-prefill", action="store_true")
    battery.add_argument("--allow-estimated-sizing", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    client = HttpClient(args.endpoint, args.model)

    def persist(result: dict[str, Any]) -> None:
        result.setdefault("profile", args.profile)
        result.setdefault("thinking", False)
        emit(args.out, result, label=args.label, endpoint=client.endpoint)

    try:
        if args.probe == "determinism":
            persist(probe_determinism(client))
        elif args.probe == "prose":
            for _ in range(args.repeat):
                persist({
                    "probe": f"prose_c{args.concurrency}",
                    **run_concurrent(
                        client, prose_prompt, args.concurrency,
                        1024 if args.concurrency > 1 else 768,
                    ),
                })
        elif args.probe == "peak":
            for _ in range(args.repeat):
                persist({
                    "probe": f"peak_c{args.concurrency}",
                    **run_concurrent(
                        client, peak_prompt, args.concurrency, PEAK_MAX_TOKENS,
                        temperature=PEAK_TEMPERATURE,
                    ),
                })
        elif args.probe == "prefill":
            persist(probe_prefill(
                client,
                args.target_tokens or PROFILES[args.profile],
                timeout=args.timeout,
                allow_estimated=args.allow_estimated_sizing,
            ))
        else:
            persist(probe_determinism(client))
            for concurrency, repeat in ((1, 2), (4, 2), (2, 1)):
                for _ in range(repeat):
                    persist({
                        "probe": f"prose_c{concurrency}",
                        **run_concurrent(
                            client, prose_prompt, concurrency,
                            1024 if concurrency > 1 else 768,
                        ),
                    })
            for concurrency in (1, 4):
                persist({
                    "probe": f"peak_c{concurrency}",
                    **run_concurrent(
                        client, peak_prompt, concurrency, PEAK_MAX_TOKENS,
                        temperature=PEAK_TEMPERATURE,
                    ),
                })
            if not args.skip_prefill:
                persist(probe_prefill(
                    client, PROFILES[args.profile], timeout=1800,
                    allow_estimated=args.allow_estimated_sizing,
                ))
            require_idle(client)
            healthy = client.health()
            persist({"probe": "post_battery_health", "valid": healthy, "post_idle": True})
            if not healthy:
                raise HealthFailure("post-battery health check failed")
    except Exception as exc:
        persist(_failure_record(exc))
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
