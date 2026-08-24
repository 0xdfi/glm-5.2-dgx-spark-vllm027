# Champion benchmark protocol

This is the promotion protocol for reproducing the historical Fast, Balanced, and Large references. A local evaluation is provisional until its exact runtime subject and evidence are authenticated by the trusted workflow.

## Identity and preflight

Bind every attempt to the same source release, build subject/lock, source manifest, image digest, model ID, sidecar ID, preset ID, tokenizer ID, sampling ID, pin state where applicable, and run identity. Capture current-window engine/rank/IB/clock/health evidence and refuse if the server is not idle.

Prompts carry a unique nonce to defeat prefix-cache reuse. Throughput comes from reconciled server counters, not client wall-clock token division. Request counts, generated-token counts, unexpected API connections, worker exceptions, and post-run idle state are retained.

## One complete battery

Run in this order:

1. **Exact determinism triplet.** Run the same fixed prompt three times with `temperature=0` and `thinking=false`; compare the complete outputs exactly. This is diagnostic and does not create a bit-reproducibility claim.
2. **Prose C1 and C4.** Cold unpredictable prompts at the recorded production sampling settings, with C4 reported as aggregate server-counter throughput.
3. **Peak C1 and C4.** Exactly 900 output tokens, `temperature=0`, `thinking=false`, on the fixed predictable peak prompt. A peak metric is favorable-content throughput, not typical chat speed.
4. **Cold prefill.** Generate each profile's exact target length using the model's `/tokenize` endpoint. Targets are 187022 tokens for Fast, 209022 for Balanced, and 187022 for Large. Verify the server-accepted token count rather than estimating characters.
5. **Post-battery gates.** Prove health, idle, ranks, IB, clocks, current-window logs, and a real generation response.

For prefill timing, use server TTFT/histogram evidence when the runtime exposes it. If unavailable, use the defined request-latency fallback and label the metric `fallback`; never present fallback timing as native TTFT. Exclude decode tail only when the retained counters support that separation.

## Attempts, validity, and thresholds

Retain all attempts, including raw output and failure evidence. A completed slow response is valid; threshold failure is not an invalidation reason. Validity may be lost only for machine-evidenced preflight refusal, foreign traffic/counter mismatch, unexpected connection, node/IB/GPU/engine failure, clock-cap drift, or client failure before server acceptance.

Each profile has at most five attempts and must produce exactly three valid complete batteries. Stop once three valid batteries exist; do not keep sampling for a better result. Fewer than three valid attempts or exhaustion of five attempts fails the profile. No cherry-picking, hidden retries, carried-forward cells, or metric-by-metric splicing is allowed.

Every valid individual primary metric must reach 90% of its [historical reference](baselines.json); the median of the three valid batteries must reach 95%. A slow complete battery that misses either floor remains valid and causes the reproduction claim to fail.

## Reporting

Report the three complete valid batteries, their medians, every invalid attempt with reason/evidence, determinism output, tokenizer counts, timing source (`native TTFT` or labeled fallback), and post-battery runtime state. Only `gate_status.py` operating on issuer-verified receipts can promote `benchmark-reproduced`.
