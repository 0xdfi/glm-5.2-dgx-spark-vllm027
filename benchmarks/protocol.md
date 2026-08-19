# Benchmark protocol — exactly how the speed matrix was produced

This document describes the measurement protocol behind every number in the README's speed
matrix, so the numbers are reproducible rather than just asserted. If you re-run this protocol
on your own hardware and get different numbers, that's expected — see the README's "what this
is not" section — but the *methodology* should transfer.

## Probe design

Every probe is a small, purpose-built Python script hitting the server's OpenAI-compatible API
directly (`urllib`, no client SDK in the loop), for two reasons: it keeps client-side overhead
out of the measurement, and it lets the probe read the server's own counters instead of
computing throughput from client-observed wall time.

**Cache-busted prompts.** Every request embeds a random nonce (e.g. a session ID string) in
the prompt text, so no request can ever hit vLLM's prefix cache and get an artificially fast
answer. This matters a lot on this stack specifically — `--enable-prefix-caching` is on in
every preset — so an un-busted probe would silently measure cache-hit speed, not decode speed.

**Server-counter measurement, not client wall-clock division.** Each probe reads
`/metrics` before and after the measurement window and computes throughput from the **delta**
in server-side counters (`vllm:generation_tokens_total`, `vllm:request_success_total`,
`vllm:spec_decode_num_accepted_tokens_total`, etc.), not from `completion_tokens / wall_time`
on the client. This avoids client-side queuing, connection setup, and JSON-parsing overhead
contaminating the number, and it's the only way to get a trustworthy **aggregate** throughput
figure across concurrent requests (client-side per-request division doesn't compose correctly
under concurrency).

**Idle-required, refuse-if-not.** Every probe checks `vllm:num_requests_running` before
starting and refuses (prints a `"refused": "not idle"` line and exits non-zero) rather than
measuring against a contaminated baseline. This is a necessary but not sufficient safeguard —
see the "contamination detection" section below for what catches the traffic that arrives
*during* a measurement window instead of before it.

**Contamination detection.** A probe also reconciles what *it* asked for against what the
server *delivered*: if `requests_completed_delta` doesn't match the number of requests the
probe itself issued, or `server_gen_tokens` doesn't match the client-observed
`completion_tokens` sum, the probe marks its own result `"contaminated": true` and that result
is discarded, not averaged in. This is what caught the campaign's two live-traffic incidents
(see `docs/operations.md`) — in both cases the probe's own arithmetic flagged the mismatch
before a human had to notice a suspicious number.

## The battery, in order

Each row in the speed matrix was produced by running these steps, in this order, against an
already-booted, already-healthy, confirmed-idle server:

1. **Temperature-0 determinism spot-check.** A small fixed set of prompts (typically 3), each
   run twice at `temperature=0`, diffed for exact-match output. Not a speed measurement — this
   is the control data behind the nondeterminism anomaly in `docs/findings.md`. Run first,
   before any load-generating probe, so it measures the freshest possible server state.
2. **Prose C1, x2.** Concurrency 1, ~768 max_tokens, temperature 1.0 / top_p 0.95 / top_k 40
   (production sampling), cold unpredictable prose content (a rotating set of essay-style
   topics — "the history of wooden boat building," "how mountain weather forms," etc., chosen
   specifically to be hard to predict token-by-token, unlike code). Run twice; report both
   numbers as `a / b` in the matrix, not averaged, so the reader can see the actual spread.
3. **Prose C4, x2.** Same content style, concurrency 4, ~1024 max_tokens. This is the
   aggregate-throughput number under concurrent load, computed from the server-counter delta
   across all 4 simultaneous requests.
4. **Prose C2** (where run). Concurrency 2, same content style — added partway through the
   campaign to fill in the C1→C4 scaling curve; not present for every config, hence the `n/m`
   cells in the matrix.
5. **Peak C1 / Peak C4.** Code-class, highly predictable content, run at concurrency 1 and 4.
   "Peak" is not a different sampling configuration — it's a different *content class* chosen
   to land the adaptive MTP speculation controller in its deep-k regime (more speculative
   tokens verified and accepted per step), which is close to this stack's throughput ceiling.
   Report this as what it is: a best-case number under favorable content, not what typical
   chat/completion traffic will see.
6. **Cold prefill gate.** A single large prompt (~190-220K tokens, varying slightly by
   config — exact token counts are recorded per battery), submitted cold (no prior context,
   fresh nonce), timed and measured via the server counters, and gated to **PASS** only if it
   completes without a host OOM or timeout. This is the number the README reports as
   "prefill" — it is a *pass/fail gate first, a speed number second*: an OOM or timeout here on
   an undersized config is exactly the historical failure mode that motivated the capacity
   frontier work in the first place (see `docs/how-it-works.md`).
7. **Post-battery health check.** Confirm `/health` still returns 200 and
   `num_requests_running` is back to 0 after the full battery — a battery that leaves the
   server in a bad state invalidates trust in every number it produced.

Some batteries also run a repeat pass of the peak probes (see "fresh-boot under-read" below)
or a memory snapshot (`free -g` on all 4 nodes) at the end, to confirm the battery didn't
leave the cluster in a different memory state than it found it.

## Noise band

**±5% at n=1-2 batteries.** Most rows in the matrix report 1-2 repeats per probe (shown as
`a / b` where both numbers are given). Treat single-digit-percent deltas between adjacent
matrix rows as noise, not a real ranking, unless the campaign explicitly ran n≥3 and called
it a confirmed result. A handful of findings in `docs/findings.md` (the mnbt-correlated
nondeterminism pattern, in particular) were specifically escalated to n=3 because a single
data point wasn't enough to trust.

## The fresh-boot peak-probe under-read quirk

**Peak-decode probes measured immediately after a fresh boot read low, and recover on a
repeat pass later in the same boot.** This was directly observed: on one config, peak C1 read
34.61 tok/s on the first pass after boot and 36.36 tok/s (+5.0%) on a repeat pass later in the
same session with no config change; peak C4 read 73.01 first-pass and 83.22 (+14.0%)
second-pass. This is a warm-up artifact of probing immediately after boot — likely JIT/CUDA
graph replay paths not yet fully warmed, or an early-session scheduling transient — **not a
property of the profile itself**.

**Practical consequence for reproducing this matrix:** don't trust a first-pass peak-C4 number
taken right after boot as representative. Re-probe after some uptime (the campaign generally
waited for at least one full battery's worth of other traffic, or a few minutes of idle time,
before treating a peak number as final) before using it in any promotion/comparison claim.
Rows in the README matrix that show two peak numbers separated by `/` are showing exactly this
first-pass/repeat-pass pair where it was measured; a single number means either only one pass
was run, or the server had already been up and stable through prior battery steps (no
fresh-boot artifact to correct for).

## The cold-prefill gate, in more detail

The prefill number is deliberately the *last* speed-relevant thing measured in some batteries
(after decode probes, not before) specifically because it is memory-riskiest step in the whole
battery — a large cold prompt is the closest thing to the historical failure mode (host OOM
during prefill on an undersized config) that motivated most of the capacity-frontier work.
Two operational notes worth preserving:

- On at least one config in this matrix (DCP1 320K@2048 — the campaign's thinnest-margin
  cell), the cold-prefill gate was **not re-run** in a later battery pass specifically to
  protect a very thin memory margin (1-2GB available at idle); the prefill number reported for
  that row is **carried forward** from an earlier pass that already gated PASS at the same
  configuration. This is noted per-row in `recipes/presets.md` where it applies — it does not
  mean the number is stale or invalid, just that it wasn't re-measured every single time.
- A gate **PASS** is the primary signal; the tok/s number is secondary. A config that gates
  PASS at a mediocre prefill speed is still a usable config. A config that fails this gate is
  not usable at that setting regardless of how fast its decode numbers look — don't promote a
  config based on decode speed alone without a passing prefill gate at your intended context
  length.
