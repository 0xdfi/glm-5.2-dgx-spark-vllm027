# Findings — what the campaign actually learned

Research results worth sharing beyond the raw matrix, plus honest open questions. Where a
number appears here it's repeated from the README/presets for convenience — the README matrix
is the source of truth if these ever drift.

---

## The chunk-size plateau: 2048 ≈ 4096, everywhere

`max_num_batched_tokens` (mnbt) 2048 vs 4096 was tested at matched KV/DCP configuration on
three separate DCP levels, and the result is consistent across all three: **4096 buys nothing,
and actively hurts peak-C4.**

- **DCP1 320K:** prose C1 26.97/26.16 @2048 vs 25.02/25.31 @4096 (flat-to-slightly-down);
  prose C4 55.96/55.94 vs 56.48/57.33 (flat, within noise); **peak C4 78.35/79.14 @2048 vs
  74.03/76.28 @4096 — a real, consistent drop**; cold prefill 694.1 vs 688.7 (flat, within
  noise — the carried-forward 688.7 is itself evidence the campaign judged this comparison
  safe to make without a fresh gate).
- **DCP2 650K→600K:** prose C1/C4 essentially flat between the 2048 and 4096 rows; peak C4
  74.12/71.06 @2048 vs 74.98/73.93 @4096 (roughly flat this time, the one row where 4096
  didn't clearly lose) — but note this comparison also changed KV budget (650K→600K)
  alongside mnbt, so it's a weaker isolation than the DCP1 pair.
- **DCP4 1M:** peak C4 at mnbt=4096 (62.64 avg) is the **single weakest peak-C4 figure in the
  entire matrix** — clearly worse than 1M@512's 69.73 and every other DCP4 row.

**Mechanism (documented, not yet proven with a targeted trace):** higher mnbt raises the
prefill admission chunk size inside every mixed decode+prefill scheduling step. Peak-C4 probes
run 4 concurrent streams, and each stream's prefill gets admitted in chunks alongside ongoing
decode; a larger `decode-prefill-token-budget`-bounded chunk drags more prefill tokens through
the same memory-bandwidth-bound pipeline in the window where peak decode throughput is being
measured, at exactly the concurrency level (C4) where the effect is easiest to see. Prefill
throughput itself doesn't suffer from this — the cold-prefill gate numbers stay flat or improve
slightly with mnbt — because pure prefill isn't sharing pipeline time with decode. The
degradation is specific to the *mixed* decode+prefill regime that peak-C4 probes exercise.

**An adjacent, related, and explicitly untested lever:** `--decode-prefill-token-budget`
(DPTB) tracked `min(mnbt, 1024)` for every row in this matrix — it was never independently
swept as its own variable. There is a documented, un-flown hypothesis from the campaign's own
speed-hunting work that **DPTB=256** (rather than the 1024 used throughout this matrix) would
recover some or all of the peak-C4 degradation at higher mnbt, based on precedent: an earlier,
pre-DCP-era promotion on this same stack measured DPTB 1024→256 as a real win (prose C1
23.9→25.8, prose C4 52.4→54.4, prefill 644→663.7 tok/s) — the exact same knob, on the same
hardware, in a different configuration era. **If you're extending this matrix, sweeping DPTB
independently of mnbt is the single highest-expected-value untested arm.**

---

## KV amount is nearly free within a fixed DCP level

Doubling the KV budget at a constant DCP degree and mnbt costs close to nothing: DCP4 500K@512
(prose C1 22.66/22.82, prose C4 48.83/49.97, prefill 360.5) vs. DCP4 1M@512 (22.10/22.06,
48.08/49.29, 361.7) — every delta sits inside the ±5% noise band, including prefill, despite
doubling the context window. Once you've paid the fixed cost of a given DCP degree (the
per-attention-call merge overhead), filling that degree's capacity ceiling is close to free.
This is why the "choosing a config" guidance in the README treats DCP degree, not KV size
within a degree, as the primary lever.

---

## DCP scaling costs, summarized

Across the matrix as a whole: **decode ≈ −5% per DCP doubling**, **prefill ≈ −20% per DCP
doubling**, fairly consistently from DCP1→DCP2 and DCP2→DCP4. See `docs/how-it-works.md` for
the mechanism (the cross-rank top-k merge step that DCP adds to every attention call) and the
README's "choosing a config" section for the practical tradeoff table.

---

## The "97GB mystery" — solved, not a bug

Early in the DCP4 memory-debugging work, ~97GB of torch-resident memory per rank (measured
just from the model being loaded, before any KV allocation) looked suspicious — high enough
that "shave 10-20GB off the baseline" looked like a plausible lever. It isn't one.

The GLM-5.2 W4A16 checkpoint is **379GiB on disk**. Sharded 4 ways under TP4 (experts and
attention heads split across ranks), that's **~94.75GiB of weights resident per rank** — the
overwhelming majority of the observed ~97GB baseline. Subtracting weights from the measured
baseline leaves roughly **~2.2GiB/rank of non-weight torch residency** (allocator reserve
slack, repack churn, buffers) — which is *lean*, not fat. There is no meaningful "baseline
bloat" hiding in this number; the per-rank capacity ceiling documented in the README's
capacity-frontier section (~10.2-10.4GB/rank of KV at mnbt 2048) is honest arithmetic:
121GB total, minus ~94.75GB weights, minus the fixed 4GB host reserve, minus a few GB of
runtime/graph/system overhead, leaves what's left for KV. **Stop looking for a shave here — it
doesn't exist.** The remaining real levers on this axis are all either KV-dtype policy trades
(a different quantization for KV, not a free lunch) or the checkpoint's own quantization
(a re-quant project, not a config change) — see the README's "what this is not" for why this
repo doesn't attempt either.

---

## The temperature-0 nondeterminism anomaly

**Observation:** running the same fixed prompts twice at `temperature=0` does not always
produce byte-identical output on this stack. The divergence rate is not constant — it
correlates with `max_num_batched_tokens` (mnbt), not with DCP degree, and the campaign has
three independent controlled data points supporting this:

| Config pair (matched everything except mnbt) | mnbt 2048 | mnbt 4096 |
|---|---|---|
| DCP1 320K | 2/3 diverged | 3/3 diverged |
| DCP2 650K→600K | 3/3 diverged | 3/3 diverged (ceiling, no further headroom to show growth) |

The initial working hypothesis going into the campaign was "this is a DCP>1-specific
artifact" — every DCP4 row tested showed the same 2/3-diverged pattern, and DCP1-250K@2048 was
initially clean. That hypothesis was **falsified**, not confirmed, by DCP1-320K@2048 showing
the identical 2/3-diverged signature at DCP degree 1. The pattern that survived is
**mnbt-linked, not DCP-linked**: raising mnbt from 2048 to 4096 at fixed KV size and fixed DCP
degree consistently produced equal-or-worse divergence, and the one data point available at
the *opposite* end (DCP4 1M@2048 vs 1M@4096: 2/3 diverged at 2048, 3/3 at 4096, identical KV)
is the cleanest single confirmation — same KV, only mnbt differs, divergence gets worse.

**Two candidate mechanisms, both plausible, not yet separated:**

1. **Marlin atomic-add reduction-order jitter.** `VLLM_MOE_MARLIN_ATOMIC_ADD=1` is on
   throughout this matrix (it's a real decode-speed win at low concurrency); floating-point
   `atomicAdd` reduction order is scheduler-dependent, so bitwise output can vary run-to-run at
   fixed input. This affects run-to-run determinism at a *fixed* batch composition, and is a
   plausible contributor independent of mnbt.
2. **Batch-composition / chunk-boundary variance.** Larger mnbt changes how the scheduler
   splits work into chunks across a decode+prefill mixed step; different chunk splits can drive
   different tile/split-k decompositions in the underlying GEMM kernels, producing different
   floating-point rounding paths and, occasionally, a flipped near-tie argmax in the MTP
   verification step. This mechanism specifically predicts an mnbt correlation, which is what
   the data shows.

**An untested, cheap discriminating experiment exists and was never run in this campaign:**
boot the same config twice, once with `VLLM_MOE_MARLIN_ATOMIC_ADD=0`, and repeat the
temperature-0 spot-check. If divergence drops with atomic-add off, mechanism (1) is a real
contributor; if it doesn't move, the batch-composition mechanism (2) is doing all the work.
This also has a possible speed angle worth checking in the same experiment: MTP verification
compares draft output against target argmax, and logit jitter from atomic-add can end
speculation rounds early — so turning atomic-add off might trade a small amount of raw kernel
speed for a higher speculative-acceptance rate, a net wash or even a net win at higher
concurrency. Nobody has measured which way that trade nets out.

**Practical implication for users of this recipe:** if your application depends on bit-exact
reproducibility at temperature 0, this stack does not currently guarantee it, and the
guarantee gets *worse*, not better, at the mnbt=2048 setting this repo recommends as the
overall sweet spot. Weigh that tradeoff explicitly if determinism matters to you — don't
assume `temperature=0` means "deterministic" on this configuration without testing your own
workload's prompts.

---

## Open questions

- **The step-time law's ~50ms unattributed fixed cost** (see `docs/how-it-works.md`). Roughly
  20-25ms of the measured 73ms fixed decode-step cost is attributed to estimated compute; the
  remainder is an open trace-level question. Candidate explanations investigated and not yet
  ruled in or out: rank-wait/straggler skew inside the ~156 per-step allreduces, per-layer GPU
  body cost across 78 transformer layers. Ruled out: NCCL protocol/tuning (an `NCCL_PROTO=LL`
  pin measured exactly flat on decode). A single detailed multi-rank trace of one decode step,
  decomposing collective-kernel time / collective-wait time / non-collective kernel time / host
  gaps, would settle this and hasn't been run.
- **The DPTB=256 arm**, described above under the chunk-size plateau — the single
  highest-expected-value untested experiment against this matrix as it stands.
- **The atomic-add A/B**, described above under the nondeterminism anomaly — cheap, and
  answers two questions (determinism mechanism, possible acceptance-rate speed trade) in one
  experiment.
- **Whether `index_share_for_mtp_iteration` is actually engaging.** The model config sets this
  flag, and the overlay has the machinery to honor it (an indexer top-k sharing path that
  should save several redundant indexer passes per step at MTP k=5), but whether it's
  confirmed *live* in a given boot depends on the draft model's own config copy carrying the
  same flag — this was flagged as worth a five-minute log grep during any future boot
  (`grep` the worker logs for the sharing-engaged log line) and was never confirmed either way
  during this campaign.
- **DCP4 1M@2048's peak-decode legs.** The final matrix cell (see the README) has confirmed
  prose C1/C2/C4 and cold-prefill numbers but no peak C1/peak C4 — the battery was interrupted
  by the live-traffic incident described in `docs/operations.md` before reaching that step.
