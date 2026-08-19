# How it works — DCP, the memory model, and the step-time law

## Decode-context-parallelism (DCP), plainly

At tensor-parallel degree 4 (TP4), each of the 4 nodes holds one-quarter of every weight
matrix — the attention heads and MoE experts are sharded across ranks, and every forward pass
does cross-rank collectives to reassemble a full result. That's TP; it splits *compute and
weights*, not the KV cache. Without DCP, **every rank still holds a full copy of the KV cache**
for the current sequence — TP shards attention heads, but the KV cache indexed by those heads
is replicated in full across all 4 ranks' attention state.

**DCP splits the KV cache itself across ranks.** At DCP degree `D`, each rank stores roughly
`1/D` of the sequence's KV cache — rank 0 holds one quarter of the tokens' key/value state (at
DCP4), not a copy of all of it. When the sparse attention indexer needs to score candidate
tokens against the query, each rank does a **local top-k** against its own KV shard, then the
ranks **all-gather and merge** the candidate lists (`_merge_b12x_dcp_topk` in the B12X overlay)
into one globally-correct top-k before the actual attention computation runs. This is the
extra cross-rank traffic DCP costs you at prefill and decode: an all-gather-and-merge step per
attention call that scales with DCP degree, instead of a free local lookup.

The payoff: **KV memory per rank scales down by roughly `1/D`** for the same total context
length, or equivalently, **you can serve `D`x the context length in the same per-rank memory
budget**. That's the entire DCP1 → DCP2 → DCP4 tradeoff in this repo: more ranks sharing the
KV cache buys more context capacity, at the cost of the merge step's cross-rank communication
overhead on every attention call.

### The `tp % dcp == 0` rule, and why DCP3 doesn't exist here

DCP does **not** expand the number of processes in the cluster. Without prefill-context-
parallelism (PCP) enabled — which this recipe doesn't use — **DCP reuses the existing TP
ranks**. The DCP group and the TP group are the same 4 processes; DCP just partitions what
each of those 4 processes holds in its KV cache, not how many processes exist.

vLLM enforces `tensor_parallel_size % decode_context_parallel_size == 0` at config-validation
time. On a 4-node TP4 cluster, the only integer divisors of 4 are 1, 2, and 4 — **DCP3 is not
a legal configuration on this cluster, full stop**, not a performance choice that happens to
be untested. (It would be legal on a TP6 or TP12 cluster, for example — this is a property of
your TP degree, not of DCP or GLM-5.2 in general.)

### DCP1 / DCP2 / DCP4 in one line each

- **DCP1** — no split. Every rank holds the full KV cache for its attention heads; no
  cross-rank merge step. Fastest decode and prefill, smallest per-rank context capacity ceiling
  (~320K tokens on this stack at mnbt 2048; see the capacity frontier in the README).
- **DCP2** — KV split 2 ways. Local block width covers 128 global tokens per rank (see the KV
  byte formula below); roughly 2x the context capacity of DCP1, ~5% decode cost, ~20% prefill
  cost per doubling.
  - `--decode-context-parallel-size 2 --dcp-comm-backend a2a --dcp-kv-cache-interleave-size 1`
- **DCP4** — KV split 4 ways. Local block width covers 256 global tokens per rank; the largest
  context capacity this stack supports (up to the model's native 1,048,576-token position cap,
  not a memory-driven ceiling — see the README's capacity frontier table), at the cumulative
  decode/prefill cost of two DCP doublings from DCP1.
  - `--decode-context-parallel-size 4 --dcp-comm-backend a2a --dcp-kv-cache-interleave-size 1`

The `--dcp-comm-backend a2a --dcp-kv-cache-interleave-size 1` pair is only valid — and only
needed — at DCP>1; at DCP1 there is no cross-rank merge to configure, and passing it anyway
will fail validation. `recipes/launch.sh` handles this with a conditional:

```bash
$( [ "${DCP_SIZE}" -gt 1 ] && printf -- "--dcp-comm-backend a2a --dcp-kv-cache-interleave-size 1" || true )
```

The trailing `|| true` matters under `set -e`: at `DCP_SIZE=1` the `[ ... -gt 1 ]` test itself
returns non-zero (false), and without the `|| true` fallback that non-zero exit would kill the
whole script under `set -e` before the argv array was even fully built. This bit the campaign
in production — it's why this guard exists in the launcher at all, not a defensive-coding
nicety.

---

## The KV byte formula

Each rank's KV pool is a fixed-size allocation, `--kv-cache-memory-bytes`, computed as:

```
blocks = ceil(max_model_len / N)
KV_CACHE_MEMORY_BYTES = blocks * 2,046,464
```

where `2,046,464` bytes is the size of one local KV block under this checkpoint's nvfp4
compact-KV layout (all layers, one 64-local-token block, per rank), and `N` is how many
**global** sequence tokens one local block covers at a given DCP degree:

| DCP degree | N (global tokens / local block) |
|---|---|
| DCP1 | 64 (no split — one local block = one 64-token global block) |
| DCP2 | 128 |
| DCP4 | 256 |

Worked examples, checked against the actual measured/booted configs in `recipes/presets.md`:

| DCP | max_model_len | blocks | formula KV_CACHE_MEMORY_BYTES | measured (presets.md) | match? |
|---|---:|---:|---:|---:|---|
| DCP4 | 500,000 | ceil(500000/256)=1,954 | 3,998,790,656 | 3,998,790,656 | exact |
| DCP4 | 750,000 | ceil(750000/256)=2,930 | 5,996,139,520 | 5,996,139,520 | exact |
| DCP4 | 800,000 | ceil(800000/256)=3,125 | 6,395,200,000 | 6,395,200,000 | exact |
| DCP4 | 1,000,000 | ceil(1000000/256)=3,907 | 7,995,534,848 | 7,995,534,848 | exact |
| DCP2 | 500,000 | ceil(500000/128)=3,907 | 7,995,534,848 | 7,995,534,848 | exact |
| DCP2 | 650,000 | ceil(650000/128)=5,079 | 10,393,990,656 | 10,395,482,112 | off by 1,491,456 B |
| DCP2 | 600,000 | ceil(600000/128)=4,688 | 9,593,823,232 | 9,596,774,400 | off by 2,951,168 B |
| DCP1 | 320,000 | ceil(320000/64)=5,000 | 10,232,320,000 | 10,232,320,000 | exact |
| DCP1 | 249,000 | ceil(249000/64)=3,891 | 7,962,791,424 | 7,995,534,848 (reused) | does not match |

The formula is **exact for every DCP4 row and the DCP2 500K row**, and exact for the DCP1
320K row (`N=64` — a single, unsplit local block covers 64 global tokens, the `64 x D`
pattern holding at `D=1`). Two real discrepancies are worth flagging rather than hiding:

- **DCP2 650K and 600K land a few blocks' worth of bytes above the naive formula.** Small,
  consistent, and not explained by anything in this campaign's source review — likely a
  rounding/alignment detail specific to the DCP2 code path (interleave padding is a plausible
  suspect, unverified). Use the measured values in `recipes/presets.md`, not the formula, for
  any DCP2 row above 500K.
- **The DCP1 249K row does not fit the formula at all** — the RUNBOOK explicitly notes this
  KV byte value was **reused verbatim from an earlier "proven-safe" 250K-class allocation**,
  not recomputed for this specific `max_model_len`. It is a legacy constant, not evidence
  against the formula.

**Bottom line: the formula is a genuine derivation for DCP4 (trust it for new DCP4 rungs you
compute yourself), a good approximation for DCP2, and not reliable for DCP1. Always prefer
the measured byte values in `recipes/presets.md` over recomputing when a preset already
exists for the config you want.**

---

## The memory model on UMA

Every DGX Spark GB10 node has **121GB of unified memory** — CPU and GPU see the same physical
RAM (LPDDR5, ~273GB/s), not separate host/device pools connected by PCIe. This changes what
"out of memory" means compared to a discrete-GPU box: there's no fixed VRAM ceiling separate
from host RAM, but there also isn't infinite headroom just because "it's all one pool" — the
OS still needs enough free/reclaimable memory to avoid thrashing, and vLLM's own allocator
guard reads Linux's notion of available memory to decide how much headroom exists before a
risky allocation.

**Budget breakdown at TP4, per rank, on this checkpoint:**

- **Weights: ~94.75GB/rank.** The GLM-5.2 W4A16 checkpoint is 379GiB on disk; sharded 4 ways
  under TP4 (experts and attention heads split across ranks), that's ~94.75GiB resident per
  rank. This is the dominant consumer by a wide margin and is **not** a place to look for
  "hidden bloat" — see `docs/findings.md` for the campaign's own investigation that confirmed
  this number is honest (~2.2GB/rank of non-weight torch residency on top of it, which is lean,
  not fat).
- **Fixed host-reserve: 4GB.** A hard-coded, always-active guard
  (`VLLM_UNIFIED_MEMORY_HOST_RESERVE_BYTES=4294967296`) that refuses to proceed if it would
  eat into this reserve. This is a fail-closed backstop, not a target to shave — it exists
  because host-level OOM (the kernel OOM-killer taking out a Ray worker mid-boot) is a far
  worse failure mode than a clean, logged allocator refusal.
- **KV cache: the `--kv-cache-memory-bytes` pool**, sized per the formula above. This is the
  one budget line you directly control per boot.
- **Activations, CUDA graphs, JIT/compile caches, Ray/NCCL overhead: the remainder.** At
  `max_num_batched_tokens` (mnbt) 2048, this leaves an empirical KV ceiling of roughly
  **10.2-10.4GB/rank**, consistent across DCP1/DCP2/DCP4 — see the README's capacity-frontier
  section for the arithmetic.

**The allocator guard's conservatism, not physics, is frequently the actual wall.** Linux's
page cache grows to fill idle RAM (normal, reclaimable), but `MemAvailable` in
`/proc/meminfo` — which the guard reads — can systematically under-report what's actually
allocatable on GB10's UMA. This was directly demonstrated during the campaign: an older,
unguarded image ran `mnbt=2048` with a **larger** KV allocation than a newer, guarded image
refused on the same physical nodes at the same moment. See `docs/operations.md` for the
compaction (`sync; echo 3 > /proc/sys/vm/drop_caches`) countermeasure and when to reach for it.

---

## The step-time law

Decode step time on this stack fits a simple two-term model:

```
T ≈ 73ms fixed + 9.2ms × (verify tokens per step)
```

with the fixed term essentially batch-invariant (73.4ms at batch 1, 73.6ms at batch 4,
measured from production window telemetry).

- **The 9.2ms/verify-token term is well understood: it's memory-bandwidth-bound MoE expert
  weight streaming**, close to the hardware's ~273GB/s floor (roughly 2.5GB read per
  verification position per rank, at this expert count/size). This is physics, not a
  configuration knob — the only ways to move it are route-coalesced grouped GEMM or expert
  route-locality work, neither of which is part of this recipe.
- **Of the 73ms fixed term, only ~20-25ms is attributed to estimated compute.** Roughly
  **50ms is unaccounted for** — candidates investigated include rank-wait/straggler skew
  inside the ~156 per-step allreduces, NCCL protocol/tuning effects (ruled out — an
  `NCCL_PROTO=LL` pin measured exactly flat on decode), and per-layer GPU-body cost across the
  model's 78 transformer layers. This remains genuinely open; see `docs/findings.md` for what's
  been ruled out and what hasn't.
- **Practical consequence:** decode throughput at a given concurrency is close to a fixed
  function of verify-token count per step (i.e. of the MTP speculation depth actually
  achieved), not something batch-size or chunk-size tuning moves much — which is consistent
  with the matrix's own finding that `mnbt` 2048 vs 4096 barely moves prose/prefill numbers.
  Where you *can* move decode throughput is by changing how many verify tokens land per step
  (deeper/shallower accepted speculation), which is a function of content predictability
  (prose vs "peak") more than of any launcher knob in this recipe.
