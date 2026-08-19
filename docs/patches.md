# Patches — what makes big-batch, big-KV DCP4 possible

Three source patches, applied on top of vLLM 0.27.2.dev + the B12X sparse-MLA overlay, turned
DCP4 from "OOMs during profile at any useful batch size" into the DCP4 rows in the speed
matrix. This document describes each patch's mechanism and diff-level shape — apply them
against your own overlay's source tree; exact line numbers will differ by build.

Also covered: the B12X DCP-world-size guard generalization that makes DCP4 legal at all, and
a historical int32-overflow fix that any `max_num_batched_tokens >= 2048` deployment needs.

---

## Patch (a) — profile-phase allocator budget gate removal

**Symptom:** `max_num_batched_tokens` (mnbt) above 512 was unreachable at *any* KV budget,
even with an explicit, generous `--kv-cache-memory-bytes` operator budget set.

**Root cause:** in the GPU worker's memory-determination path, the profile phase passes
`phase_budget_bytes=None` specifically when `phase == "profile"` — the operator-supplied
allocator budget environment variable is **deliberately ignored during the profile pass**.
Consequence: no KV step-down and no explicit budget setting could help a profile-phase OOM,
because the budget value was never consulted there in the first place.

**Fix:** thread the operator budget env through to *all* phases, not just steady-state
serving:

```diff
- phase_budget_bytes = env_budget if phase != "profile" else None
+ phase_budget_bytes = env_budget
```

Unset the env and behavior is byte-identical to stock (the host-reserve fail-closed backstop
is untouched either way — this patch only stops silently discarding an operator's explicit
choice).

---

## Patch (b) — explicit-budget ceiling formula fixed at the real root

**Symptom:** with patch (a) applied and an explicit `ALLOCATOR_BUDGET_BYTES` set, mnbt≥1024
boots *still* OOMed during profile.

**Root cause:** decoding the OOM receipt showed the actual ceiling formula was
`E = min(P, C+I, B)` — where `P` is the pool size, `C` is the current torch-resident baseline
(post weight-load), `I` is the "growth room" derived from `MemAvailable` minus the fixed
reserve, and `B` is the operator budget. Because the formula takes a **min** across all three
terms, an operator-supplied budget `B` could only ever *lower* the effective ceiling relative
to the stock `min(P, C+I)` calculation — it could never raise it past what `MemAvailable`
already implied. On GB10 UMA, `MemAvailable` systematically under-reports true allocatable
memory (reclaimable page cache isn't fully counted the way the guard assumes — see
`docs/how-it-works.md` and `docs/operations.md`), so `I` was often the actual bottleneck, and
no operator budget could route around it.

**Fix:** when a budget is explicitly set, change the formula's shape rather than just its
inputs — relax the ceiling to `E = min(pool - reserve, max(C+I, B))` with a matching
invariant change, so an explicit, deliberate operator budget can raise the ceiling past the
conservative `MemAvailable`-derived estimate, while an *unset* budget still falls back to the
original stock formula byte-for-byte (unit-tested both paths in-image). The 4GiB absolute
host-reserve remains the fail-closed cap in both cases — this patch changes how far above the
conservative estimate an explicit choice can push, not the hard floor.

```diff
- ceiling = min(pool, baseline + growth_room)
+ if budget_bytes is not None:
+     ceiling = min(pool - host_reserve, max(baseline + growth_room, budget_bytes))
+ else:
+     ceiling = min(pool, baseline + growth_room)   # byte-identical to stock
```

**Sizing rule for `ALLOCATOR_BUDGET_BYTES`:** compact first (`docs/operations.md`), read
post-trim `MemAvailable`, subtract the fixed 4GiB reserve, subtract 2GiB slack. This is a
per-cluster, per-boot-time number — do not hardcode a value copied from another cluster's run.

---

## Patch (c) — the phantom page-table fix

This is the patch that actually unblocked the DCP4 rows in the speed matrix. Patches (a) and
(b) made the allocator ceiling *tunable*; this patch fixes the thing that was overflowing it.

**Symptom:** even with a large explicit budget, DCP4 boots at `mnbt >= 1024` OOMed inside the
B12X sparse-attention-indexer's profile-run prewarm, in a `torch.empty()` call allocating a
scratch buffer no runtime request path ever needs.

**Root cause — exact reproduction:** the profile branch computes a synthetic page-table width
as `profile_k_rows = max(max_model_len, total_seq_lens)`, where `total_seq_lens =
max_model_len * 40`. That `40x` multiplier is the **K-gather buffer capacity constant** for
an entirely different (non-B12X) attention path — the profile branch reuses it as a
**page-table width**, which is a unit-mismatched number: no runtime B12X call can ever present
a page table anywhere near that wide, because runtime page tables are bounded by
`ceil(max_model_len / block_size)`, not by `max_model_len * 40`.

At `max_model_len = 750,000`: synthetic width = 30,000,000 tokens = 468,750 pages, chunked at
512 pages/chunk into `ceil(468750/512) = 916` slices. With `q_rows = min(mnbt, 2048) = 1024`
and `topk = 2048`, the fold-scratch allocation is:

```
1024 * 916 * 2048 * 4 bytes = 7,683,964,928 B = 7.16 GiB
```

— matching the observed `torch.OutOfMemoryError: ... Tried to allocate 7.16 GiB` message
exactly. Two such buffers are allocated per profile pass (`fold_values` fp32, `fold_indices`
int32), so the real phantom demand is ~14.3GiB at mnbt 1024 and doubles again at mnbt≥2048.

**Fix:** an env-gated cap (`VLLM_R17_PROFILE_DRAFTER_CAP` in the original build — the name
reflects the task brief that commissioned the fix, not the actual site, which is the indexer
profile prewarm, not the MTP drafter) that caps the synthetic profile page-table width at
`max_model_len` instead of `max_model_len * 40`:

```diff
+ def _profile_pagetable_cap_enabled() -> bool:
+     return os.environ.get("VLLM_R17_PROFILE_DRAFTER_CAP", "0").strip().lower() in ("1","true","yes","on")
+
  profile_k_rows = _get_profile_k_rows(max_model_len=max_model_len, total_seq_lens=total_seq_lens)
+ if _profile_pagetable_cap_enabled():
+     profile_k_rows = min(profile_k_rows, max(1, max_model_len))
  warmup_k_rows = _get_profile_warmup_k_rows(profile_k_rows)
```

Default off = byte-identical stock behavior; the runtime (non-profile) code path is untouched
in every case.

**Why this is safe** — the load-bearing claim of the patch: for every consumer downstream of
`profile_k_rows` (the fold scratch, the workspace-scratch reservation, the DCP local-rows
prewarm), **runtime can never present a wider page table than the capped rehearsal width**,
because runtime page tables are bounded by `ceil(max_model_len / block_size)` per request —
strictly narrower than even the *capped* `max_model_len`-wide rehearsal, let alone the
uncapped `40x` one. The capped profile pass still proves peak headroom for every shape
runtime can actually request; it just stops proving headroom for shapes runtime can never
reach.

Expected profile-phase arithmetic after the cap, vs. stock:

| max_model_len | mnbt | capped fold pair | stock fold pair |
|---:|---:|---:|---:|
| 750,000 | 1024 | 0.72 GiB | 14.31 GiB |
| 750,000 | 2048 | 1.44 GiB | 28.62 GiB |
| 1,000,000 | 2048 | 1.91 GiB | 38.15 GiB |
| 1,000,000 | 4096 | 1.91 GiB | 38.15 GiB |

This is the single highest-leverage patch of the three — without it, no DCP4 row above
`mnbt=512` in the speed matrix would have booted at all.

---

## The B12X DCP-world-size guard generalization

**What it is:** the B12X sparse-MLA overlay ships a fail-closed guard,
`_assert_dcp_world_size_supported` (naming varies by build), that hard-raises unless
`decode_context_parallel_size` is in an explicit allowlist. In the upstream-adjacent overlay
this recipe descends from, that allowlist originally contained only `{2}` — DCP4 was not a
rank-math or topology impossibility (`tp % dcp == 0` passes fine at `4 % 4 == 0`; see
`docs/how-it-works.md`), it was a **one-constant policy guard**, deliberately fail-closed
because 4-way merge correctness (stable-tie handling, cross-rank candidate packing at 4 ranks
instead of 2, graph-capture/prewarm paths at world size 4) had never been audited.

**What changed:** the allowlist was generalized to `{2, 4}` *after* auditing the merge body
(`_merge_b12x_dcp_topk` and its workspace-sizing math) and confirming it was already
`dcp_world_size`-generic in its arithmetic — the guard was stricter than the code underneath
it needed. A second, easy-to-miss guard site exists in `sparse_utils.py`
(`triton_convert_dcp_local_topk_to_global` or equivalently named topk-conversion helper) that
is **not** covered by the main indexer file's guard and will crash the first DCP4 decode step
if missed — patch both sites together, not just the primary one.

```diff
- _DCP_SUPPORTED_WORLD_SIZE = 2
+ _DCP_SUPPORTED_WORLD_SIZE = (2, 4)
```

applied at both the primary indexer guard and the `sparse_utils.py` conversion-guard site.

**Do not stop at the constant edit.** Changing the allowlist alone makes DCP4 *run*; it does
not by itself prove 4-way stable-tie and CuteDSL/graph-prewarm correctness. Follow the constant
edit with an exact-match correctness gate (byte-faithful outputs vs. a trusted DCP2 baseline
at temperature 0 on fixed prompts) before trusting any DCP4 speed number — this is what the
campaign did before running the battery that produced the DCP4 rows in the matrix.

Also required for a real DCP4 build: a **fresh CuteDSL/JIT cache directory**.
`dcp_world_size` and `dcp_rank` are part of the B12X prewarm cache key — pointing a DCP4 boot
at a DCP2 (or any other config's) JIT cache directory is not just wasteful, it risks a
cross-config JIT-poisoning failure. `recipes/launch.sh` uses a `JIT_ROOT` keyed by `RUN_ID`
specifically to make this mistake structurally hard to make.

---

## The int32-overflow tiled_topk fix

**Symptom (historical):** any batch/chunk configuration that pushed `q_rows` at or above 2048
in the B12X tiled top-k kernel would silently corrupt or crash — a big-batch ceiling that
predates this campaign's DCP work entirely.

**Root cause:** the tiled top-k kernel's row-major flat-index arithmetic,
`q_rows * 1,050,624`, overflows a signed 32-bit integer once `q_rows >= 2048` (`2048 *
1,050,624 = 2,151,677,952`, just over `INT32_MAX = 2,147,483,647`). Below 2048 rows the
arithmetic stayed in range and the bug was invisible; every config in this repo's speed
matrix that runs `max_num_batched_tokens >= 2048` depends on this fix being present.

**Fix:** widen the index arithmetic to int64 and use a static kernel descriptor instead of a
dynamically-computed one that could re-trigger the same class of overflow elsewhere:

```diff
- flat_idx = q_row_idx * 1050624 + col_idx      # int32, overflows at q_rows >= 2048
+ flat_idx = q_row_idx.to(tl.int64) * 1050624 + col_idx
```

If you are building your own image against a different overlay lineage, **verify this fix is
present before trusting any row in this repo's matrix with `mnbt >= 2048`** — it is easy to
inherit an image that has the DCP-era memory patches above but predates this one, since they
were fixed independently and at different times.
