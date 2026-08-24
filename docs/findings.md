# Historical campaign findings

These findings describe the earlier live-b4 campaign. They are references for the public-source reproduction effort, not fresh measurements from this checkout.

## Canonical profile references

| Profile | Prose C1 | Prose C4 | Peak C1 | Peak C4 | Prefill |
|---|---:|---:|---:|---:|---:|
| Fast 320K / DCP1 | 26.565 | 55.95 | 40.47 | 81.02 | 694.1 |
| Balanced 650K / DCP2 | 23.78 | 52.56 | 34.965 | 72.59 | 585.3 |
| Large 1M / DCP4 | 22.25 | 48.775 | 36.60 | 73.23 | 422.8 |

The current promotion floors are machine-readable in [baselines.json](../benchmarks/baselines.json). New measurements must use the complete protocol and trusted evidence before they can replace these references.

## What was promoted

- `max_num_batched_tokens=2048` is the common champion value. Historical 4096 rows did not justify a 4096 champion; Large is 1M at 2048.
- DPTB remains 1024. A DPTB=256 arm was investigated later but was not promoted.
- Effective `fuse_gemm_comms` remains false because the public configuration path did not prove a true effective enablement or benefit.
- Fast post-health big-core pinning is retained as a guarded action with rollback. The historical samples suggested lower variance, but they are not a new source-build reproduction.
- Build A v0 is selected. Build A v1 is excluded pending its own correctness and live evidence.

## Capacity and DCP

Within a DCP degree, additional KV was much cheaper than changing DCP degree. DCP2/DCP4 pay communication and sparse-indexer merge costs in exchange for larger context. The public compact NVFP4 sparse-MLA ABI is 368 bytes/token, or 23,552 bytes per 64-token page; explicit per-rank budgets remain part of each preset.

## Determinism remains an open runtime property

Temperature zero did not guarantee byte-identical output in the historical campaign. Atomic reductions, batching/scheduling, and speculative paths remain plausible sources. The current battery starts with an exact determinism triplet and retains divergence evidence, but a spot check is diagnostic rather than a bit-reproducibility claim.

Bit reproducibility has its own blocked publication row and requires two distinct trusted clean builds plus normalized comparison evidence. It cannot be inferred from temperature-zero generation.
