# How the champion runtime works

This page describes the published source selection. It does not claim that the checkout has reproduced the historical live-b4 service; that parity remains `not-verified` in [the O14 import manifest](../runtime/manifests/o14-import.json).

## DCP and the three profiles

Decode-context parallelism shards KV ownership across ranks while tensor parallelism stays at four. `TP % DCP == 0`, so the supported DCP values on TP4 are 1, 2, and 4.

- DCP1 keeps one request's decode KV local and is the Fast 320K profile.
- DCP2 splits ownership across pairs and uses the `a2a` exchange with interleave 1 for Balanced 650K.
- DCP4 splits ownership across all ranks and uses the same exchange/interleave for Large 1M.

All three profiles share TP4 / PP1 / PCP1, `max_num_seqs=4`, `max_num_batched_tokens=2048`, decode-prefill budget 1024, FULL graphs `6,12,18,24`, adaptive MTP depths `2,4,5`, and effective `fuse_gemm_comms=false`. The machine contracts under [presets](../presets) are authoritative.

## Compact NVFP4 sparse-MLA KV

The public compact ABI stores 368 bytes/token. A 64-token page therefore occupies 23,552 bytes. The published runtime overlay carries the cache layout, concat/cache operator binding, sparse-indexer plumbing, and NVFP4 attention paths; their exact target hashes are listed in [overlay-files.json](../runtime/manifests/overlay-files.json).

The explicit per-rank budgets are profile contracts rather than allocator guesses:

| Profile | KV bytes/rank |
|---|---:|
| Fast | 10232320000 |
| Balanced | 10395482112 |
| Large | 7995534848 |

Large additionally uses a per-host computed unified-memory allocator budget and a profile-phase drafter cap. That mechanism avoids proving impossible profile shapes while preserving the deployed runtime shapes.

## Decode and speculative execution

Build A v0 replaces two small MLA absorb/decode batch matrix multiplies on its supported shape/stride envelope and falls back outside it. Its source SHA-256 is `965b6aeaf3a0c41abefec89a31592f6b8061a053053831be7c5e6f7560857515`. Build A v1 is deliberately excluded from the effective champion.

Adaptive MTP selects depths 2, 4, or 5 within a 32-step window. The overlay also carries packed-drafter mapping, acceptance-length control, top-64 lm-head rescoring, scheduler, and graph warmup integration. Static tests prove source contracts; real first-fire evidence is still required before a live claim can pass.

## CUDA graphs and JIT caches

The repository publishes capture sizes and integration source, not captured graph objects. CUDA graphs and CuteDSL/Triton JIT caches depend on the hardware, driver, process addresses, imported modules, and exact preset. They are generated locally in a fresh per-run/per-profile cache and are never treated as portable release artifacts.

Generated-cache presence is not success by itself. Live verification must bind the graph census, Build A first fire, sidecar first fire, effective EngineCore configuration, and current run identity to the accepted image/model/preset subject.
