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

Fast uses one fleet-uniform unified-memory allocator floor derived from the weakest verified host after explicitly approved idle compaction, less 4 GiB plus a further 2 GiB operator deduction. The latter is a floor-derivation margin, not guaranteed free memory. The exact R17 function treats the explicit, legacy-named byte setting as `F` in `E=min(T-R, max(C+max(0,A-R), F))`, where `T` is the smaller of CUDA-visible and host total, `R` is the hard reserve, `C` is current allocator residency, and `A` is fresh host availability. The controller passes the same `F` and Fast profile-phase drafter cap to every container before Ray starts and reads both back before allowing worker creation. Balanced retains the upstream runtime defaults, while Large retains its per-host computed floor and profile-phase drafter cap.

The accepted R17 live receipts recorded `F=requested=applied=119605592064` bytes on every rank. Rank 0's `A=18183303168`, `R=4294967296`, and `C=104014544896` yielded a conservative allowance of `117902880768`, so the explicit floor raised it to the observed receipt. The runtime still bounded that result by `T-R`; the explicit value was not an independent hard cap.

The effective overlay includes the exact operator-observed `vllm/envs.py`, `vllm/utils/mem_utils.py`, and `vllm/v1/worker/gpu_worker.py` derivatives that parse the byte setting and wire it into integrated-GPU allocator phases. The manifest binds those files to both their observed target hashes and their bases in pinned vLLM commit `6e448d0ea9bf3d88d898b65449ca6dc2aec170ac`. That three-file source reconciliation is narrower than live-image parity and does not change the public reproduction status.

## Decode and speculative execution

Build A v0 replaces two small MLA absorb/decode batch matrix multiplies on its supported shape/stride envelope and falls back outside it. Its source SHA-256 is `965b6aeaf3a0c41abefec89a31592f6b8061a053053831be7c5e6f7560857515`. Build A v1 is deliberately excluded from the effective champion.

Adaptive MTP selects depths 2, 4, or 5 within a 32-step window. The overlay also carries packed-drafter mapping, acceptance-length control, top-64 lm-head rescoring, scheduler, and graph warmup integration. Static tests prove source contracts; real first-fire evidence is still required before a live claim can pass.

## CUDA graphs and JIT caches

The repository publishes capture sizes and integration source, not captured graph objects. CUDA graphs and CuteDSL/Triton JIT caches depend on the hardware, driver, process addresses, imported modules, and exact preset. They are generated locally in a fresh per-run/per-profile cache and are never treated as portable release artifacts.

Generated-cache presence is not success by itself. Live verification must bind the graph census, Build A first fire, sidecar first fire, effective EngineCore configuration, and current run identity to the accepted image/model/preset subject.
