# Credits and research lineages

Copyright 2026 dfi / 0xdfi

This file records broader technical and community lineages. It is not a license
grant, a statement that every referenced work is distributed here, or a claim
of authorship beyond the cited evidence.

## Runtime foundations

- The [vLLM project and contributors](https://github.com/vllm-project/vllm)
  provide the serving-system foundation discussed throughout this repository.
  The retained MIT documentation cites specific vLLM pull requests and issues
  where they informed the investigation.
- The B12X/local-inference-lab lineage and Luke Alonso are acknowledged for the
  sparse-attention runtime lineage used by the champion work. Three modified
  B12X files are now distributed over commit
  `334a2d75d166becea0aa640b402d521ea0a290eb`; their exact origins and hashes
  are recorded in the source manifests.
- vllm-project/flash-attention, MiniMax's MSA, and DeepSeek's FlashMLA are
  credited for the BSD-3-Clause and MIT donor source retained in the vLLM
  overlay. Their immutable source commits and file mapping are recorded in
  `runtime/manifests/upstream-lock.json` and `runtime/manifests/overlay-files.json`.

## DGX Spark community work

- The existing README identifies CosmicRaisins, penguinchang, tonyd2wild, and
  danielwoz with the
  [GLM-5.2 QuantTrio TP4/DCP2 repository](https://github.com/joesinvestments/GLM-5.2-QuantTrio-TP4-DCP2-4x-DGX-Spark).
- It also cites [ciprianveg/gb10-vllm](https://github.com/ciprianveg/gb10-vllm)
  and [drowzeys/vllm-gb10-spin-wait-fix](https://github.com/drowzeys/vllm-gb10-spin-wait-fix)
  as community engineering references.

## External benchmark contribution

- Jordan Miller (`ecohash-co` / `jdmpike`) authored external PR #2 commit
  [`525b210dc100b75792ac5a0818d4eab0cdddc5a8`](https://github.com/0xdfi/glm-5.2-dgx-spark-vllm027/commit/525b210dc100b75792ac5a0818d4eab0cdddc5a8),
  which lists Claude Fable 5 as co-author. `benchmarks/probe.py` integrates and
  hardens that contribution while retaining Jordan Miller's MIT license and
  authorship. The upstream Git blob is
  `cec454c44bb17e98fd721e75a0e801963347a7e0`; the exact raw SHA-256 and
  integrated target SHA-256 are recorded in the provenance ledger.
