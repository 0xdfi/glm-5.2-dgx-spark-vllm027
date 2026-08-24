# Runtime source and patch closure

The source release is rooted at public O14 commit `9ba54db8f7c852eabb90372f7dcf47f87eafbe20`, tree `46e1d7837747c080d7300496abe9aa5f7c0e127d`. [o14-import.json](../runtime/manifests/o14-import.json) records the public inventory; [overlay-files.json](../runtime/manifests/overlay-files.json) records the effective champion selection. No serving host or live image was inspected during this publication, so live-b4 parity remains `not-verified`.

## Effective source selection

- 74 effective vLLM overlay files: the public O14 selection plus the sealed
  DCP2/DCP4 caller and global-top-k updates from the later champion lineage.
- 5 effective B12X overlay files: the 3 public O14 files plus the sealed
  caller-owned two-level fold scratch repair used by the later champion.
- All 5 B12X overlay files are derived from B12X commit
  `334a2d75d166becea0aa640b402d521ea0a290eb`.
- One 14-path native SM121 patch over the pinned vLLM-native tree.
- Build A v0 at SHA-256 `965b6aeaf3a0c41abefec89a31592f6b8061a053053831be7c5e6f7560857515`.
- Build A v1 excluded from the effective runtime.

The imported vLLM closure covers configuration and argument plumbing, DCP groups and global top-k ownership exchange, sparse MLA/indexer paths, compact NVFP4 cache layout and attention, Marlin MoE, adaptive MTP and packed mapping, lm-head rescoring, scheduling, graph capture/warmup, and the donor FlashAttention/MSA/FlashMLA files. The B12X closure carries tiled top-k, compiler, and sparse-MLA scratch changes. The native patch carries the SM121 cache/operator and build integration needed by the Python overlay.

## Provenance and verification

Each distributed target has its hash and origin in the overlay manifest and [source provenance](../compliance/source-provenance.json). Donor roots and immutable commits are first-class entries in [upstream-lock.json](../runtime/manifests/upstream-lock.json), rather than being attributed wholesale to vLLM.

`tools/import_runtime.py --verify-only` recomputes the expected closure from the exact O14 Git object and rejects missing, changed, or extra effective files. `scripts/verify_source.py` rejects Build A v1, a changed native patch, source/model bundling drift, or an undocumented live-b4 claim.

## Generated objects are outside the patch set

CUDA graph captures and JIT cache objects are generated locally from the accepted source build. They are hardware- and process-specific and are never uploaded as portable kernels. A cache from another profile or run is not part of this release and must not be used as evidence of source completeness.
