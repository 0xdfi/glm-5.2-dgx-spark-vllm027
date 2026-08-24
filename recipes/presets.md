# Canonical champion presets

The JSON files in [`presets/`](../presets/) are authoritative. This page is a navigation aid, not a second configuration implementation. Render the complete ordered vLLM argv, container environment, and expected EngineCore view with:

```bash
./bin/champion render fast
./bin/champion render balanced
./bin/champion render large
```

| Profile | DCP | Max length | KV bytes/rank | Seqs | Mnbt | DPTB | Graphs | Profile-only behavior |
|---|---:|---:|---:|---:|---:|---:|---|---|
| Fast 320K | 1 | 320000 | 10232320000 | 4 | 2048 | 1024 | 6,12,18,24 | boot unpinned; after readiness pin CPUs `5-9,15-19`, memory node `0`; rollback to CPUs `0-19`, memory node `0` |
| Balanced 650K | 2 | 650000 | 10395482112 | 4 | 2048 | 1024 | 6,12,18,24 | DCP `a2a`, KV interleave `1`, no CPU pin |
| Large 1M | 4 | 1000000 | 7995534848 | 4 | 2048 | 1024 | 6,12,18,24 | DCP `a2a`, KV interleave `1`, per-host computed allocator, profile cap `1` |

All three use TP4/PP1/PCP1, FULL CUDA graphs, adaptive MTP depths `2,4,5` with maximum 5 and window 32, `nvfp4_ds_mla` KV, `B12X_MLA_SPARSE`, Marlin MoE, prefix caching, async scheduling, and the W8 sidecar requirement. `fuse_gemm_comms` is false.

The current published checkpoint and sidecar locks are deliberately blocked: their complete whole-file checksums and governing rights are not yet established. The repository does not contain model weights and does not convert partial hashes into artifact identities.

Baseline values remain DPTB `1024`, `VLLM_B12X_MLA_SPEC_EXTEND_AS_DECODE=0`, Marlin atomic-add enabled, and `NCCL_BUFFSIZE=4194304`. Experimental speed-hunt arms are not champion defaults.

Site-specific host names, fabric addresses, device names, storage paths, and accepted image IDs belong only in the operator's untracked `cluster.env`; none is hard-coded here.
