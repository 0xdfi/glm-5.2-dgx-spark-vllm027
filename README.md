# GLM-5.2 champion runtime for 4× DGX Spark

This repository publishes the locked source-build inputs, manifests, canonical presets, and BuildKit path for a GLM-5.2 vLLM 0.27 runtime on four DGX Spark GB10 nodes. It is a source publication in progress, not a claim that the public checkout has reproduced the live champion.

Current boundary: `source-complete` is pending until a trusted clean-public-checkout attestation exists. No trusted clean ARM64 build, live-b4 parity proof, exact-model rights evidence, four-node live launch, benchmark reproduction, bit reproducibility result, or anonymous readback has passed. Fast, Balanced, and Large are historical live-b4 references and are **documented-not-reproduced** for this source build. See the generated [release status](docs/release-status.md).

The exact checkpoint and W8 lm-head sidecar are not bundled. Both are `external-required`; their whole-artifact hashes are incomplete and their rights are unknown. This is not permission to obtain, use, or redistribute them. No GHCR convenience image is published, and `bin/champion pull` remains unavailable until component, registry, build, and readback gates pass.

## What is published

- Public-source Docker build path: [docker/Dockerfile.champion](docker/Dockerfile.champion), [build lock](runtime/manifests/build-lock.json), source overlays, native patch, and hash-locked runtime requirements.
- Exact public O14 import root and effective selection: [O14 import](runtime/manifests/o14-import.json) and [overlay manifest](runtime/manifests/overlay-files.json). Three allocator-related vLLM files are byte-reconciled to one operator-observed 2026-08-24 live champion container; that source observation is not complete image parity or a public live reproduction, so live-b4 reconciliation remains `not-verified`.
- Three canonical presets: [Fast](presets/fast-320k.json), [Balanced](presets/balanced-650k.json), and [Large](presets/large-1m.json).
- Fail-closed controller, static/runtime/correctness evaluators, benchmark evaluator, provenance ledger, license bundle, and release gates.
- Build A v0, the measured decode-path Triton BMM kernel, at SHA-256 `965b6aeaf3a0c41abefec89a31592f6b8061a053053831be7c5e6f7560857515`. Build A v1 is excluded.

CUDA graphs and JIT caches are generated locally for the exact hardware, process, address layout, and preset. They are never uploaded or represented as portable artifacts.

## Quickstart

These commands describe the public-source path. `verify-hosts` and `launch` fail closed until the operator supplies a complete mode-0600 site configuration, exact external artifact locks, locally available artifacts, and an accepted image ID. No command silently substitutes a private image or unknown model.

```bash
git clone https://github.com/0xdfi/glm-5.2-dgx-spark-vllm027.git
cd glm-5.2-dgx-spark-vllm027

./bin/champion bootstrap
./bin/champion configure
# Edit cluster.env: four symbolic SSH targets, four fabric addresses, RDMA devices,
# exact local model/sidecar locations and locks, image ID, API endpoint, fresh run ID.

./bin/champion verify-hosts fast
./bin/champion build --tag glm52-champion:local --receipt build-produced.receipt.json
COMPACTION_NONCE="$(./.venv/bin/python -c 'import secrets; print(secrets.token_hex(16))')"
./bin/champion launch fast --compaction-nonce "$COMPACTION_NONCE"
unset COMPACTION_NONCE
./bin/champion verify fast --static
```

The static verification command proves source/AST contracts only. A live promotion additionally requires authenticated runtime, correctness, stress, pin-state, and benchmark evidence. The launcher itself requires health, four ranks, IB transport, current-attempt log checks, and a real generated response before readiness; an HTTP health response alone is insufficient.

Legacy [recipes/launch.sh](recipes/launch.sh) and [recipes/presets.md](recipes/presets.md) are retained as MIT-licensed wrappers and campaign history. The canonical machine-readable controller and presets live under `bin/`, `scripts/`, and `presets/`.

## Canonical profiles

All profiles use TP4 / PP1 / PCP1, four sequences, `mnbt=2048`, DPTB `1024`, FULL CUDA graphs at `6, 12, 18, 24`, adaptive MTP depths `2, 4, 5`, `nvfp4_ds_mla`, B12X sparse MLA, Marlin MoE, prefix caching, and async scheduling. Effective `fuse_gemm_comms` is false. DPTB 256 and speculative-extend changes are experiments, not champion defaults.

| Profile | DCP | Max model length | KV bytes/rank | DCP details | Host action |
|---|---:|---:|---:|---|---|
| Fast (`fast-320k`) | 1 | 320000 | 10232320000 | no DCP exchange; fleet-uniform computed allocator floor | one-use approved idle compaction; boot unpinned; post-health pin to big cores with verified rollback |
| Balanced (`balanced-650k`) | 2 | 650000 | 10395482112 | `a2a`, interleave 1 | none |
| Large (`large-1m`) | 4 | 1000000 | 7995534848 | `a2a`, interleave 1 | per-host allocator floor and profile-only drafter cap |

Large is the 1M-at-2048 profile. There is no 1M-at-4096 champion, no DPTB=256 champion, no `fuse_gemm_comms=true` promotion, and no create-time CPU cpuset.

## 2026-08-24 Fast allocator and liveness repair

A historical Fast production run stopped completing requests while `/health`, `/v1/models`, Docker, and Ray still appeared healthy. The failing request crossed the 2,048-token prefill boundary and required two 320 MiB sparse-indexer fold buffers. The runtime's explicit unified-memory floor was absent on the recovery path, leaving the weakest rank just below the transient-allocation requirement. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` had also reached only the rank-0 API process after Ray started.

The public Fast launcher now requires a fresh one-use nonce for idle all-node cache compaction, samples all four hosts afterward, and supplies one identical floor to every rank:

```text
min(post-compaction MemAvailable bytes across all four hosts)
  - 4 GiB floor-derivation reserve term
  - 2 GiB additional operator deduction
```

That launcher subtraction derives a floor; it does **not** guarantee 2 GiB remains free. In the observed R17 runtime, with `T=min(CUDA total, host total)`, fresh host availability `A`, hard host reserve `R`, current allocator baseline `C=max(allocated,reserved)`, and explicit floor `F`, the requested allowance is `E=min(T-R, max(C+max(0,A-R), F))`. Thus `F` may raise the conservative availability-derived allowance, while `T-R` remains the hard cap. The accepted live receipts on all four ranks recorded `F=requested=applied=119605592064` bytes. Rank 0 recorded `A=18183303168`, `C=104014544896`, and conservative `C+(A-R)=117902880768`, so the floor won exactly.

The launcher injects `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` and the Fast drafter profile cap into each container before Ray or any worker starts, then reads the allocator floor environment back before starting Ray. Fast pins the runtime hard host reserve to `4294967296` bytes and disables the load-phase refresh only; profile, KV, and graph refreshes remain active. Balanced retains its existing defaults; Large uses a computed per-host floor. Readiness also includes a cache-busted semantic completion with more than 2,048 prompt tokens, plus the immediate and post-soak completions; process-only health cannot pass on its own.

The exact observed `envs.py`, `utils/mem_utils.py`, and `v1/worker/gpu_worker.py` files that parse and consume the legacy-named `VLLM_UNIFIED_MEMORY_ALLOCATOR_BUDGET_BYTES` floor control are included as Apache-2.0 vLLM derivatives. Their target hashes, pinned upstream base hashes, and limited container-source observation are recorded in the runtime manifests. This closes the published source path for the allocator control without claiming the whole live image or service was publicly reproduced.

Memory compaction is deliberately opt-in because dropping host page cache affects the whole node. The launcher validates the nonce and protected-container ERE before SSH, then holds a persistent `flock` at `CHAMPION_FLEET_LOCK` on the first `CHAMPION_HOSTS` entry from preflight through acceptance or rollback. Under that lock it atomically consumes only a domain-separated SHA-256 of the nonce on rank 0; replay or lock-liveness loss blocks compaction. The nonce itself is not written to logs or receipt paths, and every retry requires a fresh value. `stop` and authenticated image staging use the same canonical lock. This repair records the historical incident and hardens the public launcher; it does not change this checkout's `documented-not-reproduced` source-build status.

## Historical reference measurements

These are references from the earlier live-b4 campaign, not measurements of the source build in this checkout. Units are server-counter tokens/second; columns are prose C1, prose C4 aggregate, peak C1, peak C4 aggregate, and cold prefill.

| Profile | Prose C1 | Prose C4 | Peak C1 | Peak C4 | Prefill |
|---|---:|---:|---:|---:|---:|
| Fast | 26.565 | 55.95 | 40.47 | 81.02 | 694.1 |
| Balanced | 23.78 | 52.56 | 34.965 | 72.59 | 585.3 |
| Large | 22.25 | 48.775 | 36.60 | 73.23 | 422.8 |

The promotion protocol requires exactly three valid complete batteries within five attempts, 95% median floors, and 90% per-battery floors. It retains every attempt and never reclassifies a completed slow response as invalid. See [benchmark protocol](benchmarks/protocol.md) and [baseline contract](benchmarks/baselines.json).

## Documentation

- [How it works](docs/how-it-works.md): DCP, NVFP4 KV geometry, graphs, and runtime selection.
- [Runtime patches](docs/patches.md): exact overlay/native closure and Build A selection.
- [Operations](docs/operations.md): host preflight, headroom, launch, pin rollback, and stress gates.
- [Reproducibility](docs/reproducibility.md): source, build, live, benchmark, and readback evidence ladder.
- [Licensing](docs/licensing.md): Apache/MIT/BSD attribution plus separate model and OCI boundaries.
- [Findings](docs/findings.md): historical campaign findings and experiments not promoted.

## Attribution and license

New champion-publication work is Copyright 2026 dfi / 0xdfi and Apache-2.0. Existing recipe documentation remains MIT; imported and derived files retain their Apache-2.0, MIT, or BSD-3-Clause terms. [CREDITS.md](CREDITS.md), [NOTICE](NOTICE), [REUSE.toml](REUSE.toml), and [the provenance ledger](compliance/provenance.json) record exact boundaries.

Upstream and community lineages include vLLM, B12X, FlashAttention, MiniMax MSA, DeepSeek FlashMLA, QuantTrio community work, and the cited DGX Spark repositories and pull requests. Jordan Miller's MIT benchmark contribution at commit `525b210dc100b75792ac5a0818d4eab0cdddc5a8` is retained with authorship, and Claude Fable 5 is credited as the recorded co-author. See [CREDITS.md](CREDITS.md).

Model artifacts and OCI redistribution are separate legal and integrity decisions. Apache-2.0 coverage for original source does not grant model rights and does not clear a container image for GHCR.
