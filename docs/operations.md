# Operations and fail-closed launch discipline

The controller treats the four Spark nodes as production infrastructure. It does not remove unknown containers, reuse an ambiguous run identity, accept mixed image/model identities, or call an HTTP health response sufficient proof.

## Before any launch

Use a fresh mode-0600 `cluster.env` containing four symbolic SSH targets, unique fabric addresses, declared RDMA interface/HCA/device paths, an exact local image ID, external checkpoint and sidecar locks, API port, storage root, and a fresh run identity. The first `CHAMPION_HOSTS` entry is rank 0 and owns the canonical remote lock. For this Spark fleet, `CHAMPION_FLEET_LOCK` is `/home/dfi/.cache/nous-fleet-mutation.lock`, shared with recovery, freeze, and other serving mutations. Credentials do not belong in this file.

`./bin/champion verify-hosts PROFILE` requires:

- four reachable AArch64 hosts, one GB10/SM121 device each, and 20 online CPUs;
- active declared RDMA devices and a free target API port;
- no live GPU compute application or protected co-resident container;
- the exact image, checkpoint, and sidecar identities on every host;
- at least 116 GiB `MemAvailable`, peer spread no greater than 512 MiB, swap use no greater than 6 GiB, and polkit RSS below 128 MiB;
- for Fast, the declared big-core topology on memory node 0.

Memory compaction is a maintenance-window operation, not a blanket "cache dropping is always safe" recipe. Investigate ownership, active services, dirty pages, swap, polkit growth, and guardian state first. Never run a co-resident compiler or container build at a tight serving profile; use an isolated builder. Fast therefore refuses to compact or launch unless the operator supplies `--compaction-nonce` with a fresh 32-hex value. Generate it at invocation time as shown in the README. The inherited `CHAMPION_APPROVE_MEMORY_COMPACTION` boolean cannot authorize a launch, and persisting it in `cluster.env` remains rejected.

The launcher validates the nonce format and the protected-container pattern as a POSIX ERE before any SSH work. It then opens a persistent SSH session to rank 0, acquires `flock -n` on the canonical path, and keeps that descriptor live from before preflight through final acceptance or rollback. Lock contention, connection loss, or an indeterminate lock state blocks mutation. `bin/champion stop` and authenticated `stage-image` use the same path.

While that lock is held, Fast completes the non-memory safety preflight. GPU-application, target-port, Docker-container, and protected-pattern queries are fail-closed. Protected matching distinguishes grep status 0 (protected match), 1 (no match), and greater than 1 (query error); only status 1 can proceed. Under the same lock, rank 0 atomically creates a receipt directory named by a domain-separated SHA-256 of the nonce. A replay is refused, the raw nonce is never sent in the remote command or written to logs/paths, and an ambiguous or failed attempt requires a new nonce. Only after that receipt is consumed does the launcher run `sync`, drop clean page cache on each idle host, and sample every host again.

Every post-compaction value must be numeric and at least 116 GiB, and the peer spread must remain no greater than 512 MiB. The controller computes exactly one floor from the weakest host:

```text
min(post-compaction MemAvailable KiB) * 1024 - 4 GiB - 2 GiB
```

Both subtractions derive `F`; neither makes the final 2 GiB deduction a guaranteed free-memory reserve. The same positive floor is attached to all four containers using the byte-observed legacy environment name `VLLM_UNIFIED_MEMORY_ALLOCATOR_BUDGET_BYTES`. The observed R17 arithmetic is:

```text
T = min(CUDA-visible total, host total)
A = fresh host MemAvailable
R = VLLM_UNIFIED_MEMORY_HOST_RESERVE_BYTES = 4294967296
C = max(current torch allocated, current torch reserved)
F = explicit allocator floor
E = min(T - R, max(C + max(0, A - R), F))
```

`F` can therefore raise the conservative `C+max(0,A-R)` allowance, but `T-R` remains the hard cap. The accepted live receipts on ranks 0–3 all reported `F=requested=applied=119605592064`. Rank 0 reported `A=18183303168` and `C=104014544896`, making its conservative allowance `117902880768` before the floor was applied. Fast disables the load-phase refresh with `VLLM_UNIFIED_MEMORY_LOAD_CEILING=0`; profile, KV, and graph refreshes remain active. Balanced keeps upstream defaults, and Large uses a computed per-host floor. The fleet floor is never recomputed independently after ranks begin starting.

Run a fabric preflight before a maintenance launch: link state, MTU/GID/HCA selection, and a bidirectional microbenchmark between the declared peers. A link being `ACTIVE` is necessary but does not prove the expected bandwidth or that NCCL selected IB.

## Launch and readiness

The launcher holds the canonical rank-0 lock, creates only exact run-ID containers, uses a fresh per-profile cache, starts four Ray ranks, and checks:

1. four exact container/rank identities with no restart or OOM state;
2. Ray and API processes plus IB-only NCCL selection;
3. expected EngineCore settings and current-window logs without fatal markers;
4. `/health`, a real generation response, and for Fast a cache-busted semantic completion whose reported prompt length is greater than 2,048 tokens;
5. model, graph, Build A, sidecar, MTP, and runtime evidence when producing a promotion receipt.

`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is part of the common container environment, so Ray and every worker inherit it at process creation. Before Ray starts, the controller reads every container environment back and requires exactly one matching allocator setting. A computed Fast floor must also be present exactly once and equal on all ranks, and the Fast drafter profile cap must read back as enabled. Any mismatch leaves the exact containers preserved for diagnosis and blocks Ray startup.

Keep logs and connection snapshots scoped to the exact run identity and the current attempt window. Historical warnings from another attempt cannot fail a run, and old clean logs cannot rescue the current one.

## Fast post-health CPU pin

Fast deliberately boots unpinned. Only after health and real generation pass does the controller update all four containers to big cores `5-9,15-19` and memory node 0. It reads the applied state back and repeats health plus generation.

If any update or recheck fails, rollback is explicit: restore CPUs `0-19` and memory node 0 on all four exact containers, then prove health and generation again. A healthy unpinned rollback is preserved for diagnosis but cannot satisfy the Fast champion pin-state gate. There is no create-time cpuset.

## Stress gates

Every context request is followed by health and idle checks. The profile suite is:

- Fast: immediate generation, C4, 8K, 65K, 240K, and at least 299K prompt tokens.
- Balanced: immediate generation, C4, and at least 600K prompt tokens.
- Large: immediate generation, C4, and at least 900K prompt tokens.
- All: five minutes idle followed by another real generation.

The exact image/build/model/sidecar/preset/run subject must remain constant. A profile stress failure blocks that live claim; it is not reclassified as a benchmark outlier.

## Monitoring and recovery

Monitor host headroom, swap trend, polkit RSS, guardian/controller health, GPU processes, rank restart/OOM state, Ray membership, bounded semantic API generation, IB selection, and current-window fatal markers. `/health`, `/v1/models`, container state, and Ray membership can all remain green during a wedged inference path. A production watchdog should therefore use idle-only bounded completions plus corroborating current-window signals and fail closed when identity or workload state is uncertain; the portable launcher does not include a site-specific recovery service. Preserve the last accepted image and exact run identity for rollback. `bin/champion stop RUN_ID` stops only owned ranks in reverse order and preserves containers/images for diagnosis.
