# Operations and fail-closed launch discipline

The controller treats the four Spark nodes as production infrastructure. It does not remove unknown containers, reuse an ambiguous run identity, accept mixed image/model identities, or call an HTTP health response sufficient proof.

## Before any launch

Use a fresh mode-0600 `cluster.env` containing four symbolic SSH targets, unique fabric addresses, declared RDMA interface/HCA/device paths, an exact local image ID, external checkpoint and sidecar locks, API port, storage root, and a fresh run identity. Credentials do not belong in this file.

`./bin/champion verify-hosts PROFILE` requires:

- four reachable AArch64 hosts, one GB10/SM121 device each, and 20 online CPUs;
- active declared RDMA devices and a free target API port;
- no live GPU compute application or protected co-resident container;
- the exact image, checkpoint, and sidecar identities on every host;
- at least 116 GiB `MemAvailable`, peer spread no greater than 512 MiB, swap use no greater than 6 GiB, and polkit RSS below 128 MiB;
- for Fast, the declared big-core topology on memory node 0.

Memory compaction is a maintenance-window operation, not a blanket "cache dropping is always safe" recipe. Investigate ownership, active services, dirty pages, swap, polkit growth, and guardian state first. Never run a co-resident compiler or container build at a tight serving profile; use an isolated builder.

Run a fabric preflight before a maintenance launch: link state, MTU/GID/HCA selection, and a bidirectional microbenchmark between the declared peers. A link being `ACTIVE` is necessary but does not prove the expected bandwidth or that NCCL selected IB.

## Launch and readiness

The launcher acquires an ownership lock, creates only exact run-ID containers, uses a fresh per-profile cache, starts four Ray ranks, and checks:

1. four exact container/rank identities with no restart or OOM state;
2. Ray and API processes plus IB-only NCCL selection;
3. expected EngineCore settings and current-window logs without fatal markers;
4. `/health` and a real generation response;
5. model, graph, Build A, sidecar, MTP, and runtime evidence when producing a promotion receipt.

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

Monitor host headroom, swap trend, polkit RSS, guardian/controller health, GPU processes, rank restart/OOM state, Ray membership, API generation, IB selection, and current-window fatal markers. Preserve the last accepted image and exact run identity for rollback. `bin/champion stop RUN_ID` stops only owned ranks in reverse order and preserves containers/images for diagnosis.
