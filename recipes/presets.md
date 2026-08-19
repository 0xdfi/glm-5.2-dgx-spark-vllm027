# Presets — the 12 measured configurations

Copy-paste env lines for `recipes/launch.sh launch`. Every row here was actually booted and
put through the full probe battery described in `benchmarks/protocol.md` — these are not
computed/estimated configs. Speeds are repeated from the README matrix for convenience;
treat the README as the source of truth if the two ever drift.

All rows share: `MAX_NUM_SEQS=4`, `CUDAGRAPH_CAPTURE_SIZES=6,12,18,24` (the `(MTP_k+1)*seq`
formula for k=5, seqs 1..4 — see `docs/how-it-works.md`), MTP k=5 adaptive depths [2,4,5],
`--kv-cache-dtype nvfp4_ds_mla`, `--attention-backend B12X_MLA_SPARSE --moe-backend marlin`.

`DECODE_PREFILL_TOKEN_BUDGET` follows one consistent rule across the whole campaign:
**`min(MAX_NUM_BATCHED_TOKENS, 1024)`** — i.e. it tracks `mnbt` up to 512, then holds at
1024 for every `mnbt >= 1024` row. This was never swept as its own variable (see
`docs/findings.md` open questions — an untested `DECODE_PREFILL_TOKEN_BUDGET=256` arm is
flagged there as a plausible free win on peak-C4 throughput).

`ALLOCATOR_BUDGET_BYTES` and `PROFILE_DRAFTER_CAP` are DCP4-era memory-safety knobs (see
`docs/patches.md`). They are required for any DCP4 boot with `MAX_NUM_BATCHED_TOKENS > 512`;
harmless to set unconditionally. Size `ALLOCATOR_BUDGET_BYTES` per node right before boot:
post-compaction `MemAvailable` minus 4GiB fixed reserve minus 2GiB slack (see
`docs/operations.md` §UMA hygiene). The values below (`111` GiB-ish) are what this campaign's
121GB-per-node cluster produced after compaction — **re-derive this number for your own
cluster, do not copy it verbatim.**

---

## DCP1 — decode-context-parallel degree 1 (no KV split; fastest decode, smallest capacity)

### DCP1 250K @ mnbt 2048
```bash
DCP_SIZE=1 MAX_MODEL_LEN=249000 KV_CACHE_MEMORY_BYTES=7995534848 \
MAX_NUM_SEQS=4 MAX_NUM_BATCHED_TOKENS=2048 DECODE_PREFILL_TOKEN_BUDGET=1024 \
CUDAGRAPH_CAPTURE_SIZES=6,12,18,24 \
  RUN_ID=dcp1-250k-2048 bash recipes/launch.sh launch
```
prose C1 26.10/25.34 · prose C4 55.70/53.65 · peak C1 36.36 · peak C4 83.22 · prefill 686.9 tok/s

### DCP1 320K @ mnbt 2048 — fastest config measured this campaign
```bash
DCP_SIZE=1 MAX_MODEL_LEN=320000 KV_CACHE_MEMORY_BYTES=10232320000 \
MAX_NUM_SEQS=4 MAX_NUM_BATCHED_TOKENS=2048 DECODE_PREFILL_TOKEN_BUDGET=1024 \
CUDAGRAPH_CAPTURE_SIZES=6,12,18,24 \
  RUN_ID=dcp1-320k-2048 bash recipes/launch.sh launch
```
prose C1 26.97/26.16 · prose C4 55.96/55.94 · peak C1 40.47 · peak C4 78.35/79.14 · prefill 694.1 tok/s (campaign record)

⚠️ Thinnest memory margin measured in the campaign: 1-2GB `MemAvailable` at idle on all 4
nodes. Ran clean, but do not serve this unattended without a watchdog (`docs/operations.md`).

### DCP1 320K @ mnbt 4096
```bash
DCP_SIZE=1 MAX_MODEL_LEN=320000 KV_CACHE_MEMORY_BYTES=10232320000 \
MAX_NUM_SEQS=4 MAX_NUM_BATCHED_TOKENS=4096 DECODE_PREFILL_TOKEN_BUDGET=1024 \
CUDAGRAPH_CAPTURE_SIZES=6,12,18,24 \
  RUN_ID=dcp1-320k-4096 bash recipes/launch.sh launch
```
prose C1 25.02/25.31 · prose C4 56.48/57.33 · peak C1 39.28/39.74 · peak C4 74.03/76.28 · prefill 688.7 tok/s
(carried from the 2048 gate — mnbt 4096 buys nothing at DCP1; see `docs/findings.md`)

---

## DCP2 — decode-context-parallel degree 2 (KV split 2 ways; balanced)

### DCP2 500K @ mnbt 512
```bash
DCP_SIZE=2 MAX_MODEL_LEN=500000 KV_CACHE_MEMORY_BYTES=7995534848 \
MAX_NUM_SEQS=4 MAX_NUM_BATCHED_TOKENS=512 DECODE_PREFILL_TOKEN_BUDGET=512 \
CUDAGRAPH_CAPTURE_SIZES=6,12,18,24 \
  RUN_ID=dcp2-500k-512 bash recipes/launch.sh launch
```
prose C1 23.96/24.11 · C2 n/m · prose C4 50.52/52.86 · peak C1 34.16 · peak C4 72.44 · prefill 466.5 tok/s

### DCP2 650K @ mnbt 2048 — balanced preset
```bash
DCP_SIZE=2 MAX_MODEL_LEN=650000 KV_CACHE_MEMORY_BYTES=10395482112 \
MAX_NUM_SEQS=4 MAX_NUM_BATCHED_TOKENS=2048 DECODE_PREFILL_TOKEN_BUDGET=1024 \
CUDAGRAPH_CAPTURE_SIZES=6,12,18,24 \
  RUN_ID=dcp2-650k-2048 bash recipes/launch.sh launch
```
prose C1 23.45/24.11 · C2 35.6 · prose C4 51.47/53.65 · peak C1 35.26/34.67 · C2 50.78 · peak C4 74.12/71.06 · prefill 585.3 tok/s

### DCP2 600K @ mnbt 4096
```bash
DCP_SIZE=2 MAX_MODEL_LEN=600000 KV_CACHE_MEMORY_BYTES=9596774400 \
MAX_NUM_SEQS=4 MAX_NUM_BATCHED_TOKENS=4096 DECODE_PREFILL_TOKEN_BUDGET=1024 \
CUDAGRAPH_CAPTURE_SIZES=6,12,18,24 \
  RUN_ID=dcp2-600k-4096 bash recipes/launch.sh launch
```
prose C1 23.40/23.11 · C2 34.92 · prose C4 51.89/53.03 · peak C1 37.43/32.44 · C2 44.92 · peak C4 74.98/73.93 · prefill 577.0 tok/s
(700K and 650K@4096 both refuse — this is the DCP2@4096 capacity frontier)

---

## DCP4 — decode-context-parallel degree 4 (KV split 4 ways; max context)

All DCP4 rows require the two b4-era memory-safety envs (`docs/patches.md`):
`VLLM_R17_PROFILE_DRAFTER_CAP=1` always; `ALLOCATOR_BUDGET_BYTES` required for `mnbt > 512`.

### DCP4 500K @ mnbt 512
```bash
DCP_SIZE=4 MAX_MODEL_LEN=500000 KV_CACHE_MEMORY_BYTES=3998790656 \
MAX_NUM_SEQS=4 MAX_NUM_BATCHED_TOKENS=512 DECODE_PREFILL_TOKEN_BUDGET=512 \
CUDAGRAPH_CAPTURE_SIZES=6,12,18,24 PROFILE_DRAFTER_CAP=1 \
  RUN_ID=dcp4-500k-512 bash recipes/launch.sh launch
```
prose C1 22.66/22.82 · prose C4 48.83/49.97 · peak C1 34.09 · peak C4 70.19 · prefill 360.5 tok/s

### DCP4 750K @ mnbt 1024 — first mnbt>512 DCP4 boot (needed the phantom-page-table fix)
```bash
DCP_SIZE=4 MAX_MODEL_LEN=750000 KV_CACHE_MEMORY_BYTES=5996139520 \
MAX_NUM_SEQS=4 MAX_NUM_BATCHED_TOKENS=1024 DECODE_PREFILL_TOKEN_BUDGET=1024 \
CUDAGRAPH_CAPTURE_SIZES=6,12,18,24 PROFILE_DRAFTER_CAP=1 \
  ALLOCATOR_BUDGET_BYTES=119185342464 \
  RUN_ID=dcp4-750k-1024 bash recipes/launch.sh launch
```
prose C1 23.19/23.14 · prose C4 49.76/49.45 · peak C1 35.06 · peak C4 71.90 · prefill 393.6 tok/s

KV bytes computed via the DCP4 formula (`docs/how-it-works.md`): `ceil(750000/256)=2930`
blocks × 2,046,464 B/block = 5,996,139,520 B/rank.

### DCP4 800K @ mnbt 2048 — Don's target pairing (max KV × mnbt sweet spot)
```bash
DCP_SIZE=4 MAX_MODEL_LEN=800000 KV_CACHE_MEMORY_BYTES=6395200000 \
MAX_NUM_SEQS=4 MAX_NUM_BATCHED_TOKENS=2048 DECODE_PREFILL_TOKEN_BUDGET=1024 \
CUDAGRAPH_CAPTURE_SIZES=6,12,18,24 PROFILE_DRAFTER_CAP=1 \
  ALLOCATOR_BUDGET_BYTES=119185342464 \
  RUN_ID=dcp4-800k-2048 bash recipes/launch.sh launch
```
prose C1 22.34/23.03 · prose C4 49.33/49.01 · peak C1 36.44 · peak C4 68.28 · prefill 425.1 tok/s

### DCP4 1M @ mnbt 512 — "1M is free" (no measurable cost vs 500K@512)
```bash
DCP_SIZE=4 MAX_MODEL_LEN=1000000 KV_CACHE_MEMORY_BYTES=7995534848 \
MAX_NUM_SEQS=4 MAX_NUM_BATCHED_TOKENS=512 DECODE_PREFILL_TOKEN_BUDGET=512 \
CUDAGRAPH_CAPTURE_SIZES=6,12,18,24 PROFILE_DRAFTER_CAP=1 \
  RUN_ID=dcp4-1m-512 bash recipes/launch.sh launch
```
prose C1 22.10/22.06 · prose C4 48.08/49.29 · peak C1 33.62 · peak C4 69.73 · prefill 361.7 tok/s

### DCP4 1M @ mnbt 2048 — PENDING, final campaign cell
```bash
DCP_SIZE=4 MAX_MODEL_LEN=1000000 KV_CACHE_MEMORY_BYTES=7995534848 \
MAX_NUM_SEQS=4 MAX_NUM_BATCHED_TOKENS=2048 DECODE_PREFILL_TOKEN_BUDGET=1024 \
CUDAGRAPH_CAPTURE_SIZES=6,12,18,24 PROFILE_DRAFTER_CAP=1 \
  ALLOCATOR_BUDGET_BYTES=119185342464 \
  RUN_ID=dcp4-1m-2048 bash recipes/launch.sh launch
```
<!-- FINAL_ROW_PENDING -->
Boots and serves; battery interrupted mid-run by a live-traffic contamination incident
(see `docs/operations.md` — the "mystery client" incident). Decode legs (C1/C2/C4 prose)
and the cold-prefill gate completed and are verified uncontaminated; the peak-decode legs
(peak C1/peak C4) were not yet run when the battery was halted.

Confirmed: prose C1 21.96/22.54 · prose C2 34.34 sat-est / 33.43 wall · prose C4 49.12/48.43
sat-est (48.22/48.24 wall) · cold prefill 422.8 tok/s (187,022 tok — ties 1M@4096's 421.5,
the DCP4 chunk-size plateau holds at this cell too) · accepted/step 1.21–1.35 · temp-0
determinism spot-check 2/3 diverged (consistent with the mnbt-linked anomaly, see
`docs/findings.md`). **peak C1 / peak C4: still TBD**, pending a follow-up battery leg.

### DCP4 1M @ mnbt 4096 — summit config (max context + max chunk, left serving)
```bash
DCP_SIZE=4 MAX_MODEL_LEN=1000000 KV_CACHE_MEMORY_BYTES=7995534848 \
MAX_NUM_SEQS=4 MAX_NUM_BATCHED_TOKENS=4096 DECODE_PREFILL_TOKEN_BUDGET=1024 \
CUDAGRAPH_CAPTURE_SIZES=6,12,18,24 PROFILE_DRAFTER_CAP=1 \
  ALLOCATOR_BUDGET_BYTES=119185342464 \
  RUN_ID=dcp4-1m-4096 bash recipes/launch.sh launch
```
prose C1 22.43/22.19 · prose C4 49.97/50.54 · peak C1 36.33 avg · peak C4 62.64 avg (weakest
DCP4 peak-C4 single battery, flagged not repeated) · prefill 421.5 tok/s

---

## Filled example — what real config values look like

The env-driven `recipes/launch.sh` never hardcodes site values, but here is a filled example
from the cluster this recipe was measured on, for orientation (values are illustrative of the
*shape* of a real config, not a live target):

```bash
NODE1=spark-node-a NODE2=spark-node-b NODE3=spark-node-c NODE4=spark-node-d \
FABRIC_IP_0=192.168.100.10 FABRIC_IP_1=192.168.100.11 \
FABRIC_IP_2=192.168.100.12 FABRIC_IP_3=192.168.100.13 \
MODEL_PATH=/mnt/models/glm52-w4a16-int4-lmheadw8v2 \
IMAGE_TAG=glm52-vllm027:dcp4-b4 \
RDMA_IFNAME=enp1s0f1np1 IB_HCA=rocep1s0f1 \
  DCP_SIZE=1 MAX_MODEL_LEN=320000 KV_CACHE_MEMORY_BYTES=10232320000 \
  MAX_NUM_SEQS=4 MAX_NUM_BATCHED_TOKENS=2048 DECODE_PREFILL_TOKEN_BUDGET=1024 \
  RUN_ID=dcp1-320k-2048 bash recipes/launch.sh launch
```
