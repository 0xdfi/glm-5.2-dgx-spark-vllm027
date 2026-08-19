#!/usr/bin/env bash
# GLM-5.2 on vLLM 0.27 — generic DCP1/DCP2/DCP4 launcher for 4x DGX Spark GB10.
#
# Adapted from a production launcher ("launch-dcp4.sh") that was itself forked
# from a DCP2-500K standalone launcher. This version is site-neutral: fill in
# the CONFIG block below for your cluster, then use `render` to sanity-check
# the argv/env before ever touching hardware, and `launch` to actually bring
# the fleet up.
#
# The vLLM `serve` argv assembled in ARGS below (see "GOLDEN ARGV") is the
# actual recipe — it is copied faithfully from the batteries that produced the
# speed matrix in the repo README. Everything else in this script (preflight
# checks, Ray bootstrap, docker plumbing) exists only to get that argv running
# safely and reproducibly across 4 nodes.
#
# What this script deliberately is NOT: it carries no distributed mutation
# lock, no transactional/sealed-hash launch contract, no automatic rollback,
# no rung-ladder state machine. It is a direct, readable launcher in the style
# of a hand-run runbook — appropriate for a single operator who reads the
# preflight output before proceeding. Build a heavier harness around it if you
# need unattended/concurrent-operator safety.
#
# Usage:
#   bash launch.sh render     # print the argv/env this run would use; touches nothing
#   bash launch.sh launch     # bring up Ray + vLLM serve across all 4 nodes
#
# All tuning is environment-overridable (see "TUNABLE ENVS" below). Nothing
# else in ARGS should be treated as safe to hand-edit per boot — those values
# are the frozen envelope that the speed matrix was measured against.

set -Eeuo pipefail

# ============================================================================
# CONFIG BLOCK — EDIT THIS BEFORE USE. Everything below this block is meant
# to work unmodified once these are correct for your cluster.
# ============================================================================

# SSH-reachable identifiers for the 4 nodes, rank-ordered (rank0 = API host).
# Use hostnames from your SSH config, or IPs directly. Do NOT commit real
# hostnames/IPs for a cluster you don't want identified — see the filled
# EXAMPLE block in recipes/presets.md for what real values look like.
NODES=("${NODE1:-NODE1}" "${NODE2:-NODE2}" "${NODE3:-NODE3}" "${NODE4:-NODE4}")

# RoCE/InfiniBand fabric IPs, one per node, same rank order as NODES. Rank0's
# fabric IP is also the API-serving address.
FABRIC_IPS=("${FABRIC_IP_0:-FABRIC_IP_0}" "${FABRIC_IP_1:-FABRIC_IP_1}" \
            "${FABRIC_IP_2:-FABRIC_IP_2}" "${FABRIC_IP_3:-FABRIC_IP_3}")

PORT="${PORT:-8211}"
HOST="${FABRIC_IPS[0]}"
ENDPOINT="http://${HOST}:${PORT}"

# Model checkpoint path — must be identical on all 4 nodes (local disk or a
# shared/NFS mount all 4 can read). This recipe assumes the W4A16 INT4-Marlin
# GLM-5.2 checkpoint with the lm_head w8v2 sidecar described in the README.
MODEL="${MODEL_PATH:-/path/to/glm-5.2-checkpoint}"

# Container image tag, identical on all 4 nodes. See docs/patches.md for what
# has to be baked into this image (B12X DCP-world-size guard generalization,
# the profile-phase allocator patches, the int32-overflow tiled_topk fix).
IMAGE_TAG="${IMAGE_TAG:-glm52-vllm027:local}"

# RDMA fabric device names — check with `ip -br a` (interface) and
# `ibv_devices` / `show_gids` (HCA) on each node; they are usually identical
# across identically-configured nodes but verify, don't assume.
RDMA_IFNAME="${RDMA_IFNAME:-enp1s0f1np1}"     # NCCL_SOCKET_IFNAME / GLOO_SOCKET_IFNAME
IB_HCA="${IB_HCA:-rocep1s0f1}"                # NCCL_IB_HCA
RDMA_DEV_UVERBS="${RDMA_DEV_UVERBS:-/dev/infiniband/uverbs1}"
RDMA_DEV_RDMACM="${RDMA_DEV_RDMACM:-/dev/infiniband/rdma_cm}"

# Regex of container names this script must NEVER stop, remove, or otherwise
# touch on any node — e.g. a separately-managed retained-container fleet you
# keep warm for a different profile. Leave empty to disable the check.
PROTECTED_CONTAINER_PATTERN="${PROTECTED_CONTAINER_PATTERN:-}"

# ============================================================================
# End of CONFIG BLOCK.
# ============================================================================

MODE="${1:-}"
RUN_ID="${RUN_ID:-glm52-run}"

ROOT="${ROOT:-$HOME/glm52-candidate}"
RUN_DIR="${ROOT}/runs/${RUN_ID}"
# Fresh-per-config JIT cache dir. dcp_world_size/dcp_rank are part of the
# B12X CuteDSL prewarm cache key — NEVER point this at another DCP config's
# cache dir. A prior cross-config JIT cache reuse crashed production; treat
# this as load-bearing, not cosmetic. See docs/patches.md.
JIT_ROOT="${ROOT}/jit-cache-${RUN_ID}"

RAY_PORT=26479
RAY_NODE_MANAGER_PORT=26480
RAY_OBJECT_MANAGER_PORT=26481
RAY_SESSION_ID="${RUN_ID}-$(date -u +%Y%m%dT%H%M%SZ)-$$"

NAMES=("${RUN_ID}-rank0" "${RUN_ID}-rank1" "${RUN_ID}-rank2" "${RUN_ID}-rank3")

remote() {
  local h="$1"; shift
  local cmd; printf -v cmd '%q ' "$@"
  ssh -o BatchMode=yes -o ConnectTimeout=10 -o ServerAliveInterval=5 -o ServerAliveCountMax=2 "$h" "$cmd"
}
docker() { local h="$1"; shift; remote "$h" sudo -n docker "$@"; }
log() { printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" >&2; }

# ── TUNABLE ENVS — all env-overridable at invocation time ──────────────────
#
#   DCP_SIZE                     decode-context-parallel degree: 1, 2, or 4.
#                                 Must divide --tensor-parallel-size (fixed at
#                                 4 below). DCP3 is not a legal value on a
#                                 TP4 cluster — see docs/how-it-works.md.
#   MAX_MODEL_LEN                 max context length (tokens).
#   KV_CACHE_MEMORY_BYTES         fixed KV pool size, bytes, PER RANK. See
#                                 recipes/presets.md for the exact byte value
#                                 to pair with each MAX_MODEL_LEN/DCP_SIZE —
#                                 do not guess this number, it is derived from
#                                 a block-size formula that differs by DCP
#                                 degree (see docs/how-it-works.md).
#   MAX_NUM_SEQS                  scheduler max concurrent sequences.
#   MAX_NUM_BATCHED_TOKENS         scheduler chunk/batch budget ("mnbt").
#   DECODE_PREFILL_TOKEN_BUDGET    decode-aware-prefill per-step token budget.
#   CUDAGRAPH_CAPTURE_SIZES        comma-separated graph capture shapes.
#                                 Formula used throughout the speed matrix:
#                                 (MTP_K+1) * seq for seq in 1..MAX_NUM_SEQS,
#                                 i.e. at k=5, seqs=4: "6,12,18,24".
#   ALLOCATOR_BUDGET_BYTES         explicit profile-phase allocator ceiling.
#                                 Required for any MAX_NUM_BATCHED_TOKENS>512
#                                 boot — see docs/patches.md patch (b).
#                                 Sizing rule: post-trim MemAvailable minus 4
#                                 GiB fixed reserve minus 2 GiB slack.
#   PROFILE_DRAFTER_CAP            set to 1 to enable the phantom page-table
#                                 fix (docs/patches.md patch (c)). Required
#                                 for any MAX_NUM_BATCHED_TOKENS>=1024 boot on
#                                 an unpatched-profile image; harmless / no-op
#                                 if your image already caps this unconditionally.
DCP_SIZE="${DCP_SIZE:-1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32000}"          # SAFE smoke-test default; see presets.md for real values
KV_CACHE_MEMORY_BYTES="${KV_CACHE_MEMORY_BYTES:-255808000}"  # matches the 32K smoke-test default above
MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-256}"
LONG_PREFILL_TOKEN_THRESHOLD="${LONG_PREFILL_TOKEN_THRESHOLD:-2048}"
DECODE_PREFILL_TOKEN_BUDGET="${DECODE_PREFILL_TOKEN_BUDGET:-256}"
IDLE_PREFILL_TOKEN_BUDGET="${IDLE_PREFILL_TOKEN_BUDGET:-2048}"
MAX_LONG_PREFILLS_PER_STEP="${MAX_LONG_PREFILLS_PER_STEP:-1}"
CUDAGRAPH_CAPTURE_SIZES="${CUDAGRAPH_CAPTURE_SIZES:-6}"
ALLOCATOR_BUDGET_BYTES="${ALLOCATOR_BUDGET_BYTES:-}"
PROFILE_DRAFTER_CAP="${PROFILE_DRAFTER_CAP:-}"

if [[ "$DCP_SIZE" != 1 && "$DCP_SIZE" != 2 && "$DCP_SIZE" != 4 ]]; then
  echo "refusing: DCP_SIZE must be 1, 2, or 4 (tp_size=4 requires tp %% dcp == 0 — DCP3 is impossible on TP4, see docs/how-it-works.md); got '${DCP_SIZE}'" >&2
  exit 64
fi

INDEX_TOPK_PATTERN='FFFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSS'
HF_OVERRIDES="{\"use_index_cache\":true,\"index_topk_pattern\":\"${INDEX_TOPK_PATTERN}\"}"
SPEC='{"method":"mtp","quantization":"compressed-tensors","num_speculative_tokens":5,"moe_backend":"marlin","attention_backend":"B12X_MLA_SPARSE","draft_sample_method":"probabilistic","rejection_sample_method":"block","adaptive_speculative_tokens_window":32}'
COMPILATION_CONFIG="{\"cudagraph_mode\":\"FULL\",\"cudagraph_capture_sizes\":[${CUDAGRAPH_CAPTURE_SIZES}],\"pass_config\":{\"fuse_gemm_comms\":true}}"

# ── GOLDEN ARGV — this IS the recipe. Every row in the speed matrix is this
# argv with only DCP_SIZE / MAX_MODEL_LEN / KV_CACHE_MEMORY_BYTES /
# MAX_NUM_SEQS / MAX_NUM_BATCHED_TOKENS / DECODE_PREFILL_TOKEN_BUDGET /
# CUDAGRAPH_CAPTURE_SIZES varied. Do not hand-edit anything else here without
# re-running the full battery (benchmarks/protocol.md) — this is a faithful,
# measured configuration, not a template to improvise from. ──────────────────
ARGS=(
  python3 -m vllm.entrypoints.openai.api_server
  --model /models --tokenizer /models --served-model-name glm-5.2
  --trust-remote-code --download-dir /models --load-format auto
  --quantization compressed-tensors --distributed-executor-backend ray
  --tensor-parallel-size 4
  --decode-context-parallel-size "${DCP_SIZE}"
  --prefill-context-parallel-size 1
  --pipeline-parallel-size 1
  $( [ "${DCP_SIZE}" -gt 1 ] && printf -- "--dcp-comm-backend a2a --dcp-kv-cache-interleave-size 1" || true )
  --gpu-memory-utilization 0.90
  --max-model-len "${MAX_MODEL_LEN}"
  --max-num-seqs "${MAX_NUM_SEQS}"
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}"
  --generation-config vllm
  --override-generation-config '{"temperature":1.0,"top_p":0.95,"top_k":40}'
  --hf-overrides "${HF_OVERRIDES}"
  --port "${PORT}" --host "${HOST}"
  --no-enable-flashinfer-autotune
  --no-enable-log-requests
  --kv-cache-memory-bytes "${KV_CACHE_MEMORY_BYTES}"
  --kv-cache-dtype nvfp4_ds_mla
  --attention-backend B12X_MLA_SPARSE --moe-backend marlin
  --reasoning-parser glm45 --tool-call-parser glm47 --enable-auto-tool-choice
  --speculative-config "${SPEC}"
  --long-prefill-token-threshold "${LONG_PREFILL_TOKEN_THRESHOLD}"
  --async-scheduling
  --enable-decode-aware-prefill
  --decode-prefill-token-budget "${DECODE_PREFILL_TOKEN_BUDGET}"
  --idle-prefill-token-budget "${IDLE_PREFILL_TOKEN_BUDGET}"
  --max-long-prefills-per-step "${MAX_LONG_PREFILLS_PER_STEP}"
  --compilation-config "${COMPILATION_CONFIG}"
  --enable-prefix-caching
)

render_argv() { printf '%q ' "${ARGS[@]}"; printf '\n'; }

if [[ "$MODE" == render ]]; then
  render_argv
  exit 0
fi

[[ "$MODE" == launch ]] || { echo "usage: RUN_ID=<id> DCP_SIZE={1|2|4} $0 {render|launch}" >&2; exit 64; }

IMAGE_ID="$(docker "${NODES[0]}" image inspect "${IMAGE_TAG}" --format '{{.Id}}')"

install -d -m 0755 "$RUN_DIR" "$JIT_ROOT"
TS="$(date -u +%Y%m%dT%H%M%SZ)"
SERVE_LOG="/run/serve-${TS}.log"   # container-visible path; host-side: ${RUN_DIR}/
exec > >(tee -a "${RUN_DIR}/launch-${RUN_ID}-${TS}.log") 2>&1

log "launch start run_id=${RUN_ID} dcp=${DCP_SIZE} image=${IMAGE_TAG}"
log "argv: $(render_argv)"

# ── Preflight (fail-closed, read/inspect only until the docker-run loop). ──
if [[ -n "$PROTECTED_CONTAINER_PATTERN" ]]; then
  log 'preflight: checking protected containers are not Running on any node'
  for h in "${NODES[@]}"; do
    running="$(remote "$h" sudo -n docker ps --format '{{.Names}}' | grep -E "$PROTECTED_CONTAINER_PATTERN" || true)"
    [[ -z "$running" ]] || { echo "refusing: $h has a protected container running: $running — stop it first, this launcher never touches protected fleets" >&2; exit 65; }
  done
fi

log 'preflight: checking nvidia-smi compute-apps empty on all 4 nodes'
for h in "${NODES[@]}"; do
  apps="$(remote "$h" nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits)"
  [[ -z "$apps" ]] || { echo "refusing: $h has live GPU compute apps: $apps" >&2; exit 65; }
done

log 'preflight: checking image ID matches on all 4 nodes'
for h in "${NODES[@]}"; do
  got="$(docker "$h" image inspect "$IMAGE_TAG" --format '{{.Id}}' 2>/dev/null)" || {
    echo "refusing: $h missing image $IMAGE_TAG" >&2; exit 65
  }
  [[ "$got" == "$IMAGE_ID" ]] || { echo "refusing: $h image id mismatch: got $got want $IMAGE_ID" >&2; exit 65; }
done

log 'preflight: checking port and Ray ports are free on all 4 nodes'
for h in "${NODES[@]}"; do
  busy="$(remote "$h" ss -ltnH 2>/dev/null | grep -E ":(${PORT}|${RAY_PORT}|${RAY_NODE_MANAGER_PORT}|${RAY_OBJECT_MANAGER_PORT})[[:space:]]" || true)"
  [[ -z "$busy" ]] || { echo "refusing: $h has a listener on ${PORT}/${RAY_PORT}/${RAY_NODE_MANAGER_PORT}/${RAY_OBJECT_MANAGER_PORT}: $busy" >&2; exit 65; }
done

log 'preflight: checking model path, RDMA devices, and no stale candidate containers'
for i in 0 1 2 3; do
  h="${NODES[$i]}"; n="${NAMES[$i]}"
  remote "$h" test -r "$MODEL/config.json"
  remote "$h" test -c "$RDMA_DEV_UVERBS"
  remote "$h" test -c "$RDMA_DEV_RDMACM"
  if docker "$h" inspect "$n" >/dev/null 2>&1; then
    echo "refusing: existing container $h/$n; tear it down explicitly first (this launcher never removes containers itself)" >&2
    exit 65
  fi
done
log 'all preflight checks passed'

# ── docker run common block. This is the vLLM/NCCL/B12X tuning env — keep it
# byte-identical across boots of the same profile; only the TUNABLE ENVS
# section above should vary run-to-run. ─────────────────────────────────────
COMMON=(
  --gpus all --network host --ipc host --shm-size 16g --ulimit memlock=-1:-1 --pids-limit=-1
  --device="${RDMA_DEV_UVERBS}" --device="${RDMA_DEV_RDMACM}"
  --label com.recipe.glm52.run_id="$RUN_ID"
  --label com.recipe.glm52.dcp_size="${DCP_SIZE}"
  --label com.recipe.glm52.max_model_len="${MAX_MODEL_LEN}"
  --label com.recipe.glm52.kv_bytes="${KV_CACHE_MEMORY_BYTES}"
  -e NCCL_SOCKET_IFNAME="${RDMA_IFNAME}" -e GLOO_SOCKET_IFNAME="${RDMA_IFNAME}"
  -e NCCL_IB_HCA="${IB_HCA}" -e NCCL_IB_DISABLE=0 -e NCCL_IB_GID_INDEX=3 -e NCCL_IB_TC=106
  -e NCCL_SHM_DISABLE=1 -e NCCL_P2P_DISABLE=0 -e NCCL_NET_GDR_LEVEL=PHB -e NCCL_IB_PCI_RELAXED_ORDERING=1
  -e NCCL_IB_QPS_PER_CONNECTION=4 -e NCCL_IB_SPLIT_DATA_ON_QPS=1
  -e NCCL_MIN_NCHANNELS=4 -e NCCL_MAX_NCHANNELS=4 -e NCCL_BUFFSIZE=4194304
  -e NCCL_CUMEM_ENABLE=0 -e NCCL_CUMEM_HOST_ENABLE=0 -e NCCL_GRAPH_REGISTER=0 -e NCCL_LOCAL_REGISTER=0
  -e NCCL_SET_THREAD_NAME=1 -e NCCL_DMABUF_ENABLE=0 -e NCCL_SOCKET_RETRY_CNT=100
  -e NCCL_SOCKET_RETRY_SLEEP_MSEC=100 -e NCCL_RAS_ENABLE=0 -e TORCH_NCCL_ASYNC_ERROR_HANDLING=1
  -e UCX_NET_DEVICES="${IB_HCA}:1" -e UCX_TLS=rc,tcp,self -e RAY_memory_monitor_refresh_ms=0
  -e RAY_DEDUP_LOGS=0 -e VLLM_DISABLE_COMPILE_CACHE=0 -e VLLM_WORKER_MULTIPROC_METHOD=spawn
  -e VLLM_DISTRIBUTED_USE_SPLIT_GROUP=0 -e VLLM_DCP_Q_REPLICATE=0 -e VLLM_DISABLE_PYNCCL=0
  -e PYTHONDONTWRITEBYTECODE=1
  -e KV_FP8_ROPE=1 -e VLLM_DCP_GLOBAL_TOPK=1 -e VLLM_B12X_MLA_SPEC_EXTEND_AS_DECODE=0
  -e CUDA_DEVICE_MAX_CONNECTIONS=4 -e CUDA_MODULE_LOADING=LAZY
  -e VLLM_LMHEAD_V2_SIDECAR=/models/lmhead_w8v2_sidecar.safetensors -e VLLM_LMHEAD_V2_TOPM=64 -e VLLM_LMHEAD_V2_REQUIRE=1
  -e VLLM_BUILDA_BMM=1 -e VLLM_MOE_MARLIN_ATOMIC_ADD=1 -e VLLM_USE_B12X_SPARSE_INDEXER=1 -e PYTHONUNBUFFERED=1
  -e VLLM_NVFP4_GEMM_BACKEND=flashinfer-cutlass -e VLLM_FLASHINFER_MOE_BACKEND=throughput
  -e VLLM_NVFP4_ALLOW_SLOW_FALLBACK=0 -e VLLM_QUANTIZATION_DISABLE_FUSED_MOE=0
  -e VLLM_USE_NVFP4_FLASHINFER_MOE=1 -e VLLM_NVFP4_MOE_DETERMINISTIC=0 -e FLASHINFER_MOE_CUMSUM=1
  -e FLASHINFER_MOE_CUMSUM_DEVICE=1 -e FLASHINFER_MOE_CLEANUP=0 -e FLASHINFER_ENABLE_AOT=1
  -e "CUDA_CACHE_PATH=${JIT_ROOT}/cuda" -e "TORCHINDUCTOR_CACHE_DIR=${JIT_ROOT}/torchinductor"
  -e "TRITON_CACHE_DIR=${JIT_ROOT}/triton" -e "XDG_CACHE_HOME=${JIT_ROOT}/xdg"
  -e "B12X_CUTE_COMPILE_CACHE_DIR=${JIT_ROOT}/xdg/b12x/cute_compile"
  # Memory-safety guard rails — see docs/patches.md for what these interact with.
  -e VLLM_UNIFIED_MEMORY_HOST_RESERVE_BYTES=4294967296
  -e VLLM_UNIFIED_MEMORY_ALLOCATOR_BUDGET_BYTES="${ALLOCATOR_BUDGET_BYTES}"
  -e VLLM_R17_PROFILE_DRAFTER_CAP="${PROFILE_DRAFTER_CAP}"
  -e MAX_JOBS=1 -e CMAKE_BUILD_PARALLEL_LEVEL=1 -e NINJAFLAGS=-j1 -e MAKEFLAGS=-j1
  -v "$MODEL:/models:ro"
  -v "$RUN_DIR:/run"
  -v "$JIT_ROOT:/root/.cache/vllm"
  -v "$JIT_ROOT:$JIT_ROOT"
  --entrypoint /bin/bash "$IMAGE_ID" -lc 'sleep infinity'
)

for i in 0 1 2 3; do
  h="${NODES[$i]}"; n="${NAMES[$i]}"
  log "creating $h/$n"
  remote "$h" install -d -m 0755 "$RUN_DIR/r$i" "$RUN_DIR/ray-spill/rank$i" "$JIT_ROOT"
  cid="$(docker "$h" run -d --name "$n" --hostname "$h" "${COMMON[@]}")"
  [[ "$cid" =~ ^[0-9a-f]{64}$ ]]
  observed="$(docker "$h" inspect "$cid" --format '{{.Id}}|{{.Image}}|{{.State.Running}}')"
  [[ "$observed" == "$cid|$IMAGE_ID|true" ]]
  docker "$h" exec "$n" bash -lc "printf '%s\n' '$RAY_SESSION_ID' >/tmp/ray-session-id"
done

log 'starting Ray head'
docker "${NODES[0]}" exec -d "${NAMES[0]}" bash -lc "rm -rf /run/r0/*; mkdir -p /run/r0 /run/ray-spill/rank0; exec ray start --head --node-ip-address=${HOST} --port=${RAY_PORT} --node-manager-port=${RAY_NODE_MANAGER_PORT} --object-manager-port=${RAY_OBJECT_MANAGER_PORT} --object-store-memory=134217728 --object-spilling-directory=/run/ray-spill/rank0 --num-cpus=1 --num-gpus=1 --include-dashboard=false --include-log-monitor=false --disable-usage-stats --temp-dir=/run/r0 --block >/run/ray-head.log 2>&1"

log 'waiting for Ray head GCS/raylet liveness'
head_ready=0
for _ in $(seq 1 30); do
  if docker "${NODES[0]}" exec "${NAMES[0]}" bash -lc "pgrep -x gcs_server >/dev/null && pgrep -x raylet >/dev/null && ray status --address=${HOST}:${RAY_PORT} >/dev/null 2>&1"; then
    head_ready=1; break
  fi
  sleep 1
done
[[ "$head_ready" == 1 ]] || { log 'Ray head GCS/raylet did not become live'; exit 72; }

for i in 1 2 3; do
  h="${NODES[$i]}"; n="${NAMES[$i]}"
  log "starting Ray worker $h"
  docker "$h" exec -d "$n" bash -lc "rm -rf /run/r$i/*; mkdir -p /run/r$i /run/ray-spill/rank$i; exec ray start --address=${HOST}:${RAY_PORT} --node-ip-address=${FABRIC_IPS[$i]} --node-manager-port=${RAY_NODE_MANAGER_PORT} --object-manager-port=${RAY_OBJECT_MANAGER_PORT} --object-spilling-directory=/run/ray-spill/rank$i --num-cpus=1 --num-gpus=1 --include-log-monitor=false --disable-usage-stats --temp-dir=/run/r$i --block >/run/ray-worker.log 2>&1"
done

log 'waiting for four Ray GPU nodes'
ready=0
for _ in $(seq 1 120); do
  ready="$(docker "${NODES[0]}" exec -i "${NAMES[0]}" python3 -B - 2>/dev/null <<'PY' || true
import ray
ray.init(address='auto', ignore_reinit_error=True)
ns=[n for n in ray.nodes() if n.get('Alive') and n.get('Resources',{}).get('GPU',0)>=1]
print(len(ns))
PY
)"
  ready="${ready##*$'\n'}"
  [[ "$ready" == 4 ]] && break
  sleep 2
done
[[ "$ready" == 4 ]] || { log 'did not reach 4 Ray GPU nodes'; exit 72; }

log 'waiting for two consecutive Ray CLI 4-GPU receipts'
stable=0
for _ in $(seq 1 30); do
  status_out="$(docker "${NODES[0]}" exec "${NAMES[0]}" ray status --address="${HOST}:${RAY_PORT}" 2>&1 || true)"
  if [[ "$status_out" == *"/4.0 GPU"* ]]; then
    stable=$((stable + 1)); (( stable >= 2 )) && break
  else
    stable=0
  fi
  sleep 1
done
(( stable >= 2 )) || { log 'Ray CLI did not stabilize at four GPUs'; exit 72; }

log "launching vLLM serve on rank0; argv: $(render_argv)"
printf -v esc '%q ' "${ARGS[@]}"
launch="set -euo pipefail; test \"\$(cat /tmp/ray-session-id)\" = '${RAY_SESSION_ID}'; printf '%s\n' '${esc}' > '${SERVE_LOG}'; exec ${esc} >> '${SERVE_LOG}' 2>&1"

docker "${NODES[0]}" exec -i "${NAMES[0]}" bash -lc "test \"\$(cat /tmp/ray-session-id)\" = '${RAY_SESSION_ID}'; ray status --address=${HOST}:${RAY_PORT} | grep -q '/4.0 GPU'"

docker "${NODES[0]}" exec -d \
  -e CUDA_VISIBLE_DEVICES= \
  -e SAFETENSORS_FAST_GPU=1 -e CUDA_DEVICE_ORDER=PCI_BUS_ID -e CUDA_DEVICE_MAX_CONNECTIONS=32 \
  -e CUTE_DSL_ARCH=sm_121a -e TORCH_CUDA_ARCH_LIST=12.1a -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
  -e NCCL_SOCKET_IFNAME="${RDMA_IFNAME}" -e GLOO_SOCKET_IFNAME="${RDMA_IFNAME}" -e NCCL_IB_DISABLE=0 \
  -e NCCL_MAX_NCHANNELS=4 -e NCCL_MIN_NCHANNELS=4 \
  -e VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800 \
  -e NCCL_CUMEM_ENABLE=0 -e NCCL_CUMEM_HOST_ENABLE=0 -e NCCL_DMABUF_ENABLE=0 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
  -e VLLM_USE_FLASHINFER_SAMPLER=1 -e VLLM_USE_V2_MODEL_RUNNER=1 -e VLLM_USE_B12X_SPARSE_INDEXER=1 -e VLLM_DCP_GLOBAL_TOPK=1 \
  -e VLLM_ADAPTIVE_SPEC_DEPTHS=2,4,5 \
  -e VLLM_MARLIN_USE_ATOMIC_ADD=1 \
  -e VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=256 \
  -e GLM52_PAGED_MQA_TOPK_CHUNK_SIZE=8192 \
  -e VLLM_B12X_MLA_SPEC_EXTEND_AS_DECODE=0 \
  -e VLLM_UNIFIED_MEMORY_HOST_RESERVE_BYTES=4294967296 \
  -e VLLM_UNIFIED_MEMORY_ALLOCATOR_BUDGET_BYTES="${ALLOCATOR_BUDGET_BYTES}" \
  -e VLLM_R17_PROFILE_DRAFTER_CAP="${PROFILE_DRAFTER_CAP}" \
  -e MAX_JOBS=1 -e CMAKE_BUILD_PARALLEL_LEVEL=1 -e NINJAFLAGS=-j1 -e MAKEFLAGS=-j1 \
  -e VLLM_DISABLE_COMPILE_CACHE=0 \
  -e VLLM_ENABLE_PCIE_ALLREDUCE=0 -e RAY_ADDRESS="${HOST}:${RAY_PORT}" \
  "${NAMES[0]}" bash -lc "${launch}"

log "launch submitted run=${RUN_ID}; endpoint=${ENDPOINT}; serve log (in-container)=${SERVE_LOG}; host-side=${RUN_DIR}/"
log 'this script does not poll for readiness; check with: curl -s '"${ENDPOINT}"'/v1/models'
log 'poll health with: curl -s -o /dev/null -w "%{http_code}\n" '"${ENDPOINT}"'/health'
