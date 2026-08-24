#!/usr/bin/env bash
# Copyright 2026 dfi / 0xdfi
# SPDX-License-Identifier: Apache-2.0
set -Eeuo pipefail

CHAMPION_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CHAMPION_PYTHON="$CHAMPION_ROOT/.venv/bin/python"
[[ -x "$CHAMPION_PYTHON" ]] || { printf '%s\n' 'development environment missing; run ./bin/champion bootstrap' >&2; exit 69; }
# shellcheck source=scripts/lib/fleet.sh
source "$CHAMPION_ROOT/scripts/lib/fleet.sh"

MODE="${1:-launch}"
PROFILE="${2:-fast}"
if (( $# >= 2 )); then shift 2; else set --; fi
STARTED=0
READINESS_PASSED=0
CHAMPION_COMPACTION_NONCE_HASH=''
CHAMPION_COMPACTION_RUN_HASH=''

parse_launch_authorization() {
  local nonce=''
  if [[ "$MODE" != launch ]]; then
    (( $# == 0 )) || { printf '%s\n' 'verify-hosts accepts no authorization arguments' >&2; exit 64; }
    return 0
  fi
  while (( $# > 0 )); do
    case "$1" in
      --compaction-nonce)
        [[ -z "$nonce" && $# -ge 2 ]] || {
          printf '%s\n' 'launch: --compaction-nonce must appear exactly once with a value' >&2
          exit 64
        }
        nonce="$2"
        shift 2
        ;;
      *) printf 'launch: unknown argument: %s\n' "$1" >&2; exit 64 ;;
    esac
  done
  case "$PROFILE" in
    fast|fast-320k)
      [[ -n "$nonce" ]] || {
        printf '%s\n' 'launch fast requires --compaction-nonce with a fresh 32 hexadecimal character value' >&2
        exit 64
      }
      [[ "$nonce" =~ ^[0-9A-Fa-f]{32}$ ]] || {
        printf '%s\n' 'launch: --compaction-nonce must be exactly 32 hexadecimal characters' >&2
        exit 64
      }
      CHAMPION_COMPACTION_NONCE_HASH="$(printf '%s' "$nonce" | "$CHAMPION_PYTHON" -c '
import hashlib,sys
raw=sys.stdin.buffer.read()
print(hashlib.sha256(b"champion-fast-compaction-v1\\0" + raw).hexdigest())
')" || exit 70
      nonce=''
      ;;
    *)
      [[ -z "$nonce" ]] || {
        printf '%s\n' 'launch: --compaction-nonce is valid only for Fast' >&2
        exit 64
      }
      ;;
  esac
}

parse_launch_authorization "$@"

on_exit() {
  local rc=$?
  set +e
  if (( rc != 0 && STARTED == 1 && READINESS_PASSED == 0 )); then
    if fleet_observe_failed_three_times "$CHAMPION_RUN_ID"; then
      printf '%s\n' 'three consecutive known rank failures; stopping exact failed attempt only' >&2
      fleet_stop_exact_ordered "$CHAMPION_RUN_ID" || true
    else
      printf '%s\n' 'rank state UNKNOWN or not conclusively failed; preserving fleet' >&2
    fi
  fi
  fleet_lock_release
  exit "$rc"
}
trap on_exit EXIT

fleet_load_cluster
fleet_validate_protected_pattern
if [[ -n "$CHAMPION_COMPACTION_NONCE_HASH" ]]; then
  CHAMPION_COMPACTION_RUN_HASH="$(printf '%s' "$CHAMPION_RUN_ID" | "$CHAMPION_PYTHON" -c '
import hashlib,sys
print(hashlib.sha256(b"champion-run-v1\\0" + sys.stdin.buffer.read()).hexdigest())
')" || exit 70
fi

read_mem_available_kib() {
  fleet_remote "$1" cat /proc/meminfo | awk '/MemAvailable:/ {print $2}'
}

read_swap_used_kib() {
  fleet_remote "$1" cat /proc/meminfo | awk '
    /SwapTotal:/ {total=$2}
    /SwapFree:/ {free=$2}
    END {print total-free}
  '
}

verify_hosts() {
  local memory_policy="${1:-strict}" index host arch cpus gpu_line gpu capability mem polkit swap apps busy protected containers image protected_rc
  local min_mem=999999999 max_mem=0 reports=()
  for index in 0 1 2 3; do
    if [[ "$MODE" == launch ]]; then fleet_lock_assert_held || return; fi
    host="${CHAMPION_HOST_ARRAY[$index]}"
    arch="$(fleet_remote "$host" uname -m)"
    [[ "$arch" == aarch64 ]] || fleet_die "$host architecture is $arch, expected aarch64" || return
    cpus="$(fleet_remote "$host" getconf _NPROCESSORS_ONLN)"
    [[ "$cpus" == 20 ]] || fleet_die "$host has $cpus online CPUs, expected 20" || return
    gpu_line="$(fleet_remote "$host" nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader)"
    IFS=',' read -r gpu capability <<<"$gpu_line"
    gpu="${gpu% }"; capability="${capability# }"
    [[ "$gpu" == 'NVIDIA GB10' && "$capability" == '12.1' ]] || \
      fleet_die "$host GPU is $gpu/$capability, expected NVIDIA GB10/12.1" || return
    fleet_remote "$host" test -S /var/run/docker.sock || fleet_die "$host Docker socket unavailable" || return
    fleet_remote "$host" test -d "/sys/class/net/$CHAMPION_RDMA_IFNAME" || fleet_die "$host RDMA interface missing" || return
    fleet_remote "$host" test -c "$CHAMPION_RDMA_UVERBS" || fleet_die "$host uverbs device missing" || return
    fleet_remote "$host" test -c "$CHAMPION_RDMA_CM" || fleet_die "$host rdma_cm device missing" || return
    fleet_remote "$host" rdma link show "$CHAMPION_IB_HCA/1" | grep -q 'state ACTIVE.*physical_state LINK_UP' || \
      fleet_die "$host RDMA link is not active" || return
    mem="$(read_mem_available_kib "$host")"
    [[ "$mem" =~ ^[1-9][0-9]{0,11}$ ]] || \
      fleet_die "$host MemAvailable is invalid: ${mem:-unknown}" || return
    if [[ "$memory_policy" == strict && "$mem" -lt 121634816 ]]; then
      fleet_die "$host MemAvailable $mem KiB is below 116 GiB" || return
    fi
    (( mem < min_mem )) && min_mem=$mem
    (( mem > max_mem )) && max_mem=$mem
    polkit="$(fleet_remote "$host" ps -C polkitd -o rss= | awk 'NR==1 {print $1+0}')"
    [[ "$polkit" =~ ^[0-9]+$ && "$polkit" -lt 131072 ]] || fleet_die "$host polkit RSS exceeds 128 MiB" || return
    swap="$(read_swap_used_kib "$host")"
    [[ "$swap" =~ ^[0-9]+$ && "$swap" -le 6291456 ]] || fleet_die "$host swap use exceeds 6 GiB" || return
    if ! apps="$(fleet_remote "$host" nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits)"; then
      fleet_die "$host GPU application query failed"
      return 1
    fi
    [[ -z "$apps" ]] || fleet_die "$host has a live GPU compute application" || return
    if ! busy="$(fleet_remote "$host" ss -ltnH "sport = :$CHAMPION_API_PORT")"; then
      fleet_die "$host target port query failed"
      return 1
    fi
    [[ -z "$busy" ]] || fleet_die "$host target port $CHAMPION_API_PORT is occupied" || return
    if ! containers="$(fleet_docker "$host" ps --format '{{.Names}}')"; then
      fleet_die "$host protected container query failed"
      return 1
    fi
    if protected="$(fleet_match_protected_containers "$containers")"; then
      protected_rc=0
    else
      protected_rc=$?
    fi
    case "$protected_rc" in
      0) fleet_die "$host protected container is running: $protected"; return 1 ;;
      1) ;;
      *) fleet_die "$host protected container pattern evaluation failed"; return 1 ;;
    esac
    image="$(fleet_docker "$host" image inspect "$CHAMPION_IMAGE_REF" --format '{{.Id}}' 2>/dev/null || true)"
    [[ "$image" == "$CHAMPION_IMAGE_ID" ]] || fleet_die "$host image identity mismatch" || return
    fleet_verify_model_files "$host" "$CHAMPION_CHECKPOINT_LOCK_PATH" || return
    fleet_verify_model_files "$host" "$CHAMPION_SIDECAR_LOCK_PATH" || return
    reports+=("$host|$arch|$cpus|$gpu|$capability|$mem|$polkit|$swap")
  done
  if [[ "$memory_policy" == strict ]]; then
    (( max_mem - min_mem <= 524288 )) || fleet_die 'peer MemAvailable spread exceeds 512 MiB' || return
  elif [[ "$memory_policy" != pre-compaction ]]; then
    fleet_die "unknown memory verification policy: $memory_policy" || return
  fi
  if [[ "$PROFILE" == fast ]]; then
    for host in "${CHAMPION_HOST_ARRAY[@]}"; do
      fleet_remote "$host" lscpu -p=CPU,MAXMHZ,NODE | "$CHAMPION_PYTHON" -c '
import sys
rows={}
for line in sys.stdin:
    if line.startswith("#") or not line.strip(): continue
    cpu,mhz,node=line.strip().split(",")
    rows[int(cpu)]=(float(mhz),int(node))
wanted=set(range(5,10))|set(range(15,20))
raise SystemExit(0 if wanted <= rows.keys() and all(rows[c][0]>=3900 and rows[c][1]==0 for c in wanted) else 1)
' || fleet_die "$host Fast big-core/NUMA topology mismatch" || return
    done
  fi
  "$CHAMPION_PYTHON" - "${reports[@]}" <<'PY'
import json,sys
hosts=[]
for row in sys.argv[1:]:
    host,arch,cpus,gpu,cc,mem,polkit,swap=row.split('|')
    hosts.append({'host':host,'architecture':arch,'online_cpus':int(cpus),'gpu':gpu,'compute_capability':cc,'mem_available_kib':int(mem),'polkit_rss_kib':int(polkit),'swap_used_kib':int(swap)})
print(json.dumps({'schema_version':'1.0.0','result':'PASS','lock_scope':'canonical-rank0-required-for-mutation','comparison_environment':'drift-recorded','hosts':hosts},sort_keys=True))
PY
}

profile_name() {
  case "$1" in
    fast|fast-320k) printf '%s\n' fast-320k ;;
    balanced|balanced-650k) printf '%s\n' balanced-650k ;;
    large|large-1m) printf '%s\n' large-1m ;;
    *) fleet_die "unknown profile $1"; return 1 ;;
  esac
}

api_health() {
  "${CHAMPION_CURL_BIN:-curl}" -fsS --max-time 5 "http://$CHAMPION_API_HOST:$CHAMPION_API_PORT/health" >/dev/null
}

chat_probe() {
  "${CHAMPION_CURL_BIN:-curl}" -fsS --max-time 120 \
    -H 'Content-Type: application/json' \
    --data '{"model":"glm-5.2","messages":[{"role":"user","content":"Reply briefly with SPARK_OK."}],"temperature":1.0,"max_tokens":96}' \
    "http://$CHAMPION_API_HOST:$CHAMPION_API_PORT/v1/chat/completions" \
    | "$CHAMPION_PYTHON" -c 'import json,sys; o=json.load(sys.stdin); assert (o.get("choices") or [{}])[0].get("message",{}).get("content")'
}

cache_busted_sparse_indexer_probe() {
  local payload raw
  payload="$("$CHAMPION_PYTHON" - "$CHAMPION_RUN_ID" <<'PY'
import json,secrets,sys
nonce=f"{sys.argv[1]}-{secrets.token_hex(16)}"
payload={
    "model":"glm-5.2",
    "messages":[{"role":"user","content":f"cache-bust={nonce} " + ("probe " * 4096) + "\nReply briefly with SPARK_INDEX_OK."}],
    "temperature":1.0,
    "max_tokens":16,
    "stream":False,
}
print(json.dumps(payload,separators=(",",":")))
PY
)" || return
  raw="$("${CHAMPION_CURL_BIN:-curl}" -fsS --max-time 180 \
    -H 'Content-Type: application/json' \
    -H 'X-Champion-Probe: cache-busted-sparse-indexer' \
    --data-binary "$payload" \
    "http://$CHAMPION_API_HOST:$CHAMPION_API_PORT/v1/chat/completions")" || return
  "$CHAMPION_PYTHON" -c 'import json,sys
o=json.load(sys.stdin)
choice=(o.get("choices") or [{}])[0]
message=choice.get("message") or {}
text=next((v.strip() for v in (message.get("content"),message.get("reasoning_content"),message.get("reasoning")) if isinstance(v,str) and v.strip()),"")
prompt_tokens=(o.get("usage") or {}).get("prompt_tokens")
assert isinstance(prompt_tokens,int) and prompt_tokens>2048, prompt_tokens
assert "SPARK_INDEX_OK" in text, text' <<<"$raw"
}

compact_and_size_uniform_allocator_floor() {
  local index host mem min_mem=999999999999 max_mem=0 floor
  for index in 0 1 2 3; do
    fleet_lock_assert_held || return
    host="${CHAMPION_HOST_ARRAY[$index]}"
    fleet_remote "$host" sudo -n sh -c "'sync; printf 3 > /proc/sys/vm/drop_caches'" || {
      fleet_die "$host approved memory compaction failed"
      return 1
    }
  done
  for index in 0 1 2 3; do
    fleet_lock_assert_held || return
    host="${CHAMPION_HOST_ARRAY[$index]}"
    mem="$(read_mem_available_kib "$host")"
    [[ "$mem" =~ ^[1-9][0-9]{0,11}$ ]] || {
      fleet_die "$host post-compaction MemAvailable is invalid: ${mem:-unknown}"
      return 1
    }
    (( mem >= 121634816 )) || {
      fleet_die "$host post-compaction MemAvailable $mem KiB is below 116 GiB"
      return 1
    }
    (( mem < min_mem )) && min_mem=$mem
    (( mem > max_mem )) && max_mem=$mem
  done
  (( max_mem - min_mem <= 524288 )) || {
    fleet_die 'post-compaction peer MemAvailable spread exceeds 512 MiB'
    return 1
  }
  floor="$((min_mem * 1024 - 6 * 1024 * 1024 * 1024))"
  [[ "$floor" =~ ^[1-9][0-9]*$ ]] || {
    fleet_die 'fleet-uniform allocator floor calculation failed'
    return 1
  }
  printf '%s\n' "$floor"
}

allocator_readback_gate() {
  local profile_drafter_cap="$1" index host name env_dump expected floor_lines
  for index in 0 1 2 3; do
    host="${CHAMPION_HOST_ARRAY[$index]}"; name="$CHAMPION_RUN_ID-rank$index"
    env_dump="$(fleet_docker "$host" inspect "$name" --format '{{range .Config.Env}}{{println .}}{{end}}')" || {
      fleet_die "$host/$name allocator environment readback failed"
      return 1
    }
    [[ "$(grep -Fxc 'PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True' <<<"$env_dump" || true)" == 1 ]] || {
      fleet_die "$host/$name PyTorch allocator environment mismatch"
      return 1
    }
    expected="${CHAMPION_ALLOCATOR_FLOOR_ARRAY[$index]-}"
    floor_lines="$(grep '^VLLM_UNIFIED_MEMORY_ALLOCATOR_BUDGET_BYTES=' <<<"$env_dump" || true)"
    if [[ -n "$expected" ]]; then
      [[ "$floor_lines" == "VLLM_UNIFIED_MEMORY_ALLOCATOR_BUDGET_BYTES=$expected" ]] || {
        fleet_die "$host/$name allocator floor environment mismatch"
        return 1
      }
    elif [[ -n "$floor_lines" ]]; then
      fleet_die "$host/$name has an unexpected allocator floor environment"
      return 1
    fi
    if [[ "$profile_drafter_cap" == 1 ]]; then
      [[ "$(grep -Fxc 'VLLM_R17_PROFILE_DRAFTER_CAP=1' <<<"$env_dump" || true)" == 1 ]] || {
        fleet_die "$host/$name drafter profile cap environment mismatch"
        return 1
      }
    fi
  done
}

rank_gate() {
  local index host name state
  for index in 0 1 2 3; do
    host="${CHAMPION_HOST_ARRAY[$index]}"; name="$CHAMPION_RUN_ID-rank$index"
    fleet_assert_exact "$host" "$name" "$CHAMPION_RUN_ID" || return
    state="$(fleet_docker "$host" inspect "$name" --format '{{.State.Running}}|{{.State.OOMKilled}}|{{.RestartCount}}')"
    [[ "$state" == 'true|false|0' ]] || fleet_die "$host/$name restart/OOM state is $state" || return
    fleet_docker "$host" exec "$name" test -S /tmp/ray/session_latest/sockets/raylet || fleet_die "$host/$name Ray socket missing" || return
    fleet_docker "$host" top "$name" -eo pid,args | grep -q ray || fleet_die "$host/$name Ray PID/argv missing" || return
  done
  fleet_docker "${CHAMPION_HOST_ARRAY[0]}" exec "$CHAMPION_RUN_ID-rank0" ray status | grep -q '/4.0 GPU' || \
    fleet_die 'four Ray GPU ranks were not observed'
  fleet_docker "${CHAMPION_HOST_ARRAY[0]}" top "$CHAMPION_RUN_ID-rank0" -eo pid,args | grep -q 'vllm.entrypoints.openai.api_server' || \
    fleet_die 'coordinator API PID/argv missing'
}

transport_and_log_gate() {
  local logs
  logs="$(fleet_docker "${CHAMPION_HOST_ARRAY[0]}" exec "$CHAMPION_RUN_ID-rank0" cat /run/champion/serve.log 2>&1)"
  [[ "$logs" == *'Using network IB'* && "$logs" != *'Using network Socket'* ]] || \
    fleet_die 'NCCL did not select IB exclusively' || return
  if grep -Eqi 'Xid|CUDA out of memory|OutOfMemoryError|oom-kill' <<<"$logs"; then
    fleet_die 'current-attempt fatal marker detected'
    return 1
  fi
  if grep -Fq 'No available shared memory broadcast block' <<<"$logs"; then
    printf '%s\n' 'pre-readiness allocation warning recorded; acceptance remains conditional on all readiness gates' >&2
  fi
  CHAMPION_READINESS_LOG_LINES="$(wc -l <<<"$logs" | tr -d ' ')"
}

rollback_fast_pin() {
  local index host name
  for index in 0 1 2 3; do
    fleet_lock_assert_held || return 1
    host="${CHAMPION_HOST_ARRAY[$index]}"; name="$CHAMPION_RUN_ID-rank$index"
    fleet_assert_exact "$host" "$name" "$CHAMPION_RUN_ID" || return 1
    fleet_docker "$host" update --cpuset-cpus 0-19 --cpuset-mems 0 "$name" >/dev/null || return 1
  done
  api_health && chat_probe
}

apply_fast_pin() {
  local index host name failed=0 observed
  set +e
  for index in 0 1 2 3; do
    fleet_lock_assert_held || { failed=1; break; }
    host="${CHAMPION_HOST_ARRAY[$index]}"; name="$CHAMPION_RUN_ID-rank$index"
    fleet_assert_exact "$host" "$name" "$CHAMPION_RUN_ID" || { failed=1; break; }
    fleet_docker "$host" update --cpuset-cpus 5-9,15-19 --cpuset-mems 0 "$name" >/dev/null || { failed=1; break; }
  done
  if (( failed == 0 )); then
    for index in 0 1 2 3; do
      fleet_lock_assert_held || { failed=1; break; }
      host="${CHAMPION_HOST_ARRAY[$index]}"; name="$CHAMPION_RUN_ID-rank$index"
      observed="$(fleet_docker "$host" inspect "$name" --format '{{.HostConfig.CpusetCpus}}|{{.HostConfig.CpusetMems}}')"
      [[ "$observed" == '5-9,15-19|0' ]] || { failed=1; break; }
    done
  fi
  (( failed == 0 )) && api_health && chat_probe || failed=1
  set -e
  if (( failed != 0 )); then
    if rollback_fast_pin; then
      printf '%s\n' 'unpinned-rollback: Fast pin failed; healthy fleet restored to CPUs 0-19/mem0; champion claim blocked' >&2
    else
      printf '%s\n' 'unpinned-rollback: rollback verification failed; fleet preserved for diagnosis' >&2
    fi
    return 1
  fi
}

launch_profile() {
  local canonical argv env_lines index host name existing allocator='' logs models_json mem
  local allocator_contract allocator_scope allocator_floor allocator_cap uniform_floor=''
  canonical="$(profile_name "$PROFILE")" || return
  PROFILE="${canonical%%-*}"
  allocator_contract="$("$CHAMPION_PYTHON" - "$CHAMPION_ROOT/presets/$canonical.json" <<'PY'
import json,sys
allocator=json.load(open(sys.argv[1]))["effective_engine"]["allocator"]
print("|".join((allocator["scope"],allocator["floor"],str(allocator["profile_drafter_cap"]))))
PY
)" || return
  IFS='|' read -r allocator_scope allocator_floor allocator_cap <<<"$allocator_contract"
  argv="$("$CHAMPION_PYTHON" "$CHAMPION_ROOT/scripts/render_launch.py" --preset "$canonical" --cluster "${CHAMPION_CLUSTER_ENV:-$CHAMPION_ROOT/cluster.env}" --format argv)" || return
  env_lines="$("$CHAMPION_PYTHON" "$CHAMPION_ROOT/scripts/render_launch.py" --preset "$canonical" --cluster "${CHAMPION_CLUSTER_ENV:-$CHAMPION_ROOT/cluster.env}" --format env)" || return
  fleet_lock_acquire || return
  if [[ "$allocator_scope" == fleet-uniform && "$allocator_floor" == computed ]]; then
    verify_hosts pre-compaction >/dev/null || return
    for index in 0 1 2 3; do
      fleet_lock_assert_held || return
      host="${CHAMPION_HOST_ARRAY[$index]}"; name="$CHAMPION_RUN_ID-rank$index"
      existing="$(fleet_docker "$host" ps -a --filter "name=^/$name$" --format '{{.Names}}')"
      [[ -z "$existing" ]] || fleet_die "existing target container $host/$name is preserved; choose a fresh run id" || return
    done
    fleet_consume_compaction_nonce_hash \
      "$CHAMPION_COMPACTION_NONCE_HASH" "$CHAMPION_COMPACTION_RUN_HASH" || return
    uniform_floor="$(compact_and_size_uniform_allocator_floor)" || return
  else
    verify_hosts >/dev/null || return
  fi
  if [[ -n "${CHAMPION_REPLACE_RUN_ID:-}" ]]; then
    fleet_stop_exact_ordered "$CHAMPION_REPLACE_RUN_ID" || return
  fi
  CHAMPION_ALLOCATOR_FLOOR_ARRAY=()
  for index in 0 1 2 3; do
    fleet_lock_assert_held || return
    host="${CHAMPION_HOST_ARRAY[$index]}"; name="$CHAMPION_RUN_ID-rank$index"
    existing="$(fleet_docker "$host" ps -a --filter "name=^/$name$" --format '{{.Names}}')"
    [[ -z "$existing" ]] || fleet_die "existing target container $host/$name is preserved; choose a fresh run id" || return
    fleet_remote "$host" install -d -m 0755 "$CHAMPION_RUN_ROOT/$CHAMPION_RUN_ID/rank$index" "$CHAMPION_RUN_ROOT/$CHAMPION_RUN_ID/cache/$canonical"
    allocator="$uniform_floor"
    if [[ "$allocator_scope" == per-host && "$allocator_floor" == computed ]]; then
      mem="$(read_mem_available_kib "$host")"
      [[ "$mem" =~ ^[1-9][0-9]{0,11}$ ]] || fleet_die "$host per-host allocator input failed" || return
      allocator="$((mem * 1024 - 6 * 1024 * 1024 * 1024))"
      [[ "$allocator" =~ ^[1-9][0-9]*$ ]] || fleet_die "$host per-host allocator calculation failed" || return
    fi
    CHAMPION_ALLOCATOR_FLOOR_ARRAY[index]="$allocator"
    run_args=(run -d --name "$name" --hostname "$host" --gpus all --network host --ipc host --shm-size 16g --pids-limit=-1
      --device "$CHAMPION_RDMA_UVERBS" --device "$CHAMPION_RDMA_CM"
      --label "org.0xdfi.champion.run_id=$CHAMPION_RUN_ID"
      --label "org.0xdfi.champion.preset=$canonical"
      -v "$CHAMPION_MODEL_PATH:/models:ro"
      -v "$CHAMPION_RUN_ROOT/$CHAMPION_RUN_ID/rank$index:/run/champion"
      -v "$CHAMPION_RUN_ROOT/$CHAMPION_RUN_ID/cache/$canonical:/root/.cache/vllm")
    while IFS= read -r line; do [[ -n "$line" ]] && run_args+=(--env "$line"); done <<<"$env_lines"
    [[ -z "$allocator" ]] || run_args+=(--env "VLLM_UNIFIED_MEMORY_ALLOCATOR_BUDGET_BYTES=$allocator")
    # Deliberately no create-time --cpuset: Fast is pinned only after readiness.
    run_args+=(--entrypoint sleep "$CHAMPION_IMAGE_ID" infinity)
    fleet_docker "$host" "${run_args[@]}" >/dev/null
    STARTED=1
  done
  fleet_lock_assert_held || return
  allocator_readback_gate "$allocator_cap" || return
  fleet_lock_assert_held || return
  fleet_docker "${CHAMPION_HOST_ARRAY[0]}" exec -d "$CHAMPION_RUN_ID-rank0" ray start --head --node-ip-address "${CHAMPION_FABRIC_ARRAY[0]}" --port 6379 --block >/dev/null
  for index in 1 2 3; do
    fleet_lock_assert_held || return
    fleet_docker "${CHAMPION_HOST_ARRAY[$index]}" exec -d "$CHAMPION_RUN_ID-rank$index" ray start --address "${CHAMPION_FABRIC_ARRAY[0]}:6379" --node-ip-address "${CHAMPION_FABRIC_ARRAY[$index]}" --block >/dev/null
  done
  fleet_lock_assert_held || return
  fleet_docker "${CHAMPION_HOST_ARRAY[0]}" exec -d "$CHAMPION_RUN_ID-rank0" bash -lc "exec $argv >/run/champion/serve.log 2>&1" >/dev/null

  for _ in $(seq 1 "${CHAMPION_READY_ATTEMPTS:-210}"); do
    api_health && break
    "${CHAMPION_SLEEP_BIN:-sleep}" 10
  done
  api_health || fleet_die 'API readiness timeout' || return
  models_json="$("${CHAMPION_CURL_BIN:-curl}" -fsS "http://$CHAMPION_API_HOST:$CHAMPION_API_PORT/v1/models")"
  "$CHAMPION_PYTHON" -c 'import json,sys
name,root,raw=sys.argv[1:]
p=json.load(open(f"{root}/presets/{name}.json"))
rows=(json.loads(raw).get("data") or [])
assert len(rows)==1 and rows[0].get("id")=="glm-5.2"
assert rows[0].get("max_model_len")==p["effective_engine"]["max_model_len"]' \
    "$canonical" "$CHAMPION_ROOT" "$models_json"
  rank_gate || return
  transport_and_log_gate || return
  chat_probe || fleet_die 'immediate chat failed' || return
  if [[ "$PROFILE" == fast ]]; then
    cache_busted_sparse_indexer_probe || fleet_die 'cache-busted sparse-indexer semantic probe failed' || return
  fi
  READINESS_PASSED=1
  "${CHAMPION_SLEEP_BIN:-sleep}" "${CHAMPION_IDLE_SECONDS:-300}"
  api_health && chat_probe || fleet_die 'post-idle health/chat failed' || return
  logs="$(fleet_docker "${CHAMPION_HOST_ARRAY[0]}" exec "$CHAMPION_RUN_ID-rank0" cat /run/champion/serve.log 2>&1)"
  logs="$(tail -n "+$((CHAMPION_READINESS_LOG_LINES + 1))" <<<"$logs")"
  if grep -Eqi 'Xid|CUDA out of memory|OutOfMemoryError|oom-kill|No available shared memory broadcast block' <<<"$logs"; then
    fleet_die 'post-readiness fatal/allocation warning detected' || return
  fi
  if [[ "$PROFILE" == fast ]]; then
    apply_fast_pin || return
    fleet_lock_assert_held || return
    printf '%s\n' 'launch-result=PASS profile=fast pin-state=pinned'
  else
    fleet_lock_assert_held || return
    printf 'launch-result=PASS profile=%s pin-state=not-applicable\n' "$PROFILE"
  fi
}

case "$MODE" in
  verify-hosts) verify_hosts ;;
  launch) launch_profile ;;
  *) printf 'usage: %s {verify-hosts|launch} [fast|balanced|large]\n' "$0" >&2; exit 64 ;;
esac
