#!/usr/bin/env bash
# Copyright 2026 dfi / 0xdfi
# SPDX-License-Identifier: Apache-2.0

fleet_die() { printf 'champion-controller: BLOCKED: %s\n' "$*" >&2; return 1; }

fleet_load_cluster() {
  local file="${CHAMPION_CLUSTER_ENV:-$CHAMPION_ROOT/cluster.env}" line name value
  [[ -r "$file" ]] || fleet_die "cluster configuration is unreadable: $file" || return
  "$CHAMPION_PYTHON" "$CHAMPION_ROOT/scripts/render_launch.py" \
    --preset fast --cluster "$file" --format engine >/dev/null || return
  while IFS= read -r line || [[ -n "$line" ]]; do
    line="${line#"${line%%[![:space:]]*}"}"
    [[ -z "$line" || "$line" == \#* ]] && continue
    name="${line%%=*}"; value="${line#*=}"
    if [[ "$value" == \"*\" && "$value" == *\" ]]; then value="${value:1:${#value}-2}"; fi
    if [[ "$value" == \'*\' && "$value" == *\' ]]; then value="${value:1:${#value}-2}"; fi
    printf -v "$name" '%s' "$value"
  done < "$file"
  IFS=',' read -r -a CHAMPION_HOST_ARRAY <<<"$CHAMPION_HOSTS"
  IFS=',' read -r -a CHAMPION_FABRIC_ARRAY <<<"$CHAMPION_FABRIC_IPS"
  [[ ${#CHAMPION_HOST_ARRAY[@]} -eq 4 && ${#CHAMPION_FABRIC_ARRAY[@]} -eq 4 ]] || \
    fleet_die 'exactly four hosts and fabric addresses are required'
  CHAMPION_CHECKPOINT_LOCK_PATH="$CHAMPION_CHECKPOINT_LOCK"
  CHAMPION_SIDECAR_LOCK_PATH="$CHAMPION_SIDECAR_LOCK"
  [[ "$CHAMPION_CHECKPOINT_LOCK_PATH" == /* ]] || CHAMPION_CHECKPOINT_LOCK_PATH="$CHAMPION_ROOT/$CHAMPION_CHECKPOINT_LOCK_PATH"
  [[ "$CHAMPION_SIDECAR_LOCK_PATH" == /* ]] || CHAMPION_SIDECAR_LOCK_PATH="$CHAMPION_ROOT/$CHAMPION_SIDECAR_LOCK_PATH"
}

fleet_remote() {
  local host="$1"; shift
  "${CHAMPION_SSH_BIN:-ssh}" "$host" "$@" </dev/null
}

fleet_remote_with_stdin() {
  local host="$1"; shift
  "${CHAMPION_SSH_BIN:-ssh}" "$host" "$@"
}

fleet_docker() {
  local host="$1"; shift
  fleet_remote "$host" "${CHAMPION_REMOTE_DOCKER_BIN:-docker}" "$@"
}

fleet_validate_protected_pattern() {
  local pattern="${CHAMPION_PROTECTED_CONTAINER_PATTERN:-^(hermes|openclaw|protected)}"
  local grep_bin="${CHAMPION_PATTERN_GREP_BIN:-grep}" rc
  if "$grep_bin" -E -- "$pattern" </dev/null >/dev/null 2>&1; then
    rc=0
  else
    rc=$?
  fi
  case "$rc" in
    0|1) return 0 ;;
    *) fleet_die 'protected container pattern is not valid ERE'; return 1 ;;
  esac
}

fleet_match_protected_containers() {
  local containers="$1"
  local pattern="${CHAMPION_PROTECTED_CONTAINER_PATTERN:-^(hermes|openclaw|protected)}"
  local grep_bin="${CHAMPION_PATTERN_GREP_BIN:-grep}"
  printf '%s\n' "$containers" | "$grep_bin" -E -- "$pattern"
}

fleet_validate_lock_path() {
  local lock_path="${CHAMPION_FLEET_LOCK:-}"
  [[ "$lock_path" =~ ^/[A-Za-z0-9._/-]+$ ]] || {
    fleet_die 'CHAMPION_FLEET_LOCK must be a safe absolute path'
    return 1
  }
  [[ "$lock_path" != *'//'* && "$lock_path" != */../* && "$lock_path" != */.. && "$lock_path" != */./* && "$lock_path" != */. ]] || {
    fleet_die 'CHAMPION_FLEET_LOCK must be canonical'
    return 1
  }
}

fleet_lock_acquire() {
  local host command _ ready='' error_text=''
  [[ -z "${CHAMPION_REMOTE_LOCK_PID:-}" ]] || {
    fleet_die 'canonical fleet lock is already held by this coordinator'
    return 1
  }
  fleet_validate_lock_path || return
  host="${CHAMPION_HOST_ARRAY[0]}"
  CHAMPION_REMOTE_LOCK_STATE_DIR="$(mktemp -d "${TMPDIR:-/tmp}/champion-fleet-lock.XXXXXX")" || {
    fleet_die 'cannot create local fleet-lock state directory'
    return 1
  }
  CHAMPION_REMOTE_LOCK_FIFO="$CHAMPION_REMOTE_LOCK_STATE_DIR/stdin.fifo"
  CHAMPION_REMOTE_LOCK_READY_FILE="$CHAMPION_REMOTE_LOCK_STATE_DIR/stdout"
  CHAMPION_REMOTE_LOCK_ERROR_FILE="$CHAMPION_REMOTE_LOCK_STATE_DIR/stderr"
  mkfifo "$CHAMPION_REMOTE_LOCK_FIFO" || {
    fleet_die 'cannot create local fleet-lock control pipe'
    fleet_lock_release
    return 1
  }
  command=": CHAMPION_FLEET_LOCK_PROTOCOL_V1; umask 077; lock_path=$CHAMPION_FLEET_LOCK; lock_parent=\${lock_path%/*}; command -v flock >/dev/null 2>&1 || exit 69; mkdir -p -- \"\$lock_parent\" || exit 70; : >>\"\$lock_path\" || exit 70; chmod 0600 -- \"\$lock_path\" || exit 70; exec 9>\"\$lock_path\"; if ! flock -n 9; then printf '%s\\n' CHAMPION_FLEET_LOCK_BUSY >&2; exit 73; fi; printf '%s\\n' CHAMPION_FLEET_LOCK_READY; cat >/dev/null"
  "${CHAMPION_SSH_BIN:-ssh}" "$host" "$command" \
    <"$CHAMPION_REMOTE_LOCK_FIFO" \
    >"$CHAMPION_REMOTE_LOCK_READY_FILE" \
    2>"$CHAMPION_REMOTE_LOCK_ERROR_FILE" &
  CHAMPION_REMOTE_LOCK_PID=$!
  exec 8>"$CHAMPION_REMOTE_LOCK_FIFO"
  CHAMPION_REMOTE_LOCK_WRITER_OPEN=1
  for _ in $(seq 1 "${CHAMPION_LOCK_READY_ATTEMPTS:-100}"); do
    if [[ -r "$CHAMPION_REMOTE_LOCK_READY_FILE" ]]; then
      ready="$(awk 'NR==1 {print; exit}' "$CHAMPION_REMOTE_LOCK_READY_FILE")"
    fi
    if [[ "$ready" == CHAMPION_FLEET_LOCK_READY ]]; then
      fleet_lock_assert_held || {
        fleet_lock_release
        return 1
      }
      return 0
    fi
    if ! kill -0 "$CHAMPION_REMOTE_LOCK_PID" 2>/dev/null; then
      break
    fi
    "${CHAMPION_SLEEP_BIN:-sleep}" "${CHAMPION_LOCK_READY_DELAY_SECONDS:-0.05}"
  done
  if [[ -r "$CHAMPION_REMOTE_LOCK_ERROR_FILE" ]]; then
    error_text="$(<"$CHAMPION_REMOTE_LOCK_ERROR_FILE")"
  fi
  fleet_lock_release
  if [[ "$error_text" == *CHAMPION_FLEET_LOCK_BUSY* ]]; then
    fleet_die "canonical fleet lock is busy on $host: $CHAMPION_FLEET_LOCK"
  else
    fleet_die "canonical fleet lock could not be held on $host"
  fi
  return 1
}

fleet_lock_assert_held() {
  [[ -n "${CHAMPION_REMOTE_LOCK_PID:-}" && "${CHAMPION_REMOTE_LOCK_WRITER_OPEN:-0}" == 1 ]] || {
    fleet_die 'canonical fleet lock is not held'
    return 1
  }
  kill -0 "$CHAMPION_REMOTE_LOCK_PID" 2>/dev/null || {
    fleet_die 'canonical fleet lock liveness was lost'
    return 1
  }
}

fleet_lock_release() {
  local _
  if [[ "${CHAMPION_REMOTE_LOCK_WRITER_OPEN:-0}" == 1 ]]; then
    exec 8>&-
    CHAMPION_REMOTE_LOCK_WRITER_OPEN=0
  fi
  if [[ -n "${CHAMPION_REMOTE_LOCK_PID:-}" ]]; then
    for _ in $(seq 1 "${CHAMPION_LOCK_RELEASE_ATTEMPTS:-40}"); do
      kill -0 "$CHAMPION_REMOTE_LOCK_PID" 2>/dev/null || break
      "${CHAMPION_SLEEP_BIN:-sleep}" "${CHAMPION_LOCK_READY_DELAY_SECONDS:-0.05}"
    done
    if kill -0 "$CHAMPION_REMOTE_LOCK_PID" 2>/dev/null; then
      kill "$CHAMPION_REMOTE_LOCK_PID" 2>/dev/null || true
    fi
    wait "$CHAMPION_REMOTE_LOCK_PID" 2>/dev/null || true
  fi
  [[ -z "${CHAMPION_REMOTE_LOCK_FIFO:-}" ]] || rm -f "$CHAMPION_REMOTE_LOCK_FIFO"
  [[ -z "${CHAMPION_REMOTE_LOCK_READY_FILE:-}" ]] || rm -f "$CHAMPION_REMOTE_LOCK_READY_FILE"
  [[ -z "${CHAMPION_REMOTE_LOCK_ERROR_FILE:-}" ]] || rm -f "$CHAMPION_REMOTE_LOCK_ERROR_FILE"
  [[ -z "${CHAMPION_REMOTE_LOCK_STATE_DIR:-}" ]] || rmdir "$CHAMPION_REMOTE_LOCK_STATE_DIR" 2>/dev/null || true
  unset CHAMPION_REMOTE_LOCK_PID CHAMPION_REMOTE_LOCK_FIFO
  unset CHAMPION_REMOTE_LOCK_READY_FILE CHAMPION_REMOTE_LOCK_ERROR_FILE
  unset CHAMPION_REMOTE_LOCK_STATE_DIR CHAMPION_REMOTE_LOCK_WRITER_OPEN
}

fleet_consume_compaction_nonce_hash() {
  local nonce_hash="$1" run_hash="$2" host command output rc
  [[ "$nonce_hash" =~ ^[0-9a-f]{64}$ && "$run_hash" =~ ^[0-9a-f]{64}$ ]] || {
    fleet_die 'internal compaction receipt hash is invalid'
    return 1
  }
  fleet_lock_assert_held || return
  host="${CHAMPION_HOST_ARRAY[0]}"
  command=": CHAMPION_COMPACTION_RECEIPT_V1 receipt_hash=$nonce_hash run_hash=$run_hash; umask 077; receipt_root=$CHAMPION_FLEET_LOCK.compaction-receipts; receipt_path=\$receipt_root/$nonce_hash; mkdir -p -- \"\$receipt_root\" || exit 70; chmod 0700 -- \"\$receipt_root\" || exit 70; if ! mkdir -- \"\$receipt_path\" 2>/dev/null; then printf '%s\\n' CHAMPION_COMPACTION_RECEIPT_REPLAY >&2; exit 74; fi; printf 'schema_version=1\\nnonce_sha256=$nonce_hash\\nrun_sha256=$run_hash\\n' >\"\$receipt_path/receipt\" || exit 70; chmod 0600 -- \"\$receipt_path/receipt\" || exit 70; printf '%s\\n' CHAMPION_COMPACTION_RECEIPT_CONSUMED"
  if output="$(fleet_remote "$host" "$command" 2>&1)"; then
    rc=0
  else
    rc=$?
  fi
  fleet_lock_assert_held || return
  if (( rc == 0 )) && [[ "$output" == *CHAMPION_COMPACTION_RECEIPT_CONSUMED* ]]; then
    printf 'compaction-receipt=CONSUMED nonce-sha256=%s run-sha256=%s\n' "${nonce_hash:0:16}" "${run_hash:0:16}" >&2
    return 0
  fi
  if [[ "$output" == *CHAMPION_COMPACTION_RECEIPT_REPLAY* ]]; then
    fleet_die 'compaction nonce was already consumed; retry with a fresh nonce'
  else
    fleet_die 'canonical compaction receipt could not be written'
  fi
  return 1
}

fleet_exact_identity() {
  local host="$1" name="$2" expected_run="$3" state
  if ! state="$(fleet_docker "$host" inspect "$name" --format '{{.Image}}|{{index .Config.Labels "org.0xdfi.champion.run_id"}}|{{.State.Running}}' 2>/dev/null)"; then
    printf '%s\n' ABSENT
    return 0
  fi
  if [[ "$state" == "$CHAMPION_IMAGE_ID|$expected_run|true" || "$state" == "$CHAMPION_IMAGE_ID|$expected_run|false" ]]; then
    printf '%s\n' EXACT
  else
    printf 'UNKNOWN:%s\n' "$state"
  fi
}

fleet_assert_exact() {
  local state
  state="$(fleet_exact_identity "$1" "$2" "$3")"
  [[ "$state" == EXACT ]] || fleet_die "UNKNOWN identity for $1/$2 ($state)"
}

fleet_stop_exact_ordered() {
  local run="$1" index host name state
  for index in 3 2 1 0; do
    fleet_lock_assert_held || return
    host="${CHAMPION_HOST_ARRAY[$index]}"; name="$run-rank$index"
    state="$(fleet_exact_identity "$host" "$name" "$run")"
    case "$state" in
      EXACT) fleet_docker "$host" stop -t 60 "$name" >/dev/null ;;
      ABSENT) ;;
      *) fleet_die "UNKNOWN identity for $host/$name; no mutation authorized"; return 1 ;;
    esac
  done
}

fleet_observe_failed_three_times() {
  local run="$1" index host name state
  for _ in 1 2 3; do
    for index in 0 1 2 3; do
      host="${CHAMPION_HOST_ARRAY[$index]}"; name="$run-rank$index"
      state="$(fleet_exact_identity "$host" "$name" "$run")"
      [[ "$state" == EXACT ]] || return 1
      status="$(fleet_docker "$host" inspect "$name" --format '{{.State.Running}}|{{.State.OOMKilled}}|{{.RestartCount}}' 2>/dev/null || true)"
      [[ "$status" != 'true|false|0' ]] || return 1
    done
    "${CHAMPION_SLEEP_BIN:-sleep}" "${CHAMPION_FAILURE_OBSERVATION_SECONDS:-2}"
  done
}

fleet_verify_model_files() {
  local host="$1" lock="$2" lines relative expected actual
  lines="$("$CHAMPION_PYTHON" - "$lock" <<'PY'
import json,sys
p=json.load(open(sys.argv[1]))
i=p.get('integrity',{})
if i.get('status')!='complete' or not i.get('files'):
    raise SystemExit(3)
for row in i['files']:
    print(row['path']+'|'+row['sha256'])
PY
)" || { fleet_die "model lock lacks complete whole-file hashes: $lock"; return 1; }
  while IFS='|' read -r relative expected; do
    [[ -n "$relative" ]] || continue
    actual="$(fleet_remote "$host" sha256sum "$CHAMPION_MODEL_PATH/$relative" 2>/dev/null | awk '{print $1}')"
    [[ "$actual" == "$expected" ]] || { fleet_die "$host model artifact mismatch: $relative"; return 1; }
  done <<<"$lines"
}
