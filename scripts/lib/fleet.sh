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

fleet_docker() {
  local host="$1"; shift
  fleet_remote "$host" "${CHAMPION_REMOTE_DOCKER_BIN:-docker}" "$@"
}

fleet_lock_acquire() {
  CHAMPION_LOCK_DIR="${CHAMPION_FLEET_LOCK}.coordinator-lock"
  if ! mkdir "$CHAMPION_LOCK_DIR" 2>/dev/null; then
    fleet_die "coordinator-owned fleet lock is busy: $CHAMPION_LOCK_DIR"
    return 1
  fi
  printf 'pid=%s run=%s\n' "$$" "$CHAMPION_RUN_ID" >"$CHAMPION_LOCK_DIR/owner"
}

fleet_lock_release() {
  [[ -n "${CHAMPION_LOCK_DIR:-}" ]] || return 0
  rm -f "$CHAMPION_LOCK_DIR/owner"
  rmdir "$CHAMPION_LOCK_DIR" 2>/dev/null || true
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
