#!/usr/bin/env bash
# Copyright 2026 dfi / 0xdfi
# SPDX-License-Identifier: Apache-2.0
set -Eeuo pipefail

CHAMPION_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CHAMPION_PYTHON="$CHAMPION_ROOT/.venv/bin/python"
[[ -x "$CHAMPION_PYTHON" ]] || { printf '%s\n' 'development environment missing; run ./bin/champion bootstrap' >&2; exit 69; }
# shellcheck source=scripts/lib/fleet.sh
source "$CHAMPION_ROOT/scripts/lib/fleet.sh"
fleet_load_cluster
trap fleet_lock_release EXIT

STATUS="$CHAMPION_STATUS_FILE"
[[ "$STATUS" == /* ]] || STATUS="$CHAMPION_ROOT/$STATUS"
accepted="$($CHAMPION_PYTHON - "$STATUS" 2>/dev/null <<'PY' || true
import json,sys
try: p=json.load(open(sys.argv[1]))
except Exception: raise SystemExit(1)
b=p.get('accepted_build') or {}
if b.get('status')!='passed' or b.get('authenticated') is not True: raise SystemExit(1)
if not b.get('build_subject_id') or not b.get('image_id') or not b.get('builder_host'): raise SystemExit(1)
print('|'.join((b['build_subject_id'],b['image_id'],b['builder_host'])))
PY
)"
[[ -n "$accepted" ]] || { printf '%s\n' 'stage-image: BLOCKED: authenticated accepted build status is required' >&2; exit 3; }
IFS='|' read -r build_subject image_id builder_host <<<"$accepted"
[[ "$image_id" == "$CHAMPION_IMAGE_ID" ]] || { printf '%s\n' 'stage-image: BLOCKED: status image does not match configured image' >&2; exit 3; }
fleet_lock_acquire

# Staging changes only image storage. It never stops, updates, creates, or removes containers.
for host in "${CHAMPION_HOST_ARRAY[@]}"; do
  fleet_lock_assert_held
  fleet_remote "$host" docker system df >/dev/null
  fleet_remote "$builder_host" docker image inspect "$image_id" --format '{{.Id}}' >/dev/null
  fleet_remote "$builder_host" docker save "$image_id" | fleet_remote "$host" docker load >/dev/null
  got="$(fleet_remote "$host" docker image inspect "$image_id" --format '{{.Id}}')"
  [[ "$got" == "$image_id" ]] || { printf 'stage-image: BLOCKED: %s loaded image mismatch\n' "$host" >&2; exit 1; }
  label="$(fleet_remote "$host" docker image inspect "$image_id" --format '{{index .Config.Labels "org.0xdfi.champion.build_subject_id"}}')"
  [[ "$label" == "$build_subject" ]] || { printf 'stage-image: BLOCKED: %s build subject mismatch\n' "$host" >&2; exit 1; }
done
fleet_lock_assert_held
printf 'stage-image-result=PASS image=%s images-preserved=true\n' "$image_id"
