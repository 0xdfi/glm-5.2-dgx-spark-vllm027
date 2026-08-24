#!/usr/bin/env bash
# Copyright 2026 dfi / 0xdfi
# SPDX-License-Identifier: Apache-2.0
set -Eeuo pipefail

CHAMPION_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CHAMPION_PYTHON="$CHAMPION_ROOT/.venv/bin/python"
[[ -x "$CHAMPION_PYTHON" ]] || { printf '%s\n' 'development environment missing; run ./bin/champion bootstrap' >&2; exit 69; }
# shellcheck source=scripts/lib/fleet.sh
source "$CHAMPION_ROOT/scripts/lib/fleet.sh"
trap fleet_lock_release EXIT
fleet_load_cluster
fleet_lock_acquire
fleet_stop_exact_ordered "${1:-$CHAMPION_RUN_ID}"
printf 'stop-result=PASS run=%s containers=preserved images=preserved\n' "${1:-$CHAMPION_RUN_ID}"
