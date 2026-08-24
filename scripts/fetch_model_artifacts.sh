#!/usr/bin/env bash
# Copyright 2026 dfi / 0xdfi
# SPDX-License-Identifier: Apache-2.0
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_DIR=''
CHECKPOINT_LOCK="$ROOT/models/checkpoint.lock.json"
SIDECAR_LOCK="$ROOT/models/sidecar.lock.json"

while (($#)); do
  case "$1" in
    --model-dir) MODEL_DIR="${2:?missing --model-dir value}"; shift 2 ;;
    --checkpoint-lock) CHECKPOINT_LOCK="${2:?missing lock path}"; shift 2 ;;
    --sidecar-lock) SIDECAR_LOCK="${2:?missing lock path}"; shift 2 ;;
    --remote)
      printf '%s\n' 'refusing: no approved remote is recorded; stage artifacts through the rights holder and verify locally' >&2
      exit 3
      ;;
    *) printf 'unknown argument: %s\n' "$1" >&2; exit 64 ;;
  esac
done

[[ -n "$MODEL_DIR" ]] || { printf '%s\n' 'missing --model-dir' >&2; exit 64; }
PYTHON="$ROOT/.venv/bin/python"
[[ -x "$PYTHON" ]] || { printf '%s\n' 'development environment missing; run ./bin/champion bootstrap' >&2; exit 69; }
"$PYTHON" "$ROOT/scripts/verify_model.py" \
  --checkpoint-lock "$CHECKPOINT_LOCK" \
  --sidecar-lock "$SIDECAR_LOCK" \
  --model-dir "$MODEL_DIR" >/dev/null
printf '%s\n' 'model-artifacts: PASS source=already-staged'
