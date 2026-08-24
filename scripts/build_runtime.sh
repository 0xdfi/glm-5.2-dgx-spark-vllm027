#!/usr/bin/env bash
# Copyright 2026 dfi / 0xdfi
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
TAG=""
RECEIPT=""
SECRET_ARGS=()
while (($#)); do
  case "$1" in
    --tag) TAG=${2:?missing --tag value}; shift 2 ;;
    --receipt) RECEIPT=${2:?missing --receipt value}; shift 2 ;;
    --secret-id)
      [[ ${2:-} =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] || { echo "invalid BuildKit secret ID" >&2; exit 2; }
      SECRET_ARGS+=(--secret "id=$2,env=$2"); shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ -n "$TAG" && -n "$RECEIPT" ]] || { echo "usage: $0 --tag TAG --receipt PATH [--secret-id ENV_NAME]" >&2; exit 2; }
docker buildx version >/dev/null 2>&1 || { echo "Docker BuildKit/buildx is required" >&2; exit 2; }

IDS=$(
  cd "$ROOT"
  .venv/bin/python scripts/build_subject.py
)
BUILD_SUBJECT_ID=$(printf '%s' "$IDS" | "$ROOT/.venv/bin/python" -c 'import json,sys; print(json.load(sys.stdin)["build_subject_id"])')
SOURCE_RELEASE_ID=$(printf '%s' "$IDS" | "$ROOT/.venv/bin/python" -c 'import json,sys; print(json.load(sys.stdin)["source_release_id"])')

docker buildx build --load \
  --file "$ROOT/docker/Dockerfile.champion" \
  --tag "$TAG" \
  --build-arg "BUILD_SUBJECT_ID=$BUILD_SUBJECT_ID" \
  --build-arg "SOURCE_RELEASE_ID=$SOURCE_RELEASE_ID" \
  "${SECRET_ARGS[@]}" "$ROOT"
IMAGE_DIGEST=$(docker image inspect --format '{{.Id}}' "$TAG")
(cd "$ROOT" && "$ROOT/.venv/bin/python" scripts/build_subject.py --image-digest "$IMAGE_DIGEST" --receipt "$RECEIPT" >/dev/null)
echo "build-produced: $IMAGE_DIGEST"
