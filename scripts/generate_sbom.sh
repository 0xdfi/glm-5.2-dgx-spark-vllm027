#!/usr/bin/env bash
# Copyright 2026 dfi / 0xdfi
# SPDX-License-Identifier: Apache-2.0
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE_DIGEST=""
BUILD_RECEIPT=""
OUTPUT=""
RECEIPT=""
while (($#)); do
  case "$1" in
    --image-digest) IMAGE_DIGEST="${2:-}"; shift 2 ;;
    --build-receipt) BUILD_RECEIPT="${2:-}"; shift 2 ;;
    --output) OUTPUT="${2:-}"; shift 2 ;;
    --receipt) RECEIPT="${2:-}"; shift 2 ;;
    *) printf 'sbom: unknown argument %s\n' "$1" >&2; exit 64 ;;
  esac
done
[[ "$IMAGE_DIGEST" =~ ^sha256:[0-9a-f]{64}$ ]] || { printf '%s\n' 'sbom: immutable image digest required' >&2; exit 78; }
[[ -r "$BUILD_RECEIPT" && -n "$OUTPUT" && -n "$RECEIPT" ]] || { printf '%s\n' 'sbom: matching build receipt, output, and receipt paths are required' >&2; exit 78; }
SYFT_IMAGE="$("$ROOT/.venv/bin/python" -c 'import json,sys; print(json.load(open(sys.argv[1]))["syft"]["tool_image"] or "")' "$ROOT/compliance/sbom-policy.json")"
[[ "$SYFT_IMAGE" =~ @sha256:[0-9a-f]{64}$ ]] || { printf '%s\n' 'sbom: BLOCKED: immutable Syft tool image digest is not approved' >&2; exit 78; }
"$ROOT/.venv/bin/python" - "$BUILD_RECEIPT" "$IMAGE_DIGEST" <<'PY'
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
from champion_runtime.receipts import verify_receipt
r = verify_receipt(json.load(open(sys.argv[1])))
if r["receipt_type"] != "build-produced" or r["result"] != "PASS" or r["subject"].get("image_digest") != sys.argv[2]:
    raise SystemExit("sbom: build receipt does not match image digest")
PY
# No invocation can reach this line until compliance/sbom-policy.json carries an
# audited immutable Syft image. When approved, Docker is constrained to inspect
# the immutable subject and produces SPDX JSON directly from the image.
docker run --rm --network=none -v /var/run/docker.sock:/var/run/docker.sock "$SYFT_IMAGE" "$IMAGE_DIGEST" -o spdx-json="$OUTPUT"
printf '%s\n' 'sbom: generated; receipt emission requires the trusted build workflow' >&2
exit 78
