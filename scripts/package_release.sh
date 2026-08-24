#!/usr/bin/env bash
# Copyright 2026 dfi / 0xdfi
# SPDX-License-Identifier: Apache-2.0
set -Eeuo pipefail

CHAMPION_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT=""
while (($#)); do
  case "$1" in
    --output) OUTPUT=${2:?missing --output value}; shift 2 ;;
    *) printf 'package-release: unknown argument: %s\n' "$1" >&2; exit 2 ;;
  esac
done
[[ -n "$OUTPUT" ]] || { printf '%s\n' 'usage: scripts/package_release.sh --output DIRECTORY' >&2; exit 2; }

case "$OUTPUT" in
  /*) ;;
  *) OUTPUT="$PWD/$OUTPUT" ;;
esac
mkdir -p "$OUTPUT"
OUTPUT="$(cd "$OUTPUT" && pwd)"
CHAMPION_PYTHON="${CHAMPION_PYTHON:-$CHAMPION_ROOT/.venv/bin/python}"
[[ -x "$CHAMPION_PYTHON" ]] || { printf '%s\n' 'package-release: run ./bin/champion bootstrap first' >&2; exit 69; }

PACKAGE_TMP="$(mktemp -d "${TMPDIR:-/tmp}/champion-source-package.XXXXXX")"
cleanup() {
  [[ -n "${PACKAGE_TMP:-}" && -d "$PACKAGE_TMP" ]] && rm -rf -- "$PACKAGE_TMP"
}
trap cleanup EXIT

export CHAMPION_PACKAGE_ROOT="$CHAMPION_ROOT"
export CHAMPION_PACKAGE_TMP="$PACKAGE_TMP"
export CHAMPION_PACKAGE_OUTPUT="$OUTPUT"
"$CHAMPION_PYTHON" - <<'PY'
from __future__ import annotations

import gzip
import hashlib
import io
import json
import os
import re
import subprocess
import tarfile
from pathlib import Path, PurePosixPath

root = Path(os.environ["CHAMPION_PACKAGE_ROOT"])
stage = Path(os.environ["CHAMPION_PACKAGE_TMP"]) / "tree"
output = Path(os.environ["CHAMPION_PACKAGE_OUTPUT"])
stage.mkdir(mode=0o755)

prohibited = (
    ".safetensors", ".bin", ".pt", ".pth", ".onnx", ".gguf", ".whl",
    ".so", ".a", ".o", ".tar", ".tar.gz", ".tgz", ".zip", ".7z",
)
excluded_roots = {".git", "evidence"}
excluded_exact = {"cluster.env"}

raw = subprocess.run(
    ["git", "-C", str(root), "ls-files", "--stage", "-z"],
    check=True,
    capture_output=True,
).stdout
entries = []
for item in raw.split(b"\0"):
    if not item:
        continue
    metadata, separator, raw_path = item.partition(b"\t")
    if not separator:
        raise SystemExit("package-release: malformed Git index")
    mode, oid, stage_number = metadata.decode("ascii").split()
    path = raw_path.decode("utf-8")
    relative = PurePosixPath(path)
    if stage_number != "0" or relative.is_absolute() or ".." in relative.parts:
        raise SystemExit(f"package-release: unsafe/conflicted index entry: {path}")
    if relative.parts[0] in excluded_roots or path in excluded_exact:
        continue
    if path.casefold().endswith(prohibited):
        raise SystemExit(f"package-release: prohibited binary/model/archive path: {path}")
    if mode not in {"100644", "100755"}:
        raise SystemExit(f"package-release: unsupported non-regular index entry: {path}")
    content = subprocess.run(
        ["git", "-C", str(root), "cat-file", "blob", oid],
        check=True,
        capture_output=True,
    ).stdout
    target = stage.joinpath(*relative.parts)
    target.parent.mkdir(parents=True, exist_ok=True, mode=0o755)
    target.write_bytes(content)
    target.chmod(0o755 if mode == "100755" else 0o644)
    entries.append({
        "path": path,
        "mode": mode,
        "sha256": hashlib.sha256(content).hexdigest(),
        "size": len(content),
    })

entries.sort(key=lambda item: item["path"])
manifest = {
    "schema_version": "1.0.0",
    "archive_root": "glm52-champion-source",
    "files": entries,
}
manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
manifest_path = output / "source-manifest.json"
manifest_path.write_bytes(manifest_bytes)
manifest_path.chmod(0o644)

archive_path = output / "glm52-champion-source.tar.gz"
with archive_path.open("wb") as raw_stream:
    with gzip.GzipFile(filename="", mode="wb", fileobj=raw_stream, mtime=0) as compressed:
        with tarfile.open(fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT) as archive:
            directories = {PurePosixPath("glm52-champion-source")}
            for entry in entries:
                current = PurePosixPath("glm52-champion-source") / entry["path"]
                directories.update(current.parents)
            for directory in sorted(directories, key=lambda value: (len(value.parts), value.as_posix())):
                info = tarfile.TarInfo(directory.as_posix() + "/")
                info.type = tarfile.DIRTYPE
                info.mode = 0o755
                info.uid = info.gid = 0
                info.uname = info.gname = ""
                info.mtime = 0
                archive.addfile(info)
            for entry in entries:
                content = stage.joinpath(*PurePosixPath(entry["path"]).parts).read_bytes()
                info = tarfile.TarInfo(f"glm52-champion-source/{entry['path']}")
                info.size = len(content)
                info.mode = 0o755 if entry["mode"] == "100755" else 0o644
                info.uid = info.gid = 0
                info.uname = info.gname = ""
                info.mtime = 0
                archive.addfile(info, io.BytesIO(content))
archive_path.chmod(0o644)

checksums = []
for path in (archive_path, manifest_path):
    checksums.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}")
checksum_path = output / "SHA256SUMS"
checksum_path.write_text("\n".join(checksums) + "\n", encoding="utf-8")
checksum_path.chmod(0o644)
PY

# Scan the exact staged archive contents, not the developer working tree.
"$CHAMPION_ROOT/scripts/scan_secrets.sh" \
  --mode image \
  --root "$PACKAGE_TMP/tree" \
  --report "$PACKAGE_TMP/package-secret-scan.json" >/dev/null

printf '%s\n' "package-release: PASS: $OUTPUT"
