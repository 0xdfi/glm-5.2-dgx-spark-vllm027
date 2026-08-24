#!/usr/bin/env python3
"""Verify a locally staged model and sidecar against complete lock manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import sys
from pathlib import Path, PurePosixPath
from typing import Any


class ModelVerificationError(ValueError):
    """Raised when an artifact lock or local artifact is incomplete or wrong."""


def load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ModelVerificationError(f"cannot load lock {path}: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != "1.0.0":
        raise ModelVerificationError(f"{path}: unsupported or missing schema_version")
    return payload


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_artifact(root: Path, relative_value: str) -> Path:
    relative = PurePosixPath(relative_value)
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise ModelVerificationError(f"unsafe artifact path: {relative_value!r}")
    current = root
    for part in relative.parts:
        current /= part
        try:
            mode = current.lstat().st_mode
        except FileNotFoundError as exc:
            raise ModelVerificationError(f"missing artifact: {relative.as_posix()}") from exc
        if stat.S_ISLNK(mode):
            raise ModelVerificationError(f"symbolic-link artifact is forbidden: {relative}")
    if not current.is_file():
        raise ModelVerificationError(f"artifact is not a regular file: {relative}")
    return current


def verify_lock(lock_path: Path, model_dir: Path, expected_type: str) -> list[dict[str, Any]]:
    lock = load_json(lock_path)
    if lock.get("artifact_type") != expected_type:
        raise ModelVerificationError(
            f"{lock_path}: expected artifact_type {expected_type!r}"
        )
    integrity = lock.get("integrity")
    if not isinstance(integrity, dict) or integrity.get("status") != "complete":
        raise ModelVerificationError(
            f"{lock_path}: complete whole-file checksum manifest is unavailable"
        )
    files = integrity.get("files")
    if not isinstance(files, list) or not files:
        raise ModelVerificationError(f"{lock_path}: complete manifest has no files")
    seen: set[str] = set()
    verified: list[dict[str, Any]] = []
    for index, item in enumerate(files):
        if not isinstance(item, dict) or set(item) != {"path", "sha256", "size"}:
            raise ModelVerificationError(f"{lock_path}: malformed integrity.files[{index}]")
        relative = item["path"]
        if not isinstance(relative, str) or relative in seen:
            raise ModelVerificationError(f"{lock_path}: duplicate or invalid path {relative!r}")
        seen.add(relative)
        expected_hash = item["sha256"]
        expected_size = item["size"]
        if (
            not isinstance(expected_hash, str)
            or len(expected_hash) != 64
            or any(character not in "0123456789abcdef" for character in expected_hash)
            or not isinstance(expected_size, int)
            or expected_size < 0
        ):
            raise ModelVerificationError(f"{lock_path}: invalid checksum entry for {relative}")
        path = safe_artifact(model_dir, relative)
        actual_size = path.stat().st_size
        if actual_size != expected_size:
            raise ModelVerificationError(
                f"{relative}: size mismatch: got {actual_size}, expected {expected_size}"
            )
        actual_hash = sha256_file(path)
        if actual_hash != expected_hash:
            raise ModelVerificationError(
                f"{relative}: SHA-256 mismatch: got {actual_hash}, expected {expected_hash}"
            )
        verified.append({"path": relative, "sha256": actual_hash, "size": actual_size})
    return verified


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-lock", type=Path, default=Path("models/checkpoint.lock.json"))
    parser.add_argument("--sidecar-lock", type=Path, default=Path("models/sidecar.lock.json"))
    parser.add_argument("--model-dir", "--fixture", dest="model_dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        root = args.model_dir.resolve(strict=True)
        if not root.is_dir() or stat.S_ISLNK(args.model_dir.lstat().st_mode):
            raise ModelVerificationError(f"model directory is not a real directory: {args.model_dir}")
        verified = [
            *verify_lock(args.checkpoint_lock, root, "checkpoint"),
            *verify_lock(args.sidecar_lock, root, "sidecar"),
        ]
        print(json.dumps({"result": "PASS", "verified_files": len(verified), "files": verified}, sort_keys=True))
        return 0
    except (ModelVerificationError, OSError) as exc:
        print(f"model-verification: FAIL: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
