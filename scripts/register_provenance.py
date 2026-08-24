#!/usr/bin/env python3
"""Register explicitly classified staged files in the provenance ledger."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any


OWNER = "Copyright 2026 dfi / 0xdfi"
MODIFIER = "dfi / 0xdfi"
LEGACY_BASE_COMMIT = "d1aca44241a5100bf7af6a5388985cb42ffca4cd"
LEGACY_REPOSITORY = "https://github.com/0xdfi/glm-5.2-dgx-spark-vllm027.git"
SPDX_REPOSITORY = "https://github.com/spdx/license-list-data"
LEDGER_PATH = "compliance/provenance.json"
PROJECTION_PATH = "compliance/source-provenance.json"
SELF_EXEMPT_PATHS = {LEDGER_PATH, PROJECTION_PATH}
SUBJECT_EXCLUDED_PATHS = {
    *SELF_EXEMPT_PATHS,
    "docs/release-status.md",
    "publication/status.json",
}
KNOWN_CLASSIFICATIONS = {
    "project-apache",
    "existing-mit",
    "contributed-mit",
    "upstream-unmodified",
    "upstream-derivative",
    "generated-evidence",
    "license-text-apache",
    "license-text-mit",
    "license-text-bsd-3-clause",
}
POLICY = {
    "project-apache": (OWNER, "Apache-2.0"),
    "existing-mit": ("Copyright (c) 2026", "MIT"),
    "generated-evidence": (OWNER, "Apache-2.0"),
    "license-text-apache": (OWNER, "Apache-2.0"),
    "license-text-mit": ("Copyright (c) 2026", "MIT"),
    "license-text-bsd-3-clause": (OWNER, "BSD-3-Clause"),
}
REQUIRES_PRESEEDED_RECORD = {
    "contributed-mit",
    "upstream-unmodified",
    "upstream-derivative",
}


class ProvenanceError(ValueError):
    """Raised when staged provenance cannot be updated safely."""


def run_git(root: Path, *args: str, input_bytes: bytes | None = None) -> bytes:
    result = subprocess.run(
        ["git", "-C", str(root), *args],
        input=input_bytes,
        capture_output=True,
    )
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise ProvenanceError(f"git {' '.join(args)} failed: {detail}")
    return result.stdout


def repository_root() -> Path:
    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise ProvenanceError("--staged requires a Git working tree")
    return Path(result.stdout.strip()).resolve()


def validate_path(value: str) -> str:
    path = PurePosixPath(value)
    if not value or path.is_absolute() or ".." in path.parts or value.endswith("/"):
        raise ProvenanceError(f"unsafe repository path: {value!r}")
    return path.as_posix()


def parse_classification_maps(values: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values:
        for item in value.split(","):
            if not item:
                continue
            if "=" not in item:
                raise ProvenanceError(
                    f"classification map entry must be PATH=CLASSIFICATION: {item!r}"
                )
            prefix, classification = item.split("=", 1)
            prefix = validate_path(prefix.strip().rstrip("/"))
            classification = classification.strip()
            if classification not in KNOWN_CLASSIFICATIONS:
                choices = ", ".join(sorted(KNOWN_CLASSIFICATIONS))
                raise ProvenanceError(
                    f"unknown classification {classification!r}; expected one of: {choices}"
                )
            previous = result.get(prefix)
            if previous is not None and previous != classification:
                raise ProvenanceError(f"conflicting classifications for {prefix!r}")
            result[prefix] = classification
    return result


def classification_for(path: str, mapping: dict[str, str]) -> str | None:
    candidates = [
        (prefix, classification)
        for prefix, classification in mapping.items()
        if path == prefix or path.startswith(prefix + "/")
    ]
    if not candidates:
        return None
    prefix, classification = max(candidates, key=lambda item: (len(item[0]), item[0]))
    del prefix
    return classification


def staged_changes(root: Path) -> list[tuple[str, str | None, str | None]]:
    raw = run_git(root, "diff", "--cached", "--name-status", "-z", "--find-renames")
    fields = raw.decode("utf-8").split("\0")
    if fields and fields[-1] == "":
        fields.pop()
    changes: list[tuple[str, str | None, str | None]] = []
    index = 0
    while index < len(fields):
        status = fields[index]
        index += 1
        code = status[0]
        if code in {"R", "C"}:
            if index + 1 >= len(fields):
                raise ProvenanceError("malformed staged rename/copy record")
            old_path = validate_path(fields[index])
            new_path = validate_path(fields[index + 1])
            index += 2
            changes.append((code, old_path, new_path))
        else:
            if index >= len(fields):
                raise ProvenanceError("malformed staged path record")
            path = validate_path(fields[index])
            index += 1
            if code == "D":
                changes.append((code, path, None))
            else:
                changes.append((code, None, path))
    return changes


def load_ledger(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProvenanceError(f"cannot load {path}: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != "1.0.0":
        raise ProvenanceError(f"{path}: unsupported or missing schema_version")
    records = payload.get("records")
    if not isinstance(records, list):
        raise ProvenanceError(f"{path}: records must be an array")
    seen: set[str] = set()
    for record in records:
        if not isinstance(record, dict) or not isinstance(record.get("path"), str):
            raise ProvenanceError(f"{path}: every record requires a path")
        record_path = validate_path(record["path"])
        if record_path in seen:
            raise ProvenanceError(f"{path}: duplicate record for {record_path}")
        seen.add(record_path)
    return payload


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def index_bytes(root: Path, path: str) -> bytes:
    listing = run_git(root, "ls-files", "--stage", "-z", "--", path)
    entries = [entry for entry in listing.split(b"\0") if entry]
    stage_zero = []
    for entry in entries:
        metadata, _, stored_path = entry.partition(b"\t")
        parts = metadata.split()
        if len(parts) == 3 and parts[2] == b"0" and stored_path.decode("utf-8") == path:
            stage_zero.append(parts[1].decode("ascii"))
    if len(stage_zero) != 1:
        raise ProvenanceError(f"staged distributed path is absent or conflicted: {path}")
    return run_git(root, "cat-file", "blob", stage_zero[0])


def head_bytes(root: Path, path: str) -> bytes | None:
    result = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "--verify", f"HEAD:{path}"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    return run_git(root, "cat-file", "blob", result.stdout.strip())


def revision_bytes(root: Path, revision: str, path: str) -> bytes | None:
    result = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "--verify", f"{revision}:{path}"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    return run_git(root, "cat-file", "blob", result.stdout.strip())


def head_commit(root: Path) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "--verify", "HEAD"],
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def origin_repository(root: Path) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(root), "remote", "get-url", "origin"],
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def new_record(root: Path, path: str, classification: str) -> dict[str, Any]:
    if classification in REQUIRES_PRESEEDED_RECORD:
        raise ProvenanceError(
            f"{path}: {classification} requires a pre-seeded record with explicit "
            "upstream copyright, license, origin, and base hash"
        )
    copyright_text, license_id = POLICY[classification]
    base_content = head_bytes(root, path)
    commit = head_commit(root) if base_content is not None else None
    repository = origin_repository(root)
    origin_path = path
    if classification == "existing-mit":
        legacy_content = revision_bytes(root, LEGACY_BASE_COMMIT, path)
        if legacy_content is not None:
            base_content = legacy_content
            commit = LEGACY_BASE_COMMIT
            repository = LEGACY_REPOSITORY
    elif classification == "license-text-mit" and path == "LICENSES/MIT.txt":
        base_content = revision_bytes(root, LEGACY_BASE_COMMIT, "LICENSE")
        commit = LEGACY_BASE_COMMIT
        repository = LEGACY_REPOSITORY
        origin_path = "LICENSE"
    elif classification.startswith("license-text-"):
        base_content = index_bytes(root, path)
        commit = None
        repository = SPDX_REPOSITORY
        suffix = {
            "license-text-apache": "Apache-2.0.txt",
            "license-text-bsd-3-clause": "BSD-3-Clause.txt",
        }[classification]
        origin_path = f"text/{suffix}"
    return {
        "path": path,
        "classification": classification,
        "copyright": copyright_text,
        "license": license_id,
        "origin_repository": repository,
        "origin_commit": commit,
        "origin_path": origin_path,
        "base_sha256": sha256_bytes(base_content) if base_content is not None else None,
        "target_sha256": "",
        "modified_by": MODIFIER,
        "evidence": [],
    }


def subject_included(record: dict[str, Any]) -> bool:
    return (
        record.get("classification") != "generated-evidence"
        and record.get("path") not in SUBJECT_EXCLUDED_PATHS
    )


def normalized_payload(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": "1.0.0",
        "records": sorted(records, key=lambda record: record["path"]),
    }


def projection_for(ledger: dict[str, Any]) -> dict[str, Any]:
    return normalized_payload(
        [dict(record) for record in ledger["records"] if subject_included(record)]
    )


def json_bytes(payload: dict[str, Any]) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


def atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def verify_check(
    root: Path,
    ledger: dict[str, Any],
    changes: list[tuple[str, str | None, str | None]],
    mapping: dict[str, str],
) -> None:
    records = {record["path"]: record for record in ledger["records"]}
    errors: list[str] = []
    for code, old_path, new_path in changes:
        if old_path in SELF_EXEMPT_PATHS or new_path in SELF_EXEMPT_PATHS:
            continue
        if code == "D" or code == "R":
            if old_path in records:
                errors.append(f"stale removed provenance record: {old_path}")
        if new_path is None:
            continue
        record = records.get(new_path)
        if record is None:
            errors.append(f"staged distributed path is absent from provenance: {new_path}")
            continue
        declared = classification_for(new_path, mapping)
        if declared is not None and declared != record.get("classification"):
            errors.append(
                f"refusing to relicense {new_path}: ledger has "
                f"{record.get('classification')}, map declares {declared}"
            )
        digest = sha256_bytes(index_bytes(root, new_path))
        if record.get("target_sha256") != digest:
            errors.append(f"stale target_sha256 for staged path: {new_path}")
    expected_projection = json_bytes(projection_for(ledger))
    projection_path = root / PROJECTION_PATH
    try:
        actual_projection = projection_path.read_bytes()
    except OSError as exc:
        errors.append(f"cannot read {PROJECTION_PATH}: {exc}")
    else:
        if actual_projection != expected_projection:
            errors.append(f"{PROJECTION_PATH} is not the deterministic ledger projection")
    if errors:
        raise ProvenanceError("\n".join(errors))


def update_ledger(
    root: Path,
    ledger: dict[str, Any],
    changes: list[tuple[str, str | None, str | None]],
    mapping: dict[str, str],
) -> dict[str, Any]:
    records = {record["path"]: dict(record) for record in ledger["records"]}
    if not records:
        staged_targets = {new_path for _, _, new_path in changes if new_path is not None}
        tracked = [
            item.decode("utf-8")
            for item in run_git(root, "ls-files", "-z").split(b"\0")
            if item
        ]
        for path in sorted(tracked):
            if path not in SELF_EXEMPT_PATHS and path not in staged_targets:
                changes.append(("A", None, validate_path(path)))
    for code, old_path, new_path in changes:
        if old_path in SELF_EXEMPT_PATHS or new_path in SELF_EXEMPT_PATHS:
            continue
        prior: dict[str, Any] | None = None
        if code in {"D", "R"} and old_path is not None:
            prior = records.pop(old_path, None)
            if prior is None:
                raise ProvenanceError(f"removed distributed path is absent from ledger: {old_path}")
        if new_path is None:
            continue
        declared = classification_for(new_path, mapping)
        if declared is None:
            raise ProvenanceError(
                f"{new_path}: explicit classification required via --classification-map"
            )
        existing = records.get(new_path)
        record = prior if code == "R" else existing
        if record is None:
            record = new_record(root, new_path, declared)
        elif record.get("classification") != declared:
            raise ProvenanceError(
                f"refusing to relicense {new_path}: ledger has "
                f"{record.get('classification')}, map declares {declared}"
            )
        previous_digest = record.get("target_sha256")
        target_digest = sha256_bytes(index_bytes(root, new_path))
        if declared == "upstream-unmodified" and previous_digest not in {None, target_digest}:
            raise ProvenanceError(
                f"{new_path}: modified bytes cannot retain upstream-unmodified classification"
            )
        record["path"] = new_path
        record["target_sha256"] = target_digest
        if declared in {"project-apache", "existing-mit", "upstream-derivative"}:
            record["modified_by"] = MODIFIER
        records[new_path] = record
    return normalized_payload(list(records.values()))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staged", action="store_true", help="inspect the Git index")
    parser.add_argument(
        "--classification-map",
        action="append",
        default=[],
        metavar="PATH=CLASSIFICATION,...",
    )
    parser.add_argument("--check", action="store_true", help="validate without writing")
    args = parser.parse_args(argv)
    try:
        if not args.staged:
            raise ProvenanceError("--staged is required")
        root = repository_root()
        mapping = parse_classification_maps(args.classification_map)
        changes = staged_changes(root)
        ledger_path = root / LEDGER_PATH
        ledger = load_ledger(ledger_path)
        if any(record["path"] in SELF_EXEMPT_PATHS for record in ledger["records"]):
            raise ProvenanceError("ledger/projection target-hash records are forbidden")
        if args.check:
            verify_check(root, ledger, changes, mapping)
            print("provenance-check: PASS")
            return 0
        updated = update_ledger(root, ledger, changes, mapping)
        atomic_write(ledger_path, json_bytes(updated))
        atomic_write(root / PROJECTION_PATH, json_bytes(projection_for(updated)))
        print(f"provenance-register: PASS ({len(changes)} staged change(s))")
        return 0
    except ProvenanceError as exc:
        print(f"provenance-register: FAIL\n{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
