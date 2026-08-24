#!/usr/bin/env python3
"""Audit license, REUSE, and per-file provenance invariants."""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import re
import stat
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from champion_runtime.receipts import construct_receipt, source_release_id

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.9 bootstrap environments provide tomli.
    import tomli as tomllib

from register_provenance import (
    KNOWN_CLASSIFICATIONS,
    LEDGER_PATH,
    MODIFIER,
    OWNER,
    PROJECTION_PATH,
    SELF_EXEMPT_PATHS,
    json_bytes,
    projection_for,
)


BASE_COMMIT = "d1aca44241a5100bf7af6a5388985cb42ffca4cd"
BASE_REPOSITORY = "https://github.com/0xdfi/glm-5.2-dgx-spark-vllm027.git"
REQUIRED_FIELDS = {
    "path",
    "classification",
    "copyright",
    "license",
    "origin_repository",
    "origin_commit",
    "origin_path",
    "base_sha256",
    "target_sha256",
    "modified_by",
    "evidence",
}
SHA256 = re.compile(r"[0-9a-f]{64}")
GIT_SHA = re.compile(r"[0-9a-f]{40}")
REUSE_IGNORED = {"LICENSE", "REUSE.toml"}
FORBIDDEN_ATTRIBUTION = re.compile(
    r"^\s*(?:(?:#|//|/\*|\*|;|<!--)\s*)?"
    r"(?:(?:SPDX-FileCopyrightText:\s*)?Copyright(?: \(c\))?\s+\d{4}\s+@dfi\b"
    r"|(?:name|author|owner|modified_by):\s*[\"']?@dfi\b)",
    re.IGNORECASE | re.MULTILINE,
)


class AuditError(ValueError):
    """Raised for an unreadable audit input."""


def run_git(root: Path, *args: str) -> bytes:
    result = subprocess.run(["git", "-C", str(root), *args], capture_output=True)
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise AuditError(f"git {' '.join(args)} failed: {detail}")
    return result.stdout


def repository_root(requested: Path | None = None) -> Path:
    if requested is not None:
        result = subprocess.run(
            ["git", "-C", str(requested), "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0 or Path(result.stdout.strip()).resolve() != requested.resolve():
            raise AuditError("--root must be the root of a Git working tree")
        return requested.resolve()
    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True
    )
    if result.returncode != 0:
        raise AuditError("license audit requires a Git working tree")
    return Path(result.stdout.strip()).resolve()


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AuditError(f"cannot load {path}: {exc}") from exc


def tracked_paths(root: Path) -> set[str]:
    paths: set[str] = set()
    for item in run_git(root, "ls-files", "-z").split(b"\0"):
        if not item:
            continue
        try:
            path = PurePosixPath(item.decode("utf-8"))
        except UnicodeDecodeError as exc:
            raise AuditError("Git returned a non-UTF-8 path") from exc
        if path.is_absolute() or ".." in path.parts:
            raise AuditError(f"Git returned unsafe path: {path}")
        paths.add(path.as_posix())
    return paths


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_blob_sha256(root: Path, revision: str, path: str) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "--verify", f"{revision}:{path}"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    return hashlib.sha256(run_git(root, "cat-file", "blob", result.stdout.strip())).hexdigest()


def validate_record_shape(record: Any, index: int, errors: list[str]) -> str | None:
    label = f"record[{index}]"
    if not isinstance(record, dict):
        errors.append(f"{label}: must be an object")
        return None
    missing = REQUIRED_FIELDS - set(record)
    extra = set(record) - REQUIRED_FIELDS
    if missing:
        errors.append(f"{label}: missing fields {sorted(missing)}")
    if extra:
        errors.append(f"{label}: unknown fields {sorted(extra)}")
    path = record.get("path")
    if not isinstance(path, str) or not path or PurePosixPath(path).is_absolute() or ".." in PurePosixPath(path).parts:
        errors.append(f"{label}: invalid path")
        return None
    classification = record.get("classification")
    if classification not in KNOWN_CLASSIFICATIONS:
        errors.append(f"{path}: unknown classification {classification!r}")
    for field in ("copyright", "license"):
        if not isinstance(record.get(field), str) or not record[field]:
            errors.append(f"{path}: {field} must be non-empty")
    for field in ("origin_repository", "origin_path", "modified_by"):
        if record.get(field) is not None and not isinstance(record[field], str):
            errors.append(f"{path}: {field} must be a string or null")
    origin_commit = record.get("origin_commit")
    if origin_commit is not None and (
        not isinstance(origin_commit, str) or not GIT_SHA.fullmatch(origin_commit)
    ):
        errors.append(f"{path}: origin_commit must be a lowercase full Git SHA or null")
    for field in ("base_sha256", "target_sha256"):
        value = record.get(field)
        if value is None and field == "base_sha256":
            continue
        if not isinstance(value, str) or not SHA256.fullmatch(value):
            errors.append(f"{path}: {field} must be a lowercase SHA-256")
    evidence = record.get("evidence")
    if not isinstance(evidence, list):
        errors.append(f"{path}: evidence must be an array")
    else:
        for receipt in evidence:
            if not isinstance(receipt, dict) or set(receipt) != {"path", "sha256"}:
                errors.append(f"{path}: malformed evidence receipt")
                continue
            if receipt["path"] in SELF_EXEMPT_PATHS:
                errors.append(f"{path}: recursive ledger/projection evidence hash is forbidden")
            if not isinstance(receipt["sha256"], str) or not SHA256.fullmatch(receipt["sha256"]):
                errors.append(f"{path}: evidence sha256 is invalid")
    return path


def validate_classification(record: dict[str, Any], root: Path, errors: list[str]) -> None:
    path = record["path"]
    classification = record["classification"]
    license_id = record["license"]
    copyright_text = record["copyright"]
    if license_id == "NOASSERTION":
        errors.append(f"{path}: NOASSERTION is forbidden")
    if "@dfi" in json.dumps(record, sort_keys=True):
        errors.append(f"{path}: @dfi is forbidden; use dfi / 0xdfi")
    if classification in {
        "project-apache",
        "generated-evidence",
        "license-text-apache",
        "license-text-bsd-3-clause",
    } and copyright_text != OWNER:
        errors.append(f"{path}: project copyright must be exactly {OWNER!r}")
    if classification in {"project-apache", "generated-evidence", "license-text-apache"} and license_id != "Apache-2.0":
        errors.append(f"{path}: {classification} must use Apache-2.0")
    if classification in {"existing-mit", "contributed-mit", "license-text-mit"} and license_id != "MIT":
        errors.append(f"{path}: {classification} must retain MIT")
    if classification == "license-text-bsd-3-clause" and license_id != "BSD-3-Clause":
        errors.append(f"{path}: BSD license text must use BSD-3-Clause")
    if classification == "existing-mit":
        base_digest = git_blob_sha256(root, BASE_COMMIT, path)
        if base_digest is None:
            errors.append(f"{path}: existing-mit path was not present at {BASE_COMMIT}")
        if record.get("base_sha256") != base_digest:
            errors.append(f"{path}: existing-mit base_sha256 does not match {BASE_COMMIT}")
        if record.get("origin_repository") != BASE_REPOSITORY:
            errors.append(f"{path}: existing-mit origin repository changed")
        if record.get("origin_commit") != BASE_COMMIT or record.get("origin_path") != path:
            errors.append(f"{path}: existing-mit origin identifiers changed")
        if copyright_text != "Copyright (c) 2026":
            errors.append(f"{path}: unnamed MIT notice must be preserved exactly")
    if classification in {"contributed-mit", "upstream-unmodified", "upstream-derivative"}:
        for field in ("origin_repository", "origin_commit", "origin_path", "base_sha256"):
            if not record.get(field):
                errors.append(f"{path}: retained upstream record requires {field}")
    if classification == "upstream-derivative" and record.get("modified_by") != MODIFIER:
        errors.append(f"{path}: upstream derivative must record modification by {MODIFIER}")


def annotation_matches(pattern: str, path: str) -> bool:
    if pattern == "**":
        return True
    return fnmatch.fnmatchcase(path, pattern)


def reuse_license_for(reuse: dict[str, Any], path: str) -> tuple[str | None, str | None]:
    selected: tuple[str | None, str | None] = (None, None)
    for annotation in reuse.get("annotations", []):
        patterns = annotation.get("path", [])
        if isinstance(patterns, str):
            patterns = [patterns]
        if any(annotation_matches(pattern, path) for pattern in patterns):
            selected = (
                annotation.get("SPDX-License-Identifier"),
                annotation.get("SPDX-FileCopyrightText"),
            )
    return selected


def reuse_ignored(path: str) -> bool:
    name = PurePosixPath(path).name
    return (
        name in REUSE_IGNORED
        or name.startswith("LICENSE-")
        or path.startswith("LICENSES/")
    )


def audit(root: Path, strict: bool) -> list[str]:
    errors: list[str] = []
    ledger = load_json(root / LEDGER_PATH)
    if not isinstance(ledger, dict) or ledger.get("schema_version") != "1.0.0" or not isinstance(ledger.get("records"), list):
        return [f"{LEDGER_PATH}: invalid top-level ledger shape"]
    records: dict[str, dict[str, Any]] = {}
    for index, record in enumerate(ledger["records"]):
        path = validate_record_shape(record, index, errors)
        if path is None or not isinstance(record, dict):
            continue
        if path in records:
            errors.append(f"{path}: duplicate provenance record")
        records[path] = record
        if record.get("classification") in KNOWN_CLASSIFICATIONS:
            validate_classification(record, root, errors)

    if SELF_EXEMPT_PATHS.intersection(records):
        errors.append("ledger/projection target-hash records are forbidden")

    tracked = tracked_paths(root)
    expected = tracked - SELF_EXEMPT_PATHS
    missing = sorted(expected - set(records))
    extra = sorted(set(records) - expected)
    if missing:
        errors.append(f"distributed paths absent from provenance: {', '.join(missing)}")
    if extra:
        errors.append(f"provenance records for absent paths: {', '.join(extra)}")

    for path in sorted(expected.intersection(records)):
        full_path = root / path
        try:
            mode = full_path.lstat().st_mode
        except OSError as exc:
            errors.append(f"{path}: cannot stat distributed path: {exc}")
            continue
        if not stat.S_ISREG(mode):
            errors.append(f"{path}: distributed paths must be regular files")
            continue
        if records[path].get("target_sha256") != file_sha256(full_path):
            errors.append(f"{path}: target_sha256 is stale")

    expected_projection = json_bytes(projection_for(ledger))
    try:
        actual_projection = (root / PROJECTION_PATH).read_bytes()
    except OSError as exc:
        errors.append(f"cannot read {PROJECTION_PATH}: {exc}")
    else:
        if actual_projection != expected_projection:
            errors.append(f"{PROJECTION_PATH}: non-deterministic or stale projection")

    try:
        reuse = tomllib.loads((root / "REUSE.toml").read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        errors.append(f"REUSE.toml: {exc}")
        reuse = {}
    if reuse.get("version") != 1:
        errors.append("REUSE.toml: version must be 1")
    for path, record in sorted(records.items()):
        if reuse_ignored(path):
            continue
        license_id, copyright_text = reuse_license_for(reuse, path)
        if license_id != record.get("license"):
            errors.append(f"{path}: REUSE license {license_id!r} != ledger license {record.get('license')!r}")
        if copyright_text != record.get("copyright"):
            errors.append(f"{path}: REUSE copyright does not match ledger")

    for license_id in {record.get("license") for record in records.values()}:
        if isinstance(license_id, str) and not (root / "LICENSES" / f"{license_id}.txt").is_file():
            errors.append(f"missing canonical REUSE license text: LICENSES/{license_id}.txt")

    try:
        original_mit = run_git(root, "show", f"{BASE_COMMIT}:LICENSE")
        preserved_mit = (root / "LICENSES/MIT.txt").read_bytes()
        if original_mit != preserved_mit:
            errors.append("LICENSES/MIT.txt does not preserve the d1aca44 MIT text exactly")
    except (AuditError, OSError) as exc:
        errors.append(str(exc))
    try:
        root_license = (root / "LICENSE").read_text(encoding="utf-8")
        if not root_license.startswith("Apache License\nVersion 2.0, January 2004"):
            errors.append("root LICENSE is not Apache-2.0")
    except OSError as exc:
        errors.append(f"LICENSE: {exc}")

    components = load_json(root / "compliance/container-components.json")
    if "NOASSERTION" in json.dumps(components, sort_keys=True):
        errors.append("compliance/container-components.json contains NOASSERTION")
    if not isinstance(components, dict) or not isinstance(components.get("components"), list):
        errors.append("compliance/container-components.json has invalid shape")

    schema = load_json(root / "schemas/provenance.schema.json")
    try:
        schema_classes = set(schema["$defs"]["record"]["properties"]["classification"]["enum"])
    except (KeyError, TypeError):
        errors.append("schemas/provenance.schema.json has no classification enum")
    else:
        if schema_classes != KNOWN_CLASSIFICATIONS:
            errors.append("provenance schema classification enum is stale")

    notice = (root / "NOTICE").read_text(encoding="utf-8", errors="replace")
    has_upstream_distribution = any(
        record.get("classification") in {"upstream-unmodified", "upstream-derivative"}
        for record in records.values()
    )
    if not has_upstream_distribution and ("vLLM" in notice or "B12X" in notice):
        errors.append("NOTICE names upstream content that is not distributed")

    if strict:
        for path in sorted(tracked):
            full_path = root / path
            if not full_path.is_file():
                continue
            try:
                content = full_path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            if FORBIDDEN_ATTRIBUTION.search(content):
                errors.append(f"{path}: forbidden @dfi attribution")
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--root", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args(argv)
    try:
        root = repository_root(args.root)
        errors = audit(root, args.strict)
    except AuditError as exc:
        errors = [str(exc)]
        root = args.root.resolve() if args.root else Path.cwd()
    report = {
        "schema_version": "1.0.0",
        "scope": "tracked-source-provenance-license-reuse",
        "result": "FAIL" if errors else "PASS",
        "errors": errors,
    }
    rendered = (json.dumps(report, indent=2, sort_keys=True) + "\n").encode()
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_bytes(rendered)
    if errors:
        print("license-audit: FAIL", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1
    if args.receipt:
        if not args.report:
            parser.error("--receipt requires --report")
        evidence_paths = [
            (args.report.name, rendered),
            (LEDGER_PATH, (root / LEDGER_PATH).read_bytes()),
            (PROJECTION_PATH, (root / PROJECTION_PATH).read_bytes()),
            ("REUSE.toml", (root / "REUSE.toml").read_bytes()),
            ("NOTICE", (root / "NOTICE").read_bytes()),
        ]
        for path in sorted((root / "LICENSES").glob("*.txt")):
            evidence_paths.append((path.relative_to(root).as_posix(), path.read_bytes()))
        receipt = construct_receipt({
            "schema_version": "1.0.0",
            "receipt_type": "provenance-license",
            "command": "provenance-license",
            "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "subject": {"source_release_id": source_release_id(root)},
            "evidence": [
                {"path": path, "sha256": hashlib.sha256(content).hexdigest()}
                for path, content in evidence_paths
            ],
            "result": "PASS",
            "metadata": {},
        })
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        args.receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print("license-audit: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
