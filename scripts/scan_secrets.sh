#!/usr/bin/env bash
# Copyright 2026 dfi / 0xdfi
# SPDX-License-Identifier: Apache-2.0
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export CHAMPION_REPO_ROOT="$ROOT"
exec "${CHAMPION_PYTHON:-$ROOT/.venv/bin/python}" - "$@" <<'PY'
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

root_module = Path(os.environ["CHAMPION_REPO_ROOT"])
sys.path.insert(0, str(root_module))
from champion_runtime.receipts import construct_receipt, source_release_id

RULES = (
    ("private-key", re.compile(r"BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY")),
    ("bearer-credential", re.compile(r"(?i)(?:authorization\s*[:=]|(?:--header|-H)\s+['\"]?authorization:?)\s*['\"]?bearer\s+[A-Za-z0-9._~+/=-]{6,}")),
    ("credential", re.compile(r"(?i)\b(?:api[_-]?key|client[_-]?secret|password|access[_-]?token|secret|token)\b\s*[:=]\s*['\"]?[A-Za-z0-9+/=._-]{8,}['\"]?\s*(?:#.*)?$")),
    ("credential", re.compile(r"(?:ghp_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}|hf_[A-Za-z0-9]{20,}|xox[baprs]-[A-Za-z0-9-]{20,}|AKIA[A-Z0-9]{16})")),
    ("test-canary", re.compile(r"CHAMPION_TEST_SECRET_[A-Za-z0-9]+")),
    ("private-path", re.compile(r"/(?:Users|home)/[^/\s]+(?:/|\b)")),
    ("private-network", re.compile(r"\b(?:10(?:\.\d{1,3}){3}|192\.168(?:\.\d{1,3}){2}|172\.(?:1[6-9]|2\d|3[01])(?:\.\d{1,3}){2})\b")),
)

def git(root: Path, *args: str, data: bytes | None = None, check: bool = True) -> bytes:
    result = subprocess.run(["git", "-C", str(root), *args], input=data, capture_output=True)
    if check and result.returncode:
        raise RuntimeError(result.stderr.decode(errors="replace").strip())
    return result.stdout

def source_blobs(root: Path, revision: str):
    values = []
    if subprocess.run(["git", "-C", str(root), "rev-parse", "--verify", revision], capture_output=True).returncode == 0:
        for line in git(root, "rev-list", "--objects", revision).decode(errors="surrogateescape").splitlines():
            oid, _, path = line.partition(" ")
            if path:
                values.append((oid, path))
    for entry in git(root, "ls-files", "--stage", "-z").split(b"\0"):
        if not entry:
            continue
        meta, _, path = entry.partition(b"\t")
        parts = meta.split()
        if len(parts) == 3 and parts[2] == b"0":
            values.append((parts[1].decode(), path.decode(errors="surrogateescape")))
    seen = set()
    for oid, path in values:
        if (oid, path) in seen:
            continue
        seen.add((oid, path))
        content = git(root, "cat-file", "blob", oid, check=False)
        yield oid, path, content

def bundle_blobs(root: Path):
    for path in sorted(root.rglob("*")):
        if path.is_file() and not path.is_symlink():
            yield hashlib.sha256(path.read_bytes()).hexdigest(), path.relative_to(root).as_posix(), path.read_bytes()

def git_commit_exists(root: Path, revision: str) -> bool:
    return subprocess.run(
        ["git", "-C", str(root), "rev-parse", "--verify", f"{revision}^{{commit}}"],
        capture_output=True,
    ).returncode == 0

def baseline_entries(root: Path, revision: str, path: Path | None):
    if path is None:
        return set(), []
    errors = []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return set(), [f"cannot load exact history baseline: {exc}"]
    required = {"schema_version", "base_commit_scope", "justification", "accepted"}
    if not isinstance(payload, dict) or set(payload) != required:
        return set(), ["baseline must contain exactly schema_version, base_commit_scope, justification, accepted"]
    if payload.get("schema_version") != "1.0.0":
        errors.append("baseline schema_version must be 1.0.0")
    scope = payload.get("base_commit_scope")
    if not isinstance(scope, str) or not re.fullmatch(r"[0-9a-f]{40}", scope):
        errors.append("baseline base_commit_scope must be one full lowercase Git commit")
    elif not git_commit_exists(root, scope):
        errors.append("baseline base_commit_scope is unavailable in the scanned repository")
    elif not git_commit_exists(root, revision) or subprocess.run(
        ["git", "-C", str(root), "merge-base", "--is-ancestor", scope, revision],
        capture_output=True,
    ).returncode != 0:
        errors.append("baseline scope is not an ancestor of the scanned revision")
    justification = payload.get("justification")
    if not isinstance(justification, str) or len(justification.strip()) < 20:
        errors.append("baseline requires a substantive non-secret justification")
    accepted = payload.get("accepted")
    if not isinstance(accepted, list) or not accepted:
        errors.append("baseline accepted must be a non-empty array")
        return set(), errors
    entries = set()
    scoped_objects = set()
    if isinstance(scope, str) and re.fullmatch(r"[0-9a-f]{40}", scope) and git_commit_exists(root, scope):
        scoped_objects = {
            tuple(line.split(" ", 1))
            for line in git(root, "rev-list", "--objects", scope).decode(errors="surrogateescape").splitlines()
            if " " in line
        }
    for index, item in enumerate(accepted):
        label = f"baseline accepted[{index}]"
        if not isinstance(item, dict) or set(item) != {"rule", "object", "path", "line"}:
            errors.append(f"{label} must contain exactly rule, object, path, line")
            continue
        rule, oid, item_path, line = item["rule"], item["object"], item["path"], item["line"]
        # Historical exceptions are intentionally narrower than scanner rules:
        # only exact non-secret RFC1918 documentation locations can qualify.
        if rule != "private-network":
            errors.append(f"{label} may acknowledge only private-network")
        if not isinstance(oid, str) or not re.fullmatch(r"[0-9a-f]{40}", oid):
            errors.append(f"{label} object must be one exact Git blob ID")
        relative = PurePosixPath(item_path) if isinstance(item_path, str) else None
        if relative is None or relative.is_absolute() or ".." in relative.parts or any(char in item_path for char in "*?["):
            errors.append(f"{label} path must be one exact safe path without wildcards")
        if not isinstance(line, int) or isinstance(line, bool) or line < 1:
            errors.append(f"{label} line must be one exact positive integer")
        if (
            rule == "private-network"
            and isinstance(oid, str) and re.fullmatch(r"[0-9a-f]{40}", oid)
            and isinstance(item_path, str) and relative is not None
            and not relative.is_absolute() and ".." not in relative.parts
            and not any(char in item_path for char in "*?[")
            and isinstance(line, int) and not isinstance(line, bool) and line >= 1
        ):
            entry = (rule, oid, item_path, line)
            if entry in entries:
                errors.append(f"{label} duplicates another baseline tuple")
            entries.add(entry)
            if (oid, item_path) not in scoped_objects:
                errors.append(f"{label} object/path is not reachable from base_commit_scope")
    return entries, errors

parser = argparse.ArgumentParser(description="redacted source/image bundle secret scan")
parser.add_argument("--mode", choices=("source", "image"), required=True)
parser.add_argument("--root", type=Path, required=True)
parser.add_argument("--revision", default="HEAD")
parser.add_argument("--report", type=Path, required=True)
parser.add_argument("--receipt", type=Path)
parser.add_argument("--baseline", type=Path)
args = parser.parse_args()
scan_root = args.root.resolve()
records = source_blobs(scan_root, args.revision) if args.mode == "source" else bundle_blobs(scan_root)
findings = []
seen = set()
for oid, path, content in records:
    if b"\0" in content:
        continue
    text = content.decode("utf-8", errors="replace")
    for number, line in enumerate(text.splitlines(), 1):
        for rule, pattern in RULES:
            if rule == "private-network" and "==" in line and not re.search(r"(?:https?|ssh)://", line):
                continue
            if pattern.search(line):
                key = (rule, oid, path, number)
                if key not in seen:
                    findings.append({"rule": rule, "object": oid, "path": path, "line": number})
                    seen.add(key)
baseline_path = args.baseline
if baseline_path is None and args.mode == "source":
    default_baseline = scan_root / "compliance/secret-history-baseline.json"
    baseline_path = default_baseline if default_baseline.is_file() else None
if args.mode != "source" and baseline_path is not None:
    baseline, baseline_errors = set(), ["history baseline is accepted only in source mode"]
else:
    baseline, baseline_errors = baseline_entries(scan_root, args.revision, baseline_path)
finding_rows = sorted(findings, key=lambda row: (row["path"], row["line"], row["rule"], row["object"]))
finding_tuples = {(row["rule"], row["object"], row["path"], row["line"]) for row in finding_rows}
stale = baseline - finding_tuples
for item in sorted(stale):
    baseline_errors.append(f"stale or mismatched baseline tuple: {item[0]} {item[1]} {item[2]}:{item[3]}")
accepted_rows = [row for row in finding_rows if (row["rule"], row["object"], row["path"], row["line"]) in baseline]
unaccepted_rows = [row for row in finding_rows if (row["rule"], row["object"], row["path"], row["line"]) not in baseline]
result = "FAIL" if unaccepted_rows or baseline_errors else "PASS"
report = {
    "schema_version": "1.0.0",
    "mode": args.mode,
    "result": result,
    "redacted": True,
    "accepted_baseline": accepted_rows,
    "baseline_errors": baseline_errors,
    "findings": unaccepted_rows,
}
rendered = (json.dumps(report, indent=2, sort_keys=True) + "\n").encode()
args.report.parent.mkdir(parents=True, exist_ok=True)
args.report.write_bytes(rendered)
if args.receipt:
    if args.mode != "source":
        raise SystemExit("image receipts require a complete build subject and are intentionally unavailable in source-only mode")
    receipt = construct_receipt({
        "schema_version": "1.0.0", "receipt_type": "source-secret-scan", "command": "source-secret-scan",
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "subject": {"source_release_id": source_release_id(scan_root)},
        "evidence": [{"path": args.report.name, "sha256": hashlib.sha256(rendered).hexdigest()}],
        "result": report["result"], "metadata": {},
    })
    args.receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
print(json.dumps(report, indent=2, sort_keys=True))
raise SystemExit(1 if result == "FAIL" else 0)
PY
