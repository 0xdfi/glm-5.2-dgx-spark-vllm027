# Copyright 2026 dfi / 0xdfi
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from champion_runtime.receipts import construct_receipt
from scripts.verify_publication_source import PublicationSourceError, compose_source_candidate
from scripts.verify_source import SourceVerificationError, verify_source_tree


ROOT = Path(__file__).resolve().parents[1]
SOURCE = "a" * 64
TYPES = ("source-closure", "provenance-license", "source-component-license", "source-secret-scan", "large-file-scan")


def _write_receipt(root: Path, kind: str, *, source=SOURCE, result="PASS") -> Path:
    evidence = root / f"{kind}.report.json"
    evidence.write_text("{}\n")
    body = {
        "schema_version": "1.0.0", "receipt_type": kind, "command": kind,
        "created_at": "2026-08-23T12:00:00Z", "subject": {"source_release_id": source},
        "evidence": [{"path": evidence.name, "sha256": hashlib.sha256(evidence.read_bytes()).hexdigest()}],
        "result": result, "metadata": {},
    }
    path = root / f"{kind}.receipt.json"
    path.write_text(json.dumps(construct_receipt(body), indent=2, sort_keys=True) + "\n")
    return path


def test_repository_static_source_closure_preserves_unverified_live_b4():
    report = verify_source_tree(ROOT, require_clean=False, run_license_audit=False)
    assert report["result"] == "PASS"
    assert report["live_b4_parity"] == "not-verified"


def test_composer_requires_exact_five_pass_receipts_and_emits_provisional(tmp_path: Path):
    paths = [_write_receipt(tmp_path, kind) for kind in TYPES]
    result = compose_source_candidate(paths, tmp_path)
    assert result["result"] == "PROVISIONAL"
    assert result["receipt_type"] == "source-complete-candidate"
    with pytest.raises(PublicationSourceError, match="exactly one"):
        compose_source_candidate(paths[:-1], tmp_path)
    paths[-1] = _write_receipt(tmp_path, TYPES[-1], source="b" * 64)
    with pytest.raises(PublicationSourceError, match="source_release_id"):
        compose_source_candidate(paths, tmp_path)


def test_composer_rejects_forged_receipt_and_path_escape(tmp_path: Path):
    paths = [_write_receipt(tmp_path, kind) for kind in TYPES]
    forged = json.loads(paths[0].read_text())
    forged["result"] = "FAIL"
    paths[0].write_text(json.dumps(forged))
    with pytest.raises(PublicationSourceError, match="receipt_id"):
        compose_source_candidate(paths, tmp_path)
    outside = tmp_path.parent / "outside.receipt.json"
    outside.write_text(paths[1].read_text())
    with pytest.raises(PublicationSourceError, match="contained"):
        compose_source_candidate([outside, *paths[1:]], tmp_path)
