from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from champion_runtime.contracts import ContractError
from champion_runtime.receipts import construct_receipt
from scripts.verify_attestation import AttestationError, verify_attestation


SHA = "a" * 64


def write_receipt(tmp_path: Path) -> tuple[Path, Path]:
    evidence = tmp_path / "evidence.json"
    evidence.write_text('{"ok":true}\n', encoding="utf-8")
    body = {
        "schema_version": "1.0.0",
        "receipt_type": "source-closure",
        "command": "verify-source",
        "created_at": "2026-08-23T12:00:00Z",
        "subject": {"source_release_id": SHA},
        "evidence": [
            {
                "path": "evidence.json",
                "sha256": hashlib.sha256(evidence.read_bytes()).hexdigest(),
            }
        ],
        "result": "PASS",
        "metadata": {"workflow": ".github/workflows/attest-evidence.yml"},
    }
    receipt = tmp_path / "receipt.json"
    receipt.write_text(json.dumps(construct_receipt(body)), encoding="utf-8")
    return receipt, evidence


def policy() -> dict:
    return {
        "schema_version": "1.0.0",
        "repository": "0xdfi/glm-5.2-dgx-spark-vllm027",
        "workflow": ".github/workflows/attest-evidence.yml",
        "refs": ["refs/heads/main", "refs/tags/v*"],
        "issuer": "https://token.actions.githubusercontent.com",
        "protected_environment": "champion-evidence",
        "runner_policy": {
            "groups": ["champion-builders", "champion-controllers"],
            "labels": ["self-hosted", "ephemeral"],
            "enforcement": "committed-workflow-policy",
            "cryptographically_proven_by_dsse": False,
        },
    }


def verifier_output(receipt: Path, **overrides) -> dict:
    result = {
        "verified": True,
        "subject_sha256": hashlib.sha256(receipt.read_bytes()).hexdigest(),
        "repository": "0xdfi/glm-5.2-dgx-spark-vllm027",
        "workflow": ".github/workflows/attest-evidence.yml",
        "ref": "refs/heads/main",
        "issuer": "https://token.actions.githubusercontent.com",
        "envelope_present": True,
    }
    result.update(overrides)
    return result


def test_attestation_verifies_identity_subject_and_evidence(tmp_path):
    receipt, _ = write_receipt(tmp_path)

    verified = verify_attestation(
        receipt,
        policy(),
        evidence_root=tmp_path,
        verifier=lambda *_: verifier_output(receipt),
    )

    assert verified.receipt["receipt_id"]
    assert verified.repository == policy()["repository"]
    assert verified.runner_constraints_source == "committed-workflow-policy"


@pytest.mark.parametrize(
    "override,match",
    [
        ({"workflow": ".github/workflows/ci.yml"}, "untrusted workflow"),
        ({"ref": "refs/pull/2/merge"}, "untrusted ref"),
        ({"envelope_present": False}, "DSSE envelope"),
        ({"subject_sha256": "0" * 64}, "subject digest"),
        ({"verified": False}, "not verified"),
    ],
)
def test_attestation_fails_closed_for_untrusted_or_missing_proof(tmp_path, override, match):
    receipt, _ = write_receipt(tmp_path)

    with pytest.raises(AttestationError, match=match):
        verify_attestation(
            receipt,
            policy(),
            evidence_root=tmp_path,
            verifier=lambda *_: verifier_output(receipt, **override),
        )


def test_attestation_rejects_recomputed_but_unattested_receipt(tmp_path):
    receipt, _ = write_receipt(tmp_path)
    body = json.loads(receipt.read_text(encoding="utf-8"))
    body["metadata"]["workflow"] = "local-recomputed"
    body.pop("receipt_id")
    receipt.write_text(json.dumps(construct_receipt(body)), encoding="utf-8")

    with pytest.raises(AttestationError, match="missing verifier output"):
        verify_attestation(receipt, policy(), evidence_root=tmp_path, verifier=lambda *_: None)


def test_attestation_rejects_missing_or_changed_evidence(tmp_path):
    receipt, evidence = write_receipt(tmp_path)
    evidence.write_text('{"ok":false}\n', encoding="utf-8")

    with pytest.raises(AttestationError, match="evidence hash mismatch"):
        verify_attestation(
            receipt,
            policy(),
            evidence_root=tmp_path,
            verifier=lambda *_: verifier_output(receipt),
        )


def test_attestation_rejects_forged_canonical_receipt_id(tmp_path):
    receipt, _ = write_receipt(tmp_path)
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    payload["receipt_id"] = "0" * 64
    receipt.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ContractError, match="receipt_id mismatch"):
        verify_attestation(
            receipt,
            policy(),
            evidence_root=tmp_path,
            verifier=lambda *_: verifier_output(receipt),
        )
