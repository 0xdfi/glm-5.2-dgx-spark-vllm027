from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from champion_runtime.receipts import construct_receipt
from champion_runtime.contracts import load_contract
from scripts.gate_status import GateError, promote
from scripts.verify_attestation import VerifiedReceipt


def ident(ch: str) -> str:
    return ch * 64


SOURCE = ident("a")
BUILD = ident("b")
LOCK = ident("c")
MANIFEST = ident("d")
IMAGE = "sha256:" + ident("e")
MODEL = ident("f")
SIDECAR = ident("1")
PRESET = ident("2")
PIN = ident("3")
RUN = "live-run-1"
ROOT = Path(__file__).resolve().parents[1]


def receipt(receipt_type: str, subject: dict, *, metadata=None, result="PASS"):
    body = {
        "schema_version": "1.0.0",
        "receipt_type": receipt_type,
        "command": receipt_type,
        "created_at": "2026-08-23T12:00:00Z",
        "subject": subject,
        "evidence": [{"path": f"{receipt_type}.json", "sha256": ident("9")}],
        "result": result,
        "metadata": metadata or {},
    }
    return VerifiedReceipt(
        receipt=construct_receipt(body),
        repository="0xdfi/glm-5.2-dgx-spark-vllm027",
        workflow=".github/workflows/attest-evidence.yml",
        ref="refs/heads/main",
        attested_subject_sha256=ident("8"),
        runner_constraints_source="committed-workflow-policy",
    )


def empty_status():
    claims = []
    for claim in (
        "source-complete",
        "model-license",
        "clean-build",
        "bit-reproducible",
        "three-profile-suite",
        "anonymous-readback",
        "convenience-image-published",
        "convenience-image-readback",
    ):
        claims.append({"claim": claim, "status": "pending", "subject": {}, "receipt_ids": []})
    for profile in ("fast-320k", "balanced-650k", "large-1m"):
        for claim in ("live-launch", "benchmark-reproduced"):
            claims.append({"claim": claim, "profile": profile, "status": "pending", "subject": {}, "receipt_ids": []})
    return {"schema_version": "1.0.0", "updated_at": "2026-08-23T00:00:00Z", "claims": claims}


def test_repository_status_is_truthfully_unpromoted():
    status = load_contract(ROOT / "publication/status.json", "status")
    assert len(status["claims"]) == 14
    assert not [item for item in status["claims"] if item["status"] == "passed"]
    source = next(item for item in status["claims"] if item["claim"] == "source-complete")
    model = next(item for item in status["claims"] if item["claim"] == "model-license")
    assert source["status"] == "pending"
    assert model["status"] == "blocked" and "not yet verified" in model["reason"]


def source_receipts(source=SOURCE):
    subject = {"source_release_id": source}
    return [receipt(kind, subject) for kind in (
        "source-closure", "provenance-license", "source-component-license",
        "source-secret-scan", "large-file-scan"
    )]


def model_receipts():
    return [
        receipt("checkpoint-license", {"model_id": MODEL}, metadata={"rights_status": "verified"}),
        receipt("sidecar-license", {"sidecar_id": SIDECAR}, metadata={"rights_status": "verified"}),
    ]


def build_receipts(image=IMAGE):
    subject = {
        "source_release_id": SOURCE,
        "build_subject_id": BUILD,
        "build_lock_id": LOCK,
        "source_manifest_id": MANIFEST,
        "image_digest": image,
    }
    return [receipt(kind, subject) for kind in (
        "build-produced", "verify-image", "sbom", "container-license", "image-secret-scan"
    )]


def live_receipts(profile="fast-320k"):
    subject = {
        "source_release_id": SOURCE,
        "build_subject_id": BUILD,
        "build_lock_id": LOCK,
        "source_manifest_id": MANIFEST,
        "image_digest": IMAGE,
        "model_id": MODEL,
        "sidecar_id": SIDECAR,
        "preset_id": PRESET,
        "pin_id": PIN,
        "run_id": RUN,
    }
    kinds = ["launch", "verify-runtime", "correctness", "stability"]
    if profile == "fast-320k":
        kinds.append("pin-state")
    return [receipt(kind, subject, metadata={"profile": profile}) for kind in kinds]


def test_source_promotion_requires_all_attested_matching_receipts():
    result = promote(empty_status(), "source-complete", source_receipts())
    claim = next(item for item in result["claims"] if item["claim"] == "source-complete")
    assert claim["status"] == "passed"
    assert claim["subject"] == {"source_release_id": SOURCE}


def test_gate_rejects_raw_or_cross_subject_receipts():
    with pytest.raises(GateError, match="attested"):
        promote(empty_status(), "source-complete", [item.receipt for item in source_receipts()])
    mismatched = source_receipts()
    mismatched[-1] = source_receipts(ident("4"))[-1]
    with pytest.raises(GateError, match="subject mismatch"):
        promote(empty_status(), "source-complete", mismatched)


def test_model_license_unknown_is_hard_failure():
    receipts = model_receipts()
    receipts[1] = receipt("sidecar-license", {"sidecar_id": SIDECAR}, metadata={"rights_status": "unknown"})
    with pytest.raises(GateError, match="unknown rights"):
        promote(empty_status(), "model-license", receipts)


def test_clean_build_requires_complete_chain_and_shared_build_subject():
    all_receipts = source_receipts() + build_receipts()
    promoted = promote(empty_status(), "clean-build", all_receipts)
    claim = next(item for item in promoted["claims"] if item["claim"] == "clean-build")
    assert claim["status"] == "passed"
    assert claim["subject"]["image_digest"] == IMAGE

    cross = build_receipts()
    cross[-1] = build_receipts("sha256:" + ident("7"))[-1]
    with pytest.raises(GateError, match="subject mismatch"):
        promote(empty_status(), "clean-build", source_receipts() + cross)


def test_live_and_benchmark_chains_bind_every_identity_and_attempt():
    pool = source_receipts() + model_receipts() + build_receipts() + live_receipts()
    live = promote(empty_status(), "live-launch", pool, profile="fast-320k")
    batteries = [
        receipt(
            "battery",
            live_receipts()[0].receipt["subject"],
            metadata={
                "profile": "fast-320k", "attempt": attempt, "valid": True,
                "thresholds_passed": True, "median_threshold_passed": True,
            },
        )
        for attempt in (1, 2, 3)
    ]
    result = promote(live, "benchmark-reproduced", pool + batteries, profile="fast-320k")
    claim = next(item for item in result["claims"] if item["claim"] == "benchmark-reproduced" and item["profile"] == "fast-320k")
    assert claim["status"] == "passed"

    duplicate = batteries[:2] + [batteries[1]]
    with pytest.raises(GateError, match="duplicate"):
        promote(live, "benchmark-reproduced", pool + duplicate, profile="fast-320k")


def test_live_chain_rejects_cross_preset_and_run_subjects():
    pool = source_receipts() + model_receipts() + build_receipts() + live_receipts()
    broken = pool[-1]
    body = dict(broken.receipt)
    body.pop("receipt_id")
    body["subject"] = dict(body["subject"]) | {"preset_id": ident("6"), "run_id": "other-run"}
    pool[-1] = VerifiedReceipt(
        receipt=construct_receipt(body),
        repository=broken.repository,
        workflow=broken.workflow,
        ref=broken.ref,
        attested_subject_sha256=broken.attested_subject_sha256,
        runner_constraints_source=broken.runner_constraints_source,
    )
    with pytest.raises(GateError, match="subject mismatch"):
        promote(empty_status(), "live-launch", pool, profile="fast-320k")


def test_slow_valid_battery_cannot_promote():
    pool = source_receipts() + model_receipts() + build_receipts() + live_receipts()
    live = promote(empty_status(), "live-launch", pool, profile="fast-320k")
    batteries = [
        receipt("battery", live_receipts()[0].receipt["subject"], metadata={
            "profile": "fast-320k", "attempt": i, "valid": True,
            "thresholds_passed": i != 2, "median_threshold_passed": True,
        }) for i in (1, 2, 3)
    ]
    with pytest.raises(GateError, match="threshold"):
        promote(live, "benchmark-reproduced", pool + batteries, profile="fast-320k")


def test_manual_pass_in_status_is_rejected_without_its_evidence():
    edited = deepcopy(empty_status())
    edited["claims"][0].update(status="passed", subject={"source_release_id": SOURCE}, receipt_ids=[ident("8")])
    with pytest.raises(GateError, match="manually edited|unsupported passed"):
        promote(edited, "model-license", model_receipts())
