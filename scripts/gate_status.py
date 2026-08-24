#!/usr/bin/env python3
"""Pure dependency evaluator for evidence-backed publication status claims."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from champion_runtime.contracts import ContractError, load_contract
from scripts.verify_attestation import VerifiedReceipt, verify_attestation


PROFILES = ("fast-320k", "balanced-650k", "large-1m")
SOURCE_TYPES = (
    "source-closure",
    "provenance-license",
    "source-component-license",
    "source-secret-scan",
    "large-file-scan",
)
BUILD_TYPES = ("build-produced", "verify-image", "sbom", "container-license", "image-secret-scan")
LIVE_TYPES = ("launch", "verify-runtime", "correctness", "stability")
SOURCE_KEYS = ("source_release_id",)
BUILD_KEYS = ("source_release_id", "build_subject_id", "build_lock_id", "source_manifest_id", "image_digest")
LIVE_KEYS = BUILD_KEYS + ("model_id", "sidecar_id", "preset_id", "run_id")


class GateError(ValueError):
    """Raised when a claim cannot be derived from trusted evidence."""


def _pool(receipts: Iterable[VerifiedReceipt]) -> list[VerifiedReceipt]:
    result = list(receipts)
    if any(not isinstance(item, VerifiedReceipt) for item in result):
        raise GateError("every receipt must be attested and issuer-verified")
    identifiers = [item.receipt["receipt_id"] for item in result]
    if len(set(identifiers)) != len(identifiers):
        raise GateError("duplicate attested receipt")
    return result


def _of_type(pool: list[VerifiedReceipt], receipt_type: str) -> list[VerifiedReceipt]:
    return [item for item in pool if item.receipt["receipt_type"] == receipt_type]


def _one_pass(pool: list[VerifiedReceipt], receipt_type: str) -> VerifiedReceipt:
    matches = _of_type(pool, receipt_type)
    if len(matches) != 1:
        raise GateError(f"expected exactly one {receipt_type} receipt, found {len(matches)}")
    if matches[0].receipt["result"] != "PASS":
        raise GateError(f"{receipt_type} did not pass")
    return matches[0]


def _common_subject(items: Iterable[VerifiedReceipt], keys: tuple[str, ...]) -> dict[str, str]:
    receipts = list(items)
    if not receipts:
        raise GateError("receipt chain is empty")
    first = receipts[0].receipt["subject"]
    missing = [key for key in keys if key not in first]
    if missing:
        raise GateError(f"receipt subject is missing identities: {missing}")
    subject = {key: first[key] for key in keys}
    for item in receipts[1:]:
        candidate = item.receipt["subject"]
        if any(candidate.get(key) != value for key, value in subject.items()):
            raise GateError("receipt subject mismatch across evidence chain")
    return subject


def _profile(item: VerifiedReceipt, profile: str) -> bool:
    return item.receipt["metadata"].get("profile") == profile


def _derive_source(pool: list[VerifiedReceipt]) -> tuple[dict[str, str], list[str]]:
    items = [_one_pass(pool, kind) for kind in SOURCE_TYPES]
    return _common_subject(items, SOURCE_KEYS), [item.receipt["receipt_id"] for item in items]


def _derive_model(pool: list[VerifiedReceipt]) -> tuple[dict[str, str], list[str]]:
    checkpoint = _one_pass(pool, "checkpoint-license")
    sidecar = _one_pass(pool, "sidecar-license")
    for item in (checkpoint, sidecar):
        if item.receipt["metadata"].get("rights_status") != "verified":
            raise GateError("model or sidecar has unknown rights")
    try:
        subject = {
            "model_id": checkpoint.receipt["subject"]["model_id"],
            "sidecar_id": sidecar.receipt["subject"]["sidecar_id"],
        }
    except KeyError as exc:
        raise GateError(f"license receipt subject is missing {exc.args[0]}") from exc
    return subject, [checkpoint.receipt["receipt_id"], sidecar.receipt["receipt_id"]]


def _derive_clean(pool: list[VerifiedReceipt]) -> tuple[dict[str, str], list[str]]:
    source, source_ids = _derive_source(pool)
    items = [_one_pass(pool, kind) for kind in BUILD_TYPES]
    subject = _common_subject(items, BUILD_KEYS)
    if subject["source_release_id"] != source["source_release_id"]:
        raise GateError("clean-build source subject mismatch")
    return subject, source_ids + [item.receipt["receipt_id"] for item in items]


def _derive_live(pool: list[VerifiedReceipt], profile: str) -> tuple[dict[str, str], list[str]]:
    if profile not in PROFILES:
        raise GateError(f"unknown profile: {profile}")
    clean, clean_ids = _derive_clean(pool)
    model, model_ids = _derive_model(pool)
    kinds = LIVE_TYPES + (("pin-state",) if profile == "fast-320k" else ())
    items = []
    for kind in kinds:
        matches = [item for item in _of_type(pool, kind) if _profile(item, profile)]
        if len(matches) != 1 or matches[0].receipt["result"] != "PASS":
            raise GateError(f"expected one passing {profile} {kind} receipt")
        items.append(matches[0])
    keys = LIVE_KEYS + (("pin_id",) if profile == "fast-320k" else ())
    subject = _common_subject(items, keys)
    for key, value in clean.items():
        if subject[key] != value:
            raise GateError("live-launch build subject mismatch")
    for key, value in model.items():
        if subject[key] != value:
            raise GateError("live-launch model subject mismatch")
    return subject, clean_ids + model_ids + [item.receipt["receipt_id"] for item in items]


def _derive_benchmark(pool: list[VerifiedReceipt], profile: str) -> tuple[dict[str, str], list[str]]:
    live, live_ids = _derive_live(pool, profile)
    profile_attempts = [item for item in _of_type(pool, "battery") if _profile(item, profile)]
    if any(
        any(item.receipt["subject"].get(key) != value for key, value in live.items())
        for item in profile_attempts
    ):
        raise GateError("battery subject mismatch with live-launch chain")
    attempts = profile_attempts
    if len(attempts) > 5:
        raise GateError("benchmark exceeds five attempts")
    attempt_numbers = [item.receipt["metadata"].get("attempt") for item in attempts]
    if any(not isinstance(number, int) or not 1 <= number <= 5 for number in attempt_numbers):
        raise GateError("battery attempt number must be an integer from one through five")
    if len(set(attempt_numbers)) != len(attempt_numbers):
        raise GateError("duplicate benchmark attempt")
    valid = [item for item in attempts if item.receipt["metadata"].get("valid") is True]
    if len(valid) != 3:
        raise GateError(f"benchmark requires exactly three valid attempts; found {len(valid)}")
    for item in valid:
        metadata = item.receipt["metadata"]
        if item.receipt["result"] != "PASS" or metadata.get("thresholds_passed") is not True:
            raise GateError("valid battery failed an individual threshold")
        if metadata.get("median_threshold_passed") is not True:
            raise GateError("battery median threshold did not pass")
    return live, live_ids + [item.receipt["receipt_id"] for item in attempts]


def _derive(claim: str, pool: list[VerifiedReceipt], profile: str | None) -> tuple[dict[str, str], list[str]]:
    if claim == "source-complete":
        return _derive_source(pool)
    if claim == "model-license":
        return _derive_model(pool)
    if claim == "clean-build":
        return _derive_clean(pool)
    if claim == "live-launch":
        return _derive_live(pool, profile or "")
    if claim == "benchmark-reproduced":
        return _derive_benchmark(pool, profile or "")
    if claim == "three-profile-suite":
        results = [_derive_benchmark(pool, selected) for selected in PROFILES]
        subjects = [subject for subject, _ in results]
        shared = {key: subjects[0][key] for key in BUILD_KEYS + ("model_id", "sidecar_id")}
        if any(any(subject[key] != value for key, value in shared.items()) for subject in subjects[1:]):
            raise GateError("three-profile suite subject mismatch")
        return shared, [receipt_id for _, ids in results for receipt_id in ids]
    if claim == "bit-reproducible":
        comparison = _one_pass(pool, "build-comparison")
        metadata = comparison.receipt["metadata"]
        if metadata.get("normalized_outputs_equal") is not True:
            raise GateError("builds are source-equivalent only; bit reproducibility did not pass")
        source, source_ids = _derive_source(pool)
        digests = {
            item.receipt["subject"].get("image_digest")
            for item in _of_type(pool, "build-produced")
        }
        if len(digests) != 2 or None in digests:
            raise GateError("bit reproducibility requires two distinct build outputs")
        chains: list[VerifiedReceipt] = []
        subjects: list[dict[str, str]] = []
        for digest in sorted(digests):
            items = []
            for kind in BUILD_TYPES:
                matches = [
                    item for item in _of_type(pool, kind)
                    if item.receipt["subject"].get("image_digest") == digest
                ]
                if len(matches) != 1 or matches[0].receipt["result"] != "PASS":
                    raise GateError(f"build {digest} lacks one passing {kind} receipt")
                items.append(matches[0])
            subject = _common_subject(items, BUILD_KEYS)
            if subject["source_release_id"] != source["source_release_id"]:
                raise GateError("bit reproducibility source subject mismatch")
            subjects.append(subject)
            chains.extend(items)
        stable = {key: subjects[0][key] for key in BUILD_KEYS[:-1]}
        if any(subject[key] != value for key, value in stable.items() for subject in subjects[1:]):
            raise GateError("bit reproducibility build subject mismatch")
        comparison_subject = comparison.receipt["subject"]
        if any(comparison_subject.get(key) != value for key, value in stable.items()):
            raise GateError("build comparison subject mismatch")
        return stable, source_ids + [item.receipt["receipt_id"] for item in chains] + [comparison.receipt["receipt_id"]]
    if claim == "anonymous-readback":
        live, live_ids = _derive_live(pool, "fast-320k")
        items = [_one_pass(pool, kind) for kind in ("anonymous-configure", "anonymous-build", "anonymous-verify", "anonymous-live-fast")]
        subject = _common_subject(items, BUILD_KEYS)
        if any(subject[key] != live[key] for key in BUILD_KEYS):
            raise GateError("anonymous readback subject mismatch")
        return subject, live_ids + [item.receipt["receipt_id"] for item in items]
    if claim == "convenience-image-published":
        clean, clean_ids = _derive_clean(pool)
        license_receipt = _one_pass(pool, "registry-license")
        publication = _one_pass(pool, "registry-publish")
        if license_receipt.receipt["metadata"].get("registry_license_cleared") is not True:
            raise GateError("registry license clearance is absent")
        subject = _common_subject((license_receipt, publication), BUILD_KEYS)
        if subject != clean:
            raise GateError("registry publication subject mismatch")
        for item in (license_receipt, publication):
            registry_digest = item.receipt["metadata"].get("registry_digest")
            if registry_digest != subject["image_digest"]:
                raise GateError("registry digest is absent or does not match image subject")
        return subject, clean_ids + [license_receipt.receipt["receipt_id"], publication.receipt["receipt_id"]]
    if claim == "convenience-image-readback":
        published, ids = _derive("convenience-image-published", pool, None)
        readback = _one_pass(pool, "registry-readback")
        subject = _common_subject((readback,), BUILD_KEYS)
        if subject != published:
            raise GateError("registry readback subject mismatch")
        return subject, ids + [readback.receipt["receipt_id"]]
    raise GateError(f"unknown claim: {claim}")


def _claim_key(item: dict[str, Any]) -> tuple[str, str | None]:
    return item["claim"], item.get("profile")


def promote(
    status: dict[str, Any],
    claim: str,
    receipts: Iterable[VerifiedReceipt],
    *,
    profile: str | None = None,
    now: str | None = None,
) -> dict[str, Any]:
    """Return status with one claim promoted only from a complete trusted pool."""

    pool = _pool(receipts)
    for existing in status.get("claims", []):
        if existing.get("status") == "passed":
            expected_ids = set(existing.get("receipt_ids", []))
            existing_pool = [
                item for item in pool if item.receipt["receipt_id"] in expected_ids
            ]
            if {item.receipt["receipt_id"] for item in existing_pool} != expected_ids:
                raise GateError("manually edited or stale passed status references missing evidence")
            try:
                subject, ids = _derive(existing["claim"], existing_pool, existing.get("profile"))
            except GateError as exc:
                raise GateError(f"manually edited or unsupported passed status: {exc}") from exc
            if existing.get("subject") != subject or set(existing.get("receipt_ids", [])) != set(ids):
                raise GateError("manually edited or stale passed status")

    subject, ids = _derive(claim, pool, profile)
    wanted = (claim, profile)
    matches = [item for item in status.get("claims", []) if _claim_key(item) == wanted]
    if len(matches) != 1:
        raise GateError(f"status must contain exactly one claim entry for {wanted}")
    result = deepcopy(status)
    target = next(item for item in result["claims"] if _claim_key(item) == wanted)
    target.update(status="passed", subject=subject, receipt_ids=list(dict.fromkeys(ids)))
    target.pop("reason", None)
    result["updated_at"] = now or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return result


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, 0o644)
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["promote"])
    parser.add_argument("claim")
    parser.add_argument("--profile", choices=PROFILES)
    parser.add_argument("--status", type=Path, default=ROOT / "publication/status.json")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--policy", type=Path, default=ROOT / "publication/trusted-issuers.json")
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--attested", nargs=2, action="append", metavar=("RECEIPT", "ATTESTATION"), default=[])
    args = parser.parse_args(argv)
    try:
        status = load_contract(args.status, "status")
        verified = []
        for receipt_path, attestation_path in args.attested:
            verified.append(verify_attestation(
                receipt_path, args.policy, evidence_root=args.evidence_root,
                bundle_path=attestation_path,
            ))
        updated = promote(status, args.claim, verified, profile=args.profile)
        destination = args.output or args.status
        _atomic_json(destination, updated)
    except (OSError, json.JSONDecodeError, ContractError, GateError, ValueError) as exc:
        print(f"status-promotion: FAIL: {exc}", file=sys.stderr)
        return 1
    print(f"status-promotion: PASS: {args.claim}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
