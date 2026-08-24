#!/usr/bin/env python3
"""Verify one receipt's GitHub attestation, issuer identity, and evidence hashes."""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from champion_runtime.contracts import ContractError, load_contract


class AttestationError(ValueError):
    """Raised when evidence authentication fails closed."""


@dataclass(frozen=True)
class VerifiedReceipt:
    """A receipt authenticated to a trusted GitHub workflow identity."""

    receipt: dict[str, Any]
    repository: str
    workflow: str
    ref: str
    attested_subject_sha256: str
    runner_constraints_source: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_evidence(root: Path, value: str) -> Path:
    relative = PurePosixPath(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise AttestationError(f"unsafe evidence path: {value}")
    path = root.joinpath(*relative.parts)
    if not path.is_file() or path.is_symlink():
        raise AttestationError(f"missing evidence file: {value}")
    return path


def _run_gh(receipt_path: Path, policy: dict[str, Any], bundle_path: Path | None) -> Any:
    if bundle_path is None or not bundle_path.is_file():
        raise AttestationError("DSSE envelope bundle is missing")
    command = [
        "gh", "attestation", "verify", str(receipt_path),
        "--repo", policy["repository"], "--bundle", str(bundle_path),
        "--cert-oidc-issuer", policy["issuer"], "--format", "json",
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        detail = result.stderr.strip() or "GitHub attestation verification failed"
        raise AttestationError(detail)
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise AttestationError("GitHub verifier returned invalid JSON") from exc


def _normalized_verification(payload: Any) -> dict[str, Any]:
    """Normalize the small authenticated claim set used by publication gates.

    Tests inject this exact flat shape. Real invocations parse the verified gh
    JSON result and fail closed when any required certificate/statement field
    is absent.
    """

    if isinstance(payload, list) and len(payload) == 1:
        entry = payload[0]
        if isinstance(entry, dict) and "verificationResult" in entry:
            try:
                verification = entry["verificationResult"]
                certificate = verification["signature"]["certificate"]
                subjects = verification["statement"]["subject"]
                digests = {
                    subject["digest"]["sha256"]
                    for subject in subjects
                    if isinstance(subject, dict)
                    and isinstance(subject.get("digest"), dict)
                    and isinstance(subject["digest"].get("sha256"), str)
                }
                if len(digests) != 1:
                    raise AttestationError("verified statement lacks one SHA-256 subject")
                identity = certificate.get("subjectAlternativeName") or certificate.get("certificateIdentity")
                match = re.fullmatch(
                    r"https://github\.com/([^/]+/[^/]+)/(\.github/workflows/[^@]+)@(refs/.+)",
                    identity or "",
                )
                if match is None:
                    raise AttestationError("certificate workflow identity is malformed")
                repository_uri = certificate.get("sourceRepositoryUri")
                repository = (
                    repository_uri.removeprefix("https://github.com/")
                    if isinstance(repository_uri, str)
                    else match.group(1)
                )
                payload = {
                    "verified": True,
                    "subject_sha256": next(iter(digests)),
                    "repository": repository,
                    "workflow": match.group(2),
                    "ref": certificate.get("sourceRepositoryRef") or match.group(3),
                    "issuer": certificate.get("issuer") or "https://token.actions.githubusercontent.com",
                    "envelope_present": bool(entry.get("attestation", {}).get("bundle")),
                }
            except (KeyError, TypeError, AttributeError) as exc:
                raise AttestationError("GitHub verifier output is missing required claims") from exc
        else:
            payload = entry
    if not isinstance(payload, dict):
        raise AttestationError("missing verifier output")
    required = {
        "verified", "subject_sha256", "repository", "workflow", "ref",
        "issuer", "envelope_present",
    }
    if not required.issubset(payload):
        missing = sorted(required - set(payload))
        raise AttestationError(f"verifier output missing authenticated fields: {missing}")
    return payload


def verify_attestation(
    receipt_path: str | Path,
    policy: dict[str, Any] | str | Path,
    *,
    evidence_root: str | Path,
    bundle_path: str | Path | None = None,
    verifier: Callable[[Path, dict[str, Any], Path | None], Any] = _run_gh,
) -> VerifiedReceipt:
    """Return an authenticated receipt or raise without partial acceptance."""

    receipt_file = Path(receipt_path).resolve()
    if isinstance(policy, (str, Path)):
        policy_payload = load_contract(policy, "attestation-policy")
    else:
        policy_payload = dict(policy)
    bundle = Path(bundle_path).resolve() if bundle_path is not None else None
    result = _normalized_verification(verifier(receipt_file, policy_payload, bundle))
    if result["verified"] is not True:
        raise AttestationError("attestation is not verified")
    if result["envelope_present"] is not True:
        raise AttestationError("DSSE envelope is missing")
    actual_digest = _sha256(receipt_file)
    if result["subject_sha256"] != actual_digest:
        raise AttestationError("attested subject digest does not match receipt bytes")
    for field in ("repository", "workflow", "issuer"):
        if result[field] != policy_payload[field]:
            raise AttestationError(f"untrusted {field.replace('_', ' ')}")
    if not any(fnmatch.fnmatchcase(result["ref"], pattern) for pattern in policy_payload["refs"]):
        raise AttestationError("untrusted ref")

    receipt = load_contract(receipt_file, "receipt")
    root = Path(evidence_root).resolve()
    for item in receipt["evidence"]:
        path = _safe_evidence(root, item["path"])
        if _sha256(path) != item["sha256"]:
            raise AttestationError(f"evidence hash mismatch: {item['path']}")

    runner_policy = policy_payload["runner_policy"]
    if runner_policy["cryptographically_proven_by_dsse"] is not False:
        raise AttestationError("runner policy must not claim unsupported DSSE proof")
    return VerifiedReceipt(
        receipt=receipt,
        repository=result["repository"],
        workflow=result["workflow"],
        ref=result["ref"],
        attested_subject_sha256=actual_digest,
        runner_constraints_source=runner_policy["enforcement"],
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--policy", type=Path, default=ROOT / "publication/trusted-issuers.json")
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--evidence-root", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        verified = verify_attestation(
            args.receipt, args.policy, evidence_root=args.evidence_root,
            bundle_path=args.bundle,
        )
    except (AttestationError, ContractError, OSError) as exc:
        print(f"attestation: FAIL: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({
        "receipt_id": verified.receipt["receipt_id"],
        "repository": verified.repository,
        "workflow": verified.workflow,
        "ref": verified.ref,
        "runner_constraints_source": verified.runner_constraints_source,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
