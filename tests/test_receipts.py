from __future__ import annotations

import json
import math
import os
import subprocess
import stat
from pathlib import Path

import pytest

from champion_runtime.receipts import (
    ReceiptError,
    build_subject_id,
    canonical_bytes,
    construct_receipt,
    receipt_id,
    source_release_id,
    verify_receipt,
    write_receipt,
)


SHA256 = "a" * 64


def write(root: Path, relative_path: str, content: bytes | str) -> None:
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(content, bytes):
        path.write_bytes(content)
    else:
        path.write_text(content, encoding="utf-8")


def source_tree(tmp_path: Path) -> Path:
    write(tmp_path, "runtime/example.py", "answer = 42\n")
    write(tmp_path, "README.md", "source release\n")
    write(tmp_path, "publication/status.json", '{"status":"pending"}\n')
    write(tmp_path, "docs/release-status.md", "pending\n")
    write(tmp_path, "evidence/build/result.json", '{"result":"PASS"}\n')
    write(
        tmp_path,
        "compliance/provenance.json",
        json.dumps(
            {
                "schema_version": "1.0.0",
                "records": [
                    {
                        "path": "runtime/example.py",
                        "classification": "project-apache",
                    }
                ],
            }
        ),
    )
    write(
        tmp_path,
        "compliance/source-provenance.json",
        json.dumps(
            {
                "schema_version": "1.0.0",
                "records": [
                    {
                        "path": "runtime/example.py",
                        "classification": "project-apache",
                        "target_sha256": SHA256,
                    }
                ],
            },
            sort_keys=True,
        ),
    )
    return tmp_path


def receipt_payload():
    return {
        "schema_version": "1.0.0",
        "receipt_type": "verify-source",
        "command": "verify-source",
        "created_at": "2026-08-23T12:00:00Z",
        "subject": {"source_release_id": SHA256},
        "evidence": [{"path": "evidence.json", "sha256": "b" * 64}],
        "result": "PASS",
        "metadata": {},
    }


def test_rfc8785_serialization_is_independent_of_key_order():
    left = {"z": [3, 2, 1], "a": {"two": 2, "one": 1}}
    right = {"a": {"one": 1, "two": 2}, "z": [3, 2, 1]}

    assert canonical_bytes(left) == canonical_bytes(right)
    assert canonical_bytes(left) == b'{"a":{"one":1,"two":2},"z":[3,2,1]}'


def test_receipt_id_is_independent_of_key_order():
    payload = receipt_payload()
    reordered = dict(reversed(list(payload.items())))

    assert receipt_id(payload) == receipt_id(reordered)


def test_constructed_receipt_binds_id_to_body_without_self_hashing():
    body = receipt_payload()

    stored = construct_receipt(body)

    assert stored["receipt_id"] == receipt_id(body)
    assert receipt_id(stored) == stored["receipt_id"]
    assert verify_receipt(stored) == stored
    assert "receipt_id" not in body


def test_construct_receipt_rejects_an_existing_receipt_id():
    body = receipt_payload() | {"receipt_id": "0" * 64}

    with pytest.raises(ReceiptError, match="body must not contain receipt_id"):
        construct_receipt(body)


def test_verify_receipt_rejects_forged_id():
    stored = construct_receipt(receipt_payload())
    stored["receipt_id"] = "0" * 64

    with pytest.raises(ReceiptError, match="receipt_id mismatch"):
        verify_receipt(stored)


@pytest.mark.parametrize("field", ["evidence", "result"])
def test_verify_receipt_rejects_body_mutation(field):
    stored = construct_receipt(receipt_payload())
    if field == "evidence":
        stored[field][0]["sha256"] = "c" * 64
    else:
        stored[field] = "FAIL"

    with pytest.raises(ReceiptError, match="receipt_id mismatch"):
        verify_receipt(stored)


def test_receipt_id_changes_for_one_byte_of_evidence():
    payload = receipt_payload()
    changed = receipt_payload()
    changed["evidence"][0]["sha256"] = "b" * 63 + "c"

    assert receipt_id(payload) != receipt_id(changed)


def test_receipt_rejects_nan():
    payload = receipt_payload() | {"measurement": math.nan}

    with pytest.raises(ReceiptError, match="non-finite"):
        receipt_id(payload)


@pytest.mark.parametrize(
    "secret_key",
    [
        "password",
        "api_token",
        "apiToken",
        "auth.token",
        "client-secret",
        "authorization",
        "private_key",
    ],
)
def test_receipt_rejects_secret_like_keys(secret_key):
    payload = receipt_payload() | {"metadata": {secret_key: "canary-value"}}

    with pytest.raises(ReceiptError, match=secret_key):
        receipt_id(payload)


@pytest.mark.parametrize(
    "secret_value",
    [
        "Authorization:" + " Bearer " + "canary-secret-value",
        "https://user:canary-password@example.invalid/artifact",
        "https://example.invalid/run?access_token=canary-secret-value",
    ],
)
def test_receipt_construction_rejects_secret_like_values(secret_value):
    payload = receipt_payload() | {"command": secret_value}

    with pytest.raises(ReceiptError, match="secret-like value"):
        construct_receipt(payload)


def test_receipt_allows_non_secret_token_measurement_keys():
    payload = receipt_payload() | {"measurements": {"token_count": 1024}}

    assert len(receipt_id(payload)) == 64


def test_write_receipt_is_atomic_mode_0644_and_schema_complete(tmp_path):
    destination = tmp_path / "nested" / "receipt.json"

    stored = write_receipt(destination, receipt_payload())

    assert json.loads(destination.read_text(encoding="utf-8")) == stored
    assert stat.S_IMODE(destination.stat().st_mode) == 0o644
    assert not list(destination.parent.glob(f".{destination.name}.*"))


def test_construct_receipt_rejects_incomplete_typed_body():
    body = receipt_payload()
    del body["created_at"]

    with pytest.raises(ReceiptError, match="missing required receipt fields"):
        construct_receipt(body)


@pytest.mark.parametrize(
    "subject",
    [
        None,
        {},
        {"run_id": "comparison-only"},
        {"source_release_id": None},
        {"source_release_id": ""},
        {"source_release_id": "A" * 64},
    ],
)
def test_receipt_requires_a_durable_subject_identity(subject):
    payload = receipt_payload()
    if subject is None:
        del payload["subject"]
    else:
        payload["subject"] = subject

    with pytest.raises(ReceiptError, match="durable subject identity"):
        receipt_id(payload)


@pytest.mark.parametrize("identity", ["tokenizer_id", "sampling_id"])
def test_receipt_accepts_forward_compatible_runtime_identity(identity):
    payload = receipt_payload()
    payload["subject"] = {identity: SHA256}

    assert construct_receipt(payload)["subject"] == {identity: SHA256}


def test_source_release_ignores_generated_evidence_status_and_full_ledger(tmp_path):
    root = source_tree(tmp_path)
    initialize_git_index(root)
    original = source_release_id(root)

    write(root, "evidence/build/result.json", '{"result":"FAIL"}\n')
    write(root, "publication/status.json", '{"status":"passed"}\n')
    write(root, "docs/release-status.md", "passed\n")
    write(
        root,
        "compliance/provenance.json",
        '{"records":[{"path":"evidence/new.json","classification":"generated-evidence"}]}\n',
    )
    write(root, ".git/HEAD", "ref: refs/heads/different-merge-commit\n")

    assert source_release_id(root) == original


def test_source_release_changes_for_relevant_source_byte(tmp_path):
    root = source_tree(tmp_path)
    original = source_release_id(root)

    write(root, "runtime/example.py", "answer = 43\n")

    assert source_release_id(root) != original


def test_source_release_changes_for_included_provenance_projection(tmp_path):
    root = source_tree(tmp_path)
    original = source_release_id(root)

    projection = json.loads((root / "compliance/source-provenance.json").read_text())
    projection["records"][0]["target_sha256"] = "b" * 64
    write(root, "compliance/source-provenance.json", json.dumps(projection, sort_keys=True))

    assert source_release_id(root) != original


def initialize_git_index(root: Path) -> None:
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(["git", "-C", str(root), "add", "."], check=True)


def test_source_release_uses_tracked_files_in_a_git_checkout(tmp_path):
    root = source_tree(tmp_path)
    initialize_git_index(root)
    original = source_release_id(root)

    write(root, "untracked.txt", "not distributed\n")

    assert source_release_id(root) == original


def test_source_release_fails_when_git_enumeration_fails(tmp_path):
    root = source_tree(tmp_path)
    (root / ".git").mkdir()

    with pytest.raises(ReceiptError, match="Git file enumeration failed"):
        source_release_id(root)


def test_source_release_rejects_a_tracked_symlink(tmp_path):
    root = source_tree(tmp_path)
    os.symlink("runtime/example.py", root / "tracked-link.py")
    initialize_git_index(root)

    with pytest.raises(ReceiptError, match="symbolic link"):
        source_release_id(root)


@pytest.mark.parametrize("target_kind", ["broken", "internal", "external"])
def test_source_release_rejects_filesystem_symlinks(tmp_path, target_kind):
    root = tmp_path / "source"
    root.mkdir()
    source_tree(root)
    link = root / f"{target_kind}-link.py"
    if target_kind == "broken":
        target = "missing.py"
    elif target_kind == "internal":
        target = "runtime/example.py"
    else:
        outside = tmp_path / "outside.py"
        outside.write_text("outside\n", encoding="utf-8")
        target = str(outside)
    os.symlink(target, link)

    with pytest.raises(ReceiptError, match="symbolic link"):
        source_release_id(root)


def build_tree(tmp_path: Path):
    inputs = [
        "docker/Dockerfile.champion",
        "runtime/manifests/build-lock.json",
        "runtime/requirements.lock",
        "runtime/overlays/vllm/example.py",
        "scripts/build-runtime.sh",
        "compliance/source-provenance.json",
        "LICENSES/Apache-2.0.txt",
        "NOTICE",
    ]
    for index, relative_path in enumerate(inputs):
        write(tmp_path, relative_path, f"input {index}\n")
    write(tmp_path, "evidence/build/result.json", "evidence\n")
    write(tmp_path, "publication/status.json", "status\n")
    write(tmp_path, "compliance/provenance.json", "mutable full ledger\n")
    return inputs


def test_build_subject_ignores_non_consumed_evidence_and_status(tmp_path):
    inputs = build_tree(tmp_path)
    original = build_subject_id(tmp_path, inputs)

    write(tmp_path, "evidence/build/result.json", "different evidence\n")
    write(tmp_path, "publication/status.json", "different status\n")
    write(tmp_path, "compliance/provenance.json", "different evidence records\n")
    write(tmp_path, ".git/HEAD", "different commit\n")

    assert build_subject_id(tmp_path, reversed(inputs)) == original


@pytest.mark.parametrize(
    "relative_path",
    [
        "docker/Dockerfile.champion",
        "runtime/manifests/build-lock.json",
        "runtime/overlays/vllm/example.py",
        "scripts/build-runtime.sh",
        "compliance/source-provenance.json",
        "LICENSES/Apache-2.0.txt",
        "NOTICE",
    ],
)
def test_build_subject_changes_for_every_consumed_input(tmp_path, relative_path):
    inputs = build_tree(tmp_path)
    original = build_subject_id(tmp_path, inputs)

    write(tmp_path, relative_path, "one byte differs!\n")

    assert build_subject_id(tmp_path, inputs) != original


def test_build_subject_rejects_missing_or_outside_inputs(tmp_path):
    build_tree(tmp_path)

    with pytest.raises(ReceiptError, match="does not exist"):
        build_subject_id(tmp_path, ["missing.file"])
    with pytest.raises(ReceiptError, match="must stay within"):
        build_subject_id(tmp_path, ["../outside.file"])


def test_build_subject_rejects_external_symlink_input(tmp_path):
    inputs = build_tree(tmp_path)
    notice = tmp_path / "NOTICE"
    notice.unlink()
    outside = tmp_path.parent / f"{tmp_path.name}-outside-notice"
    outside.write_text("outside\n", encoding="utf-8")
    os.symlink(outside, notice)

    with pytest.raises(ReceiptError, match="symbolic link"):
        build_subject_id(tmp_path, inputs)


@pytest.mark.parametrize(
    "relative_path",
    [
        "compliance/provenance.json",
        "publication/status.json",
        "docs/release-status.md",
        "evidence/build/result.json",
    ],
)
def test_build_subject_rejects_mutable_or_generated_inputs(tmp_path, relative_path):
    inputs = build_tree(tmp_path)
    if not (tmp_path / relative_path).exists():
        write(tmp_path, relative_path, "excluded\n")

    with pytest.raises(ReceiptError, match="not a stable build input"):
        build_subject_id(tmp_path, inputs + [relative_path])
