"""Canonical receipts and stable content-subject identities."""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import os
import re
import stat
import subprocess
import tempfile
from collections.abc import Iterable, Mapping
from pathlib import Path, PurePosixPath
from typing import Any

import rfc8785
from jsonschema import Draft202012Validator


class ReceiptError(ValueError):
    """Raised when receipt data cannot be safely canonicalized or identified."""


_SECRET_KEYS = {
    "access_key",
    "api_key",
    "apikey",
    "authorization",
    "client_secret",
    "credential",
    "credentials",
    "passwd",
    "password",
    "private_key",
    "secret",
    "secrets",
    "token",
}
_SECRET_SUFFIXES = (
    "_access_key",
    "_api_key",
    "_authorization",
    "_credential",
    "_credentials",
    "_password",
    "_private_key",
    "_secret",
    "_token",
)
_DURABLE_SUBJECT_KEYS = {
    "source_release_id",
    "build_subject_id",
    "build_lock_id",
    "source_manifest_id",
    "image_digest",
    "model_id",
    "sidecar_id",
    "preset_id",
    "pin_id",
    "tokenizer_id",
    "sampling_id",
}
_SOURCE_EXCLUDED_TOP_LEVEL = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "evidence",
}
_SOURCE_EXCLUDED_FILES = {
    "docs/release-status.md",
    "compliance/provenance.json",
    "publication/status.json",
}
_COMMAND_IDENTIFIER = re.compile(r"[a-z][a-z0-9]*(?:-[a-z0-9]+)*")
_RECEIPT_BODY_FIELDS = {
    "schema_version",
    "receipt_type",
    "command",
    "created_at",
    "subject",
    "evidence",
    "result",
    "metadata",
}
_SECRET_VALUE_PATTERNS = (
    re.compile(r"(?i)\bauthorization\s*[:=]\s*\S+"),
    re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]{6,}"),
    re.compile(r"(?i)\b[a-z][a-z0-9+.-]*://[^/\s:@]+:[^/\s@]+@"),
    re.compile(
        r"(?i)[?&;](?:access[_-]?token|api[_-]?key|client[_-]?secret|password|token)=[^&\s]+"
    ),
)


def _normalized_key(key: str) -> str:
    separated = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", key)
    return re.sub(r"[^a-z0-9]+", "_", separated.casefold()).strip("_")


def _validate_json_value(value: Any, path: str = "$") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ReceiptError(f"{path}: non-finite numbers are forbidden in receipts")
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ReceiptError(f"{path}: JSON object keys must be strings")
            _validate_json_value(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _validate_json_value(item, f"{path}[{index}]")


def _reject_secrets(value: Any, path: str = "$") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = _normalized_key(str(key))
            if normalized in _SECRET_KEYS or normalized.endswith(_SECRET_SUFFIXES):
                raise ReceiptError(f"{path}.{key}: secret-like key {key!r} is forbidden")
            _reject_secrets(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _reject_secrets(item, f"{path}[{index}]")
    elif isinstance(value, str):
        if any(pattern.search(value) for pattern in _SECRET_VALUE_PATTERNS):
            raise ReceiptError(f"{path}: secret-like value is forbidden")


def validate_receipt_content(receipt: Mapping[str, Any]) -> None:
    """Apply the shared safety policy used for receipt construction and loading."""

    _validate_json_value(receipt)
    _reject_secrets(receipt)
    command = receipt.get("command")
    if command is not None and (
        not isinstance(command, str) or not _COMMAND_IDENTIFIER.fullmatch(command)
    ):
        raise ReceiptError("$.command: command must be a stable identifier")


def canonical_bytes(value: Any) -> bytes:
    """Serialize *value* with the RFC 8785 JSON Canonicalization Scheme."""

    _validate_json_value(value)
    try:
        return rfc8785.dumps(value)
    except Exception as exc:  # rfc8785 exposes multiple domain-specific exceptions.
        raise ReceiptError(f"value cannot be RFC 8785 canonicalized: {exc}") from exc


def receipt_id(receipt: Mapping[str, Any]) -> str:
    """Return the SHA-256 identifier of a safe canonical receipt body."""

    if not isinstance(receipt, Mapping):
        raise ReceiptError("receipt must be a JSON object")
    body = {key: value for key, value in receipt.items() if key != "receipt_id"}
    subject = body.get("subject")
    if not isinstance(subject, Mapping):
        raise ReceiptError("receipt must contain at least one durable subject identity")
    durable_identity = False
    for key in _DURABLE_SUBJECT_KEYS.intersection(subject):
        value = subject[key]
        pattern = r"sha256:[0-9a-f]{64}" if key == "image_digest" else r"[0-9a-f]{64}"
        if isinstance(value, str) and re.fullmatch(pattern, value):
            durable_identity = True
            break
    if not durable_identity:
        raise ReceiptError("receipt must contain at least one durable subject identity")
    validate_receipt_content(body)
    return hashlib.sha256(canonical_bytes(body)).hexdigest()


def construct_receipt(body: Mapping[str, Any]) -> dict[str, Any]:
    """Return a stored receipt whose ID is bound to its canonical body."""

    if not isinstance(body, Mapping):
        raise ReceiptError("receipt body must be a JSON object")
    if "receipt_id" in body:
        raise ReceiptError("receipt body must not contain receipt_id")
    missing = _RECEIPT_BODY_FIELDS - set(body)
    if missing:
        raise ReceiptError(f"missing required receipt fields: {sorted(missing)}")
    extra = set(body) - _RECEIPT_BODY_FIELDS
    if extra:
        raise ReceiptError(f"unknown receipt fields: {sorted(extra)}")
    stored = dict(body)
    stored["receipt_id"] = receipt_id(body)
    _validate_stored_receipt_schema(stored)
    return stored


def _validate_stored_receipt_schema(receipt: Mapping[str, Any]) -> None:
    schema_path = Path(__file__).resolve().parent.parent / "schemas/receipt.schema.json"
    try:
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReceiptError(f"cannot load receipt schema: {exc}") from exc
    errors = sorted(
        Draft202012Validator(schema).iter_errors(receipt),
        key=lambda error: (list(error.absolute_path), error.message),
    )
    if errors:
        detail = "; ".join(error.message for error in errors)
        raise ReceiptError(f"receipt does not satisfy typed schema: {detail}")


def write_receipt(path: str | Path, body: Mapping[str, Any]) -> dict[str, Any]:
    """Atomically write a canonical-ID receipt with public mode 0644."""

    destination = Path(path)
    stored = construct_receipt(body)
    content = (json.dumps(stored, indent=2, sort_keys=True) + "\n").encode("utf-8")
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, 0o644)
        os.replace(temporary, destination)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
    return stored


def verify_receipt(receipt: Mapping[str, Any]) -> dict[str, Any]:
    """Verify and return a stored receipt with a content-bound identifier."""

    if not isinstance(receipt, Mapping):
        raise ReceiptError("receipt must be a JSON object")
    validate_receipt_content(receipt)
    _validate_stored_receipt_schema(receipt)
    stored_id = receipt.get("receipt_id")
    if not isinstance(stored_id, str) or not re.fullmatch(r"[0-9a-f]{64}", stored_id):
        raise ReceiptError("receipt_id must be a lowercase SHA-256 digest")
    expected_id = receipt_id(receipt)
    if not hmac.compare_digest(stored_id, expected_id):
        raise ReceiptError(
            f"receipt_id mismatch: stored {stored_id}, canonical body {expected_id}"
        )
    return dict(receipt)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tracked_paths(root: Path) -> list[Path] | None:
    checkout = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "--is-inside-work-tree"],
        check=False,
        capture_output=True,
    )
    if checkout.returncode != 0:
        if (root / ".git").exists():
            raise ReceiptError(f"Git file enumeration failed for source root: {root}")
        return None
    if checkout.stdout.strip() != b"true":
        return None
    result = subprocess.run(
        ["git", "-C", str(root), "ls-files", "-z", "--", "."],
        check=False,
        capture_output=True,
    )
    if result.returncode != 0:
        raise ReceiptError(f"Git file enumeration failed for source root: {root}")
    paths: list[Path] = []
    for item in result.stdout.split(b"\0"):
        if not item:
            continue
        try:
            relative = PurePosixPath(item.decode("utf-8"))
        except UnicodeDecodeError as exc:
            raise ReceiptError("Git returned a non-UTF-8 tracked path") from exc
        if relative.is_absolute() or ".." in relative.parts:
            raise ReceiptError(f"Git returned a path outside source root: {relative}")
        paths.append(root.joinpath(*relative.parts))
    return paths


def _is_source_path(relative_path: str) -> bool:
    parts = PurePosixPath(relative_path).parts
    if not parts or parts[0] in _SOURCE_EXCLUDED_TOP_LEVEL:
        return False
    if relative_path in _SOURCE_EXCLUDED_FILES:
        return False
    return "__pycache__" not in parts and not relative_path.endswith((".pyc", ".pyo"))


def _lstat_within_root(root: Path, path: Path) -> int:
    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise ReceiptError(f"path must stay within source root: {path}") from exc
    if relative.is_absolute() or ".." in relative.parts:
        raise ReceiptError(f"path must stay within source root: {path}")
    current = root
    for part in relative.parts:
        current = current / part
        try:
            mode = current.lstat().st_mode
        except FileNotFoundError as exc:
            raise ReceiptError(f"input does not exist: {relative.as_posix()}") from exc
        if stat.S_ISLNK(mode):
            raise ReceiptError(f"symbolic link is not allowed: {relative.as_posix()}")
    return root.lstat().st_mode if not relative.parts else mode


def _source_paths(root: Path) -> list[Path]:
    candidates = _tracked_paths(root)
    if candidates is None:
        candidates = list(root.rglob("*"))
    files: list[Path] = []
    for path in candidates:
        try:
            relative = path.relative_to(root).as_posix()
        except ValueError as exc:
            raise ReceiptError(f"source path must stay within source root: {path}") from exc
        if not _is_source_path(relative):
            continue
        mode = _lstat_within_root(root, path)
        if stat.S_ISREG(mode):
            files.append(path)
        elif not stat.S_ISDIR(mode):
            raise ReceiptError(f"source input is not a regular file: {relative}")
    return files


def _content_map(root: Path, paths: Iterable[Path]) -> dict[str, str]:
    result: dict[str, str] = {}
    for path in sorted(set(paths), key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix()
        mode = _lstat_within_root(root, path)
        if not stat.S_ISREG(mode):
            raise ReceiptError(f"input is not a regular file: {relative}")
        result[relative] = _file_sha256(path)
    return result


def source_release_id(root: str | Path) -> str:
    """Hash the distributed, non-generated source tree by path and file content."""

    source_root = Path(root).resolve()
    if not source_root.is_dir():
        raise ReceiptError(f"source root does not exist or is not a directory: {source_root}")
    files = _content_map(source_root, _source_paths(source_root))
    if not files:
        raise ReceiptError(f"source root contains no distributed files: {source_root}")
    return hashlib.sha256(canonical_bytes(files)).hexdigest()


def _resolve_consumed_paths(root: Path, consumed_paths: Iterable[str | Path]) -> list[Path]:
    resolved: list[Path] = []
    for value in consumed_paths:
        relative = PurePosixPath(Path(value).as_posix())
        if relative.is_absolute() or ".." in relative.parts:
            raise ReceiptError(f"build input must stay within source root: {value}")
        if not _is_source_path(relative.as_posix()):
            raise ReceiptError(f"not a stable build input: {relative.as_posix()}")
        candidate = root.joinpath(*relative.parts)
        try:
            mode = _lstat_within_root(root, candidate)
        except ReceiptError as exc:
            if "does not exist" in str(exc):
                raise ReceiptError(f"build input does not exist: {relative.as_posix()}") from exc
            raise
        if stat.S_ISDIR(mode):
            for path in candidate.rglob("*"):
                nested_relative = path.relative_to(root).as_posix()
                if not _is_source_path(nested_relative):
                    raise ReceiptError(f"not a stable build input: {nested_relative}")
                nested_mode = _lstat_within_root(root, path)
                if stat.S_ISREG(nested_mode):
                    resolved.append(path)
                elif not stat.S_ISDIR(nested_mode):
                    raise ReceiptError(f"build input is not a regular file: {nested_relative}")
        elif stat.S_ISREG(mode):
            resolved.append(candidate)
        else:
            raise ReceiptError(f"build input is not a regular file: {relative.as_posix()}")
    return resolved


def build_subject_id(
    root: str | Path,
    consumed_paths: Iterable[str | Path],
) -> str:
    """Hash the explicit set of files consumed by the champion image build."""

    source_root = Path(root).resolve()
    if not source_root.is_dir():
        raise ReceiptError(f"source root does not exist or is not a directory: {source_root}")
    files = _content_map(
        source_root,
        _resolve_consumed_paths(source_root, consumed_paths),
    )
    if not files:
        raise ReceiptError("build subject requires at least one consumed input")
    return hashlib.sha256(canonical_bytes(files)).hexdigest()
