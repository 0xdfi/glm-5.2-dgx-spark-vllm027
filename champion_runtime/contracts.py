"""Strict loading for champion publication JSON contracts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from champion_runtime.receipts import (
    ReceiptError,
    validate_receipt_content,
    verify_receipt,
)


SCHEMA_DIRECTORY = Path(__file__).resolve().parent.parent / "schemas"
SCHEMA_PATHS = {
    "build-lock": SCHEMA_DIRECTORY / "build-lock.schema.json",
    "attestation-policy": SCHEMA_DIRECTORY / "attestation-policy.schema.json",
    "preset": SCHEMA_DIRECTORY / "preset.schema.json",
    "provenance": SCHEMA_DIRECTORY / "provenance.schema.json",
    "receipt": SCHEMA_DIRECTORY / "receipt.schema.json",
    "runtime-evidence": SCHEMA_DIRECTORY / "runtime-evidence.schema.json",
    "status": SCHEMA_DIRECTORY / "status.schema.json",
}


class ContractError(ValueError):
    """Raised when a JSON document does not satisfy its declared contract."""


class _DuplicateNameError(ValueError):
    pass


class _NonJsonConstantError(ValueError):
    pass


def _json_path(parts: Any) -> str:
    path = "$"
    for part in parts:
        if isinstance(part, int):
            path += f"[{part}]"
        else:
            path += f".{part}"
    return path


def _read_json(path: Path) -> Any:
    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for name, value in pairs:
            if name in result:
                raise _DuplicateNameError(f"duplicate object name {name!r}")
            result[name] = value
        return result

    def reject_constant(value: str) -> Any:
        raise _NonJsonConstantError(f"non-JSON constant {value}")

    try:
        with path.open("r", encoding="utf-8") as stream:
            return json.load(
                stream,
                object_pairs_hook=unique_object,
                parse_constant=reject_constant,
            )
    except (OSError, json.JSONDecodeError, _DuplicateNameError, _NonJsonConstantError) as exc:
        if isinstance(exc, json.JSONDecodeError):
            detail = f"invalid JSON at line {exc.lineno}, column {exc.colno}: {exc.msg}"
        elif isinstance(exc, (_DuplicateNameError, _NonJsonConstantError)):
            detail = f"invalid JSON: {exc}"
        else:
            detail = str(exc)
        raise ContractError(f"{path}: {detail}") from exc


def load_contract(path: str | Path, kind: str) -> dict[str, Any]:
    """Load and validate one JSON contract, returning its decoded object."""

    if kind not in SCHEMA_PATHS:
        choices = ", ".join(sorted(SCHEMA_PATHS))
        raise ContractError(f"unknown contract kind {kind!r}; expected one of: {choices}")

    contract_path = Path(path)
    payload = _read_json(contract_path)
    if kind == "receipt" and isinstance(payload, dict):
        try:
            validate_receipt_content(payload)
        except ReceiptError as exc:
            raise ContractError(f"{contract_path}: {exc}") from exc
    schema = _read_json(SCHEMA_PATHS[kind])
    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema)
    errors = sorted(
        validator.iter_errors(payload),
        key=lambda error: (list(error.absolute_path), error.message),
    )
    if errors:
        details = "; ".join(
            f"{_json_path(error.absolute_path)}: {error.message}" for error in errors
        )
        raise ContractError(f"{contract_path}: {details}")
    if not isinstance(payload, dict):
        raise ContractError(f"{contract_path}: $: contract must be a JSON object")
    if kind == "receipt":
        try:
            verify_receipt(payload)
        except ReceiptError as exc:
            raise ContractError(f"{contract_path}: {exc}") from exc
    return payload
