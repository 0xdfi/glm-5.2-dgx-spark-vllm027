# Copyright 2026 dfi / 0xdfi
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.audit_container_licenses import ComponentLicenseError, audit_components


ROOT = Path(__file__).resolve().parents[1]


def test_source_scope_passes_and_external_unknown_artifacts_are_excluded():
    matrix = json.loads((ROOT / "compliance/container-components.json").read_text())
    report = audit_components(matrix, ROOT, scope="source")
    assert report["result"] == "PASS"
    assert "glm52-dfi-hybrid-corrected-v4-lmheadw8v2-checkpoint" in report["excluded_external"]


def test_source_scope_rejects_noassertion_or_missing_text(tmp_path: Path):
    matrix = {"schema_version": "1.0.0", "components": [{
        "name": "bad", "distribution": "source", "license": "NOASSERTION",
        "license_text": "LICENSES/nope.txt", "origin": "x", "releasable": True,
    }]}
    with pytest.raises(ComponentLicenseError, match="NOASSERTION|license"):
        audit_components(matrix, tmp_path, scope="source")


def test_image_and_ghcr_scopes_fail_closed_without_exact_sbom_clearance():
    matrix = json.loads((ROOT / "compliance/container-components.json").read_text())
    with pytest.raises(ComponentLicenseError, match="SBOM"):
        audit_components(matrix, ROOT, scope="image")
    with pytest.raises(ComponentLicenseError, match="GHCR"):
        audit_components(matrix, ROOT, scope="registry", venue="ghcr")
