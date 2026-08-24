# Copyright 2026 dfi / 0xdfi
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import subprocess
from pathlib import Path

from scripts.scan_large_files import MAX_BYTES, scan_repository


def _git(root: Path, *args: str):
    subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True)


def _repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "fixture@example.invalid")
    _git(root, "config", "user.name", "Fixture")
    return root


def test_exact_ten_mib_passes_but_larger_and_prohibited_suffix_fail(tmp_path: Path):
    root = _repo(tmp_path)
    (root / "exact.dat").write_bytes(b"x" * MAX_BYTES)
    _git(root, "add", "exact.dat")
    assert scan_repository(root) == []
    (root / "weight.SAFETENSORS").write_bytes(b"x")
    _git(root, "add", "weight.SAFETENSORS")
    assert any(item["rule"] == "prohibited-suffix" for item in scan_repository(root))
    (root / "too-big.dat").write_bytes(b"x" * (MAX_BYTES + 1))
    _git(root, "add", "too-big.dat")
    assert any(item["rule"] == "blob-over-10mib" for item in scan_repository(root))


def test_deleted_reachable_history_is_scanned_but_untracked_and_unreachable_are_not(tmp_path: Path):
    root = _repo(tmp_path)
    (root / "old.whl").write_bytes(b"wheel")
    _git(root, "add", "old.whl")
    _git(root, "commit", "-qm", "old")
    (root / "old.whl").unlink()
    _git(root, "add", "-u")
    _git(root, "commit", "-qm", "remove")
    (root / "untracked.pt").write_bytes(b"ignored")
    findings = scan_repository(root)
    assert any(item["path"] == "old.whl" for item in findings)
    assert not any(item["path"] == "untracked.pt" for item in findings)
