from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REGISTER = ROOT / "scripts" / "register_provenance.py"
AUDIT = ROOT / "scripts" / "audit_licenses.py"
OWNER = "Copyright 2026 dfi / 0xdfi"
LEGACY_PATHS = {
    ".gitignore",
    "README.md",
    "benchmarks/protocol.md",
    "docs/findings.md",
    "docs/how-it-works.md",
    "docs/img/capacity-vs-speed.png",
    "docs/operations.md",
    "docs/patches.md",
    "recipes/launch.sh",
    "recipes/presets.md",
}


def git(root: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(root), *args],
        check=check,
        capture_output=True,
        text=True,
    )


def write(root: Path, relative: str, content: str) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def sha256(content: str) -> str:
    return hashlib.sha256(content.encode()).hexdigest()


def initialize_repo(tmp_path: Path) -> Path:
    git(tmp_path, "init", "-q")
    git(tmp_path, "config", "user.email", "tests@example.invalid")
    git(tmp_path, "config", "user.name", "Licensing Tests")
    write(
        tmp_path,
        "compliance/provenance.json",
        json.dumps({"schema_version": "1.0.0", "records": []}) + "\n",
    )
    write(
        tmp_path,
        "compliance/source-provenance.json",
        json.dumps({"schema_version": "1.0.0", "records": []}) + "\n",
    )
    git(tmp_path, "add", "compliance")
    git(tmp_path, "commit", "-qm", "initial ledger")
    return tmp_path


def register(
    root: Path,
    classification_map: str | None = None,
    *,
    check: bool = False,
) -> subprocess.CompletedProcess[str]:
    command = [sys.executable, str(REGISTER), "--staged"]
    if classification_map is not None:
        command.extend(["--classification-map", classification_map])
    if check:
        command.append("--check")
    return subprocess.run(command, cwd=root, capture_output=True, text=True)


def ledger(root: Path, name: str = "provenance.json") -> dict:
    return json.loads((root / "compliance" / name).read_text(encoding="utf-8"))


def records_by_path(root: Path, name: str = "provenance.json") -> dict[str, dict]:
    return {record["path"]: record for record in ledger(root, name)["records"]}


def test_root_license_and_project_attribution_are_exact():
    license_text = (ROOT / "LICENSE").read_text(encoding="utf-8")

    assert license_text.startswith("Apache License\nVersion 2.0, January 2004")
    assert OWNER in (ROOT / "NOTICE").read_text(encoding="utf-8")
    assert "dfi / 0xdfi" in (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    assert "@dfi" not in "\n".join(
        path.read_text(encoding="utf-8", errors="ignore")
        for path in [ROOT / "NOTICE", ROOT / "CREDITS.md", ROOT / "CITATION.cff"]
    )


def test_legacy_mit_text_and_file_level_reuse_annotations_are_preserved():
    original = git(ROOT, "show", "d1aca44:LICENSE").stdout
    assert (ROOT / "LICENSES/MIT.txt").read_text(encoding="utf-8") == original

    reuse = tomllib.loads((ROOT / "REUSE.toml").read_text(encoding="utf-8"))
    mit_paths = set()
    for annotation in reuse["annotations"]:
        if annotation["SPDX-License-Identifier"] == "MIT":
            assert annotation["precedence"] == "override"
            if annotation["SPDX-FileCopyrightText"] == "Copyright (c) 2026":
                mit_paths.update(annotation["path"])
    assert mit_paths == LEGACY_PATHS


def test_repository_ledger_has_full_coverage_and_no_recursive_hashes():
    result = git(ROOT, "ls-files", "-z")
    distributed = {path for path in result.stdout.split("\0") if path}
    records = records_by_path(ROOT)

    assert set(records) == distributed - {
        "compliance/provenance.json",
        "compliance/source-provenance.json",
    }
    assert "compliance/provenance.json" not in records
    assert "compliance/source-provenance.json" not in records
    assert all(record["license"] != "NOASSERTION" for record in records.values())
    assert all("@dfi" not in json.dumps(record) for record in records.values())


def test_registrar_adds_and_modifies_using_only_explicit_classification(tmp_path):
    root = initialize_repo(tmp_path)
    write(root, "src/new.py", "one\n")
    git(root, "add", "src/new.py")

    added = register(root, "src=project-apache")

    assert added.returncode == 0, added.stderr
    record = records_by_path(root)["src/new.py"]
    assert record["classification"] == "project-apache"
    assert record["copyright"] == OWNER
    assert record["license"] == "Apache-2.0"
    assert record["target_sha256"] == sha256("one\n")
    git(root, "add", "compliance")
    git(root, "commit", "-qam", "add source")

    write(root, "src/new.py", "two\n")
    git(root, "add", "src/new.py")
    modified = register(root, "src=project-apache")

    assert modified.returncode == 0, modified.stderr
    assert records_by_path(root)["src/new.py"]["target_sha256"] == sha256("two\n")


def test_registrar_handles_rename_and_remove(tmp_path):
    root = initialize_repo(tmp_path)
    write(root, "src/old.py", "content\n")
    git(root, "add", "src/old.py")
    assert register(root, "src=project-apache").returncode == 0
    git(root, "add", "compliance")
    git(root, "commit", "-qam", "add old path")

    git(root, "mv", "src/old.py", "src/new.py")
    renamed = register(root, "src/new.py=project-apache")

    assert renamed.returncode == 0, renamed.stderr
    records = records_by_path(root)
    assert "src/old.py" not in records
    assert records["src/new.py"]["origin_path"] == "src/old.py"
    git(root, "add", "compliance")
    git(root, "commit", "-qam", "rename source")

    git(root, "rm", "-q", "src/new.py")
    removed = register(root)

    assert removed.returncode == 0, removed.stderr
    assert "src/new.py" not in records_by_path(root)


def test_longest_classification_prefix_wins_and_retains_mit(tmp_path):
    root = initialize_repo(tmp_path)
    write(root, "src/legacy.py", "legacy base\n")
    git(root, "add", "src/legacy.py")
    git(root, "commit", "-qm", "legacy source")
    write(root, "src/legacy.py", "legacy modified\n")
    write(root, "src/fresh.py", "fresh\n")
    git(root, "add", "src")

    result = register(
        root,
        "src=project-apache,src/legacy.py=existing-mit",
    )

    assert result.returncode == 0, result.stderr
    records = records_by_path(root)
    assert records["src/fresh.py"]["license"] == "Apache-2.0"
    assert records["src/legacy.py"]["classification"] == "existing-mit"
    assert records["src/legacy.py"]["license"] == "MIT"
    assert records["src/legacy.py"]["base_sha256"] == sha256("legacy base\n")


def test_generated_evidence_does_not_perturb_projection(tmp_path):
    root = initialize_repo(tmp_path)
    write(root, "src/source.py", "source\n")
    git(root, "add", "src/source.py")
    assert register(root, "src=project-apache").returncode == 0
    git(root, "add", "compliance")
    git(root, "commit", "-qam", "add projected source")
    original_projection = (root / "compliance/source-provenance.json").read_bytes()

    write(root, "evidence/run.json", '{"result":"PASS"}\n')
    git(root, "add", "evidence/run.json")
    result = register(root, "evidence=generated-evidence")

    assert result.returncode == 0, result.stderr
    assert (root / "compliance/source-provenance.json").read_bytes() == original_projection
    assert "evidence/run.json" in records_by_path(root)
    assert "evidence/run.json" not in records_by_path(root, "source-provenance.json")


def test_registrar_refuses_missing_or_changed_classification(tmp_path):
    root = initialize_repo(tmp_path)
    write(root, "src/source.py", "source\n")
    git(root, "add", "src/source.py")

    absent = register(root)
    assert absent.returncode != 0
    assert "classification" in absent.stderr

    assert register(root, "src=project-apache").returncode == 0
    git(root, "add", "compliance")
    git(root, "commit", "-qam", "add source")
    write(root, "src/source.py", "modified\n")
    git(root, "add", "src/source.py")

    relicense = register(root, "src=existing-mit")
    assert relicense.returncode != 0
    assert "relicense" in relicense.stderr
    assert records_by_path(root)["src/source.py"]["license"] == "Apache-2.0"


def test_upstream_update_preserves_identifiers_and_license(tmp_path):
    root = initialize_repo(tmp_path)
    write(root, "runtime/upstream.py", "base\n")
    git(root, "add", "runtime/upstream.py")
    git(root, "commit", "-qm", "seed upstream")
    record = {
        "path": "runtime/upstream.py",
        "classification": "upstream-derivative",
        "copyright": "Copyright upstream contributors",
        "license": "BSD-3-Clause",
        "origin_repository": "https://example.invalid/upstream.git",
        "origin_commit": "a" * 40,
        "origin_path": "src/upstream.py",
        "base_sha256": sha256("base\n"),
        "target_sha256": sha256("base\n"),
        "modified_by": "dfi / 0xdfi",
        "evidence": [],
    }
    write(
        root,
        "compliance/provenance.json",
        json.dumps({"schema_version": "1.0.0", "records": [record]}) + "\n",
    )
    git(root, "add", "compliance/provenance.json")
    git(root, "commit", "-qm", "record upstream")
    write(root, "runtime/upstream.py", "modified\n")
    git(root, "add", "runtime/upstream.py")

    result = register(root, "runtime=upstream-derivative")

    assert result.returncode == 0, result.stderr
    updated = records_by_path(root)["runtime/upstream.py"]
    for field in (
        "classification",
        "copyright",
        "license",
        "origin_repository",
        "origin_commit",
        "origin_path",
        "base_sha256",
    ):
        assert updated[field] == record[field]
    assert updated["target_sha256"] == sha256("modified\n")


def test_check_is_read_only_and_rejects_stale_staged_paths(tmp_path):
    root = initialize_repo(tmp_path)
    write(root, "src/source.py", "source\n")
    git(root, "add", "src/source.py")
    before = {
        name: (root / "compliance" / name).read_bytes()
        for name in ("provenance.json", "source-provenance.json")
    }

    stale = register(root, check=True)

    assert stale.returncode != 0
    assert "absent" in stale.stderr or "stale" in stale.stderr
    assert all(
        (root / "compliance" / name).read_bytes() == content
        for name, content in before.items()
    )

    assert register(root, "src=project-apache").returncode == 0
    current = {
        name: (root / "compliance" / name).read_bytes()
        for name in ("provenance.json", "source-provenance.json")
    }
    checked = register(root, check=True)

    assert checked.returncode == 0, checked.stderr
    assert all(
        (root / "compliance" / name).read_bytes() == content
        for name, content in current.items()
    )


def test_strict_auditor_passes_repository_baseline():
    result = subprocess.run(
        [sys.executable, str(AUDIT), "--strict"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "license-audit: PASS" in result.stdout
