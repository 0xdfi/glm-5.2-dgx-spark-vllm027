# Copyright 2026 dfi / 0xdfi
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCANNER = ROOT / "scripts/scan_secrets.sh"


def _git(root: Path, *args: str):
    subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True)


def _run_scan(repo: Path, report: Path, baseline: Path | None = None):
    command = [str(SCANNER), "--mode", "source", "--root", str(repo), "--report", str(report)]
    if baseline is not None:
        command.extend(["--baseline", str(baseline)])
    return subprocess.run(command, text=True, capture_output=True)


def _baseline(repo: Path, finding: dict, path: Path, *, mutate: dict | None = None):
    entry = dict(finding)
    if mutate:
        entry.update(mutate)
    payload = {
        "schema_version": "1.0.0",
        "base_commit_scope": subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "justification": "Already-public non-secret RFC1918 documentation example.",
        "accepted": [entry],
    }
    path.write_text(json.dumps(payload) + "\n")


def test_full_reachable_history_scan_is_redacted(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "fixture@example.invalid")
    _git(repo, "config", "user.name", "Fixture")
    canary = "CHAMPION_TEST_" + "SECRET_9f34"
    (repo / "leak.txt").write_text("token=" + canary + "\n")
    _git(repo, "add", "leak.txt")
    _git(repo, "commit", "-qm", "leak")
    (repo / "leak.txt").unlink()
    _git(repo, "add", "-u")
    _git(repo, "commit", "-qm", "delete")
    report = tmp_path / "report.json"
    result = subprocess.run([str(SCANNER), "--mode", "source", "--root", str(repo), "--report", str(report)], text=True, capture_output=True)
    assert result.returncode != 0
    assert canary not in result.stdout + result.stderr + report.read_text()
    payload = json.loads(report.read_text())
    assert payload["result"] == "FAIL"
    assert {"rule", "object", "path", "line"} == set(payload["findings"][0])


def test_private_path_rfc1918_and_ssh_bearer_relay_are_rejected(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "fixture@example.invalid")
    _git(repo, "config", "user.name", "Fixture")
    private_path = "/" + "Users" + "/alice/private"
    private_ip = "192" + ".168.4.2"
    bearer = "Authorization:" + " Bearer " + "abcdefghi"
    (repo / "bad.txt").write_text(f"{private_path} {private_ip} ssh node {bearer}\n")
    _git(repo, "add", "bad.txt")
    report = tmp_path / "report.json"
    result = subprocess.run([str(SCANNER), "--mode", "source", "--root", str(repo), "--report", str(report)], capture_output=True)
    assert result.returncode != 0
    rules = {item["rule"] for item in json.loads(report.read_text())["findings"]}
    assert {"private-path", "private-network", "bearer-credential"} <= rules


def test_image_scan_allows_only_the_canonical_shared_fleet_lock_path(
    tmp_path: Path,
):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    canonical = "/home/dfi/.cache/nous-fleet-mutation.lock"
    (bundle / "config.txt").write_text(f"CHAMPION_FLEET_LOCK={canonical}\n")
    accepted_report = tmp_path / "accepted.json"

    accepted = subprocess.run(
        [
            str(SCANNER),
            "--mode",
            "image",
            "--root",
            str(bundle),
            "--report",
            str(accepted_report),
        ],
        text=True,
        capture_output=True,
    )

    assert accepted.returncode == 0, accepted.stdout + accepted.stderr
    assert json.loads(accepted_report.read_text())["findings"] == []

    rejected_private_path = "/" + "home" + "/dfi/.cache/other.lock"
    (bundle / "config.txt").write_text(
        f"CHAMPION_FLEET_LOCK={canonical}\nPRIVATE={rejected_private_path}\n"
    )
    rejected_report = tmp_path / "rejected.json"
    rejected = subprocess.run(
        [
            str(SCANNER),
            "--mode",
            "image",
            "--root",
            str(bundle),
            "--report",
            str(rejected_report),
        ],
        text=True,
        capture_output=True,
    )

    assert rejected.returncode != 0
    assert {row["rule"] for row in json.loads(rejected_report.read_text())["findings"]} == {
        "private-path"
    }


def test_exact_rfc1918_history_tuple_can_be_acknowledged(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "fixture@example.invalid")
    _git(repo, "config", "user.name", "Fixture")
    address = ".".join(("192", "168", "7", "9"))
    (repo / "legacy.md").write_text(f"retired example {address}\n")
    _git(repo, "add", "legacy.md")
    _git(repo, "commit", "-qm", "legacy")
    first_report = tmp_path / "first.json"
    assert _run_scan(repo, first_report).returncode != 0
    finding = json.loads(first_report.read_text())["findings"][0]
    baseline = tmp_path / "baseline.json"
    _baseline(repo, finding, baseline)

    report = tmp_path / "accepted.json"
    result = _run_scan(repo, report, baseline)

    payload = json.loads(report.read_text())
    assert result.returncode == 0
    assert payload["result"] == "PASS"
    assert payload["findings"] == []
    assert payload["accepted_baseline"] == [finding]
    assert payload["baseline_errors"] == []


def test_baseline_rejects_mismatch_stale_entry_and_wildcard(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "fixture@example.invalid")
    _git(repo, "config", "user.name", "Fixture")
    address = ".".join(("10", "1", "2", "3"))
    (repo / "legacy.md").write_text(f"retired example {address}\n")
    _git(repo, "add", "legacy.md")
    _git(repo, "commit", "-qm", "legacy")
    raw = tmp_path / "raw.json"
    _run_scan(repo, raw)
    finding = json.loads(raw.read_text())["findings"][0]

    for name, mutation in (("stale", {"line": 99}), ("wildcard", {"object": "*"})):
        baseline = tmp_path / f"{name}.json"
        _baseline(repo, finding, baseline, mutate=mutation)
        report = tmp_path / f"{name}-report.json"
        result = _run_scan(repo, report, baseline)
        payload = json.loads(report.read_text())
        assert result.returncode != 0
        assert payload["result"] == "FAIL"
        assert payload["baseline_errors"]


def test_baseline_never_masks_new_or_credential_findings(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "fixture@example.invalid")
    _git(repo, "config", "user.name", "Fixture")
    old_address = ".".join(("172", "20", "4", "8"))
    (repo / "legacy.md").write_text(f"retired example {old_address}\n")
    _git(repo, "add", "legacy.md")
    _git(repo, "commit", "-qm", "legacy")
    raw = tmp_path / "raw.json"
    _run_scan(repo, raw)
    finding = json.loads(raw.read_text())["findings"][0]
    baseline = tmp_path / "baseline.json"
    _baseline(repo, finding, baseline)

    new_address = ".".join(("192", "168", "8", "10"))
    credential = "ghp_" + "A" * 31
    (repo / "new.md").write_text(f"new {new_address} {credential}\n")
    _git(repo, "add", "new.md")
    report = tmp_path / "new-report.json"
    result = _run_scan(repo, report, baseline)
    payload = json.loads(report.read_text())

    assert result.returncode != 0
    assert payload["accepted_baseline"] == [finding]
    assert {item["rule"] for item in payload["findings"]} >= {"private-network", "credential"}
