# Copyright 2026 dfi / 0xdfi
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
import re
from pathlib import Path

from scripts.render_release_status import render_release_status


ROOT = Path(__file__).resolve().parents[1]


def read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def load(path: str) -> dict:
    return json.loads(read(path))


def test_release_status_is_a_deterministic_projection_of_machine_status():
    status = load("publication/status.json")
    rendered = render_release_status(status)
    assert read("docs/release-status.md") == rendered
    assert "| source-complete | — | pending |" in rendered
    assert "| clean-build | — | blocked |" in rendered
    assert "| benchmark-reproduced | fast-320k | blocked |" in rendered
    assert "passed" not in {
        claim["status"] for claim in status["claims"]
    }


def test_readme_is_truthful_about_the_publication_boundary_and_quickstart():
    text = read("README.md")
    ordered = [
        "git clone",
        "./bin/champion bootstrap",
        "./bin/champion configure",
        "./bin/champion verify-hosts fast",
        "./bin/champion build",
        "./bin/champion launch fast",
        "./bin/champion verify fast",
    ]
    positions = [text.index(token) for token in ordered]
    assert positions == sorted(positions)
    for statement in (
        "source-complete` is pending",
        "No trusted clean ARM64 build",
        "live-b4 parity",
        "exact-model rights",
        "four-node live launch",
        "benchmark reproduction",
        "bit reproducibility",
        "anonymous readback",
        "external-required",
        "rights are unknown",
        "GHCR",
        "documented-not-reproduced",
    ):
        assert statement in text
    forbidden = (
        "obtain a patched image",
        "fuse_gemm_comms:true",
    )
    assert not any(value in text for value in forbidden)
    assert "There is no 1M-at-4096 champion" in text
    assert "no DPTB=256 champion" in text


def test_documented_presets_and_historical_references_match_contracts():
    readme = read("README.md")
    common = load("presets/common.json")["effective_engine"]
    assert "TP4 / PP1 / PCP1" in readme
    assert "DPTB `1024`" in readme
    assert "`6, 12, 18, 24`" in readme
    assert "MTP depths `2, 4, 5`" in readme
    assert common["fuse_gemm_comms"] is False
    assert common["kv_cache_dtype"] == "nvfp4_ds_mla"
    expected = {
        "fast-320k": (1, 320000, 10232320000),
        "balanced-650k": (2, 650000, 10395482112),
        "large-1m": (4, 1000000, 7995534848),
    }
    for name, (dcp, length, kv_bytes) in expected.items():
        preset = load(f"presets/{name}.json")["effective_engine"]
        assert (preset["decode_context_parallel_size"], preset["max_model_len"], preset["kv_cache_bytes_per_rank"]) == (dcp, length, kv_bytes)
        assert all(str(value) in readme for value in (length, kv_bytes))
    for value in (
        "26.565", "55.95", "40.47", "81.02", "694.1",
        "23.78", "52.56", "34.965", "72.59", "585.3",
        "22.25", "48.775", "36.60", "73.23", "422.8",
    ):
        assert value in readme


def test_runtime_docs_pin_build_a_nvfp4_and_generated_cache_boundaries():
    text = "\n".join(read(path) for path in (
        "README.md", "docs/how-it-works.md", "docs/patches.md", "docs/reproducibility.md",
    ))
    assert "965b6aeaf3a0c41abefec89a31592f6b8061a053053831be7c5e6f7560857515" in text
    assert "Build A v1" in text and "excluded" in text
    assert "368 bytes/token" in text
    assert "23,552 bytes" in text
    assert "generated locally" in text
    assert "never" in text and "portable" in text


def test_model_locks_and_licensing_are_explicit_external_boundaries():
    checkpoint = load("models/checkpoint.lock.json")
    sidecar = load("models/sidecar.lock.json")
    licensing = read("docs/licensing.md")
    assert checkpoint["distribution"]["method"] == "external-required"
    assert sidecar["distribution"]["method"] == "external-required"
    assert checkpoint["rights"]["status"] == sidecar["rights"]["status"] == "unknown"
    for term in ("Apache-2.0", "MIT", "BSD-3-Clause", "Jordan Miller", "Claude Fable 5", "model", "OCI"):
        assert term in licensing
    credits = read("CREDITS.md")
    assert "Jordan Miller" in credits and "Claude Fable 5" in credits
    assert "525b210dc100b75792ac5a0818d4eab0cdddc5a8" in credits


def test_benchmark_and_operations_documents_state_fail_closed_protocol():
    protocol = read("benchmarks/protocol.md")
    for term in (
        "determinism triplet", "900", "temperature=0", "thinking", "/tokenize",
        "TTFT", "fallback", "all attempts", "slow", "valid", "three valid",
        "five attempts", "cherry-picking",
    ):
        assert term.lower() in protocol.lower()
    operations = read("docs/operations.md")
    for term in (
        "real generation", "post-health", "rollback", "headroom", "polkit",
        "guardian", "co-resident", "fabric", "900K", "run identity", "current-window",
    ):
        assert term.lower() in operations.lower()
    assert "always safe, always" not in operations


def test_distributed_operator_docs_have_no_private_locations_and_local_links_resolve():
    paths = [ROOT / "README.md", ROOT / "benchmarks/protocol.md", *sorted((ROOT / "docs").glob("*.md"))]
    private_ip = re.compile(r"\b(?:10(?:\.\d{1,3}){3}|192\.168(?:\.\d{1,3}){2}|172\.(?:1[6-9]|2\d|3[01])(?:\.\d{1,3}){2})\b")
    private_path = re.compile(r"/(?:Users|home)/[^/\s]+")
    link = re.compile(r"\[[^]]+\]\(([^)]+)\)")
    for path in paths:
        text = path.read_text(encoding="utf-8")
        assert not private_ip.search(text), path
        assert not private_path.search(text), path
        for target in link.findall(text):
            if "://" in target or target.startswith("#"):
                continue
            relative = target.split("#", 1)[0]
            if not relative:
                continue
            assert (path.parent / relative).resolve().exists(), f"{path}: broken link {target}"
