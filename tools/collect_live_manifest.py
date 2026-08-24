#!/usr/bin/env python3
"""Validate an operator-collected read-only live manifest without contacting hosts.

This deliberately inert helper never runs SSH, Docker, or inference.  Operators
may supply an independently collected JSON snapshot; publication keeps b4 parity
unverified until such evidence is reviewed.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


SHA256 = re.compile(r"^[0-9a-f]{64}$")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-json", required=True, type=Path)
    args = parser.parse_args()
    payload = json.loads(args.from_json.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != "1.0.0":
        raise SystemExit("live-manifest: FAIL: unsupported snapshot schema")
    if payload.get("collection_mode") != "operator-supplied-read-only":
        raise SystemExit("live-manifest: FAIL: collection mode must be operator-supplied-read-only")
    files = payload.get("files")
    if not isinstance(files, list) or any(
        not isinstance(item, dict)
        or set(item) != {"path", "sha256"}
        or not isinstance(item["path"], str)
        or not SHA256.fullmatch(item["sha256"])
        for item in files
    ):
        raise SystemExit("live-manifest: FAIL: malformed file inventory")
    print(f"live-manifest: PASS ({len(files)} operator-supplied files; no host contacted)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
