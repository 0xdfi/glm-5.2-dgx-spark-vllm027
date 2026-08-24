#!/usr/bin/env python3
"""Compare wheel or directory build outputs after reproducibility normalization."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path


IGNORED_WHEEL_MEMBERS = (".dist-info/RECORD",)


def digest(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def normalized(path: Path) -> dict[str, str]:
    if path.is_dir():
        return {
            item.relative_to(path).as_posix(): digest(item.read_bytes())
            for item in sorted(path.rglob("*"))
            if item.is_file() and "__pycache__" not in item.parts and item.suffix != ".pyc"
        }
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            return {
                name: digest(archive.read(name))
                for name in sorted(archive.namelist())
                if not name.endswith("/") and not name.endswith(IGNORED_WHEEL_MEMBERS)
            }
    return {path.name: digest(path.read_bytes())}


def compare(left: Path, right: Path) -> dict[str, object]:
    a, b = normalized(left), normalized(right)
    changed = sorted(key for key in a.keys() & b.keys() if a[key] != b[key])
    return {
        "equivalent": a == b,
        "missing_from_left": sorted(b.keys() - a.keys()),
        "missing_from_right": sorted(a.keys() - b.keys()),
        "changed": changed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    parser.add_argument("--require-equivalent", action="store_true")
    args = parser.parse_args()
    report = compare(args.left, args.right)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 1 if args.require_equivalent and not report["equivalent"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
