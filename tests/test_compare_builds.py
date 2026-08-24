from __future__ import annotations

import zipfile
from pathlib import Path

from scripts.compare_builds import compare


def write_wheel(path: Path, payload: bytes, timestamp: tuple[int, int, int, int, int, int]) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        info = zipfile.ZipInfo("pkg/module.py", timestamp)
        archive.writestr(info, payload)
        archive.writestr("pkg-1.0.dist-info/RECORD", b"build-specific-record")


def test_wheel_comparison_ignores_zip_timestamps_and_record(tmp_path):
    left, right = tmp_path / "a.whl", tmp_path / "b.whl"
    write_wheel(left, b"answer = 42\n", (2020, 1, 1, 0, 0, 0))
    write_wheel(right, b"answer = 42\n", (2026, 1, 1, 0, 0, 0))
    assert compare(left, right)["equivalent"] is True


def test_wheel_comparison_reports_content_change(tmp_path):
    left, right = tmp_path / "a.whl", tmp_path / "b.whl"
    write_wheel(left, b"answer = 42\n", (2020, 1, 1, 0, 0, 0))
    write_wheel(right, b"answer = 43\n", (2020, 1, 1, 0, 0, 0))
    report = compare(left, right)
    assert report["equivalent"] is False
    assert report["changed"] == ["pkg/module.py"]
