from __future__ import annotations

import base64
import hashlib
import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[1]
BOOTSTRAP = REPOSITORY / "scripts" / "bootstrap_dev.sh"


def python_312() -> str:
    if sys.version_info[:2] == (3, 12):
        return sys.executable
    executable = shutil.which("python3.12")
    assert executable is not None, "bootstrap tests require an installed Python 3.12"
    return executable


def _record_line(path: str, content: bytes) -> str:
    digest = base64.urlsafe_b64encode(hashlib.sha256(content).digest()).rstrip(b"=")
    return f"{path},sha256={digest.decode('ascii')},{len(content)}"


def make_wheelhouse(tmp_path: Path) -> tuple[Path, Path, str]:
    wheelhouse = tmp_path / "wheelhouse"
    wheelhouse.mkdir()
    wheel_name = "champion_bootstrap_fixture-1.0-py3-none-any.whl"
    wheel = wheelhouse / wheel_name
    files = {
        "champion_bootstrap_fixture/__init__.py": b'VALUE = "offline-installed"\n',
        "champion_bootstrap_fixture-1.0.dist-info/METADATA": (
            b"Metadata-Version: 2.1\n"
            b"Name: champion-bootstrap-fixture\n"
            b"Version: 1.0\n"
        ),
        "champion_bootstrap_fixture-1.0.dist-info/WHEEL": (
            b"Wheel-Version: 1.0\n"
            b"Generator: champion-bootstrap-test\n"
            b"Root-Is-Purelib: true\n"
            b"Tag: py3-none-any\n"
        ),
    }
    record_path = "champion_bootstrap_fixture-1.0.dist-info/RECORD"
    record = "\n".join(_record_line(path, content) for path, content in files.items())
    files[record_path] = f"{record}\n{record_path},,\n".encode("utf-8")
    with zipfile.ZipFile(wheel, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path, content in files.items():
            archive.writestr(path, content)
    wheel_hash = hashlib.sha256(wheel.read_bytes()).hexdigest()
    lock = tmp_path / "requirements.lock"
    lock.write_text(
        f"champion-bootstrap-fixture==1.0 --hash=sha256:{wheel_hash}\n",
        encoding="utf-8",
    )
    return wheelhouse, lock, wheel_hash


def bootstrap_environment(tmp_path: Path, wheelhouse: Path, lock: Path) -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(
        {
            "CHAMPION_PYTHON": python_312(),
            "CHAMPION_VENV": str(tmp_path / ".venv"),
            "CHAMPION_REQUIREMENTS_LOCK": str(lock),
            "CHAMPION_WHEELHOUSE": str(wheelhouse),
            "PIP_NO_INDEX": "1",
            "PIP_INDEX_URL": "https://user:canary-password@example.invalid/simple",
        }
    )
    return environment


def run_bootstrap(environment: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(BOOTSTRAP)],
        cwd=REPOSITORY,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )


def test_reuse_pin_includes_a_locked_encoding_backend():
    pyproject = (REPOSITORY / "pyproject.toml").read_text(encoding="utf-8")
    lock = (REPOSITORY / "requirements-dev.lock").read_text(encoding="utf-8")

    assert 'reuse[charset-normalizer]==6.2.0' in pyproject
    assert "charset-normalizer==" in lock


def test_clean_offline_wheelhouse_install_emits_sanitized_lock_receipt(tmp_path):
    wheelhouse, lock, _ = make_wheelhouse(tmp_path)
    environment = bootstrap_environment(tmp_path, wheelhouse, lock)

    result = run_bootstrap(environment)

    assert result.returncode == 0, result.stdout + result.stderr
    venv = Path(environment["CHAMPION_VENV"])
    installed = subprocess.run(
        [
            str(venv / "bin" / "python"),
            "-c",
            "import champion_bootstrap_fixture as fixture; print(fixture.VALUE)",
        ],
        text=True,
        capture_output=True,
        check=True,
    )
    assert installed.stdout.strip() == "offline-installed"
    receipt = json.loads((venv / "dependency-lock.receipt.json").read_text())
    expected_python_version = subprocess.check_output(
        [python_312(), "-c", "import platform; print(platform.python_version())"],
        text=True,
    ).strip()
    assert receipt == {
        "architecture": os.uname().machine,
        "lock_file": lock.name,
        "lock_sha256": hashlib.sha256(lock.read_bytes()).hexdigest(),
        "platform": sys.platform,
        "python_version": expected_python_version,
        "schema_version": "1.0.0",
    }
    transcript = result.stdout + result.stderr + json.dumps(receipt)
    assert "canary-password" not in transcript
    assert "https://user" not in transcript


def test_bootstrap_refuses_wrong_python_before_creating_venv(tmp_path):
    wheelhouse, lock, _ = make_wheelhouse(tmp_path)
    fake_python = tmp_path / "python3.11"
    fake_python.write_text(
        "#!/bin/sh\nprintf '%s\\n' '3.11.9|linux|x86_64'\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    environment = bootstrap_environment(tmp_path, wheelhouse, lock)
    environment["CHAMPION_PYTHON"] = str(fake_python)

    result = run_bootstrap(environment)

    assert result.returncode != 0
    assert "Python 3.12" in result.stderr
    assert not Path(environment["CHAMPION_VENV"]).exists()


def test_bootstrap_refuses_bad_requirement_hash(tmp_path):
    wheelhouse, lock, _ = make_wheelhouse(tmp_path)
    lock.write_text(
        f"champion-bootstrap-fixture==1.0 --hash=sha256:{'0' * 64}\n",
        encoding="utf-8",
    )
    environment = bootstrap_environment(tmp_path, wheelhouse, lock)

    result = run_bootstrap(environment)

    assert result.returncode != 0
    assert "DO NOT MATCH THE HASHES" in result.stderr.upper()
    assert not (
        Path(environment["CHAMPION_VENV"]) / "dependency-lock.receipt.json"
    ).exists()


def test_bootstrap_is_idempotent(tmp_path):
    wheelhouse, lock, _ = make_wheelhouse(tmp_path)
    environment = bootstrap_environment(tmp_path, wheelhouse, lock)
    first = run_bootstrap(environment)
    assert first.returncode == 0, first.stdout + first.stderr
    venv = Path(environment["CHAMPION_VENV"])
    marker = venv / "preserve-on-rebootstrap"
    marker.write_text("keep me\n", encoding="utf-8")
    first_receipt = (venv / "dependency-lock.receipt.json").read_bytes()

    second = run_bootstrap(environment)

    assert second.returncode == 0, second.stdout + second.stderr
    assert marker.read_text(encoding="utf-8") == "keep me\n"
    assert (venv / "dependency-lock.receipt.json").read_bytes() == first_receipt


def test_bootstrap_repairs_a_tampered_existing_environment(tmp_path):
    wheelhouse, lock, _ = make_wheelhouse(tmp_path)
    environment = bootstrap_environment(tmp_path, wheelhouse, lock)
    first = run_bootstrap(environment)
    assert first.returncode == 0, first.stdout + first.stderr
    venv = Path(environment["CHAMPION_VENV"])
    module_path = subprocess.check_output(
        [
            str(venv / "bin" / "python"),
            "-c",
            "import champion_bootstrap_fixture as fixture; print(fixture.__file__)",
        ],
        text=True,
    ).strip()
    Path(module_path).write_text('VALUE = "tampered"\n', encoding="utf-8")

    second = run_bootstrap(environment)

    assert second.returncode == 0, second.stdout + second.stderr
    repaired = subprocess.check_output(
        [
            str(venv / "bin" / "python"),
            "-c",
            "import champion_bootstrap_fixture as fixture; print(fixture.VALUE)",
        ],
        text=True,
    )
    assert repaired.strip() == "offline-installed"


def test_changed_bad_lock_invalidates_an_existing_environment_receipt(tmp_path):
    wheelhouse, lock, _ = make_wheelhouse(tmp_path)
    environment = bootstrap_environment(tmp_path, wheelhouse, lock)
    first = run_bootstrap(environment)
    assert first.returncode == 0, first.stdout + first.stderr
    receipt_path = Path(environment["CHAMPION_VENV"]) / "dependency-lock.receipt.json"
    assert receipt_path.exists()
    lock.write_text(
        f"champion-bootstrap-fixture==1.0 --hash=sha256:{'0' * 64}\n",
        encoding="utf-8",
    )

    second = run_bootstrap(environment)

    assert second.returncode != 0
    assert "DO NOT MATCH THE HASHES" in second.stderr.upper()
    assert not receipt_path.exists()


def test_bootstrap_supports_a_venv_path_with_spaces(tmp_path):
    wheelhouse, lock, _ = make_wheelhouse(tmp_path)
    environment = bootstrap_environment(tmp_path, wheelhouse, lock)
    environment["CHAMPION_VENV"] = str(tmp_path / "developer environment with spaces")

    result = run_bootstrap(environment)

    assert result.returncode == 0, result.stdout + result.stderr
    assert (Path(environment["CHAMPION_VENV"]) / "bin" / "python").is_file()


def test_make_test_uses_quoted_champion_venv_path_from_environment(tmp_path):
    custom_venv = tmp_path / "developer environment with spaces"
    bin_directory = custom_venv / "bin"
    bin_directory.mkdir(parents=True)
    marker = tmp_path / "custom-python-invoked"
    wrapper = bin_directory / "python"
    wrapper.write_text(
        "#!/bin/sh\n"
        f"printf '%s\\n' invoked >> '{marker}'\n"
        f"exec '{REPOSITORY / '.venv/bin/python'}' \"$@\"\n",
        encoding="utf-8",
    )
    wrapper.chmod(0o755)
    environment = os.environ.copy()
    environment.update(
        {
            "CHAMPION_VENV": str(custom_venv),
            "PYTEST_ADDOPTS": "tests/test_contracts.py -q",
        }
    )

    result = subprocess.run(
        ["make", "test"],
        cwd=REPOSITORY,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert marker.exists()
