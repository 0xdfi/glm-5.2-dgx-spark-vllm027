#!/usr/bin/env bash
set -euo pipefail

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${CHAMPION_PYTHON:-python3.12}"
venv_path="${CHAMPION_VENV:-${repository_root}/.venv}"
requirements_lock="${CHAMPION_REQUIREMENTS_LOCK:-${repository_root}/requirements-dev.lock}"
wheelhouse="${CHAMPION_WHEELHOUSE:-}"

if ! command -v "${python_bin}" >/dev/null 2>&1; then
  echo "bootstrap: Python 3.12 is required; set CHAMPION_PYTHON to its executable" >&2
  exit 2
fi

runtime_facts="$(${python_bin} -c 'import platform, sys; print("|".join((platform.python_version(), sys.platform, platform.machine())))')"
IFS='|' read -r python_version python_platform python_architecture <<<"${runtime_facts}"

case "${python_version}" in
  3.12.*) ;;
  *)
    echo "bootstrap: Python 3.12 is required, found ${python_version}" >&2
    exit 2
    ;;
esac

case "${python_platform}" in
  darwin|linux) ;;
  *)
    echo "bootstrap: unsupported developer platform ${python_platform}; expected darwin or linux" >&2
    exit 2
    ;;
esac

case "${python_architecture}" in
  aarch64|arm64|x86_64|AMD64) ;;
  *)
    echo "bootstrap: unsupported architecture ${python_architecture}" >&2
    exit 2
    ;;
esac

if [[ ! -f "${requirements_lock}" ]]; then
  echo "bootstrap: dependency lock not found: ${requirements_lock}" >&2
  exit 2
fi

if [[ -n "${wheelhouse}" && ! -d "${wheelhouse}" ]]; then
  echo "bootstrap: offline wheelhouse not found: ${wheelhouse}" >&2
  exit 2
fi

if [[ ! -x "${venv_path}/bin/python" ]]; then
  "${python_bin}" -m venv "${venv_path}"
fi

venv_version="$("${venv_path}/bin/python" -c 'import platform; print(platform.python_version())')"
case "${venv_version}" in
  3.12.*) ;;
  *)
    echo "bootstrap: ${venv_path} uses Python ${venv_version}; remove it and rerun bootstrap with Python 3.12" >&2
    exit 2
    ;;
esac

pip_args=(
  install
  --disable-pip-version-check
  --force-reinstall
  --require-hashes
  --quiet
)
if [[ -n "${wheelhouse}" ]]; then
  pip_args+=(--no-index --find-links "${wheelhouse}")
else
  pip_args+=(--index-url "https://pypi.org/simple")
fi
pip_args+=(-r "${requirements_lock}")

receipt_path="${venv_path}/dependency-lock.receipt.json"
rm -f "${receipt_path}"

env \
  -u PIP_INDEX_URL \
  -u PIP_EXTRA_INDEX_URL \
  -u PIP_TRUSTED_HOST \
  -u PIP_NO_INDEX \
  PIP_CONFIG_FILE=/dev/null \
  "${venv_path}/bin/python" -m pip "${pip_args[@]}"

"${venv_path}/bin/python" -m pip check --disable-pip-version-check >/dev/null

"${venv_path}/bin/python" - "${requirements_lock}" "${receipt_path}" <<'PY'
import hashlib
import json
import os
import platform
import sys
from pathlib import Path

lock_path = Path(sys.argv[1])
receipt_path = Path(sys.argv[2])
receipt = {
    "architecture": platform.machine(),
    "lock_file": lock_path.name,
    "lock_sha256": hashlib.sha256(lock_path.read_bytes()).hexdigest(),
    "platform": sys.platform,
    "python_version": platform.python_version(),
    "schema_version": "1.0.0",
}
temporary_path = receipt_path.with_suffix(".json.tmp")
temporary_path.write_text(
    json.dumps(receipt, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n",
    encoding="utf-8",
)
os.chmod(temporary_path, 0o644)
os.replace(temporary_path, receipt_path)
PY

echo "bootstrap: PASS (Python ${venv_version}, ${python_platform}/${python_architecture})"
