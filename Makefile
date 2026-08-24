CHAMPION_VENV ?= .venv
VENV_PYTHON := $(CHAMPION_VENV)/bin/python

.PHONY: bootstrap check-venv test contracts lint-manifests

bootstrap:
	./scripts/bootstrap_dev.sh

check-venv:
	@test -x "$(VENV_PYTHON)" || { echo "development environment missing; run 'make bootstrap'" >&2; exit 2; }
	@"$(VENV_PYTHON)" -c 'import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 12) else "development environment must use Python 3.12; run make bootstrap")'

test: check-venv
	@"$(VENV_PYTHON)" -m pytest

contracts: check-venv
	@"$(VENV_PYTHON)" -m pytest tests/test_contracts.py -q
	@echo "contracts: PASS"

lint-manifests: check-venv
	@"$(VENV_PYTHON)" -c 'import json; from pathlib import Path; from jsonschema import Draft202012Validator; schemas = sorted(Path("schemas").glob("*.schema.json")); [Draft202012Validator.check_schema(json.loads(path.read_text(encoding="utf-8"))) for path in schemas]; print(f"manifests: PASS ({len(schemas)} schemas)")'
