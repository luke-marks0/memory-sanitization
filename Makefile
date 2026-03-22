PYTHON ?= python
UV ?= uv
PROFILE ?= dev-small

.PHONY: check build test test-parity test-graphs test-hardware calibrate bench clean

check:
	$(UV) run python scripts/check_banned_shortcuts.py

build:
	$(UV) sync --extra dev
	$(UV) run $(PYTHON) -m compileall src tests
	$(UV) run $(PYTHON) -m pose.cli.main --help > /dev/null

test:
	$(MAKE) check
	$(UV) run pytest tests/unit tests/integration tests/e2e tests/adversarial tests/performance

test-parity:
	$(UV) run pytest tests/parity

test-graphs:
	$(UV) run pytest tests/unit tests/parity -k graph

test-hardware:
	$(UV) run pytest tests/hardware

calibrate:
	$(UV) run $(PYTHON) -m pose.cli.main verifier calibrate --profile $(PROFILE)

bench:
	$(UV) run $(PYTHON) -m pose.cli.main bench run --profile $(PROFILE)

clean:
	rm -rf .venv .pytest_cache dist build target
