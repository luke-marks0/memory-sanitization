PYTHON ?= python
UV ?= uv
PROFILE ?= dev-small

.PHONY: sync-upstream check-upstream bootstrap-upstream-rust hydrate-upstream-rust build-bridge test-bridge test-upstream-rust build test test-parity test-hardware bench clean

sync-upstream:
	./scripts/sync_upstream.sh

check-upstream:
	$(UV) run python scripts/check_upstream_integrity.py
	$(UV) run python scripts/check_banned_shortcuts.py

bootstrap-upstream-rust:
	$(PYTHON) scripts/run_upstream_rust_tests.py --bootstrap-only

hydrate-upstream-rust:
	$(PYTHON) scripts/run_upstream_rust_tests.py --hydrate-only

build-bridge:
	$(UV) sync --extra dev
	$(PYTHON) scripts/build_bridge.py

test-bridge:
	$(MAKE) hydrate-upstream-rust
	$(MAKE) build-bridge
	POSE_RUN_REAL_BRIDGE_TESTS=1 $(UV) run pytest tests/e2e/test_real_filecoin_bridge.py

test-upstream-rust:
	$(PYTHON) scripts/run_upstream_rust_tests.py

build:
	$(UV) sync --extra dev
	$(MAKE) build-bridge
	$(UV) run $(PYTHON) -m compileall src tests
	$(UV) run pose --help > /dev/null

test:
	$(MAKE) check-upstream
	$(UV) run pytest tests/unit tests/integration tests/e2e tests/adversarial tests/performance

test-parity:
	$(UV) run pytest tests/parity

test-hardware:
	$(UV) run pytest tests/hardware

bench:
	$(UV) run pose bench run --profile $(PROFILE)

clean:
	rm -rf .venv .pytest_cache dist build target
