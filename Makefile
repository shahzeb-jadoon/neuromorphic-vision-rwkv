# ── Neuromorphic Vision-RWKV for Autonomous Driving ──────────────────
# Convenience targets for common development tasks.

.PHONY: test lint format typecheck ci clean analyze train tune help setup lock env-check

help:  ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  %-14s %s\n", $$1, $$2}'

# ── Setup ────────────────────────────────────────────────────────────

LOCK_FLAGS := --extra dev --extra simulator \
	--python-version 3.11 --python-platform x86_64-manylinux_2_31 \
	--extra-index-url https://download.pytorch.org/whl/cu124 \
	--index-strategy unsafe-best-match --generate-hashes --emit-index-url \
	--custom-compile-command "make lock"

setup:  ## Install locked dependencies, the package (editable) and pre-commit hooks
	pip install -r requirements-lock.txt
	pip install -e . --no-deps
	pre-commit install
	@echo "\n✓ Dev environment ready. Pre-commit hooks installed."

lock:  ## Re-pin requirements-lock.txt from pyproject.toml (needs uv)
	uv pip compile pyproject.toml $(LOCK_FLAGS) -o requirements-lock.txt

LOCK_MACOS_FLAGS := --extra dev \
	--python-version 3.11 --python-platform aarch64-apple-darwin \
	--generate-hashes --custom-compile-command "make lock-macos"

lock-macos:  ## Re-pin requirements-lock-macos.txt (Apple silicon, no CUDA/CARLA)
	uv pip compile pyproject.toml $(LOCK_MACOS_FLAGS) -o requirements-lock-macos.txt

env-check:  ## Verify the environment: imports, offline tests, GPU smoke train
	python scripts/verify_env.py
	pytest tests/ -q -m "not online" -p no:cacheprovider --no-cov
	test -d data/toy || $(MAKE) toy-data
	rm -rf runs/smoke
	python scripts/train.py --config configs/smoke_toy.yaml --max-epochs 1
	python scripts/check_smoke_run.py runs/smoke

# ── Quality ──────────────────────────────────────────────────────────

test:  ## Run offline tests with coverage (no CARLA needed)
	pytest tests/ -v -m "not online" --cov=src/ --cov-report=term-missing --cov-config=pyproject.toml

test-online:  ## Run CARLA integration tests (requires live CARLA server)
	pytest tests/ -v -m online

lint:  ## Lint with ruff
	ruff check src/ scripts/ tests/

format:  ## Check formatting with ruff
	ruff format --check src/ scripts/ tests/

format-fix:  ## Format code with ruff
	ruff format src/ scripts/ tests/

typecheck:  ## Type-check with mypy
	mypy --config-file=pyproject.toml scripts/ src/ tests/

ci: lint format typecheck test  ## Run full CI pipeline locally

# ── Data ─────────────────────────────────────────────────────────────

toy-data:  ## Generate toy dataset for testing
	python scripts/create_toy_dataset.py --output data/toy --episodes 10 --frames 100

analyze:  ## Analyze collected dataset
	python scripts/analyze_dataset.py data/collected --save-samples 10

collect:  ## Run overnight collection campaign
	bash scripts/overnight_collection.sh

validate:  ## Validate dataset integrity
	python scripts/validate_dataset.py data/collected

# ── Training (Phase 2) ───────────────────────────────────────────────

train:  ## Start model training (Phase 2)
	python scripts/train.py --config configs/vrwkv_base.yaml

tune:  ## Run Optuna hyperparameter search (short trials)
	python scripts/tune_hparams.py --config configs/vrwkv_base.yaml --n-trials 10 --trial-epochs 2

# ── Utilities ────────────────────────────────────────────────────────

clean:  ## Remove build artifacts and caches
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .mypy_cache -exec rm -rf {} + 2>/dev/null || true
	rm -rf data/test_fixtures/ data/test_toy/
	rm -rf build/ dist/ *.egg-info
