# ── Neuromorphic Vision-RWKV for Autonomous Driving ──────────────────
# Convenience targets for common development tasks.

.PHONY: test lint format typecheck ci clean analyze train tune help setup

help:  ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  %-14s %s\n", $$1, $$2}'

# ── Setup ────────────────────────────────────────────────────────────

setup:  ## Install package in dev mode + pre-commit hooks
	pip install -e ".[dev]" --extra-index-url https://download.pytorch.org/whl/cu124
	pre-commit install
	@echo "\n✓ Dev environment ready. Pre-commit hooks installed."

# ── Quality ──────────────────────────────────────────────────────────

test:  ## Run offline tests with coverage (no CARLA needed)
	pytest tests/ -v -m "not online" --cov=src/ --cov-report=term-missing --cov-config=pyproject.toml

test-online:  ## Run CARLA integration tests (requires live CARLA server)
	pytest tests/ -v -m online

lint:  ## Run linting (flake8)
	flake8 src/ scripts/ tests/ agents/

format:  ## Check formatting (black)
	black --check --line-length 100 src/ scripts/ tests/ agents/

format-fix:  ## Auto-format code with black
	black --line-length 100 src/ scripts/ tests/ agents/

typecheck:  ## Run type checking (mypy)
	mypy src/ --config-file pyproject.toml

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
