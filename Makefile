# ── Neuromorphic Vision-RWKV for Autonomous Driving ──────────────────
# Convenience targets for common development tasks.

.PHONY: test lint format typecheck ci clean analyze train help

help:  ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  %-14s %s\n", $$1, $$2}'

# ── Quality ──────────────────────────────────────────────────────────

test:  ## Run tests with coverage
	pytest tests/ -v --cov=src/ --cov-report=term-missing --cov-fail-under=80

lint:  ## Run linting (flake8)
	flake8 src/ scripts/ tests/ --max-line-length=100 --extend-ignore=E203,W503

format:  ## Check formatting (black)
	black --check --line-length 100 src/ scripts/ tests/

format-fix:  ## Auto-format code with black
	black --line-length 100 src/ scripts/ tests/

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
	@echo "Training pipeline not yet implemented — Phase 2"

# ── Utilities ────────────────────────────────────────────────────────

clean:  ## Remove build artifacts and caches
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .mypy_cache -exec rm -rf {} + 2>/dev/null || true
	rm -rf data/test_fixtures/ data/test_toy/
	rm -rf build/ dist/ *.egg-info
