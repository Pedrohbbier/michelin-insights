PYTHON ?= .venv/bin/python
PIP ?= .venv/bin/pip
STREAMLIT ?= .venv/bin/streamlit
PYTEST ?= .venv/bin/pytest
RUFF ?= .venv/bin/ruff

.PHONY: help venv install install-dev run test lint format samples clean

help: ## Show available commands
	@awk 'BEGIN {FS = ":.*## "; printf "\nCommands:\n"} /^[a-zA-Z_-]+:.*## / {printf "  make %-14s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

venv: ## Create the virtual environment
	python3 -m venv .venv

install: venv ## Install runtime dependencies
	$(PIP) install -r requirements.txt

install-dev: venv ## Install runtime + test/lint dependencies
	$(PIP) install -r requirements-dev.txt

run: ## Start the Streamlit app
	$(STREAMLIT) run streamlit_app.py

test: ## Run the test suite
	$(PYTEST)

lint: ## Lint the code with ruff
	$(RUFF) check .

format: ## Format the code with ruff
	$(RUFF) format .

samples: ## Regenerate data/samples from the bundled CSV
	$(PYTHON) -m scripts.make_samples

clean: ## Remove caches and temporary files
	rm -rf .pytest_cache .ruff_cache .mypy_cache
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
