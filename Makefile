# Makefile for torchgeo-bench

PYTHON := python -m

.PHONY: tests lint format clean help

tests:
	$(PYTHON) pytest

lint:
	ruff check src/ tests/

format:
	ruff check --fix --select I src/ tests/
	ruff format src/ tests/

clean:
	rm -rf htmlcov .pytest_cache .coverage

help:
	@echo "Available targets:"
	@echo "  tests   - Run test suite with coverage"
	@echo "  lint    - Run ruff linter on src/ and tests/"
	@echo "  format  - Auto-fix imports and format code with ruff"
	@echo "  clean   - Remove generated files (htmlcov, .coverage, .pytest_cache)"
	@echo "  help    - Show this help message"
