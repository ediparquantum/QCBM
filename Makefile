PACKAGE = qcbm
TEST = tests

.PHONY: help
help:
	@echo "Available tasks:"
	@echo "  check:   Check code for conformance with respect to formatting, style, and type-checking."
	@echo "  format:  Format code with black."
	@echo "  test:    Run unit tests (with coverage reporting)."

.PHONY: check
check:
	poetry run black --check --diff $(PACKAGE) $(TEST)
	poetry run flake8 $(PACKAGE) $(TEST)
	poetry run mypy $(PACKAGE) $(TEST)

.PHONY: format
format:
	poetry run black $(PACKAGE) $(TEST)

.PHONY: test
test:
	poetry run pytest -v --doctest-modules --cov=$(PACKAGE) --cov-report=term-missing
