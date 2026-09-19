UV ?= uv
export MPLBACKEND := Agg
export PYTHONDONTWRITEBYTECODE := 1
.PHONY: install lint format test check
install:
	$(UV) sync --frozen --all-groups
lint:
	$(UV) run ruff check .
	$(UV) run ruff format --check .
format:
	$(UV) run ruff format .
test:
	$(UV) run pytest
check: lint test
