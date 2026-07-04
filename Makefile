.PHONY: run format lint check test sync

run:
	uv run python -m specguard.main

format:
	uv run ruff format .
	uv run ruff check --fix .

format-check:
	uv run ruff format --check .

lint:
	uv run ruff check .

test:
	uv run pytest

sync:
	uv sync
