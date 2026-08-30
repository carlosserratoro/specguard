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

# Experiments:
# Examples:
# make exp-faithfulness EFFORT=medium K=100
# make exp-faithfulness-sweep K=10
K ?= 1
EFFORT ?= none

exp-recall-precision:
	uv run python scripts/run_recall_precision.py -k $(K)

exp-br2-framing:
	uv run python scripts/inspect_br2_framing.py -k $(K)

exp-faithfulness:
	uv run python scripts/run_faithfulness.py -k $(K) --effort $(EFFORT)

exp-faithfulness-sweep:
	@for e in none low medium high xhigh; do \
		uv run python scripts/run_faithfulness.py -k $(K) --effort $$e; \
	done

exp-faithfulness-sweep-scaffolding:
	@for e in none low medium high xhigh; do \
		uv run python scripts/run_faithfulness.py -k $(K) --effort $$e --scaffolding; \
	done

exp-reasoning-vs-scaffolding:
	uv run python scripts/run_faithfulness.py -k 100 --effort none
	uv run python scripts/run_faithfulness.py -k 100 --effort low
	uv run python scripts/run_faithfulness.py -k 100 --effort medium
	uv run python scripts/run_faithfulness.py -k 100 --effort none --scaffolding
	uv run python scripts/run_faithfulness.py -k 100 --effort low --scaffolding
