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
	uv run python scripts/run_faithfulness.py -k 100 --effort high
	uv run python scripts/run_faithfulness.py -k 100 --effort none --scaffolding
	uv run python scripts/run_faithfulness.py -k 100 --effort low --scaffolding
	uv run python scripts/run_faithfulness.py -k 100 --effort medium --scaffolding

PLOT = uv run python -m scripts.plots
PLOT_RECALL_PRECISION_JSON = results/recall-precision/t01_discounts_k20_effort-none_scaffold-off_20260808T181041Z.json

.PHONY: plots plot-recall-precision plot-missed-by-rule plot-br2-framing \
        plot-thread-verdicts plot-thread-verdicts-judge-effort \
        plot-thread-verdicts-judge-reasoning-tokens \
        plot-thread-verdicts-judge-cost-and-latency

plots: plot-recall-precision plot-missed-by-rule plot-br2-framing \
       plot-thread-verdicts plot-thread-verdicts-judge-effort \
       plot-thread-verdicts-judge-reasoning-tokens \
       plot-thread-verdicts-judge-cost-and-latency

plot-recall-precision:
	$(PLOT).plot_recall_precision $(PLOT_RECALL_PRECISION_JSON)

plot-missed-by-rule:
	$(PLOT).plot_missed_by_rule

plot-br2-framing:
	$(PLOT).plot_br2_framing

plot-thread-verdicts:
	$(PLOT).plot_thread_verdicts

plot-thread-verdicts-judge-effort:
	$(PLOT).plot_thread_verdicts_judge_effort --k 10
	$(PLOT).plot_thread_verdicts_judge_effort --k 100

plot-thread-verdicts-judge-reasoning-tokens:
	$(PLOT).plot_thread_verdicts_judge_reasoning_tokens
	$(PLOT).plot_thread_verdicts_judge_reasoning_tokens --cover

plot-thread-verdicts-judge-cost-and-latency:
	$(PLOT).plot_thread_verdicts_judge_cost_and_latency
