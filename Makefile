.PHONY: run format lint check test sync

run:
	uv run python -m specguard.main

format:
	uv run ruff check --fix .
	uv run ruff format .

format-check:
	uv run ruff format --check .

lint:
	uv run ruff check .

test:
	uv run pytest

typecheck:
	uv run mypy

check: format-check lint typecheck test

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
        plot-thread-verdicts-judge-cost-and-latency \
		plot-generative-vs-decision

plots: plot-recall-precision plot-missed-by-rule plot-br2-framing \
       plot-thread-verdicts plot-thread-verdicts-judge-effort \
       plot-thread-verdicts-judge-reasoning-tokens \
       plot-thread-verdicts-judge-cost-and-latency \
	   plot-thread-verdicts-judge-cot \
	   plot-generative-vs-decision

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

plot-thread-verdicts-judge-cot:
	$(PLOT).plot_thread_verdicts_judge_cot_accuracy_k100
	$(PLOT).plot_thread_verdicts_judge_cot_accuracy_k1000
	$(PLOT).plot_thread_verdicts_judge_cot_accuracy_cover
	$(PLOT).plot_thread_verdicts_judge_cot_cost
	$(PLOT).plot_thread_verdicts_judge_cot_latency
	$(PLOT).plot_thread_verdicts_judge_cot_latency_per_call
	$(PLOT).plot_thread_verdicts_judge_cot_latency_tail

exp-generative-vs-decision:
	uv run python scripts/run_generative_vs_decision.py -k 100 --configs A A-prime B
	uv run python scripts/run_generative_vs_decision.py -k 1000 --configs B --statements RESTRICTION-FORM
	uv run python scripts/run_generative_vs_decision.py -k 100 --configs C
	uv run python scripts/run_generative_vs_decision.py -k 100 --configs D

exp-generative-vs-decision-hint-vs-no-hint:
	uv run python scripts/run_generative_vs_decision.py -k 100 --configs A A-no-hint A-prime A-prime-no-hint B B-no-hint C C-no-hint D D-no-hint E E-no-hint

PLOT_P07 = $(PLOT).faithfulness_llm_vs_jev

plot-generative-vs-decision:
	$(PLOT_P07).summarise_generative_vs_decision
	$(PLOT_P07).plot_generative_vs_decision_hint_effect
	$(PLOT_P07).plot_generative_vs_decision_probabilities
	$(PLOT_P07).plot_generative_vs_decision_confidence
	$(PLOT_P07).plot_generative_vs_decision_cost
	$(PLOT_P07).plot_generative_vs_decision_latency
	$(PLOT_P07).plot_generative_vs_decision_cover
