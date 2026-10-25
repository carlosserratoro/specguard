"""Compare GPT-5.4 mini and Jev as faithfulness judges (post 7)."""

from __future__ import annotations

import argparse

from paths import EXAMPLES_DIR, RESULTS_DIR, RESULTS_LLM_VS_JEV_HINT_VS_NO_HINT_DIR

from specguard.adapters.jev import DecisionConfig
from specguard.adapters.llm import JudgeConfig
from specguard.adapters.reporting import save_report
from specguard.adapters.statements import load_statements
from specguard.bootstrap import load_environment, make_decision_verifier, make_generative_verifier
from specguard.domain.thread import ConversationalThread
from specguard.service_layer.faithfulness import check, check_majority

load_environment()

GENERATIVE_MODEL = "openrouter/openai/gpt-5.4-mini"
DECISION_MODEL_JEV = "typesafe/jev-1.13"
DECISION_MODEL_LUNA = "openai/gpt-6-luna-decisions"

CONFIGS = {
    "A": DecisionConfig(model=DECISION_MODEL_JEV, input_style="same-as-llm"),
    "A-prime": DecisionConfig(model=DECISION_MODEL_JEV, input_style="jev"),
    "B": JudgeConfig(model=GENERATIVE_MODEL, scaffolding=True),
    "C": JudgeConfig(model=GENERATIVE_MODEL, scaffolding=True),
    "D": DecisionConfig(model=DECISION_MODEL_LUNA, input_style="same-as-llm"),
    "A-no-hint": DecisionConfig(model=DECISION_MODEL_JEV, input_style="same-as-llm", hint=False),
    "A-prime-no-hint": DecisionConfig(model=DECISION_MODEL_JEV, input_style="jev", hint=False),
    "B-no-hint": JudgeConfig(model=GENERATIVE_MODEL, scaffolding=True, hint=False),
    "C-no-hint": JudgeConfig(model=GENERATIVE_MODEL, scaffolding=True, hint=False),
    "D-no-hint": DecisionConfig(model=DECISION_MODEL_LUNA, input_style="same-as-llm", hint=False),
    "E": DecisionConfig(model=DECISION_MODEL_LUNA, input_style="jev"),
    "E-no-hint": DecisionConfig(model=DECISION_MODEL_LUNA, input_style="jev", hint=False),
}
GROUP_SIZE = 10

THRESHOLD_STATEMENTS = load_statements(EXAMPLES_DIR / "t01_threshold_statements.yaml")

parser = argparse.ArgumentParser(
    description="Compares GPT-5.4 mini, Jev and Luna Decisions on the threshold cases."
)
parser.add_argument("-k", "--reps", type=int, required=True, help="Decisions per statement.")
parser.add_argument("--configs", nargs="+", choices=list(CONFIGS), default=list(CONFIGS))
parser.add_argument(
    "--statements",
    nargs="+",
    choices=list(THRESHOLD_STATEMENTS),
    default=list(THRESHOLD_STATEMENTS),
)
args = parser.parse_args()

thread = ConversationalThread.from_file(EXAMPLES_DIR / "t01_discounts.thread.yaml")

for name in args.configs:
    config = CONFIGS[name]
    if isinstance(config, DecisionConfig):
        verify = make_decision_verifier(thread, config)
    else:
        verify = make_generative_verifier(thread, config)

    for label in args.statements:
        statement = THRESHOLD_STATEMENTS[label]
        if name in ("C", "C-no-hint"):
            report = check_majority(
                statement, label, verify, groups=args.reps, group_size=GROUP_SIZE
            )
            case_id = f"t01_discounts_{label}_groups{args.reps}x{GROUP_SIZE}"
        else:
            report = check(statement, label, verify, k=args.reps)
            case_id = f"t01_discounts_{label}_{name}_k{args.reps}"
        save_report(
            report,
            config,
            results_dir=RESULTS_DIR,
            experiment=RESULTS_LLM_VS_JEV_HINT_VS_NO_HINT_DIR.name,
            case_id=case_id,
        )
