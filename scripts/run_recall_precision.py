from __future__ import annotations

from paths import PROJECT_ROOT, RESULTS_DIR

from specguard.adapters.llm import JudgeConfig
from specguard.adapters.reporting import save_report
from specguard.domain.spec import UseCaseSpec
from specguard.domain.thread import ConversationalThread
from specguard.service_layer.evaluation import evaluate

REPS = 20

thread = ConversationalThread.from_file(PROJECT_ROOT / "examples" / "t01_discounts.thread.yaml")
gold = UseCaseSpec.from_file(PROJECT_ROOT / "examples" / "t01_discounts.gold.yaml")

judge_config = JudgeConfig()
report = evaluate(thread, gold, k=REPS, judge_config=judge_config)
path = save_report(
    report,
    judge_config,
    results_dir=RESULTS_DIR,
    experiment="recall-precision",
    case_id=f"t01_discounts_k{REPS}",
)
print(f"Report saved to {path}")
