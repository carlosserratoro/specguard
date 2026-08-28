# Script to further investigate BR2, which is the business rule that fails the most.
# It aims at seeing the reasoning being made.


from __future__ import annotations

from paths import PROJECT_ROOT, RESULTS_DIR

from specguard.adapters.reporting import save_report
from specguard.domain.spec import UseCaseSpec
from specguard.domain.thread import ConversationalThread
from specguard.service_layer.evaluation import (
    DEFAULT_EXTRACTOR_TEMPERATURE,
    JudgeConfig,
    extract,
    match_rule,
)

REPS = 30


# Keep only BR2 in the gold.
thread = ConversationalThread.from_file(PROJECT_ROOT / "examples" / "t01_discounts.thread.yaml")
gold = UseCaseSpec.from_file(PROJECT_ROOT / "examples" / "t01_discounts.gold.yaml")
gold_br2 = next(br for br in gold.business_rules if br.id == "BR2")


judge_config = JudgeConfig()
report = {
    "k": REPS,
    "extractor_temperature": DEFAULT_EXTRACTOR_TEMPERATURE,
    "runs": [],
}
for rep_no in range(REPS):
    print(f"Repetition {rep_no + 1}/{REPS}")
    predicted = extract(thread)

    decision = match_rule(gold_br2, predicted.business_rules, judge_config)
    report["runs"].append(
        {
            "matched": decision.matched,
            "reasoning": decision.reasoning,
            "predicted_rules": [
                {"id": r.id, "statement": r.statement} for r in predicted.business_rules
            ],
        }
    )

path = save_report(
    report,
    judge_config,
    results_dir=RESULTS_DIR,
    experiment="br2-framing",
    case_id=f"t01_discounts_k{REPS}",
)
print(f"Report saved to {path}")
