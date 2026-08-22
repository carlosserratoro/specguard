# Script to further investigate BR2, which is the business rule that fails the most.
# It aims at seeing the reasoning being made.


from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from specguard.domain.spec import UseCaseSpec
from specguard.domain.thread import ConversationalThread
from specguard.service_layer.evaluation import (
    DEFAULT_EXTRACTOR_TEMPERATURE,
    extract,
    match_rule,
)

REPS = 30

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"


def save_report(report: dict, case_id: str) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    k = report["k"]
    temp = report["temperature"]
    path = RESULTS_DIR / f"inspect_br2_framing_{case_id}_k{k}_temp{temp}_{timestamp}.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return path


# Keep only BR2 in the gold.
thread = ConversationalThread.from_file(PROJECT_ROOT / "examples" / "t01_discounts.thread.yaml")
gold = UseCaseSpec.from_file(PROJECT_ROOT / "examples" / "t01_discounts.gold.yaml")
gold_br2 = next(br for br in gold.business_rules if br.id == "BR2")

report = {"k": REPS, "temperature": DEFAULT_EXTRACTOR_TEMPERATURE, "runs": []}
for rep_no in range(REPS):
    print(f"Repetition {rep_no + 1}/{REPS}")
    predicted = extract(thread)

    decision = match_rule(gold_br2, predicted.business_rules)
    report["runs"].append(
        {
            "matched": decision.matched,
            "reasoning": decision.reasoning,
            "predicted_rules": [
                {"id": r.id, "statement": r.statement} for r in predicted.business_rules
            ],
        }
    )

path = save_report(report, case_id="t01_discounts")
print(f"Report saved to {path}")
