# Copyright 2026 Carlos Serra-Toro
# SPDX-License-Identifier: Apache-2.0

# Script to further investigate BR2, which is the business rule that fails the most.
# It aims at seeing the reasoning being made.


from __future__ import annotations

import logging

from cli import build_parser, judge_config_from_args
from paths import EXAMPLES_DIR, RESULTS_DIR

from specguard.adapters.llm import DEFAULT_EXTRACTOR_TEMPERATURE
from specguard.adapters.reporting import save_report
from specguard.bootstrap import load_environment
from specguard.domain.spec import UseCaseSpec
from specguard.domain.thread import ConversationalThread
from specguard.service_layer.extractor import extract
from specguard.service_layer.judges import match_rule

logging.basicConfig(level=logging.INFO, format="%(message)s")

load_environment()

parser = build_parser("Inspects BR2 framing over k extractions.")
args = parser.parse_args()
judge_config = judge_config_from_args(args)


# Keep only BR2 in the gold.
thread = ConversationalThread.from_file(EXAMPLES_DIR / "t01_discounts.thread.yaml")
gold = UseCaseSpec.from_file(EXAMPLES_DIR / "t01_discounts.gold.yaml")
gold_br2 = next(br for br in gold.business_rules if br.id == "BR2")


report = {
    "k": args.reps,
    "extractor_temperature": DEFAULT_EXTRACTOR_TEMPERATURE,
    "runs": [],
}
for rep_no in range(args.reps):
    print(f"Repetition {rep_no + 1}/{args.reps}")
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
    case_id=f"t01_discounts_k{args.reps}",
)
print(f"Report saved to {path}")
