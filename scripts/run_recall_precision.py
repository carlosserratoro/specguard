# Copyright 2026 Carlos Serra-Toro
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from cli import build_parser, judge_config_from_args
from paths import EXAMPLES_DIR, RESULTS_DIR

from specguard.adapters.reporting import save_report
from specguard.bootstrap import load_environment
from specguard.domain.spec import UseCaseSpec
from specguard.domain.thread import ConversationalThread
from specguard.service_layer.evaluation import evaluate

load_environment()

parser = build_parser("Measures recall and precision over k extractions.")
args = parser.parse_args()
judge_config = judge_config_from_args(args)

thread = ConversationalThread.from_file(EXAMPLES_DIR / "t01_discounts.thread.yaml")
gold = UseCaseSpec.from_file(EXAMPLES_DIR / "t01_discounts.gold.yaml")

report = evaluate(thread, gold, k=args.reps, judge_config=judge_config)
path = save_report(
    report,
    judge_config,
    results_dir=RESULTS_DIR,
    experiment="recall-precision",
    case_id=f"t01_discounts_k{args.reps}",
)
print(f"Report saved to {path}")
