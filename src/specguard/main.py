from __future__ import annotations

import json
from pathlib import Path

from specguard.domain.spec import UseCaseSpec
from specguard.service_layer.evaluation import evaluate_rule_precision, evaluate_rule_recall

from .domain.thread import ConversationalThread

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    thread = ConversationalThread.from_file(PROJECT_ROOT / "examples" / "t01_discounts.thread.yaml")
    gold = UseCaseSpec.from_file(PROJECT_ROOT / "examples" / "t01_discounts.gold.yaml")

    report_recall = evaluate_rule_recall(thread, gold, k=5)
    print("RECALL:")
    print(json.dumps(report_recall, indent=2))

    report_precision = evaluate_rule_precision(thread, gold, k=5)
    print("PRECISION:")
    print(json.dumps(report_precision, indent=2))


if __name__ == "__main__":
    main()
