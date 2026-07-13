from __future__ import annotations

import json
from pathlib import Path

from specguard.domain.spec import UseCaseSpec
from specguard.service_layer.evaluation import evaluate_completeness

from .domain.thread import ConversationalThread

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    thread = ConversationalThread.from_file(PROJECT_ROOT / "examples" / "t01_discounts.thread.yaml")
    gold = UseCaseSpec.from_file(PROJECT_ROOT / "examples" / "t01_discounts.gold.yaml")

    report = evaluate_completeness(thread, gold, k=10)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
