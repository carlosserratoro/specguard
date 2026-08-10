from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from specguard.domain.spec import UseCaseSpec
from specguard.domain.thread import ConversationalThread
from specguard.service_layer.evaluation import evaluate

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"


def save_report(report: dict, case_id: str) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    k = report["k"]
    temp = report["temperature"]
    path = RESULTS_DIR / f"run_recall_precision_{case_id}_k{k}_temp{temp}_{timestamp}.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return path


thread = ConversationalThread.from_file(PROJECT_ROOT / "examples" / "t01_discounts.thread.yaml")
gold = UseCaseSpec.from_file(PROJECT_ROOT / "examples" / "t01_discounts.gold.yaml")

report = evaluate(thread, gold, k=20)
path = save_report(report, case_id="t01_discounts")
print(f"Report saved to {path}")
