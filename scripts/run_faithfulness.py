from __future__ import annotations

import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from specguard.domain.spec import BusinessRule, Severity
from specguard.domain.thread import ConversationalThread
from specguard.llm import DEFAULT_JUDGE_MODEL, DEFAULT_JUDGE_TEMPERATURE
from specguard.service_layer.evaluation import verify_rule_against_thread

REPS_PERMISSION_FORM = 10
REPS_RESTRICTION_FORM = 10
REPS_FALSE_THRESHOLD = 10

REASONING_EFFORTS = (
    "none",
    "low",
    "medium",
    "high",
    "xhigh",
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"


def save_report(report: dict) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    model = report["model"].replace("/", "-")
    effort = report.get("reasoning_effort", "default")
    path = RESULTS_DIR / (
        f"run_faithfulness_isolated_{report['label']}_{model}_effort-{effort}_{timestamp}.json"
    )
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return path


def check(
    statement: str,
    label: str,
    thread: ConversationalThread,
    model: str | None = None,
    temperature: float | None = None,
    reasoning_effort: str | None = None,
    k: int = 20,
) -> dict:

    report = {
        "label": label,
        "model": model,
        "temperature": temperature,
        "reasoning_effort": reasoning_effort,
        "statement": statement,
        "k": k,
        "runs": [],
    }

    rule = BusinessRule(
        id=label,
        statement=statement,
        source_refs=[],
        severity=Severity.critical,
        precedence=[],
    )

    verdicts = []
    for rep_no in range(k):
        verdict, call_metrics = verify_rule_against_thread(
            rule, thread, model, temperature=temperature, reasoning_effort=reasoning_effort
        )
        print(f"{label} repetition {rep_no + 1}/{k}: {verdict.verdict}")
        verdicts.append(verdict)

        report["runs"].append(
            {
                "verdict": verdict.model_dump(),
                "call_metrics": call_metrics.model_dump(),
            }
        )

    verdict_counts = Counter(v.verdict for v in verdicts)
    report.update({"verdict_counts": dict(verdict_counts)})
    print(f"{label} {model} effort={reasoning_effort} verdict_counts: {dict(verdict_counts)}\n")

    return report


thread = ConversationalThread.from_file(PROJECT_ROOT / "examples" / "t01_discounts.thread.yaml")

for reasoning_effort in REASONING_EFFORTS:
    report_permission_form = check(
        "When a premium customer's discount is up to 20%, "
        "the system shall allow confirmation without approval.",
        "PERMISSION-FORM",
        thread,
        model=DEFAULT_JUDGE_MODEL,
        temperature=DEFAULT_JUDGE_TEMPERATURE,
        reasoning_effort=reasoning_effort,
        k=REPS_PERMISSION_FORM,
    )
    save_report(report_permission_form)

    report_restriction_form = check(
        "When a discount exceeds 20% for a premium customer, "
        "the system shall require approval before confirming.",
        "RESTRICTION-FORM",
        thread,
        model=DEFAULT_JUDGE_MODEL,
        temperature=DEFAULT_JUDGE_TEMPERATURE,
        reasoning_effort=reasoning_effort,
        k=REPS_RESTRICTION_FORM,
    )
    save_report(report_restriction_form)

    report_false_threshold = check(
        "Premium customers are allowed up to 35% without approval.",
        "FALSE-THRESHOLD",
        thread,
        model=DEFAULT_JUDGE_MODEL,
        temperature=DEFAULT_JUDGE_TEMPERATURE,
        reasoning_effort=reasoning_effort,
        k=REPS_FALSE_THRESHOLD,
    )
    save_report(report_false_threshold)
