from __future__ import annotations

from collections import Counter

from paths import PROJECT_ROOT, RESULTS_DIR

from specguard.adapters.llm import DEFAULT_JUDGE_MODEL, DEFAULT_JUDGE_TEMPERATURE, JudgeConfig
from specguard.adapters.reporting import save_report
from specguard.domain.spec import BusinessRule, Severity
from specguard.domain.thread import ConversationalThread
from specguard.service_layer.metrics import verify_rule_against_thread

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


def check(
    statement: str,
    label: str,
    thread: ConversationalThread,
    judge_config: JudgeConfig,
    k: int = 20,
) -> dict:

    report = {
        "label": label,
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
        verdict, call_metrics = verify_rule_against_thread(rule, thread, judge_config)
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
    print(f"{label} {judge_config.label()} verdict_counts: {dict(verdict_counts)}\n")

    return report


thread = ConversationalThread.from_file(PROJECT_ROOT / "examples" / "t01_discounts.thread.yaml")


for reasoning_effort in REASONING_EFFORTS:
    judge_config = JudgeConfig(
        model=DEFAULT_JUDGE_MODEL,
        temperature=DEFAULT_JUDGE_TEMPERATURE,
        reasoning_effort=reasoning_effort,
    )

    report_permission_form = check(
        "When a premium customer's discount is up to 20%, "
        "the system shall allow confirmation without approval.",
        "PERMISSION-FORM",
        thread,
        judge_config=judge_config,
        k=REPS_PERMISSION_FORM,
    )
    save_report(
        report_permission_form,
        judge_config,
        results_dir=RESULTS_DIR,
        experiment="faithfulness",
        case_id=f"t01_discounts_PERMISSION-FORM_k{REPS_PERMISSION_FORM}",
    )

    report_restriction_form = check(
        "When a discount exceeds 20% for a premium customer, "
        "the system shall require approval before confirming.",
        "RESTRICTION-FORM",
        thread,
        judge_config=judge_config,
        k=REPS_RESTRICTION_FORM,
    )
    save_report(
        report_restriction_form,
        judge_config,
        results_dir=RESULTS_DIR,
        experiment="faithfulness",
        case_id=f"t01_discounts_RESTRICTION-FORM_k{REPS_RESTRICTION_FORM}",
    )

    report_false_threshold = check(
        "Premium customers are allowed up to 35% without approval.",
        "FALSE-THRESHOLD",
        thread,
        judge_config=judge_config,
        k=REPS_FALSE_THRESHOLD,
    )
    save_report(
        report_false_threshold,
        judge_config,
        results_dir=RESULTS_DIR,
        experiment="faithfulness",
        case_id=f"t01_discounts_FALSE-THRESHOLD_k{REPS_FALSE_THRESHOLD}",
    )
