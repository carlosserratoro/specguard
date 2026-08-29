from __future__ import annotations

from collections import Counter

from cli import build_parser, judge_config_from_args
from paths import EXAMPLES_DIR, RESULTS_DIR

from specguard.adapters.llm import JudgeConfig
from specguard.adapters.reporting import save_report
from specguard.domain.spec import BusinessRule, Severity
from specguard.domain.thread import ConversationalThread
from specguard.service_layer.metrics import verify_rule_against_thread


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


parser = build_parser("Measures faithfulness over the threshold cases.")
args = parser.parse_args()
judge_config = judge_config_from_args(args)

thread = ConversationalThread.from_file(EXAMPLES_DIR / "t01_discounts.thread.yaml")


report_permission_form = check(
    "When a premium customer's discount is up to 20%, "
    "the system shall allow confirmation without approval.",
    "PERMISSION-FORM",
    thread,
    judge_config=judge_config,
    k=args.reps,
)
save_report(
    report_permission_form,
    judge_config,
    results_dir=RESULTS_DIR,
    experiment="faithfulness",
    case_id=f"t01_discounts_PERMISSION-FORM_k{args.reps}",
)

report_restriction_form = check(
    "When a discount exceeds 20% for a premium customer, "
    "the system shall require approval before confirming.",
    "RESTRICTION-FORM",
    thread,
    judge_config=judge_config,
    k=args.reps,
)
save_report(
    report_restriction_form,
    judge_config,
    results_dir=RESULTS_DIR,
    experiment="faithfulness",
    case_id=f"t01_discounts_RESTRICTION-FORM_k{args.reps}",
)

report_false_threshold = check(
    "Premium customers are allowed up to 35% without approval.",
    "FALSE-THRESHOLD",
    thread,
    judge_config=judge_config,
    k=args.reps,
)
save_report(
    report_false_threshold,
    judge_config,
    results_dir=RESULTS_DIR,
    experiment="faithfulness",
    case_id=f"t01_discounts_FALSE-THRESHOLD_k{args.reps}",
)
