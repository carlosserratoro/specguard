from __future__ import annotations

from cli import build_parser, judge_config_from_args
from paths import EXAMPLES_DIR, RESULTS_DIR

from specguard.adapters.reporting import save_report
from specguard.adapters.statements import load_statements
from specguard.bootstrap import load_environment, make_generative_verifier
from specguard.domain.thread import ConversationalThread
from specguard.service_layer.faithfulness import check

load_environment()

parser = build_parser("Measures faithfulness over the threshold cases.")
args = parser.parse_args()
judge_config = judge_config_from_args(args)

thread = ConversationalThread.from_file(EXAMPLES_DIR / "t01_discounts.thread.yaml")

verify = make_generative_verifier(thread, judge_config)

THRESHOLD_STATEMENTS = load_statements(EXAMPLES_DIR / "t01_threshold_statements.yaml")

report_permission_form = check(
    THRESHOLD_STATEMENTS["PERMISSION-FORM"], "PERMISSION-FORM", verify, k=args.reps
)
save_report(
    report_permission_form,
    judge_config,
    results_dir=RESULTS_DIR,
    experiment="faithfulness",
    case_id=f"t01_discounts_PERMISSION-FORM_k{args.reps}",
)

report_restriction_form = check(
    THRESHOLD_STATEMENTS["RESTRICTION-FORM"], "RESTRICTION-FORM", verify, k=args.reps
)
save_report(
    report_restriction_form,
    judge_config,
    results_dir=RESULTS_DIR,
    experiment="faithfulness",
    case_id=f"t01_discounts_RESTRICTION-FORM_k{args.reps}",
)

report_false_threshold = check(
    THRESHOLD_STATEMENTS["FALSE-THRESHOLD"], "FALSE-THRESHOLD", verify, k=args.reps
)
save_report(
    report_false_threshold,
    judge_config,
    results_dir=RESULTS_DIR,
    experiment="faithfulness",
    case_id=f"t01_discounts_FALSE-THRESHOLD_k{args.reps}",
)
