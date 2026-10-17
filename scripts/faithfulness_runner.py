from collections import Counter
from collections.abc import Callable

from specguard.domain.spec import BusinessRule, Severity

THRESHOLD_STATEMENTS = {
    "PERMISSION-FORM": "When a premium customer's discount is up to 20%, "
    "the system shall allow confirmation without approval.",
    "RESTRICTION-FORM": "When a discount exceeds 20% for a premium customer, "
    "the system shall require approval before confirming.",
    "FALSE-THRESHOLD": "Premium customers are allowed up to 35% without approval.",
}

VERDICT_ORDER = ("supported", "contradicted", "not_stated")

# Stop a batch if the API keeps failing, e.g. bad key, invalid prompt.
MAX_CONSECUTIVE_ERRORS = 5


class JudgeCallError(Exception):
    """A judge call failed."""


# A verifier takes a business rule and returns a verdict label and a record to store for this run.
Verifier = Callable[[BusinessRule], tuple[str, dict]]


def check(
    statement: str,
    label: str,
    verify: Verifier,
    k: int = 20,
) -> dict:

    report = {
        "label": label,
        "statement": statement,
        "k": k,
        "verdict_counts": {},
        "errors": 0,
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
    consecutive_errors = 0
    for rep_no in range(k):
        try:
            verdict, record = verify(rule)
        except JudgeCallError as exc:
            # Record the error and keep going, unless the API keeps failing.
            cause = exc.__cause__ or exc
            consecutive_errors += 1
            report["errors"] += 1
            report["runs"].append({"error": {"type": type(cause).__name__, "message": str(cause)}})
            print(f"{label} repetition {rep_no + 1}/{k}: ERROR {str(exc)[:120]}")
            if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                raise RuntimeError(f"{MAX_CONSECUTIVE_ERRORS} consecutive errors") from exc
        else:
            consecutive_errors = 0
            print(f"{label} repetition {rep_no + 1}/{k}: {verdict}")
            verdicts.append(verdict)
            report["runs"].append(record)

    counts = Counter(verdicts)
    report["verdict_counts"] = {v: counts[v] for v in VERDICT_ORDER}
    print(f"{label} verdict_counts: {report['verdict_counts']}, errors: {report['errors']}\n")

    return report
