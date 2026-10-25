import time
from collections import Counter
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from specguard.domain.definitions import VERDICT_ORDER
from specguard.domain.spec import BusinessRule, Severity

# Stop a batch if the API keeps failing, e.g. bad key, invalid prompt.
MAX_CONSECUTIVE_ERRORS = 5


class JudgeCallError(Exception):
    """A judge call failed."""


# A verifier takes a business rule and returns a verdict label and a record to store for this run.
Verifier = Callable[[BusinessRule], tuple[str, dict[str, Any]]]


def rule_from_statement(statement: str, label: str) -> BusinessRule:
    return BusinessRule(
        id=label, statement=statement, source_refs=[], severity=Severity.critical, precedence=[]
    )


def check(
    statement: str,
    label: str,
    verify: Verifier,
    k: int = 20,
) -> dict[str, Any]:

    report: dict[str, Any] = {
        "label": label,
        "statement": statement,
        "k": k,
        "verdict_counts": {},
        "errors": 0,
        "runs": [],
    }

    rule = rule_from_statement(statement, label)

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


def _call(verify: Verifier, rule: BusinessRule) -> tuple[str | None, dict[str, Any]]:
    """One call that never raises: a failure comes back as an error record."""
    try:
        return verify(rule)
    except JudgeCallError as exc:
        cause = exc.__cause__ or exc
        return None, {"error": {"type": type(cause).__name__, "message": str(cause)}}


def check_majority(
    statement: str, label: str, verify: Verifier, groups: int, group_size: int = 10
) -> dict[str, Any]:
    """Each decision is the majority verdict of `group_size` concurrent calls.

    One warm-up call is made first, in the calling thread, and discarded: Instructor
    fills its handler registry on the first call, and doing that from several threads
    at once fails.
    """
    rule = rule_from_statement(statement, label)
    _call(verify, rule)  # warm-up, not stored

    report: dict[str, Any] = {
        "label": label,
        "statement": statement,
        "groups": groups,
        "group_size": group_size,
        "warm_up_calls": 1,
        "verdict_counts": {},  # one majority verdict per group
        "call_verdict_counts": {},  # every individual call
        "ties": 0,
        "errors": 0,
        "decisions": [],
    }
    majorities, call_verdicts = [], []

    with ThreadPoolExecutor(max_workers=group_size) as pool:
        for group_no in range(groups):
            start = time.perf_counter()
            results = list(pool.map(lambda _: _call(verify, rule), range(group_size)))
            wall_time_s = time.perf_counter() - start

            verdicts = [v for v, _ in results if v is not None]
            report["errors"] += group_size - len(verdicts)
            if not verdicts:
                raise RuntimeError(f"{label} group {group_no + 1}: every call failed")

            ranked = Counter(verdicts).most_common()
            tie = len(ranked) > 1 and ranked[0][1] == ranked[1][1]
            majority = None if tie else ranked[0][0]
            report["ties"] += tie
            if majority:
                majorities.append(majority)
            call_verdicts.extend(verdicts)

            report["decisions"].append(
                {"majority": majority, "wall_time_s": wall_time_s, "runs": [r for _, r in results]}
            )
            print(f"{label} group {group_no + 1}/{groups}: {majority} ({wall_time_s:.2f} s)")

    report["verdict_counts"] = {v: Counter(majorities)[v] for v in VERDICT_ORDER}
    report["call_verdict_counts"] = {v: Counter(call_verdicts)[v] for v in VERDICT_ORDER}
    return report
