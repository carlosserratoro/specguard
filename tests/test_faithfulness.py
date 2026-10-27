# Copyright 2026 Carlos Serra-Toro
# SPDX-License-Identifier: Apache-2.0

import threading
from collections.abc import Sequence
from typing import Any
from unittest.mock import Mock

from specguard.service_layer.faithfulness import JudgeCallError, VerifierFn, check, check_majority


def test_check_counts_verdicts_and_stores_records_unchanged():
    record = {"verdict": {"verdict": "supported"}, "call_metrics": {}}
    report = check("A rule.", "LABEL", lambda rule: ("supported", record), k=3)

    assert set(report) == {"label", "statement", "k", "verdict_counts", "errors", "runs"}
    assert report["verdict_counts"] == {"supported": 3, "contradicted": 0, "not_stated": 0}
    assert report["runs"] == [record] * 3
    assert report["errors"] == 0


def test_check_stores_the_original_error_type():
    error = JudgeCallError("boom")
    error.__cause__ = ValueError("error")
    flaky = Mock(side_effect=[error, ("supported", {})])

    report = check("A rule.", "LABEL", flaky, k=2)

    assert report["errors"] == 1
    assert report["runs"][0] == {"error": {"type": "ValueError", "message": "error"}}


WARM_UP = ["supported"]  # Consumed by the warm-up call and never stored.


def _scripted(outcomes: Sequence[str | Exception]) -> VerifierFn:
    lock, it = threading.Lock(), iter(outcomes)

    def verify(rule) -> tuple[str, dict[str, Any]]:
        with lock:
            outcome = next(it)
        if isinstance(outcome, Exception):
            raise JudgeCallError(str(outcome)) from outcome
        return outcome, {"verdict": outcome}

    return verify


def test_majority_takes_the_most_frequent_verdict():
    verify = _scripted([*WARM_UP, "supported", "supported", "not_stated"])
    report = check_majority("A rule.", "LABEL", verify, groups=1, group_size=3)
    assert report["verdict_counts"]["supported"] == 1
    assert report["call_verdict_counts"] == {"supported": 2, "contradicted": 0, "not_stated": 1}
    assert len(report["decisions"][0]["runs"]) == 3


def test_majority_records_a_tie_as_no_decision():
    verify = _scripted([*WARM_UP, "supported", "not_stated"])
    report = check_majority("A rule.", "LABEL", verify, groups=1, group_size=2)
    assert report["ties"] == 1
    assert sum(report["verdict_counts"].values()) == 0


def test_majority_ignores_failed_calls():
    verify = _scripted([*WARM_UP, "supported", ValueError("error"), "supported"])
    report = check_majority("A rule.", "LABEL", verify, groups=1, group_size=3)
    assert report["errors"] == 1
    assert report["verdict_counts"]["supported"] == 1


def test_majority_does_not_store_the_warm_up_call():
    verify = _scripted(["contradicted"] + ["supported"] * 3)
    report = check_majority("A rule.", "LABEL", verify, groups=1, group_size=3)
    assert report["warm_up_calls"] == 1
    assert report["call_verdict_counts"]["contradicted"] == 0
    assert len(report["decisions"][0]["runs"]) == 3
