from unittest.mock import Mock

from faithfulness_runner import JudgeCallError, check


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
