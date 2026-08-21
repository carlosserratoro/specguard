from unittest.mock import patch

from specguard.domain.spec import BusinessRule, Severity
from specguard.service_layer.evaluation import SupportVerdict, rule_faithfulness

DUMMY_VAL = "."


def _rule(id: str) -> BusinessRule:
    return BusinessRule(id=id, statement=DUMMY_VAL, source_refs=[], severity=Severity.critical)


def _verdict(verdict: str) -> SupportVerdict:
    return SupportVerdict(verdict=verdict, supporting_refs=[], reasoning=DUMMY_VAL)


def test_faithfulness_all_supported():
    """Every rule extracted is supported by the thread."""
    predicted = [_rule("BR1"), _rule("BR2")]
    with patch(
        "specguard.service_layer.evaluation.verify_rule_against_thread",
        return_value=_verdict("supported"),
    ):
        result = rule_faithfulness(predicted, thread=None)
    assert result["faithfulness"] == 1.0
    assert result["contradicted_rule_ids"] == []
    assert result["not_stated_rule_ids"] == []


def test_faithfulness_empty_predicted():
    """No rules extracted: no faithfulness."""
    with patch(
        "specguard.service_layer.evaluation.verify_rule_against_thread",
    ) as m:
        result = rule_faithfulness([], thread=None)
    assert result["faithfulness"] == 0.0
    assert result["contradicted_rule_ids"] == []
    assert result["not_stated_rule_ids"] == []
    m.assert_not_called()


def test_faithfulness_sparates_verdicts():
    predicted = [_rule("BR1"), _rule("BR2"), _rule("BR3"), _rule("BR4")]
    verdicts = [
        _verdict("supported"),
        _verdict("contradicted"),
        _verdict("not_stated"),
        _verdict("supported"),
    ]
    with patch(
        "specguard.service_layer.evaluation.verify_rule_against_thread",
        side_effect=verdicts,
    ):
        result = rule_faithfulness(predicted, thread=None)
    assert result["faithfulness"] == 0.5
    assert result["contradicted_rule_ids"] == ["BR2"]
    assert result["not_stated_rule_ids"] == ["BR3"]
