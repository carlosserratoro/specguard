from unittest.mock import patch

from specguard.domain.spec import BusinessRule, Severity
from specguard.service_layer.metrics import MatchDecision, rule_precision

DUMMY_VAL = "."


def _rule(id: str) -> BusinessRule:
    return BusinessRule(id=id, statement=DUMMY_VAL, source_refs=[], severity=Severity.critical)


def test_precision_all_supported():
    """Perfect precision."""
    predicted = [_rule("BR1"), _rule("BR2")]
    gold = [_rule("G1"), _rule("G2")]
    with patch(
        "specguard.service_layer.metrics.match_rule",
        autospec=True,
        return_value=MatchDecision(gold_rule_id=DUMMY_VAL, matched=True, reasoning=DUMMY_VAL),
    ):
        result = rule_precision(predicted, gold)
    assert result["precision"] == 1.0
    assert result["hallucinated_rule_ids"] == []


def test_precision_with_one_hallucination():
    """One rule is hallucinated: precision not perfect."""
    predicted = [_rule("BR1"), _rule("BR2")]
    gold = [_rule("G1")]
    decisions = [
        MatchDecision(gold_rule_id=DUMMY_VAL, matched=True, reasoning=DUMMY_VAL),
        MatchDecision(gold_rule_id=DUMMY_VAL, matched=False, reasoning=DUMMY_VAL),
    ]
    with patch(
        "specguard.service_layer.metrics.match_rule",
        autospec=True,
        side_effect=decisions,
    ):
        result = rule_precision(predicted, gold)
    assert result["precision"] == 0.5
    assert result["hallucinated_rule_ids"] == ["BR2"]


def test_precision_empty_predicted():
    """No predicion made: precision is zero."""
    with patch(
        "specguard.service_layer.metrics.match_rule",
        autospec=True,
    ) as m:
        result = rule_precision(predicted_rules=[], gold_rules=[_rule("G1")])
    assert result["precision"] == 0
    m.assert_not_called()  # No predicted rules, tus match_rule is not called.
