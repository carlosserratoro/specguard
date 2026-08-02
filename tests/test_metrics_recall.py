from unittest.mock import patch

from specguard.domain.spec import BusinessRule, Severity
from specguard.service_layer.evaluation import MatchDecision, rule_recall

DUMMY_VAL = "."


def _rule(id: str) -> BusinessRule:
    return BusinessRule(id=id, statement=DUMMY_VAL, source_refs=[], severity=Severity.critical)


def test_recall_all_supported():
    """Perfect recall."""
    predicted = [_rule("BR1"), _rule("BR2")]
    gold = [_rule("G1"), _rule("G2")]
    with patch(
        "specguard.service_layer.evaluation.match_rule",
        return_value=MatchDecision(gold_rule_id=DUMMY_VAL, matched=True, reasoning=DUMMY_VAL),
    ):
        result = rule_recall(gold, predicted)
    assert result["recall"] == 1.0
    assert result["missed_rule_ids"] == []


def test_recall_with_one_missing():
    """One rule is missing: recall not perfect."""
    predicted = [_rule("BR1")]
    gold = [_rule("G1"), _rule("G2")]
    decisions = [
        MatchDecision(gold_rule_id=DUMMY_VAL, matched=True, reasoning=DUMMY_VAL),  # G1: match
        MatchDecision(gold_rule_id=DUMMY_VAL, matched=False, reasoning=DUMMY_VAL),  # G2: no match
    ]
    with patch(
        "specguard.service_layer.evaluation.match_rule",
        side_effect=decisions,
    ):
        result = rule_recall(gold, predicted)
    assert result["recall"] == 0.5
    assert result["missed_rule_ids"] == ["G2"]


def test_recall_empty_gold():
    """No gold given: recall is zero.

    This is to test the impossible case (we should always have a gold)
    but it's given here for completeness.
    """
    with patch("specguard.service_layer.evaluation.match_rule") as m:
        result = rule_recall(gold_rules=[], candidate_rules=[_rule("BR1")])
    assert result["recall"] == 0
    m.assert_not_called()  # No gold rules, tus match_rule is not called.
